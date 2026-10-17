from pathlib import Path

import numpy as np
import pandas as pd

# MACRO SWING LAB 4.1
# Validazione statistica dei modelli 4.0

OUTPUT = Path("output")
INPUT = OUTPUT / "scelte_modelli_40.csv"

MODELS = ["MACRO", "MOMENTUM", "IBRIDO"]

COMPARISONS = [
    ("MOMENTUM", "MACRO"),
    ("IBRIDO", "MACRO"),
    ("MOMENTUM", "IBRIDO"),
]

N_BOOTSTRAP = 10000
BLOCK_SIZE = 5
HORIZON = 20
SEED = 41


def load_data():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"File non trovato: {INPUT}"
        )

    df = pd.read_csv(INPUT)

    required = {
        "data_segnale",
        "modello",
        "orizzonte",
        "rendimento",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Colonne mancanti: {sorted(missing)}"
        )

    df["data_segnale"] = pd.to_datetime(
        df["data_segnale"]
    )

    df = df[
        (df["orizzonte"] == HORIZON)
        & (df["modello"].isin(MODELS))
    ].copy()

    if df.duplicated(
        ["data_segnale", "modello"]
    ).any():
        raise ValueError(
            "Presenti date duplicate per modello"
        )

    pivot = df.pivot(
        index="data_segnale",
        columns="modello",
        values="rendimento",
    )

    missing_models = [
        m for m in MODELS
        if m not in pivot.columns
    ]

    if missing_models:
        raise ValueError(
            f"Modelli mancanti: {missing_models}"
        )

    pivot = (
        pivot[MODELS]
        .dropna()
        .sort_index()
    )

    if len(pivot) < 30:
        raise ValueError(
            "Campione insufficiente per la validazione"
        )

    return pivot


def moving_block_bootstrap(
    values,
    block_size=BLOCK_SIZE,
    n_bootstrap=N_BOOTSTRAP,
    seed=SEED,
):
    values = np.asarray(values, dtype=float)
    n = len(values)

    if n < 2:
        return np.nan, np.nan, np.nan

    block_size = min(block_size, n)

    rng = np.random.default_rng(seed)

    possible_starts = np.arange(
        n - block_size + 1
    )

    bootstrap_means = np.empty(n_bootstrap)

    blocks_needed = int(
        np.ceil(n / block_size)
    )

    for i in range(n_bootstrap):
        starts = rng.choice(
            possible_starts,
            size=blocks_needed,
            replace=True,
        )

        indices = np.concatenate([
            np.arange(
                start,
                start + block_size
            )
            for start in starts
        ])[:n]

        bootstrap_means[i] = (
            values[indices].mean()
        )

    ci_low, ci_high = np.quantile(
        bootstrap_means,
        [0.025, 0.975],
    )

    return (
        float(ci_low),
        float(ci_high),
        float(
            np.mean(bootstrap_means <= 0)
        ),
    )


def compare_models(pivot, label):
    results = []

    for a, b in COMPARISONS:
        difference = (
            pivot[a] - pivot[b]
        ).dropna()

        if len(difference) < 2:
            continue

        ci_low, ci_high, bootstrap_prob = (
            moving_block_bootstrap(
                difference.values
            )
        )

        results.append({
            "campione": label,
            "modello_a": a,
            "modello_b": b,
            "n": len(difference),
            "media_a_pct": (
                pivot.loc[
                    difference.index, a
                ].mean() * 100
            ),
            "media_b_pct": (
                pivot.loc[
                    difference.index, b
                ].mean() * 100
            ),
            "differenza_media_pct": (
                difference.mean() * 100
            ),
            "mediana_differenza_pct": (
                difference.median() * 100
            ),
            "percentuale_a_superiore": (
                (difference > 0).mean() * 100
            ),
            "ic95_basso_pct": ci_low * 100,
            "ic95_alto_pct": ci_high * 100,
            "prob_bootstrap_media_nonpositiva": (
                bootstrap_prob
            ),
            "ic95_esclude_zero": bool(
                ci_low > 0 or ci_high < 0
            ),
        })

    return results


def annual_analysis(pivot):
    rows = []

    for year, group in pivot.groupby(
        pivot.index.year
    ):
        for model in MODELS:
            returns = group[model]

            rows.append({
                "anno": int(year),
                "modello": model,
                "n": len(returns),
                "rendimento_medio_pct": (
                    returns.mean() * 100
                ),
                "rendimento_mediano_pct": (
                    returns.median() * 100
                ),
                "percentuale_positivi": (
                    (returns > 0).mean() * 100
                ),
            })

    return pd.DataFrame(rows)


def non_overlapping_sample(pivot):
    """
    Seleziona un segnale ogni 20 sedute circa.

    Le date disponibili sono settimanali:
    vengono usate almeno 28 giornate
    di calendario tra i segnali.

    Questa e' una verifica conservativa,
    non un portafoglio simulato.
    """
    selected = []
    last_date = None

    for date in pivot.index:
        if (
            last_date is None
            or (date - last_date).days >= 28
        ):
            selected.append(date)
            last_date = date

    return pivot.loc[selected]


def save_report(pivot, comparisons, annual, sparse):
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    comparison_df = pd.DataFrame(
        comparisons
    )

    comparison_df.to_csv(
        OUTPUT / "validazione_41.csv",
        index=False,
    )

    annual.to_csv(
        OUTPUT / "validazione_41_annuale.csv",
        index=False,
    )

    sparse.reset_index().to_csv(
        OUTPUT / "validazione_41_non_overlap.csv",
        index=False,
    )

    differences = pivot.copy()

    differences["momentum_meno_macro"] = (
        pivot["MOMENTUM"] - pivot["MACRO"]
    )

    differences["ibrido_meno_macro"] = (
        pivot["IBRIDO"] - pivot["MACRO"]
    )

    differences["momentum_meno_ibrido"] = (
        pivot["MOMENTUM"] - pivot["IBRIDO"]
    )

    differences.reset_index().to_csv(
        OUTPUT / "validazione_41_differenze.csv",
        index=False,
    )

    return comparison_df


def main():
    print("=" * 60)
    print("MACRO SWING LAB 4.1")
    print("VALIDAZIONE STATISTICA")
    print("=" * 60)

    pivot = load_data()

    print(
        f"Date comuni analizzate: {len(pivot)}"
    )

    print(
        f"Periodo: "
        f"{pivot.index.min().date()} - "
        f"{pivot.index.max().date()}"
    )

    comparisons = compare_models(
        pivot,
        "TUTTE_LE_DATE",
    )

    annual = annual_analysis(pivot)

    sparse = non_overlapping_sample(
        pivot
    )

    print(
        f"Date campione distanziato: "
        f"{len(sparse)}"
    )

    comparisons += compare_models(
        sparse,
        "NON_OVERLAP",
    )

    result = save_report(
        pivot,
        comparisons,
        annual,
        sparse,
    )

    print("\nRISULTATI:")
    print(
        result[
            [
                "campione",
                "modello_a",
                "modello_b",
                "n",
                "differenza_media_pct",
                "ic95_basso_pct",
                "ic95_alto_pct",
                "ic95_esclude_zero",
            ]
        ].to_string(index=False)
    )

    print("\nValidazione 4.1 completata.")


if __name__ == "__main__":
    main()
