#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MACRO SWING LAB 4.8
VALIDAZIONE STATISTICA AVANZATA

Ricerca associativa NON point-in-time.

NON genera segnali di acquisto.
NON dimostra causalita' o capacita' predittiva.

Analisi:
- Tutte le osservazioni settimanali
- Regime contro altri regimi
- Test HAC/Newey-West
- Bootstrap per blocchi annuali
- Correzione FDR
- Verifica fuori campione
- Walk-forward
"""

from pathlib import Path

import numpy as np
import pandas as pd

import statsmodels.api as sm

from statsmodels.stats.multitest import multipletests


# ==========================================
# CONFIGURAZIONE
# ==========================================

OUT = Path(__file__).resolve().parent / "output"

SOURCE = OUT / "osservazioni_volatilita_47.csv"

SPLIT = pd.Timestamp("2022-01-01")

HORIZONS = (1, 2)

TARGETS = {
    "rialzo": "hit_rialzo",
    "espansione_1sigma": "mfe_ge_1_0sigma",
    "espansione_1_5sigma": "mfe_ge_1_5sigma",
    "espansione_2sigma": "mfe_ge_2_0sigma",
}

PRIMARY = "espansione_1_5sigma"

REGIMES = ("R1", "R2", "R3", "R4")

MIN_TRAIN = 30

MIN_OTHER_TRAIN = 30

MIN_TEST = 15

MIN_YEARS = 5

BOOT_REPS = 500

RNG = np.random.default_rng(481027)


# ==========================================
# CARICAMENTO DATI
# ==========================================

def load_data():

    if not SOURCE.is_file():

        raise FileNotFoundError(
            f"Manca {SOURCE}. "
            "Serve il CSV della versione 4.7."
        )

    df = pd.read_csv(
        SOURCE,
        parse_dates=[
            "data_segnale",
            "data_ingresso",
            "data_uscita"
        ]
    )

    required = {
        "asset",
        "regime",
        "settimane",
        "data_segnale",
        "data_ingresso",
        "data_uscita",
        "rendimento",
        "mfe",
        "mae",
        "mfe_sigma",
        "mfe_ge_1_0sigma",
        "mfe_ge_1_5sigma",
        "mfe_ge_2_0sigma"
    }

    missing = required - set(df.columns)

    if missing:

        raise ValueError(
            f"Colonne mancanti: {sorted(missing)}"
        )

    # Converte i nomi completi dei regimi
    # della versione 4.7 in R1, R2, R3, R4.

    df["regime_originale"] = (
        df["regime"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["regime"] = (
        df["regime_originale"]
        .str.extract(
            r"^(R[1-4])(?:_|$)",
            expand=False
        )
    )

    print(
        "Regimi riconosciuti:",
        df["regime"].dropna().unique()
    )

    if not set(df["regime"]).intersection(REGIMES):

        raise ValueError(
            "Nessun regime R1-R4 riconosciuto."
        )

    df["settimane"] = pd.to_numeric(
        df["settimane"],
        errors="raise"
    ).astype(int)

    df = df[
        df["settimane"].isin(HORIZONS)
        &
        df["regime"].isin(REGIMES)
    ].copy()

    numeric_columns = [
        "rendimento",
        "mfe",
        "mae",
        "mfe_sigma",
        "mfe_ge_1_0sigma",
        "mfe_ge_1_5sigma",
        "mfe_ge_2_0sigma"
    ]

    for col in numeric_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=list(required)
    ).copy()

    if df.empty:

        raise RuntimeError(
            "Nessuna osservazione utilizzabile."
        )

    keys = [
        "asset",
        "settimane",
        "data_segnale"
    ]

    if df.duplicated(keys).any():

        raise ValueError(
            "Osservazioni settimanali duplicate."
        )

    # Controllo cronologico.

    valid_dates = (
        (df["data_segnale"] < df["data_ingresso"])
        &
        (df["data_ingresso"] < df["data_uscita"])
    )

    if not valid_dates.all():

        raise ValueError(
            "Date cronologicamente incoerenti."
        )

    # Controllo indicatori binari.

    for col in [
        "mfe_ge_1_0sigma",
        "mfe_ge_1_5sigma",
        "mfe_ge_2_0sigma"
    ]:

        if not df[col].isin([0, 1]).all():

            raise ValueError(
                f"Colonna non binaria: {col}"
            )

    df["hit_rialzo"] = (
        df["rendimento"] > 0
    ).astype(int)

    df["anno"] = (
        df["data_segnale"].dt.year
    )

    df["periodo"] = np.where(
        df["data_segnale"] < SPLIT,
        "RICERCA",
        "VERIFICA"
    )

    # Evita che nel training entrino
    # risultati con uscita dopo lo split.

    df = df[
        (df["periodo"] != "RICERCA")
        |
        (df["data_uscita"] < SPLIT)
    ].copy()

    return df.sort_values(
        [
            "asset",
            "settimane",
            "data_segnale"
        ]
    ).reset_index(drop=True)


# ==========================================
# TEST STATISTICO HAC
# ==========================================

def hac_pvalue(g, col, regime, lag):

    """
    Test bilaterale sulla differenza
    delle frequenze tra un regime
    e gli altri tre.

    Errori standard HAC/Newey-West.
    """

    indicator = (
        g["regime"] == regime
    ).astype(float).to_numpy()

    outcome = (
        g[col]
        .astype(float)
        .to_numpy()
    )

    if (
        indicator.sum() < 2
        or
        (1 - indicator).sum() < 2
    ):

        return np.nan

    try:

        X = sm.add_constant(indicator)

        result = sm.OLS(
            outcome,
            X
        ).fit(
            cov_type="HAC",
            cov_kwds={
                "maxlags": int(lag)
            }
        )

        p = result.pvalues[1]

        if np.isfinite(p):

            return float(p)

        return np.nan

    except (
        ValueError,
        np.linalg.LinAlgError,
        FloatingPointError
    ):

        return np.nan


# ==========================================
# BOOTSTRAP PER BLOCCHI ANNUALI
# ==========================================

def year_block_ci(
    g,
    col,
    regime,
    reps=BOOT_REPS
):

    """
    Ricampiona interi anni.

    Preserva parte della dipendenza
    temporale delle osservazioni.

    Non garantisce copertura nominale
    con pochi anni o dati non stazionari.
    """

    years = sorted(
        g["anno"].unique()
    )

    if len(years) < 2:

        return np.nan, np.nan

    aggregates = []

    for year in years:

        subset = g[
            g["anno"] == year
        ]

        regime_data = subset[
            subset["regime"] == regime
        ][col]

        other_data = subset[
            subset["regime"] != regime
        ][col]

        aggregates.append(
            (
                float(regime_data.sum()),
                len(regime_data),
                float(other_data.sum()),
                len(other_data)
            )
        )

    arr = np.asarray(
        aggregates,
        dtype=float
    )

    picks = RNG.integers(
        0,
        len(arr),
        size=(
            reps,
            len(arr)
        )
    )

    samples = arr[picks].sum(
        axis=1
    )

    valid = (
        (samples[:, 1] > 0)
        &
        (samples[:, 3] > 0)
    )

    values = (
        samples[valid, 0]
        /
        samples[valid, 1]
        -
        samples[valid, 2]
        /
        samples[valid, 3]
    )

    if len(values) < max(
        50,
        reps // 2
    ):

        return np.nan, np.nan

    lower, upper = np.quantile(
        values,
        [0.025, 0.975]
    )

    return float(lower), float(upper)


# ==========================================
# CONFRONTO REGIME VS ALTRI REGIMI
# ==========================================

def period_metrics(
    g,
    col,
    regime
):

    selected = g[
        g["regime"] == regime
    ]

    others = g[
        g["regime"] != regime
    ]

    p_selected = (
        float(selected[col].mean())
        if len(selected)
        else np.nan
    )

    p_others = (
        float(others[col].mean())
        if len(others)
        else np.nan
    )

    return {
        "n_regime": len(selected),

        "n_altri": len(others),

        "anni_regime":
            selected["anno"].nunique(),

        "p_regime":
            p_selected,

        "p_altri":
            p_others,

        "delta":
            p_selected - p_others,

        "mfe_mediano_pct":
            100 * selected["mfe"].median()
            if len(selected)
            else np.nan,

        "mae_mediano_pct":
            100 * selected["mae"].median()
            if len(selected)
            else np.nan
    }


# ==========================================
# STABILITA' ANNUALE
# ==========================================

def annual_table(df):

    rows = []

    groups = df.groupby(
        [
            "asset",
            "settimane",
            "anno"
        ]
    )

    for (
        asset,
        weeks,
        year
    ), group in groups:

        for regime in REGIMES:

            metrics = period_metrics(
                group,
                TARGETS[PRIMARY],
                regime
            )

            rows.append({
                "asset": asset,
                "settimane": weeks,
                "anno": year,
                "regime": regime,
                **metrics
            })

    return pd.DataFrame(rows)


# ==========================================
# VERIFICA WALK-FORWARD
# ==========================================

def walk_forward(df):

    """
    Finestre temporali prefissate.

    Il training usa solo osservazioni
    con uscita precedente al periodo
    di verifica.
    """

    folds = [
        (2015, 2016),
        (2017, 2018),
        (2019, 2020),
        (2021, 2022),
        (2023, 2024),
        (2025, 2026)
    ]

    rows = []

    groups = df.groupby(
        [
            "asset",
            "settimane"
        ]
    )

    for (
        asset,
        weeks
    ), group in groups:

        for start, end in folds:

            cutoff = pd.Timestamp(
                f"{start}-01-01"
            )

            right = pd.Timestamp(
                f"{end + 1}-01-01"
            )

            train = group[
                (group["data_segnale"] < cutoff)
                &
                (group["data_uscita"] < cutoff)
            ]

            test = group[
                (group["data_segnale"] >= cutoff)
                &
                (group["data_segnale"] < right)
                &
                (group["data_uscita"] < right)
            ]

            for regime in REGIMES:

                train_stats = period_metrics(
                    train,
                    TARGETS[PRIMARY],
                    regime
                )

                test_stats = period_metrics(
                    test,
                    TARGETS[PRIMARY],
                    regime
                )

                evaluable = (
                    train_stats["n_regime"] >= MIN_TRAIN
                    and
                    train_stats["n_altri"] >= MIN_OTHER_TRAIN
                    and
                    test_stats["n_regime"] >= MIN_TEST
                )

                rows.append({
                    "asset": asset,

                    "settimane": weeks,

                    "regime": regime,

                    "fold":
                        f"{start}-{end}",

                    "n_train":
                        train_stats["n_regime"],

                    "n_test":
                        test_stats["n_regime"],

                    "delta_train":
                        train_stats["delta"],

                    "delta_test":
                        test_stats["delta"],

                    "valutabile":
                        evaluable
                })

    return pd.DataFrame(rows)


# ==========================================
# MOTORE PRINCIPALE
# ==========================================

def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 55)

    print(
        "MACRO SWING LAB 4.8"
    )

    print(
        "VALIDAZIONE STATISTICA AVANZATA"
    )

    print("=" * 55)

    df = load_data()

    print(
        "Osservazioni:",
        len(df)
    )

    print(
        "Asset:",
        df["asset"].nunique()
    )

    # Stabilita' annuale.

    annual = annual_table(df)

    # Validazione temporale.

    wf = walk_forward(df)

    records = []

    groups = df.groupby(
        [
            "asset",
            "settimane"
        ]
    )

    for (
        asset,
        weeks
    ), group in groups:

        train = group[
            group["periodo"] == "RICERCA"
        ]

        test = group[
            group["periodo"] == "VERIFICA"
        ]

        for regime in REGIMES:

            for name, col in TARGETS.items():

                a = period_metrics(
                    train,
                    col,
                    regime
                )

                b = period_metrics(
                    test,
                    col,
                    regime
                )

                enough_train = (
                    a["n_regime"] >= MIN_TRAIN
                    and
                    a["n_altri"] >= MIN_OTHER_TRAIN
                    and
                    a["anni_regime"] >= MIN_YEARS
                )

                # Finestra HAC predefinita.

                lag = (
                    2 if weeks == 1
                    else 4
                )

                if enough_train:

                    p = hac_pvalue(
                        train,
                        col,
                        regime,
                        lag
                    )

                    ci = year_block_ci(
                        train,
                        col,
                        regime
                    )

                else:

                    p = np.nan

                    ci = (
                        np.nan,
                        np.nan
                    )

                enough_test = (
                    b["n_regime"] >= MIN_TEST
                    and
                    b["n_altri"] >= MIN_TEST
                )

                records.append({
                    "asset": asset,

                    "settimane": weeks,

                    "regime": regime,

                    "target": name,

                    "n_train":
                        a["n_regime"],

                    "n_other_train":
                        a["n_altri"],

                    "years_train":
                        a["anni_regime"],

                    "p_train":
                        a["p_regime"],

                    "p_other_train":
                        a["p_altri"],

                    "delta_train":
                        a["delta"],

                    "ci95_low":
                        ci[0],

                    "ci95_high":
                        ci[1],

                    "p_hac":
                        p,

                    "n_test":
                        b["n_regime"],

                    "n_other_test":
                        b["n_altri"],

                    "p_test":
                        b["p_regime"],

                    "p_other_test":
                        b["p_altri"],

                    "delta_test":
                        b["delta"],

                    "mfe_mediano_test_pct":
                        b["mfe_mediano_pct"],

                    "mae_mediano_test_pct":
                        b["mae_mediano_pct"],

                    "sufficiente_train":
                        enough_train,

                    "sufficiente_test":
                        enough_test
                })

    results = pd.DataFrame(
        records
    )

    # ======================================
    # CORREZIONE CONFRONTI MULTIPLI
    # ======================================

    # La famiglia primaria di test riguarda
    # solo MFE >= 1,5 sigma.

    results["q_bh_primario"] = np.nan

    mask = (
        (results["target"] == PRIMARY)
        &
        results["p_hac"].notna()
    )

    if mask.any():

        adjusted = multipletests(
            results.loc[
                mask,
                "p_hac"
            ].to_numpy(),
            method="fdr_bh"
        )[1]

        results.loc[
            mask,
            "q_bh_primario"
        ] = adjusted

    # ======================================
    # CLASSIFICAZIONE RISULTATI
    # ======================================

    primary = results[
        results["target"] == PRIMARY
    ].copy()

    quality = []

    for _, row in primary.iterrows():

        eligible = (
            bool(row["sufficiente_train"])
            and
            bool(row["sufficiente_test"])
        )

        if not eligible:

            label = (
                "CAMPIONE_INSUFFICIENTE"
            )

        elif (
            row["delta_train"] > 0
            and
            row["delta_test"] > 0
            and
            row["ci95_low"] > 0
            and
            row["q_bh_primario"] <= 0.05
        ):

            label = (
                "SUPPORTATA_CON_LIMITI_NON_PIT"
            )

        elif (
            row["delta_train"] > 0
            and
            row["delta_test"] > 0
        ):

            label = (
                "PROMETTENTE_NON_CONFERMATA"
            )

        else:

            label = (
                "NON_CONFERMATA"
            )

        quality.append(label)

    primary["valutazione"] = quality

    # ======================================
    # RANKING DI RICERCA
    # ======================================

    primary["ordine_ricerca"] = (
        primary.groupby(
            [
                "regime",
                "settimane"
            ]
        )["delta_train"]
        .rank(
            method="min",
            ascending=False
        )
    )

    # ======================================
    # ESPORTAZIONE
    # ======================================

    annual.to_csv(
        OUT / "stabilita_annuale_48.csv",
        index=False
    )

    wf.to_csv(
        OUT / "walk_forward_48.csv",
        index=False
    )

    results.to_csv(
        OUT / "test_statistici_48.csv",
        index=False
    )

    primary = primary.sort_values(
        [
            "regime",
            "settimane",
            "ordine_ricerca"
        ]
    )

    primary.to_csv(
        OUT / "ranking_validato_48.csv",
        index=False
    )

    print("\nRISULTATI")

    print(
        "Test effettuati:",
        len(results)
    )

    print(
        "Combinazioni primarie:",
        len(primary)
    )

    print(
        "\nClassificazione:"
    )

    print(
        primary["valutazione"]
        .value_counts()
        .to_string()
    )

    print(
        "\nFILE GENERATI:"
    )

    print(
        "output/stabilita_annuale_48.csv"
    )

    print(
        "output/walk_forward_48.csv"
    )

    print(
        "output/test_statistici_48.csv"
    )

    print(
        "output/ranking_validato_48.csv"
    )

    print(
        "\nATTENZIONE:"
    )

    print(
        "Ricerca storica NON point-in-time."
    )

    print(
        "I risultati non sono segnali operativi."
    )

    print(
        "\nMACRO SWING LAB 4.8 COMPLETATO"
    )


if __name__ == "__main__":
    main()
