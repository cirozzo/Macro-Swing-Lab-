#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MACRO SWING LAB 4.3
VALIDAZIONE ROBUSTEZZA MOMENTUM

Analisi:
1. Campione completo
2. Segnali distanziati di almeno 28 giorni
3. Esclusione ETF SLV
4. Esclusione estremi del 5%
5. Confronto Momentum vs SPY
6. Analisi per regime
7. Analisi annuale

INPUT:
output/regimi_momentum_osservazioni_42.csv

OUTPUT:
output/robustezza_43.csv
output/robustezza_regimi_43.csv
output/robustezza_annuale_43.csv
output/robustezza_non_overlap_43.csv

ATTENZIONE:
- Ricerca esplorativa.
- Non simula un portafoglio negoziabile.
- Il campione distanziato riduce la sovrapposizione,
  ma non garantisce indipendenza statistica.
- Non include commissioni o slippage.
- I dati macro storici sono revisionati.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ==========================================
# CONFIGURAZIONE
# ==========================================

OUTPUT = Path("output")

INPUT = (
    OUTPUT
    / "regimi_momentum_osservazioni_42.csv"
)

HORIZON = 20

MIN_DAYS_BETWEEN = 28

TRIM_PERCENT = 0.05

BOOTSTRAP_REPS = 10000

BLOCK_SIZE = 5

SEED = 43


# ==========================================
# CARICAMENTO DATI
# ==========================================

def load_data():

    if not INPUT.exists():

        raise FileNotFoundError(
            f"File non trovato: {INPUT}"
        )

    df = pd.read_csv(INPUT)

    required = {
        "data_segnale",
        "modello",
        "asset",
        "regime",
        "orizzonte",
        "rendimento",
        "rendimento_spy"
    }

    missing = required - set(df.columns)

    if missing:

        raise ValueError(
            f"Colonne mancanti: {sorted(missing)}"
        )

    df["data_segnale"] = pd.to_datetime(
        df["data_segnale"]
    )

    numeric_columns = [
        "orizzonte",
        "rendimento",
        "rendimento_spy"
    ]

    for col in numeric_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df[
        (df["modello"] == "MOMENTUM")
        &
        (df["orizzonte"] == HORIZON)
    ].copy()

    df = df.dropna(
        subset=[
            "data_segnale",
            "asset",
            "regime",
            "rendimento",
            "rendimento_spy"
        ]
    )

    if df.empty:

        raise ValueError(
            "Nessuna osservazione Momentum."
        )

    if df.duplicated(
        "data_segnale"
    ).any():

        raise ValueError(
            "Date del segnale duplicate."
        )

    df = df.sort_values(
        "data_segnale"
    ).reset_index(drop=True)

    df["excess"] = (
        df["rendimento"]
        -
        df["rendimento_spy"]
    )

    df["anno"] = (
        df["data_segnale"].dt.year
    )

    return df


# ==========================================
# BOOTSTRAP
# ==========================================

def bootstrap_ci(
    values,
    block_size=BLOCK_SIZE,
    seed=SEED
):

    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    n = len(values)

    if n < 3:

        return np.nan, np.nan

    rng = np.random.default_rng(seed)

    block_size = min(
        block_size,
        n
    )

    starts = np.arange(
        n - block_size + 1
    )

    blocks_needed = int(
        np.ceil(n / block_size)
    )

    means = np.empty(
        BOOTSTRAP_REPS
    )

    for i in range(
        BOOTSTRAP_REPS
    ):

        selected_starts = rng.choice(
            starts,
            size=blocks_needed,
            replace=True
        )

        indices = np.concatenate([

            np.arange(
                start,
                start + block_size
            )

            for start in selected_starts

        ])[:n]

        means[i] = (
            values[indices].mean()
        )

    low, high = np.quantile(
        means,
        [0.025, 0.975]
    )

    return (
        float(low),
        float(high)
    )


# ==========================================
# STATISTICHE
# ==========================================

def calculate_stats(
    df,
    scenario,
    regime="TUTTI"
):

    if df.empty:

        return None

    df = df.sort_values(
        "data_segnale"
    )

    returns = df[
        "rendimento"
    ]

    spy = df[
        "rendimento_spy"
    ]

    excess = (
        returns - spy
    )

    ret_low, ret_high = bootstrap_ci(
        returns.values,
        seed=SEED
    )

    exc_low, exc_high = bootstrap_ci(
        excess.values,
        seed=SEED + 1
    )

    return {

        "scenario": scenario,

        "regime": regime,

        "n": len(df),

        "data_inizio": (
            df["data_segnale"]
            .min()
            .date()
        ),

        "data_fine": (
            df["data_segnale"]
            .max()
            .date()
        ),

        "rendimento_medio_pct":
            returns.mean() * 100,

        "rendimento_mediano_pct":
            returns.median() * 100,

        "probabilita_positiva_pct":
            (returns > 0).mean() * 100,

        "spy_medio_pct":
            spy.mean() * 100,

        "excess_medio_pct":
            excess.mean() * 100,

        "probabilita_battere_spy_pct":
            (excess > 0).mean() * 100,

        "ic95_rendimento_basso_pct":
            ret_low * 100,

        "ic95_rendimento_alto_pct":
            ret_high * 100,

        "ic95_excess_basso_pct":
            exc_low * 100,

        "ic95_excess_alto_pct":
            exc_high * 100,

        "ic95_excess_esclude_zero":
            bool(
                np.isfinite(exc_low)
                and
                np.isfinite(exc_high)
                and
                (
                    exc_low > 0
                    or
                    exc_high < 0
                )
            ),

        "rendimento_minimo_pct":
            returns.min() * 100,

        "rendimento_massimo_pct":
            returns.max() * 100

    }


# ==========================================
# CAMPIONE DISTANZIATO
# ==========================================

def non_overlapping_sample(df):

    selected = []

    last_date = None

    for _, row in df.sort_values(
        "data_segnale"
    ).iterrows():

        date = row[
            "data_segnale"
        ]

        if (
            last_date is None
            or
            (date - last_date).days
            >= MIN_DAYS_BETWEEN
        ):

            selected.append(
                row
            )

            last_date = date

    if not selected:

        return df.iloc[0:0].copy()

    return pd.DataFrame(
        selected
    ).reset_index(drop=True)


# ==========================================
# ESCLUSIONE VALORI ESTREMI
# ==========================================

def remove_extremes(df):

    if len(df) < 20:

        return df.copy()

    low = df[
        "rendimento"
    ].quantile(
        TRIM_PERCENT
    )

    high = df[
        "rendimento"
    ].quantile(
        1 - TRIM_PERCENT
    )

    return df[
        (
            df["rendimento"]
            >= low
        )
        &
        (
            df["rendimento"]
            <= high
        )
    ].copy()


# ==========================================
# SCENARI DI ROBUSTEZZA
# ==========================================

def build_scenarios(df):

    scenarios = {}

    scenarios[
        "COMPLETO"
    ] = df.copy()

    scenarios[
        "NON_OVERLAP_28G"
    ] = non_overlapping_sample(
        df
    )

    scenarios[
        "SENZA_SLV"
    ] = df[
        df["asset"] != "SLV"
    ].copy()

    scenarios[
        "SENZA_ESTREMI_5PCT"
    ] = remove_extremes(
        df
    )

    scenarios[
        "NON_OVERLAP_SENZA_SLV"
    ] = non_overlapping_sample(
        df[
            df["asset"] != "SLV"
        ]
    )

    return scenarios


# ==========================================
# ANALISI COMPLETA
# ==========================================

def overall_analysis(scenarios):

    rows = []

    for name, sample in scenarios.items():

        result = calculate_stats(
            sample,
            name
        )

        if result is not None:

            rows.append(
                result
            )

    return pd.DataFrame(
        rows
    )


# ==========================================
# ANALISI PER REGIME
# ==========================================

def regime_analysis(scenarios):

    rows = []

    for name, sample in scenarios.items():

        for regime, group in sample.groupby(
            "regime"
        ):

            result = calculate_stats(
                group,
                name,
                regime
            )

            if result is not None:

                rows.append(
                    result
                )

    return pd.DataFrame(
        rows
    )


# ==========================================
# ANALISI ANNUALE
# ==========================================

def annual_analysis(df):

    rows = []

    for year, group in df.groupby(
        "anno"
    ):

        result = calculate_stats(
            group,
            "ANNUALE"
        )

        if result is not None:

            result["anno"] = int(
                year
            )

            rows.append(
                result
            )

    return pd.DataFrame(
        rows
    )


# ==========================================
# ESECUZIONE PRINCIPALE
# ==========================================

def main():

    print("=" * 65)

    print(
        "MACRO SWING LAB 4.3"
    )

    print(
        "VALIDAZIONE ROBUSTEZZA MOMENTUM"
    )

    print("=" * 65)

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_data()

    print(
        "\nOsservazioni iniziali:",
        len(df)
    )

    print(
        "Periodo:",
        df["data_segnale"].min().date(),
        "-",
        df["data_segnale"].max().date()
    )

    scenarios = build_scenarios(
        df
    )

    overall = overall_analysis(
        scenarios
    )

    regimes = regime_analysis(
        scenarios
    )

    annual = annual_analysis(
        df
    )

    sparse = scenarios[
        "NON_OVERLAP_28G"
    ]

    # ======================================
    # SALVATAGGIO
    # ======================================

    overall.to_csv(
        OUTPUT / "robustezza_43.csv",
        index=False
    )

    regimes.to_csv(
        OUTPUT / "robustezza_regimi_43.csv",
        index=False
    )

    annual.to_csv(
        OUTPUT / "robustezza_annuale_43.csv",
        index=False
    )

    sparse.to_csv(
        OUTPUT / "robustezza_non_overlap_43.csv",
        index=False
    )

    # ======================================
    # REPORT
    # ======================================

    print(
        "\nRISULTATI COMPLESSIVI:"
    )

    columns = [
        "scenario",
        "n",
        "rendimento_medio_pct",
        "excess_medio_pct",
        "ic95_excess_basso_pct",
        "ic95_excess_alto_pct",
        "ic95_excess_esclude_zero"
    ]

    print(
        overall[columns].to_string(
            index=False
        )
    )

    print(
        "\nOSSERVAZIONI PER SCENARIO:"
    )

    for name, sample in scenarios.items():

        print(
            name,
            ":",
            len(sample)
        )

    print(
        "\nFILE GENERATI:"
    )

    print(
        "output/robustezza_43.csv"
    )

    print(
        "output/robustezza_regimi_43.csv"
    )

    print(
        "output/robustezza_annuale_43.csv"
    )

    print(
        "output/robustezza_non_overlap_43.csv"
    )

    print(
        "\nLIMITAZIONI:"
    )

    print(
        "Il campione distanziato non "
        "equivale a un backtest "
        "di portafoglio."
    )

    print(
        "L'esclusione di SLV e degli "
        "estremi e' un test di "
        "sensibilita', non una "
        "nuova strategia validata."
    )

    print(
        "Gli intervalli bootstrap "
        "sono esplorativi e non "
        "correggono per test multipli."
    )

    print(
        "\nMACRO SWING LAB 4.3 "
        "COMPLETATO."
    )


if __name__ == "__main__":
    main()
