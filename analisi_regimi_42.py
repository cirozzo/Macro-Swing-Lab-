#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MACRO SWING LAB 4.2

Analisi della strategia Momentum
condizionata al regime macroeconomico.

Input:
    output/scelte_modelli_40.csv

Output:
    output/regimi_momentum_42.csv
    output/regimi_momentum_annuale_42.csv
    output/regimi_momentum_osservazioni_42.csv

NOTE:
- Ricerca esplorativa.
- Rendimenti a 20 sedute.
- Possibile sovrapposizione delle osservazioni.
- Dati macro FRED revisionati.
- Costi di negoziazione esclusi.
- Nessuna ottimizzazione dei parametri.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ==========================================
# CONFIGURAZIONE
# ==========================================

OUTPUT = Path("output")

INPUT = OUTPUT / "scelte_modelli_40.csv"

HORIZON = 20

N_BOOTSTRAP = 10000

BLOCK_SIZE = 5

MIN_SAMPLE = 20

SEED = 42


# ==========================================
# CARICAMENTO DATI
# ==========================================

def load_data():

    if not INPUT.exists():
        raise FileNotFoundError(
            f"File mancante: {INPUT}"
        )

    df = pd.read_csv(INPUT)

    required = {
        "data_segnale",
        "modello",
        "asset",
        "regime",
        "orizzonte",
        "rendimento",
        "rendimento_spy",
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
        (df["modello"] == "MOMENTUM")
        &
        (pd.to_numeric(
            df["orizzonte"],
            errors="coerce"
        ) == HORIZON)
    ].copy()

    df["rendimento"] = pd.to_numeric(
        df["rendimento"],
        errors="coerce"
    )

    df["rendimento_spy"] = pd.to_numeric(
        df["rendimento_spy"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "data_segnale",
            "regime",
            "rendimento",
            "rendimento_spy",
        ]
    )

    if df.duplicated("data_segnale").any():
        raise ValueError(
            "Esistono più selezioni Momentum "
            "per la stessa data."
        )

    df = df.sort_values(
        "data_segnale"
    ).reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "Nessuna osservazione Momentum."
        )

    df["excess_return"] = (
        df["rendimento"]
        - df["rendimento_spy"]
    )

    df["anno"] = (
        df["data_segnale"].dt.year
    )

    return df


# ==========================================
# BLOCK BOOTSTRAP
# ==========================================

def bootstrap_ci(values, seed=SEED):

    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    n = len(values)

    if n < 2:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)

    block = min(
        BLOCK_SIZE,
        n
    )

    starts = np.arange(
        n - block + 1
    )

    needed = int(
        np.ceil(n / block)
    )

    means = np.empty(
        N_BOOTSTRAP
    )

    for i in range(N_BOOTSTRAP):

        selected = rng.choice(
            starts,
            size=needed,
            replace=True
        )

        indices = np.concatenate([
            np.arange(
                start,
                start + block
            )
            for start in selected
        ])[:n]

        means[i] = (
            values[indices].mean()
        )

    low, high = np.quantile(
        means,
        [0.025, 0.975]
    )

    return float(low), float(high)


# ==========================================
# ANALISI PER REGIME
# ==========================================

def analyze_regimes(df):

    rows = []

    for regime, group in df.groupby(
        "regime"
    ):

        group = group.sort_values(
            "data_segnale"
        )

        returns = group[
            "rendimento"
        ]

        excess = group[
            "excess_return"
        ]

        n = len(group)

        ret_low, ret_high = bootstrap_ci(
            returns.values
        )

        exc_low, exc_high = bootstrap_ci(
            excess.values,
            seed=SEED + 1
        )

        rows.append({

            "regime": regime,

            "n": n,

            "rendimento_medio_pct":
                returns.mean() * 100,

            "rendimento_mediano_pct":
                returns.median() * 100,

            "probabilita_positiva_pct":
                (returns > 0).mean() * 100,

            "rendimento_spy_medio_pct":
                group[
                    "rendimento_spy"
                ].mean() * 100,

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

            "campione_sufficiente":
                n >= MIN_SAMPLE,

            "excess_ic95_positivo":
                bool(exc_low > 0),

        })

    result = pd.DataFrame(rows)

    return result.sort_values(
        "regime"
    )


# ==========================================
# STABILITÀ ANNUALE
# ==========================================

def annual_analysis(df):

    rows = []

    for (year, regime), group in df.groupby(
        ["anno", "regime"]
    ):

        returns = group[
            "rendimento"
        ]

        excess = group[
            "excess_return"
        ]

        rows.append({

            "anno": int(year),

            "regime": regime,

            "n": len(group),

            "rendimento_medio_pct":
                returns.mean() * 100,

            "probabilita_positiva_pct":
                (returns > 0).mean() * 100,

            "excess_medio_pct":
                excess.mean() * 100,

            "probabilita_battere_spy_pct":
                (excess > 0).mean() * 100,

        })

    return pd.DataFrame(rows).sort_values(
        ["anno", "regime"]
    )


# ==========================================
# ESECUZIONE
# ==========================================

def main():

    print("=" * 60)
    print("MACRO SWING LAB 4.2")
    print("MOMENTUM PER REGIME MACRO")
    print("=" * 60)

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_data()

    print(
        "\nOsservazioni Momentum:",
        len(df)
    )

    print(
        "Periodo:",
        df["data_segnale"].min().date(),
        "-",
        df["data_segnale"].max().date()
    )

    regimes = analyze_regimes(df)

    annual = annual_analysis(df)

    regimes.to_csv(
        OUTPUT / "regimi_momentum_42.csv",
        index=False
    )

    annual.to_csv(
        OUTPUT / "regimi_momentum_annuale_42.csv",
        index=False
    )

    df.to_csv(
        OUTPUT / "regimi_momentum_osservazioni_42.csv",
        index=False
    )

    print("\nRISULTATI PER REGIME:")

    columns = [
        "regime",
        "n",
        "rendimento_medio_pct",
        "probabilita_positiva_pct",
        "excess_medio_pct",
        "ic95_excess_basso_pct",
        "ic95_excess_alto_pct",
        "campione_sufficiente",
    ]

    print(
        regimes[columns].to_string(
            index=False
        )
    )

    print("\nFILE GENERATI:")
    print("output/regimi_momentum_42.csv")
    print("output/regimi_momentum_annuale_42.csv")
    print("output/regimi_momentum_osservazioni_42.csv")

    print("\nLIMITAZIONI:")
    print(
        "Gli intervalli bootstrap sono "
        "esplorativi: i rendimenti a 20 "
        "sedute possono sovrapporsi."
    )
    print(
        "I regimi sono ricostruiti con "
        "dati macro revisionati."
    )
    print(
        "Un intervallo positivo non dimostra "
        "da solo un vantaggio negoziabile."
    )

    print(
        "\nMACRO SWING LAB 4.2 COMPLETATO."
    )


if __name__ == "__main__":
    main()
