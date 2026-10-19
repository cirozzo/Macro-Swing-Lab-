#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MACRO SWING LAB 4.5
RICERCA ASSET PER REGIME MACROECONOMICO

OBIETTIVO
Individuare asset storicamente interessanti
per movimenti rialzisti da 5 a 40 sedute.

NON È UNA STRATEGIA OPERATIVA.

Non definisce:
- ingressi
- stop loss
- take profit
- dimensionamento
- gestione del portafoglio

LIMITAZIONI
- Dati macro FRED revisionati
- Regimi non completamente point-in-time
- Campioni storici non indipendenti
- Nessuna garanzia di risultati futuri
"""

from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent

OUT = BASE / "output"

OUT.mkdir(
    parents=True,
    exist_ok=True
)

INPUT = OUT / "osservazioni_40.csv"

HORIZONS = [5, 10, 20, 40]

TEST_START = pd.Timestamp("2022-01-01")

MIN_EPISODES = 12

MIN_YEARS = 3


# ==========================================
# CARICAMENTO
# ==========================================

def load_data():

    if not INPUT.exists():
        raise FileNotFoundError(
            f"File mancante: {INPUT}"
        )

    df = pd.read_csv(INPUT)

    required = {
        "data_segnale",
        "data_ingresso",
        "data_uscita",
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

    for col in [
        "data_segnale",
        "data_ingresso",
        "data_uscita",
    ]:
        df[col] = pd.to_datetime(
            df[col],
            errors="coerce"
        )

    for col in [
        "orizzonte",
        "rendimento",
        "rendimento_spy",
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=list(required)
    ).copy()

    df = df[
        df["orizzonte"].isin(HORIZONS)
    ].copy()

    df = df[
        (df["data_segnale"] < df["data_ingresso"])
        &
        (df["data_ingresso"] < df["data_uscita"])
    ].copy()

    df["excess_return"] = (
        df["rendimento"]
        -
        df["rendimento_spy"]
    )

    df["anno"] = (
        df["data_segnale"].dt.year
    )

    df["periodo"] = np.where(
        df["data_segnale"] < TEST_START,
        "RICERCA_2008_2021",
        "VERIFICA_2022_2026"
    )

    df = df.sort_values(
        [
            "asset",
            "regime",
            "orizzonte",
            "data_ingresso"
        ]
    ).reset_index(drop=True)

    keys = [
        "data_segnale",
        "asset",
        "orizzonte"
    ]

    if df.duplicated(keys).any():
        raise ValueError(
            "Osservazioni duplicate nel dataset."
        )

    if df.empty:
        raise ValueError(
            "Nessuna osservazione valida."
        )

    return df


# ==========================================
# CAMPIONE SENZA SOVRAPPOSIZIONI
# ==========================================

def non_overlapping(df):

    selected = []

    for _, group in df.groupby(
        [
            "asset",
            "regime",
            "orizzonte"
        ]
    ):

        group = group.sort_values(
            "data_ingresso"
        )

        last_exit = None

        for idx, row in group.iterrows():

            entry = row["data_ingresso"]

            exit_date = row["data_uscita"]

            if (
                last_exit is None
                or entry > last_exit
            ):

                selected.append(idx)

                last_exit = exit_date

    return df.loc[
        selected
    ].copy()


# ==========================================
# STATISTICHE
# ==========================================

def calculate_stats(df, sample_name):

    rows = []

    groups = df.groupby(
        [
            "periodo",
            "asset",
            "regime",
            "orizzonte"
        ]
    )

    for (
        period,
        asset,
        regime,
        horizon
    ), group in groups:

        r = group["rendimento"]

        excess = group["excess_return"]

        n = len(group)

        years = group["anno"].nunique()

        positive = (r > 0).mean()

        beat_spy = (excess > 0).mean()

        rows.append({

            "campione": sample_name,

            "periodo": period,

            "asset": asset,

            "regime": regime,

            "orizzonte": horizon,

            "osservazioni": n,

            "anni_distinti": years,

            "probabilita_long_pct":
                positive * 100,

            "rendimento_medio_pct":
                r.mean() * 100,

            "rendimento_mediano_pct":
                r.median() * 100,

            "percentile_25_pct":
                r.quantile(0.25) * 100,

            "percentile_75_pct":
                r.quantile(0.75) * 100,

            "probabilita_battere_spy_pct":
                beat_spy * 100,

            "excess_medio_spy_pct":
                excess.mean() * 100,

            "excess_mediano_spy_pct":
                excess.median() * 100,

            "rendimento_minimo_pct":
                r.min() * 100,

            "rendimento_massimo_pct":
                r.max() * 100,

        })

    return pd.DataFrame(rows)


# ==========================================
# STABILITÀ ANNUALE
# ==========================================

def annual_stability(df):

    rows = []

    groups = df.groupby(
        [
            "asset",
            "regime",
            "orizzonte",
            "anno"
        ]
    )

    for (
        asset,
        regime,
        horizon,
        year
    ), group in groups:

        r = group["rendimento"]

        rows.append({

            "asset": asset,

            "regime": regime,

            "orizzonte": horizon,

            "anno": year,

            "osservazioni": len(group),

            "probabilita_long_pct":
                (r > 0).mean() * 100,

            "rendimento_medio_pct":
                r.mean() * 100,

            "rendimento_mediano_pct":
                r.median() * 100,

        })

    return pd.DataFrame(rows)


# ==========================================
# CONFRONTO PERIODI
# ==========================================

def compare_periods(stats):

    sample = stats[
        stats["campione"] == "NON_OVERLAP"
    ].copy()

    keys = [
        "asset",
        "regime",
        "orizzonte"
    ]

    research = sample[
        sample["periodo"]
        == "RICERCA_2008_2021"
    ].copy()

    verification = sample[
        sample["periodo"]
        == "VERIFICA_2022_2026"
    ].copy()

    cols = [
        "osservazioni",
        "anni_distinti",
        "probabilita_long_pct",
        "rendimento_mediano_pct",
        "excess_medio_spy_pct"
    ]

    result = research[
        keys + cols
    ].merge(

        verification[
            keys + cols
        ],

        on=keys,

        how="outer",

        suffixes=(
            "_ricerca",
            "_verifica"
        )
    )

    return result


# ==========================================
# VALUTAZIONE EVIDENZA
# ==========================================

def evidence_label(row):

    n = row["osservazioni"]

    years = row["anni_distinti"]

    if (
        n < MIN_EPISODES
        or years < MIN_YEARS
    ):
        return "CAMPIONE_LIMITATO"

    return "CAMPIONE_DESCRITTIVO"


# ==========================================
# WATCHLIST STORICA
# ==========================================

def historical_watchlist(stats):

    sample = stats[
        (
            stats["campione"]
            == "NON_OVERLAP"
        )
        &
        (
            stats["periodo"]
            == "RICERCA_2008_2021"
        )
    ].copy()

    if sample.empty:
        return sample

    sample["evidenza"] = sample.apply(
        evidence_label,
        axis=1
    )

    # Graduatoria descrittiva:
    # privilegia la mediana positiva.
    # Non rappresenta un segnale di acquisto.

    sample["mediana_positiva"] = (
        sample["rendimento_mediano_pct"] > 0
    )

    sample = sample.sort_values(
        [
            "regime",
            "orizzonte",
            "mediana_positiva",
            "rendimento_mediano_pct",
            "probabilita_long_pct"
        ],
        ascending=[
            True,
            True,
            False,
            False,
            False
        ]
    )

    sample["posizione_storica"] = (
        sample.groupby(
            [
                "regime",
                "orizzonte"
            ]
        ).cumcount() + 1
    )

    return sample


# ==========================================
# MAIN
# ==========================================

def main():

    print("=" * 60)

    print("MACRO SWING LAB 4.5")

    print(
        "RICERCA ASSET PER REGIME MACRO"
    )

    print("=" * 60)

    df = load_data()

    print(
        "Osservazioni totali:",
        len(df)
    )

    # Tutte le osservazioni

    full_stats = calculate_stats(
        df,
        "COMPLETO"
    )

    # Campione non sovrapposto:
    # separiamo prima i periodi, per
    # evitare che un episodio attraversi
    # il confine ricerca/verifica.

    non_overlap_parts = []

    for _, period_df in df.groupby(
        "periodo"
    ):

        non_overlap_parts.append(
            non_overlapping(period_df)
        )

    independent = pd.concat(
        non_overlap_parts,
        ignore_index=True
    )

    independent_stats = calculate_stats(
        independent,
        "NON_OVERLAP"
    )

    stats = pd.concat(
        [
            full_stats,
            independent_stats
        ],
        ignore_index=True
    )

    annual = annual_stability(
        independent
    )

    comparison = compare_periods(
        stats
    )

    watchlist = historical_watchlist(
        stats
    )

    # ======================================
    # SALVATAGGIO
    # ======================================

    stats.to_csv(
        OUT / "asset_regimi_45.csv",
        index=False
    )

    independent.to_csv(
        OUT / "episodi_indipendenti_45.csv",
        index=False
    )

    annual.to_csv(
        OUT / "stabilita_annuale_45.csv",
        index=False
    )

    comparison.to_csv(
        OUT / "confronto_periodi_45.csv",
        index=False
    )

    watchlist.to_csv(
        OUT / "watchlist_storica_45.csv",
        index=False
    )

    # ======================================
    # REPORT
    # ======================================

    print(
        "\nOsservazioni non sovrapposte:",
        len(independent)
    )

    print(
        "\nAsset analizzati:",
        df["asset"].nunique()
    )

    print(
        "\nRegimi analizzati:",
        df["regime"].nunique()
    )

    print(
        "\nOrizzonti:",
        HORIZONS
    )

    print("\nFILE GENERATI:")

    for filename in [
        "asset_regimi_45.csv",
        "episodi_indipendenti_45.csv",
        "stabilita_annuale_45.csv",
        "confronto_periodi_45.csv",
        "watchlist_storica_45.csv",
    ]:

        print(
            "output/" + filename
        )

    print(
        "\nATTENZIONE:"
    )

    print(
        "Le statistiche sono descrittive."
    )

    print(
        "I regimi macro non sono ancora "
        "point-in-time."
    )

    print(
        "Il campione NON_OVERLAP riduce "
        "la sovrapposizione temporale "
        "per singolo asset e regime, "
        "ma non garantisce indipendenza "
        "statistica tra episodi."
    )

    print(
        "La graduatoria non e' una "
        "raccomandazione di acquisto."
    )

    print(
        "\nMACRO SWING LAB 4.5 COMPLETATO"
    )


if __name__ == "__main__":
    main()
