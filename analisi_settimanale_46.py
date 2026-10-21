#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MACRO SWING LAB 4.6

RICERCA STORICA SWING LONG SETTIMANALI

OBIETTIVO:
Studiare quali ETF sviluppano movimenti rialzisti
nelle successive 1, 2, 3, 4, 6 e 8 settimane.

Misure:
- Forward return
- MFE
- MAE
- Tempo al massimo rialzo
- Probabilita' movimento >= 2%, 4%, 6%
- Confronto con SPY
- Classificazione per regime macro

NON E' UNA STRATEGIA OPERATIVA.

LIMITAZIONI:
- Dati macro storici revisionati
- Regimi non ancora point-in-time
- Prezzi Yahoo Finance rettificati
- Le osservazioni possono sovrapporsi
- MFE non equivale a profitto realizzato
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from macro_swing_40 import (
    ASSETS,
    START,
    macro_regimes
)


# ==========================================
# CONFIGURAZIONE
# ==========================================

BASE = Path(__file__).resolve().parent

OUT = BASE / "output"

OUT.mkdir(
    parents=True,
    exist_ok=True
)

HORIZONS = {
    1: 5,
    2: 10,
    3: 15,
    4: 20,
    6: 30,
    8: 40
}

THRESHOLDS = [2, 4, 6]

TEST_START = pd.Timestamp("2022-01-01")

MIN_EPISODES = 20


# ==========================================
# SCARICAMENTO PREZZI
# ==========================================

def download_ohlc():

    print("\nScaricamento prezzi OHLC...")

    data = {}

    for ticker in ASSETS:

        try:

            df = yf.download(
                ticker,
                start=START,
                auto_adjust=True,
                progress=False,
                threads=False
            )

            if df.empty:
                print(
                    f"ATTENZIONE: {ticker} vuoto"
                )
                continue

            # Gestione colonne MultiIndex
            if isinstance(
                df.columns,
                pd.MultiIndex
            ):

                if "Close" in df.columns.get_level_values(0):

                    df.columns = (
                        df.columns.get_level_values(0)
                    )

                else:

                    raise ValueError(
                        "Formato colonne non riconosciuto"
                    )

            required = [
                "Open",
                "High",
                "Low",
                "Close"
            ]

            if not all(
                c in df.columns
                for c in required
            ):

                raise ValueError(
                    "Colonne OHLC mancanti"
                )

            df = df[required].copy()

            for col in required:

                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )

            df.index = pd.to_datetime(
                df.index
            ).tz_localize(None)

            df = df.sort_index()

            df = df[
                ~df.index.duplicated()
            ]

            df = df.dropna()

            df = df[
                (df[required] > 0).all(axis=1)
            ]

            if df.empty:
                continue

            data[ticker] = df

            print(
                f"{ticker}: {len(df)} sedute"
            )

        except Exception as exc:

            print(
                f"ERRORE {ticker}: {exc}"
            )

    if "SPY" not in data:

        raise RuntimeError(
            "SPY non disponibile."
        )

    if len(data) < 2:

        raise RuntimeError(
            "ETF insufficienti."
        )

    return data


# ==========================================
# DATE SETTIMANALI
# ==========================================

def get_weekly_signals(dates):

    dates = pd.DatetimeIndex(
        dates
    ).sort_values()

    dates = dates[
        ~dates.duplicated()
    ]

    calendar = pd.DataFrame({
        "date": dates
    })

    calendar["week"] = (
        calendar["date"].dt.to_period(
            "W-FRI"
        )
    )

    # Prende l'ultima seduta effettiva
    # di ciascuna settimana.

    weekly = calendar.groupby(
        "week"
    )["date"].max()

    # Esclude la settimana in corso
    # se non e' ancora terminata.

    today = pd.Timestamp.now(
        tz="America/New_York"
    ).tz_localize(None).normalize()

    completed = []

    for week, date in weekly.items():

        if week.end_time.normalize() <= today:

            completed.append(date)

    return pd.DatetimeIndex(
        completed
    )


# ==========================================
# CREAZIONE OSSERVAZIONI
# ==========================================

def build_observations(
    market_data,
    regimes
):

    rows = []

    spy = market_data["SPY"]

    spy_close = spy["Close"]

    for ticker, df in market_data.items():

        print(
            f"\nAnalisi {ticker}..."
        )

        weekly_dates = get_weekly_signals(
            df.index
        )

        for signal_date in weekly_dates:

            if signal_date not in regimes.index:
                continue

            regime = regimes.at[
                signal_date,
                "regime"
            ]

            if (
                pd.isna(regime)
                or regime == "SCONOSCIUTO"
            ):
                continue

            loc = df.index.get_loc(
                signal_date
            )

            # L'osservazione inizia
            # dalla chiusura successiva.

            entry_idx = loc + 1

            if entry_idx >= len(df):
                continue

            entry_date = df.index[
                entry_idx
            ]

            if (
                entry_date not in spy.index
            ):
                continue

            entry_price = float(
                df["Close"].iloc[
                    entry_idx
                ]
            )

            spy_entry = float(
                spy_close.loc[
                    entry_date
                ]
            )

            for weeks, horizon in (
                HORIZONS.items()
            ):

                exit_idx = (
                    entry_idx + horizon
                )

                if exit_idx >= len(df):
                    continue

                exit_date = df.index[
                    exit_idx
                ]

                if exit_date not in spy.index:
                    continue

                # Controllo che entrambe
                # le serie abbiano una
                # distanza coerente
                # di sedute.

                spy_start_idx = (
                    spy.index.get_loc(
                        entry_date
                    )
                )

                spy_end_idx = (
                    spy.index.get_loc(
                        exit_date
                    )
                )

                if (
                    spy_end_idx
                    -
                    spy_start_idx
                    != horizon
                ):
                    continue

                future = df.iloc[
                    entry_idx + 1:
                    exit_idx + 1
                ]

                if len(future) != horizon:
                    continue

                exit_price = float(
                    df["Close"].iloc[
                        exit_idx
                    ]
                )

                spy_exit = float(
                    spy_close.loc[
                        exit_date
                    ]
                )

                # Rendimento finale

                forward_return = (
                    exit_price / entry_price - 1
                )

                spy_return = (
                    spy_exit / spy_entry - 1
                )

                # MFE:
                # massimo HIGH raggiunto
                # DOPO la chiusura di ingresso.

                highs = (
                    future["High"]
                    .to_numpy(
                        dtype=float
                    )
                )

                lows = (
                    future["Low"]
                    .to_numpy(
                        dtype=float
                    )
                )

                max_high = float(
                    np.max(highs)
                )

                min_low = float(
                    np.min(lows)
                )

                mfe = (
                    max_high / entry_price - 1
                )

                mae = (
                    min_low / entry_price - 1
                )

                # Prima seduta nella quale
                # viene raggiunto il massimo.

                days_to_max = (
                    int(np.argmax(highs)) + 1
                )

                row = {

                    "data_segnale":
                        signal_date,

                    "data_ingresso":
                        entry_date,

                    "data_uscita":
                        exit_date,

                    "asset":
                        ticker,

                    "regime":
                        regime,

                    "settimane":
                        weeks,

                    "sedute":
                        horizon,

                    "rendimento":
                        forward_return,

                    "rendimento_spy":
                        spy_return,

                    "excess_return":
                        forward_return
                        -
                        spy_return,

                    "mfe":
                        mfe,

                    "mae":
                        mae,

                    "sedute_al_massimo":
                        days_to_max,

                    "prezzo_ingresso":
                        entry_price,

                    "prezzo_uscita":
                        exit_price
                }

                for threshold in THRESHOLDS:

                    # Il massimo favorevole
                    # raggiunge la soglia?

                    row[
                        f"mfe_ge_{threshold}pct"
                    ] = int(
                        mfe >= threshold / 100
                    )

                rows.append(row)

    result = pd.DataFrame(rows)

    if result.empty:

        raise RuntimeError(
            "Nessuna osservazione generata."
        )

    result = result.sort_values(
        [
            "data_segnale",
            "asset",
            "settimane"
        ]
    ).reset_index(drop=True)

    if result.duplicated(
        [
            "data_segnale",
            "asset",
            "settimane"
        ]
    ).any():

        raise ValueError(
            "Osservazioni duplicate."
        )

    result["anno"] = (
        result["data_segnale"].dt.year
    )

    result["periodo"] = np.where(
        result["data_segnale"] < TEST_START,
        "RICERCA_PRE_2022",
        "PERIODO_2022_PLUS"
    )

    return result


# ==========================================
# CAMPIONE NON SOVRAPPOSTO
# ==========================================

def select_non_overlap(df):

    selected = []

    # Importante:
    # selezioniamo le osservazioni
    # senza sovrapposizione PRIMA
    # di dividerle per regime.

    groups = df.groupby(
        [
            "asset",
            "settimane",
            "periodo"
        ]
    )

    for _, group in groups:

        group = group.sort_values(
            "data_ingresso"
        )

        last_exit = None

        for idx, row in group.iterrows():

            entry = row[
                "data_ingresso"
            ]

            exit_date = row[
                "data_uscita"
            ]

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
# STATISTICHE AGGREGATE
# ==========================================

def calculate_stats(
    df,
    sample_name
):

    rows = []

    groups = df.groupby(
        [
            "periodo",
            "asset",
            "regime",
            "settimane"
        ]
    )

    for keys, group in groups:

        (
            period,
            ticker,
            regime,
            weeks
        ) = keys

        r = group["rendimento"]

        mfe = group["mfe"]

        mae = group["mae"]

        excess = group["excess_return"]

        row = {

            "campione":
                sample_name,

            "periodo":
                period,

            "asset":
                ticker,

            "regime":
                regime,

            "settimane":
                weeks,

            "osservazioni":
                len(group),

            "anni_distinti":
                group["anno"].nunique(),

            "probabilita_long_pct":
                (r > 0).mean() * 100,

            "rendimento_medio_pct":
                r.mean() * 100,

            "rendimento_mediano_pct":
                r.median() * 100,

            "mfe_medio_pct":
                mfe.mean() * 100,

            "mfe_mediano_pct":
                mfe.median() * 100,

            "mae_medio_pct":
                mae.mean() * 100,

            "mae_mediano_pct":
                mae.median() * 100,

            "sedute_mediane_al_massimo":
                group[
                    "sedute_al_massimo"
                ].median(),

            "probabilita_battere_spy_pct":
                (excess > 0).mean() * 100,

            "excess_medio_spy_pct":
                excess.mean() * 100,

            "excess_mediano_spy_pct":
                excess.median() * 100

        }

        for threshold in THRESHOLDS:

            col = (
                f"mfe_ge_{threshold}pct"
            )

            row[
                f"prob_mfe_ge_{threshold}pct"
            ] = (
                group[col].mean() * 100
            )

        row["evidenza"] = (
            "CAMPIONE_LIMITATO"
            if len(group) < MIN_EPISODES
            else "DESCRITTIVO"
        )

        rows.append(row)

    return pd.DataFrame(rows)


# ==========================================
# STABILITA' ANNUALE
# ==========================================

def calculate_annual(df):

    rows = []

    groups = df.groupby(
        [
            "anno",
            "asset",
            "regime",
            "settimane"
        ]
    )

    for keys, group in groups:

        (
            year,
            ticker,
            regime,
            weeks
        ) = keys

        rows.append({

            "anno":
                year,

            "asset":
                ticker,

            "regime":
                regime,

            "settimane":
                weeks,

            "osservazioni":
                len(group),

            "probabilita_long_pct":
                (
                    group["rendimento"] > 0
                ).mean() * 100,

            "mfe_mediano_pct":
                group["mfe"].median() * 100,

            "mae_mediano_pct":
                group["mae"].median() * 100,

            "rendimento_mediano_pct":
                group[
                    "rendimento"
                ].median() * 100

        })

    return pd.DataFrame(rows)


# ==========================================
# CONFRONTO CON BASELINE
# ==========================================

def compare_with_baseline(stats):

    # Come riferimento prendiamo
    # tutte le settimane dello
    # stesso asset, orizzonte e periodo.

    base = stats[
        stats["regime"] == "TUTTI"
    ].copy()

    regimes = stats[
        stats["regime"] != "TUTTI"
    ].copy()

    keys = [
        "campione",
        "periodo",
        "asset",
        "settimane"
    ]

    cols = [
        "probabilita_long_pct",
        "rendimento_mediano_pct",
        "mfe_mediano_pct",
        "prob_mfe_ge_4pct"
    ]

    base = base[
        keys + cols
    ]

    result = regimes.merge(
        base,
        on=keys,
        how="left",
        suffixes=(
            "",
            "_baseline"
        ),
        validate="many_to_one"
    )

    for col in cols:

        result[
            f"delta_{col}"
        ] = (
            result[col]
            -
            result[
                f"{col}_baseline"
            ]
        )

    return result


# ==========================================
# MAIN
# ==========================================

def main():

    print("=" * 65)

    print(
        "MACRO SWING LAB 4.6"
    )

    print(
        "RICERCA SWING LONG SETTIMANALI"
    )

    print("=" * 65)

    market_data = download_ohlc()

    # SPY fornisce il calendario
    # di riferimento dei mercati USA.

    calendar = market_data[
        "SPY"
    ].index

    print(
        "\nCalcolo regimi macro..."
    )

    regimes = macro_regimes(
        calendar
    )

    print(
        "\nCreazione osservazioni..."
    )

    observations = build_observations(
        market_data,
        regimes
    )

    print(
        "\nOsservazioni:",
        len(observations)
    )

    # Campione con sovrapposizioni

    full = calculate_stats(
        observations,
        "COMPLETO"
    )

    # Campione senza sovrapposizioni

    independent = select_non_overlap(
        observations
    )

    non_overlap_stats = calculate_stats(
        independent,
        "NON_OVERLAP"
    )

    # Baseline per asset e orizzonte,
    # senza distinguere il regime.

    baseline_full = (
        observations.copy()
    )

    baseline_full["regime"] = (
        "TUTTI"
    )

    baseline_non_overlap = (
        independent.copy()
    )

    baseline_non_overlap["regime"] = (
        "TUTTI"
    )

    baseline_stats = pd.concat(
        [
            calculate_stats(
                baseline_full,
                "COMPLETO"
            ),
            calculate_stats(
                baseline_non_overlap,
                "NON_OVERLAP"
            )
        ],
        ignore_index=True
    )

    all_stats = pd.concat(
        [
            full,
            non_overlap_stats,
            baseline_stats
        ],
        ignore_index=True
    )

    comparison = compare_with_baseline(
        all_stats
    )

    annual = calculate_annual(
        independent
    )

    # ======================================
    # SALVATAGGIO
    # ======================================

    observations.to_csv(
        OUT / "osservazioni_46.csv",
        index=False
    )

    independent.to_csv(
        OUT / "episodi_non_overlap_46.csv",
        index=False
    )

    all_stats.to_csv(
        OUT / "statistiche_settimanali_46.csv",
        index=False
    )

    comparison.to_csv(
        OUT / "confronto_baseline_46.csv",
        index=False
    )

    annual.to_csv(
        OUT / "stabilita_annuale_46.csv",
        index=False
    )

    # ======================================
    # REPORT
    # ======================================

    print(
        "\nAsset analizzati:",
        observations["asset"].nunique()
    )

    print(
        "\nOsservazioni non sovrapposte:",
        len(independent)
    )

    print(
        "\nOrizzonti:",
        list(HORIZONS.keys())
    )

    print("\nFILE GENERATI:")

    for name in [
        "osservazioni_46.csv",
        "episodi_non_overlap_46.csv",
        "statistiche_settimanali_46.csv",
        "confronto_baseline_46.csv",
        "stabilita_annuale_46.csv"
    ]:

        print(
            "output/" + name
        )

    print(
        "\nATTENZIONE:"
    )

    print(
        "MFE e MAE descrivono "
        "escursioni storiche."
    )

    print(
        "Non sono profitti o perdite "
        "di una strategia operativa."
    )

    print(
        "Le statistiche non dimostrano "
        "un vantaggio predittivo."
    )

    print(
        "\nMACRO SWING LAB 4.6 COMPLETATO"
    )


if __name__ == "__main__":
    main()
