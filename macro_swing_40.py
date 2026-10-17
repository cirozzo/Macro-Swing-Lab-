#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MACRO SWING LAB 4.0
Motore statistico multi-asset LONG ONLY.

FUNZIONI:
1. Scarica prezzi storici di 15 ETF.
2. Classifica quattro regimi macroeconomici.
3. Analizza rendimenti long per asset e regime.
4. Confronta MACRO, MOMENTUM e IBRIDO.
5. Genera una watchlist aggiornata.

CORREZIONI:
- Gestione robusta dei dati mancanti.
- Controllo della watchlist vuota.
- Diagnostica degli ETF disponibili.
- Controllo dei dati prima dell'ordinamento.

LIMITAZIONI:
- Ricerca esplorativa, non segnali operativi.
- Dati FRED revisionati.
- Regimi non point-in-time.
- Costi di negoziazione esclusi.
- Operazioni potenzialmente sovrapposte.
- Pullback trading non ancora implementato.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings(
    "ignore",
    category=FutureWarning
)

# =====================================================
# CONFIGURAZIONE
# =====================================================

BASE = Path(__file__).resolve().parent
OUT = BASE / "output"

OUT.mkdir(
    parents=True,
    exist_ok=True
)

START = "2007-01-01"

ASSETS = {
    "QQQ": "Nasdaq 100",
    "SPY": "S&P 500",
    "IWM": "Russell 2000",
    "SMH": "Semiconduttori",
    "XLE": "Energia",
    "XLF": "Finanziari",
    "XLV": "Sanita",
    "XLP": "Consumi essenziali",
    "GLD": "Oro",
    "SLV": "Argento",
    "USO": "Petrolio",
    "TLT": "Treasury lunghi",
    "IEF": "Treasury intermedi",
    "UUP": "Dollaro USA",
    "EEM": "Mercati emergenti",
}

HORIZONS = [5, 10, 20, 40, 60]

MIN_HISTORY = 252

MIN_TRAIN_TRADES = 20

MACRO_LAG_DAYS = 90

TEST_START = pd.Timestamp(
    "2022-01-01"
)

# =====================================================
# DOWNLOAD PREZZI
# =====================================================

def download_prices():

    print("\nSCARICAMENTO PREZZI ETF")

    frames = {}

    for ticker in ASSETS:

        try:

            data = yf.download(
                ticker,
                start=START,
                auto_adjust=True,
                progress=False,
                threads=False
            )

            if data.empty:

                print(
                    f"ATTENZIONE: {ticker} vuoto"
                )

                continue

            close = data["Close"]

            if isinstance(
                close,
                pd.DataFrame
            ):

                close = close.iloc[:, 0]

            close = pd.to_numeric(
                close,
                errors="coerce"
            )

            close.index = pd.to_datetime(
                close.index
            ).tz_localize(None)

            close = close.dropna()

            if close.empty:

                print(
                    f"ATTENZIONE: {ticker} "
                    "senza prezzi validi"
                )

                continue

            frames[ticker] = close.rename(
                ticker
            )

            print(
                f"{ticker}: "
                f"{len(close)} sedute"
            )

        except Exception as exc:

            print(
                f"ERRORE DOWNLOAD {ticker}: "
                f"{exc}"
            )

    if (
        "SPY" not in frames
        or "QQQ" not in frames
    ):

        raise RuntimeError(
            "Mancano SPY o QQQ. "
            "Impossibile procedere."
        )

    prices = pd.concat(
        frames.values(),
        axis=1
    ).sort_index()

    prices = prices[
        ~prices.index.duplicated()
    ]

    print(
        "\nETF disponibili:",
        len(prices.columns)
    )

    print(
        "Ultima data disponibile:",
        prices.index[-1]
    )

    return prices


# =====================================================
# DOWNLOAD DATI MACROECONOMICI
# =====================================================

def fred_series(code):

    url = (
        "https://fred.stlouisfed.org/"
        "graph/fredgraph.csv?id="
        f"{code}"
    )

    df = pd.read_csv(url)

    df.columns = [
        "date",
        "value"
    ]

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce"
    )

    return (
        df.set_index("date")["value"]
        .sort_index()
    )


# =====================================================
# REGIMI MACROECONOMICI
# =====================================================

def macro_regimes(dates):

    print(
        "\nSCARICAMENTO DATI MACRO"
    )

    industrial = fred_series(
        "INDPRO"
    )

    inflation = fred_series(
        "PCEPILFE"
    )

    growth = industrial.pct_change(
        12,
        fill_method=None
    ) * 100

    inflation_yoy = inflation.pct_change(
        12,
        fill_method=None
    ) * 100

    growth_change = growth.diff(3)

    inflation_change = (
        inflation_yoy.diff(3)
    )

    macro = pd.DataFrame({

        "growth": growth,

        "inflation": inflation_yoy,

        "growth_change": (
            growth_change
        ),

        "inflation_change": (
            inflation_change
        ),

    })

    macro = macro.dropna().copy()

    macro["available_date"] = (
        macro.index
        + pd.Timedelta(
            days=MACRO_LAG_DAYS
        )
    )

    macro = (
        macro.reset_index(drop=True)
        .sort_values("available_date")
    )

    calendar = pd.DataFrame({

        "date": pd.to_datetime(
            dates
        )

    }).sort_values("date")

    merged = pd.merge_asof(
        calendar,
        macro,
        left_on="date",
        right_on="available_date",
        direction="backward"
    )

    conditions = [

        (
            (merged["growth_change"] >= 0)
            &
            (merged["inflation_change"] < 0)
        ),

        (
            (merged["growth_change"] >= 0)
            &
            (merged["inflation_change"] >= 0)
        ),

        (
            (merged["growth_change"] < 0)
            &
            (merged["inflation_change"] >= 0)
        ),

        (
            (merged["growth_change"] < 0)
            &
            (merged["inflation_change"] < 0)
        ),

    ]

    merged["regime"] = np.select(

        conditions,

        [
            "R1_CRESCITA_DISINFLAZIONE",
            "R2_CRESCITA_INFLAZIONE",
            "R3_RALLENTAMENTO_INFLAZIONE",
            "R4_RALLENTAMENTO_DISINFLAZIONE",
        ],

        default="SCONOSCIUTO"

    )

    return merged.set_index("date")


# =====================================================
# INDICATORI
# =====================================================

def indicators(prices):

    sma50 = prices.rolling(
        50
    ).mean()

    sma200 = prices.rolling(
        200
    ).mean()

    momentum20 = prices.pct_change(
        20,
        fill_method=None
    )

    momentum60 = prices.pct_change(
        60,
        fill_method=None
    )

    trend = (
        (prices > sma200)
        &
        (sma50 > sma200)
    )

    return (
        sma50,
        sma200,
        momentum20,
        momentum60,
        trend
    )


# =====================================================
# OSSERVAZIONI STORICHE
# =====================================================

def build_observations(
    prices,
    macro
):

    print(
        "\nCOSTRUZIONE OSSERVAZIONI"
    )

    (
        sma50,
        sma200,
        mom20,
        mom60,
        trend
    ) = indicators(prices)

    dates = prices.index

    weekly = (
        pd.Series(
            dates,
            index=dates
        )
        .groupby(
            dates.to_period("W-FRI")
        )
        .last()
        .tolist()
    )

    rows = []

    for signal_date in weekly:

        i = dates.get_loc(
            signal_date
        )

        if i < MIN_HISTORY:
            continue

        regime = macro.loc[
            signal_date,
            "regime"
        ]

        if regime == "SCONOSCIUTO":
            continue

        entry_i = i + 1

        if entry_i >= len(dates):
            continue

        entry_date = dates[entry_i]

        for ticker in ASSETS:

            if ticker not in prices.columns:
                continue

            p = prices[ticker]

            signal_price = p.iloc[i]

            entry_price = p.iloc[
                entry_i
            ]

            if (
                not np.isfinite(
                    signal_price
                )
                or not np.isfinite(
                    entry_price
                )
                or signal_price <= 0
                or entry_price <= 0
            ):
                continue

            for horizon in HORIZONS:

                exit_i = (
                    entry_i + horizon
                )

                if exit_i >= len(dates):
                    continue

                exit_date = dates[
                    exit_i
                ]

                exit_price = p.iloc[
                    exit_i
                ]

                spy_entry = prices[
                    "SPY"
                ].iloc[entry_i]

                spy_exit = prices[
                    "SPY"
                ].iloc[exit_i]

                if (
                    not np.isfinite(
                        exit_price
                    )
                    or not np.isfinite(
                        spy_entry
                    )
                    or not np.isfinite(
                        spy_exit
                    )
                    or spy_entry <= 0
                    or spy_exit <= 0
                ):
                    continue

                asset_return = (
                    exit_price
                    / entry_price
                    - 1
                )

                spy_return = (
                    spy_exit
                    / spy_entry
                    - 1
                )

                rows.append({

                    "data_segnale": (
                        signal_date
                    ),

                    "data_ingresso": (
                        entry_date
                    ),

                    "data_uscita": (
                        exit_date
                    ),

                    "asset": ticker,

                    "regime": regime,

                    "orizzonte": horizon,

                    "trend": bool(
                        trend[ticker].iloc[i]
                    ),

                    "momentum20": (
                        mom20[ticker].iloc[i]
                    ),

                    "momentum60": (
                        mom60[ticker].iloc[i]
                    ),

                    "rendimento": (
                        asset_return
                    ),

                    "rendimento_spy": (
                        spy_return
                    ),

                    "excess_return": (
                        asset_return
                        - spy_return
                    ),

                })

    result = pd.DataFrame(rows)

    if result.empty:

        raise RuntimeError(
            "Nessuna osservazione "
            "storica generata."
        )

    print(
        "Osservazioni generate:",
        len(result)
    )

    return result


# =====================================================
# STATISTICHE PER REGIME
# =====================================================

def regime_statistics(obs):

    print(
        "\nSTATISTICHE PER REGIME"
    )

    rows = []

    groups = obs.groupby(
        [
            "asset",
            "regime",
            "orizzonte"
        ]
    )

    for (
        asset,
        regime,
        horizon
    ), g in groups:

        returns = g[
            "rendimento"
        ]

        negative = returns[
            returns < 0
        ]

        rows.append({

            "asset": asset,

            "regime": regime,

            "orizzonte": horizon,

            "numero_osservazioni": (
                len(g)
            ),

            "probabilita_long": (
                (returns > 0).mean()
            ),

            "rendimento_medio": (
                returns.mean()
            ),

            "rendimento_mediano": (
                returns.median()
            ),

            "perdita_media": (
                negative.mean()
                if len(negative)
                else np.nan
            ),

            "excess_medio_spy": (
                g["excess_return"].mean()
            ),

        })

    return pd.DataFrame(rows)


# =====================================================
# CONFRONTO MODELLI WALK-FORWARD
# =====================================================

def model_comparison(obs):

    print(
        "\nCONFRONTO MODELLI"
    )

    results = []

    horizon = 20

    sample = obs[
        obs["orizzonte"] == horizon
    ].copy()

    dates = sorted(
        sample[
            "data_segnale"
        ].unique()
    )

    for date in dates:

        if date < TEST_START:
            continue

        current = sample[
            sample["data_segnale"]
            == date
        ].copy()

        history = sample[
            sample["data_uscita"]
            <= date
        ].copy()

        if (
            current.empty
            or history.empty
        ):
            continue

        regime = current[
            "regime"
        ].iloc[0]

        regime_history = history[
            history["regime"]
            == regime
        ]

        macro_scores = (
            regime_history
            .groupby("asset")
            .agg(

                macro_score=(
                    "rendimento",
                    "mean"
                ),

                macro_n=(
                    "rendimento",
                    "size"
                )

            )
        )

        current = current.merge(
            macro_scores,
            left_on="asset",
            right_index=True,
            how="left"
        )

        current.loc[
            current["macro_n"]
            < MIN_TRAIN_TRADES,
            "macro_score"
        ] = np.nan

        current["momentum_score"] = (
            current["momentum60"]
        )

        eligible = current[
            current["trend"]
            &
            current[
                "momentum_score"
            ].notna()
        ].copy()

        if eligible.empty:
            continue

        macro_candidates = (
            eligible.dropna(
                subset=["macro_score"]
            )
        )

        momentum_candidates = (
            eligible.copy()
        )

        hybrid_candidates = (
            macro_candidates.copy()
        )

        if not hybrid_candidates.empty:

            hybrid_candidates[
                "macro_rank"
            ] = (
                hybrid_candidates[
                    "macro_score"
                ].rank(pct=True)
            )

            hybrid_candidates[
                "mom_rank"
            ] = (
                hybrid_candidates[
                    "momentum_score"
                ].rank(pct=True)
            )

            hybrid_candidates[
                "hybrid_score"
            ] = (

                hybrid_candidates[
                    "macro_rank"
                ]

                +

                hybrid_candidates[
                    "mom_rank"
                ]

            ) / 2

        selections = []

        if not macro_candidates.empty:

            selected = (
                macro_candidates
                .sort_values(
                    "macro_score",
                    ascending=False
                )
                .iloc[0]
            )

            selections.append(
                ("MACRO", selected)
            )

        if not momentum_candidates.empty:

            selected = (
                momentum_candidates
                .sort_values(
                    "momentum_score",
                    ascending=False
                )
                .iloc[0]
            )

            selections.append(
                ("MOMENTUM", selected)
            )

        if not hybrid_candidates.empty:

            selected = (
                hybrid_candidates
                .sort_values(
                    "hybrid_score",
                    ascending=False
                )
                .iloc[0]
            )

            selections.append(
                ("IBRIDO", selected)
            )

        for model, selected in selections:

            results.append({

                "data_segnale": date,

                "modello": model,

                "asset": selected[
                    "asset"
                ],

                "regime": regime,

                "orizzonte": horizon,

                "rendimento": selected[
                    "rendimento"
                ],

                "rendimento_spy": (
                    selected[
                        "rendimento_spy"
                    ]
                ),

                "excess_return": (
                    selected[
                        "excess_return"
                    ]
                ),

            })

    trades = pd.DataFrame(
        results
    )

    if trades.empty:

        raise RuntimeError(
            "Nessuna selezione "
            "nel periodo TEST."
        )

    common = (
        trades.groupby(
            "data_segnale"
        )["modello"]
        .nunique()
    )

    common_dates = common[
        common == 3
    ].index

    paired = trades[
        trades[
            "data_segnale"
        ].isin(common_dates)
    ].copy()

    summary = (
        paired.groupby("modello")
        .agg(

            osservazioni=(
                "rendimento",
                "size"
            ),

            rendimento_medio=(
                "rendimento",
                "mean"
            ),

            rendimento_mediano=(
                "rendimento",
                "median"
            ),

            probabilita_long=(
                "rendimento",
                lambda x: (
                    x > 0
                ).mean()
            ),

            excess_medio_spy=(
                "excess_return",
                "mean"
            )

        )
        .reset_index()
    )

    return trades, summary


# =====================================================
# WATCHLIST CORRETTA
# =====================================================

def current_watchlist(
    prices,
    macro
):

    print(
        "\nGENERAZIONE WATCHLIST"
    )

    (
        sma50,
        sma200,
        mom20,
        mom60,
        trend
    ) = indicators(prices)

    date = prices.index[-1]

    regime = macro.loc[
        date,
        "regime"
    ]

    rows = []

    for ticker, name in ASSETS.items():

        if ticker not in prices.columns:

            print(
                f"{ticker}: "
                "non disponibile"
            )

            continue

        price = prices[
            ticker
        ].iloc[-1]

        if not np.isfinite(price):

            print(
                f"{ticker}: "
                "prezzo finale mancante"
            )

            continue

        rows.append({

            "data": date,

            "asset": ticker,

            "descrizione": name,

            "regime": regime,

            "prezzo": price,

            "sma50": (
                sma50[ticker].iloc[-1]
            ),

            "sma200": (
                sma200[ticker].iloc[-1]
            ),

            "momentum20": (
                mom20[ticker].iloc[-1]
            ),

            "momentum60": (
                mom60[ticker].iloc[-1]
            ),

            "trend_rialzista": bool(
                trend[ticker].iloc[-1]
            ),

        })

    columns = [
        "data",
        "asset",
        "descrizione",
        "regime",
        "prezzo",
        "sma50",
        "sma200",
        "momentum20",
        "momentum60",
        "trend_rialzista",
    ]

    watchlist = pd.DataFrame(
        rows,
        columns=columns
    )

    if watchlist.empty:

        print(
            "\nATTENZIONE: "
            "watchlist vuota."
        )

        print(
            "Nessun ETF dispone "
            "di un prezzo valido "
            "nell'ultima data."
        )

        return watchlist

    watchlist = (
        watchlist.sort_values(

            [
                "trend_rialzista",
                "momentum60"
            ],

            ascending=[
                False,
                False
            ],

            na_position="last"

        )
        .reset_index(drop=True)
    )

    print(
        "Asset nella watchlist:",
        len(watchlist)
    )

    return watchlist


# =====================================================
# ESECUZIONE PRINCIPALE
# =====================================================

def main():

    print("=" * 55)

    print(
        "MACRO SWING LAB 4.0"
    )

    print(
        "MULTI-ASSET LONG RESEARCH"
    )

    print("=" * 55)

    prices = download_prices()

    macro = macro_regimes(
        prices.index
    )

    observations = (
        build_observations(
            prices,
            macro
        )
    )

    stats = regime_statistics(
        observations
    )

    trades, comparison = (
        model_comparison(
            observations
        )
    )

    watchlist = current_watchlist(
        prices,
        macro
    )

    # =========================================
    # SALVATAGGIO RISULTATI
    # =========================================

    observations.to_csv(
        OUT / "osservazioni_40.csv",
        index=False
    )

    stats.to_csv(
        OUT / "regimi_asset_40.csv",
        index=False
    )

    trades.to_csv(
        OUT / "scelte_modelli_40.csv",
        index=False
    )

    comparison.to_csv(
        OUT / "confronto_modelli_40.csv",
        index=False
    )

    watchlist.to_csv(
        OUT / "watchlist_40.csv",
        index=False
    )

    # =========================================
    # REPORT
    # =========================================

    print(
        "\nRISULTATI MODELLO:"
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    print(
        "\nWATCHLIST:"
    )

    if watchlist.empty:

        print(
            "Nessun asset disponibile "
            "per la watchlist."
        )

    else:

        print(

            watchlist[

                [
                    "asset",
                    "regime",
                    "momentum60",
                    "trend_rialzista"
                ]

            ].to_string(
                index=False
            )

        )

    print(
        "\nFILE GENERATI:"
    )

    print(
        "output/osservazioni_40.csv"
    )

    print(
        "output/regimi_asset_40.csv"
    )

    print(
        "output/scelte_modelli_40.csv"
    )

    print(
        "output/confronto_modelli_40.csv"
    )

    print(
        "output/watchlist_40.csv"
    )

    print(
        "\nATTENZIONE:"
    )

    print(
        "Risultati esplorativi. "
        "Dati macro revisionati; "
        "costi esclusi; "
        "operazioni sovrapposte."
    )

    print(
        "\nMACRO SWING LAB 4.0 "
        "COMPLETATO."
    )


if __name__ == "__main__":
    main()
