#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MACRO SWING LAB 4.4
SIMULAZIONE STORICA PORTAFOGLIO MOMENTUM

Obiettivo:
Simulare un portafoglio con una sola posizione
alla volta e confrontarlo con SPY.

Regole:
- Capitale iniziale: 10000
- Long only
- Massimo una posizione contemporanea
- Ingresso alla chiusura successiva al segnale
- Uscita dopo 20 sedute
- Nessuna sovrapposizione
- Costi stimati: 5 basis point per lato
- Nessuna leva
- Quote frazionarie consentite

LIMITAZIONI:
- Prezzi adjusted close, non esecuzioni reali
- Slippage e commissioni sono ipotesi
- Dati macro storici revisionati
- Nessuna conversione valutaria
- Nessuna tassazione
- Nessun finanziamento della liquidita'
- Ricerca esplorativa, non strategia validata
"""

from pathlib import Path

import numpy as np
import pandas as pd

from macro_swing_40 import download_prices


# ==========================================
# CONFIGURAZIONE
# ==========================================

BASE = Path(__file__).resolve().parent

OUT = BASE / "output"

OUT.mkdir(
    parents=True,
    exist_ok=True
)

INPUT_SELECTIONS = (
    OUT / "scelte_modelli_40.csv"
)

INPUT_OBSERVATIONS = (
    OUT / "osservazioni_40.csv"
)

INITIAL_CAPITAL = 10000.0

COST_BPS = 5.0

COST_RATE = COST_BPS / 10000.0

HORIZON = 20

MODEL = "MOMENTUM"

TRADING_DAYS = 252


# ==========================================
# CARICAMENTO SEGNALI
# ==========================================

def load_signals():

    selections = pd.read_csv(
        INPUT_SELECTIONS
    )

    observations = pd.read_csv(
        INPUT_OBSERVATIONS
    )

    required_selections = {
        "data_segnale",
        "modello",
        "asset",
        "orizzonte"
    }

    required_observations = {
        "data_segnale",
        "data_ingresso",
        "data_uscita",
        "asset",
        "orizzonte"
    }

    if not required_selections.issubset(
        selections.columns
    ):

        raise ValueError(
            "Colonne mancanti in scelte_modelli_40.csv"
        )

    if not required_observations.issubset(
        observations.columns
    ):

        raise ValueError(
            "Colonne mancanti in osservazioni_40.csv"
        )

    selections["data_segnale"] = pd.to_datetime(
        selections["data_segnale"]
    )

    observations["data_segnale"] = pd.to_datetime(
        observations["data_segnale"]
    )

    observations["data_ingresso"] = pd.to_datetime(
        observations["data_ingresso"]
    )

    observations["data_uscita"] = pd.to_datetime(
        observations["data_uscita"]
    )

    selections["orizzonte"] = pd.to_numeric(
        selections["orizzonte"],
        errors="coerce"
    )

    observations["orizzonte"] = pd.to_numeric(
        observations["orizzonte"],
        errors="coerce"
    )

    selections = selections[
        (selections["modello"] == MODEL)
        &
        (selections["orizzonte"] == HORIZON)
    ].copy()

    observations = observations[
        observations["orizzonte"] == HORIZON
    ].copy()

    if selections.empty:

        raise ValueError(
            "Nessun segnale Momentum disponibile."
        )

    keys = [
        "data_segnale",
        "asset",
        "orizzonte"
    ]

    if selections.duplicated(keys).any():

        raise ValueError(
            "Selezioni Momentum duplicate."
        )

    if observations.duplicated(keys).any():

        raise ValueError(
            "Osservazioni storiche duplicate."
        )

    signals = selections.merge(
        observations[
            keys + [
                "data_ingresso",
                "data_uscita"
            ]
        ],
        on=keys,
        how="left",
        validate="one_to_one"
    )

    if signals[
        ["data_ingresso", "data_uscita"]
    ].isna().any().any():

        raise ValueError(
            "Alcuni segnali non hanno date "
            "di ingresso o uscita."
        )

    signals = signals.sort_values(
        [
            "data_segnale",
            "asset"
        ]
    ).reset_index(drop=True)

    return signals


# ==========================================
# COSTRUZIONE OPERAZIONI
# ==========================================

def build_trades(signals, prices):

    trades = []

    last_exit = None

    for _, signal in signals.iterrows():

        signal_date = signal[
            "data_segnale"
        ]

        entry_date = signal[
            "data_ingresso"
        ]

        exit_date = signal[
            "data_uscita"
        ]

        asset = signal[
            "asset"
        ]

        # Il segnale deve essere disponibile
        # quando non esiste una posizione aperta.
        if last_exit is not None:

            if signal_date < last_exit:
                continue

        if asset not in prices.columns:
            continue

        if (
            entry_date not in prices.index
            or
            exit_date not in prices.index
        ):
            continue

        if not (
            signal_date < entry_date < exit_date
        ):
            continue

        entry_price = prices.at[
            entry_date,
            asset
        ]

        exit_price = prices.at[
            exit_date,
            asset
        ]

        if (
            not np.isfinite(entry_price)
            or
            not np.isfinite(exit_price)
            or
            entry_price <= 0
            or
            exit_price <= 0
        ):
            continue

        entry_index = prices.index.get_loc(
            entry_date
        )

        exit_index = prices.index.get_loc(
            exit_date
        )

        if (
            exit_index - entry_index
            != HORIZON
        ):
            continue

        trades.append({

            "data_segnale": signal_date,

            "data_ingresso": entry_date,

            "data_uscita": exit_date,

            "asset": asset,

            "regime": signal.get(
                "regime",
                "SCONOSCIUTO"
            ),

            "prezzo_ingresso": float(
                entry_price
            ),

            "prezzo_uscita": float(
                exit_price
            ),

        })

        last_exit = exit_date

    result = pd.DataFrame(trades)

    if result.empty:

        raise ValueError(
            "Nessuna operazione valida."
        )

    return result


# ==========================================
# SIMULAZIONE PORTAFOGLIO
# ==========================================

def simulate_portfolio(trades, prices):

    first_date = trades[
        "data_ingresso"
    ].min()

    last_date = trades[
        "data_uscita"
    ].max()

    calendar = prices.loc[
        first_date:last_date
    ].index

    entry_events = {}

    exit_events = {}

    for idx, trade in trades.iterrows():

        entry_events[
            trade["data_ingresso"]
        ] = idx

        exit_events[
            trade["data_uscita"]
        ] = idx

    cash = INITIAL_CAPITAL

    shares = 0.0

    current_asset = None

    invested_days = 0

    equity_rows = []

    trade_rows = []

    active_trade = None

    for date in calendar:

        # ----------------------------------
        # USCITA
        # ----------------------------------

        if date in exit_events:

            idx = exit_events[date]

            trade = trades.loc[idx]

            if (
                current_asset
                == trade["asset"]
                and shares > 0
            ):

                price = prices.at[
                    date,
                    current_asset
                ]

                gross_value = (
                    shares * price
                )

                exit_cost = (
                    gross_value * COST_RATE
                )

                cash = (
                    gross_value - exit_cost
                )

                initial_value = (
                    active_trade[
                        "capitale_pre_ingresso"
                    ]
                )

                net_return = (
                    cash / initial_value - 1
                )

                trade_rows.append({

                    "data_segnale":
                        trade["data_segnale"],

                    "data_ingresso":
                        trade["data_ingresso"],

                    "data_uscita":
                        date,

                    "asset":
                        current_asset,

                    "regime":
                        trade["regime"],

                    "capitale_iniziale":
                        initial_value,

                    "capitale_finale":
                        cash,

                    "rendimento_netto_pct":
                        net_return * 100,

                    "costo_ingresso":
                        active_trade[
                            "costo_ingresso"
                        ],

                    "costo_uscita":
                        exit_cost

                })

                shares = 0.0

                current_asset = None

                active_trade = None

        # ----------------------------------
        # INGRESSO
        # ----------------------------------

        if date in entry_events:

            idx = entry_events[date]

            trade = trades.loc[idx]

            if current_asset is None:

                asset = trade["asset"]

                price = prices.at[
                    date,
                    asset
                ]

                capital_before = cash

                entry_cost = (
                    cash
                    *
                    COST_RATE
                    /
                    (1 + COST_RATE)
                )

                invested_amount = (
                    cash - entry_cost
                )

                shares = (
                    invested_amount / price
                )

                cash = 0.0

                current_asset = asset

                active_trade = {

                    "capitale_pre_ingresso":
                        capital_before,

                    "costo_ingresso":
                        entry_cost

                }

        # ----------------------------------
        # VALORE PORTAFOGLIO
        # ----------------------------------

        if current_asset is not None:

            current_price = prices.at[
                date,
                current_asset
            ]

            if (
                not np.isfinite(current_price)
                or current_price <= 0
            ):

                raise ValueError(
                    "Prezzo mancante durante "
                    f"la posizione {current_asset} "
                    f"del {date.date()}."
                )

            equity = (
                shares * current_price
            )

            invested_days += 1

        else:

            equity = cash

        equity_rows.append({

            "data": date,

            "capitale_momentum": equity,

            "asset": (
                current_asset
                if current_asset
                else "LIQUIDITA"
            ),

            "investito": int(
                current_asset is not None
            )

        })

    equity_df = pd.DataFrame(
        equity_rows
    )

    trades_df = pd.DataFrame(
        trade_rows
    )

    if len(trades_df) != len(trades):

        raise RuntimeError(
            "Numero operazioni chiuse "
            "diverso da quelle previste."
        )

    exposure = (
        invested_days
        /
        len(equity_df)
    )

    return (
        equity_df,
        trades_df,
        exposure
    )


# ==========================================
# BENCHMARK SPY
# ==========================================

def add_benchmark(equity, prices):

    start = equity[
        "data"
    ].iloc[0]

    end = equity[
        "data"
    ].iloc[-1]

    spy = prices.loc[
        start:end,
        "SPY"
    ]

    if (
        spy.isna().any()
        or
        (spy <= 0).any()
    ):

        raise ValueError(
            "Prezzi SPY incompleti."
        )

    initial_spy_price = (
        spy.iloc[0]
    )

    entry_cost = (
        INITIAL_CAPITAL
        *
        COST_RATE
        /
        (1 + COST_RATE)
    )

    invested_capital = (
        INITIAL_CAPITAL
        -
        entry_cost
    )

    spy_shares = (
        invested_capital
        /
        initial_spy_price
    )

    spy_values = (
        spy_shares
        *
        spy
    )

    # Valutazione netta del costo di
    # liquidazione del benchmark.
    spy_values = (
        spy_values
        *
        (1 - COST_RATE)
    )

    equity = equity.copy()

    equity["capitale_spy"] = (
        equity["data"]
        .map(spy_values)
    )

    return equity


# ==========================================
# METRICHE
# ==========================================

def performance_metrics(
    values,
    dates
):

    values = pd.Series(
        values,
        dtype=float
    )

    dates = pd.to_datetime(
        dates
    )

    returns = (
        values.pct_change()
        .dropna()
    )

    final_value = (
        values.iloc[-1]
    )

    initial_value = (
        INITIAL_CAPITAL
    )

    total_return = (
        final_value
        /
        initial_value
        - 1
    )

    days = (
        dates.iloc[-1]
        -
        dates.iloc[0]
    ).days

    years = (
        days / 365.25
    )

    if years > 0:

        cagr = (
            (
                final_value
                /
                initial_value
            ) ** (1 / years)
            - 1
        )

    else:

        cagr = np.nan

    if len(returns) > 1:

        volatility = (
            returns.std(ddof=1)
            *
            np.sqrt(TRADING_DAYS)
        )

    else:

        volatility = np.nan

    if (
        np.isfinite(volatility)
        and volatility > 0
    ):

        sharpe = (
            returns.mean()
            *
            TRADING_DAYS
            /
            volatility
        )

    else:

        sharpe = np.nan

    running_max = (
        values.cummax()
    )

    drawdown = (
        values / running_max - 1
    )

    max_drawdown = (
        drawdown.min()
    )

    return {

        "capitale_finale":
            final_value,

        "rendimento_totale_pct":
            total_return * 100,

        "cagr_pct":
            cagr * 100,

        "volatilita_pct":
            volatility * 100,

        "sharpe":
            sharpe,

        "max_drawdown_pct":
            max_drawdown * 100

    }


# ==========================================
# CONFRONTO
# ==========================================

def compare_models(
    equity,
    trades,
    exposure
):

    momentum = performance_metrics(
        equity["capitale_momentum"],
        equity["data"]
    )

    spy = performance_metrics(
        equity["capitale_spy"],
        equity["data"]
    )

    rows = []

    for model, metrics in [

        ("MOMENTUM", momentum),

        ("SPY_BUY_HOLD", spy)

    ]:

        row = {

            "modello": model,

            "data_inizio":
                equity["data"].iloc[0],

            "data_fine":
                equity["data"].iloc[-1],

            "capitale_iniziale":
                INITIAL_CAPITAL,

            **metrics

        }

        if model == "MOMENTUM":

            row["numero_operazioni"] = (
                len(trades)
            )

            row["win_rate_pct"] = (
                (
                    trades[
                        "rendimento_netto_pct"
                    ] > 0
                ).mean() * 100
            )

            row["esposizione_pct"] = (
                exposure * 100
            )

        else:

            row["numero_operazioni"] = 1

            row["win_rate_pct"] = np.nan

            row["esposizione_pct"] = 100.0

        rows.append(row)

    return pd.DataFrame(
        rows
    )


# ==========================================
# RISULTATI ANNUALI
# ==========================================

def annual_results(equity):

    df = equity.copy()

    df["anno"] = (
        df["data"].dt.year
    )

    rows = []

    for year, group in df.groupby(
        "anno"
    ):

        first = group.iloc[0]

        last = group.iloc[-1]

        for model, column in [

            (
                "MOMENTUM",
                "capitale_momentum"
            ),

            (
                "SPY_BUY_HOLD",
                "capitale_spy"
            )

        ]:

            start_value = (
                first[column]
            )

            end_value = (
                last[column]
            )

            annual_return = (
                end_value
                /
                start_value
                - 1
            )

            rows.append({

                "anno": int(year),

                "modello": model,

                "rendimento_pct":
                    annual_return * 100,

                "capitale_inizio":
                    start_value,

                "capitale_fine":
                    end_value

            })

    return pd.DataFrame(
        rows
    )


# ==========================================
# MAIN
# ==========================================

def main():

    print("=" * 65)

    print(
        "MACRO SWING LAB 4.4"
    )

    print(
        "SIMULAZIONE PORTAFOGLIO MOMENTUM"
    )

    print("=" * 65)

    print(
        "\nCaricamento selezioni..."
    )

    signals = load_signals()

    print(
        "Segnali Momentum:",
        len(signals)
    )

    print(
        "\nScaricamento prezzi..."
    )

    prices = download_prices()

    prices.index = pd.to_datetime(
        prices.index
    )

    prices = prices.sort_index()

    print(
        "\nCostruzione operazioni..."
    )

    trades = build_trades(
        signals,
        prices
    )

    print(
        "Operazioni eseguibili:",
        len(trades)
    )

    print(
        "\nSimulazione capitale..."
    )

    (
        equity,
        completed_trades,
        exposure
    ) = simulate_portfolio(
        trades,
        prices
    )

    equity = add_benchmark(
        equity,
        prices
    )

    comparison = compare_models(
        equity,
        completed_trades,
        exposure
    )

    annual = annual_results(
        equity
    )

    # ======================================
    # SALVATAGGIO
    # ======================================

    equity.to_csv(
        OUT / "equity_44.csv",
        index=False
    )

    completed_trades.to_csv(
        OUT / "operazioni_44.csv",
        index=False
    )

    comparison.to_csv(
        OUT / "confronto_portafoglio_44.csv",
        index=False
    )

    annual.to_csv(
        OUT / "portafoglio_annuale_44.csv",
        index=False
    )

    # ======================================
    # REPORT
    # ======================================

    print(
        "\nRISULTATI PORTAFOGLIO:"
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    print(
        "\nRISULTATI ANNUALI:"
    )

    print(
        annual.to_string(
            index=False
        )
    )

    print(
        "\nFILE GENERATI:"
    )

    print(
        "output/equity_44.csv"
    )

    print(
        "output/operazioni_44.csv"
    )

    print(
        "output/confronto_portafoglio_44.csv"
    )

    print(
        "output/portafoglio_annuale_44.csv"
    )

    print(
        "\nATTENZIONE:"
    )

    print(
        "Simulazione esplorativa con "
        "prezzi adjusted close."
    )

    print(
        "I risultati non rappresentano "
        "operazioni reali garantite."
    )

    print(
        "\nMACRO SWING LAB 4.4 COMPLETATO."
    )


if __name__ == "__main__":
    main()
