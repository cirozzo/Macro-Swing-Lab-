#!/usr/bin/env python3
"""Macro Swing Lab 3.0 — weekly ETF research, macro vs momentum.
Exploratory research, NOT point-in-time macro: revised FRED series are used.
Run: pip install pandas numpy yfinance ; python backtest_macro.py
"""
from pathlib import Path
from datetime import datetime, timedelta, timezone
import io
import urllib.request
import numpy as np
import pandas as pd
import yfinance as yf

START = '2007-01-01'
FIRST_SIGNAL = '2010-01-01'
SPLIT = '2022-01-01'
END = (datetime.now(timezone.utc).date() + timedelta(days=1)).isoformat()

TICKERS = [
    'SPY', 'XLK', 'XLE', 'XLF',
    'XLI', 'XLB', 'XLV', 'XLP',
    'XLY', 'XLU', 'XLRE', 'XLC'
]

HORIZONS = (5, 10, 20)
TRAIN_WEEKS = 104
MIN_REGIME_TRAIN = 12

OUT = Path(__file__).resolve().parent / 'output'
OUT.mkdir(parents=True, exist_ok=True)


def fred(series):
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}'

    with urllib.request.urlopen(url, timeout=60) as response:
        csv = response.read().decode('utf-8-sig')

    df = pd.read_csv(io.StringIO(csv))
    df.columns = ['Date', series]
    df['Date'] = pd.to_datetime(df['Date'])
    df[series] = pd.to_numeric(df[series], errors='coerce')

    return df.set_index('Date')[series].dropna().sort_index()


def download_prices():
    raw = yf.download(
        TICKERS,
        start=START,
        end=END,
        auto_adjust=True,
        progress=False,
        threads=False,
        group_by='column'
    )

    if raw.empty:
        raise RuntimeError('Download prezzi fallito')

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
    else:
        close = raw[['Close']].rename(columns={'Close': 'SPY'})

    close.index = (
        pd.to_datetime(close.index)
        .tz_localize(None)
        .normalize()
    )

    close = close.loc[
        :,
        [c for c in TICKERS if c in close.columns]
    ].sort_index()

    close = close.apply(pd.to_numeric, errors='coerce')

    if 'SPY' not in close or close['SPY'].dropna().empty:
        raise RuntimeError('SPY non disponibile')

    return close


def regime_table(indpro, core):
    ip = indpro.copy()
    cp = core.copy()

    ip.index = ip.index.to_period('M')
    cp.index = cp.index.to_period('M')

    periods = pd.period_range(
        '2007-01',
        pd.Timestamp.today().strftime('%Y-%m'),
        freq='M'
    )

    ip = ip.groupby(level=0).last().reindex(periods)
    cp = cp.groupby(level=0).last().reindex(periods)

    acceleration = (ip / ip.shift(12) - 1).diff(3)
    inflation = (
        cp / cp.shift(3)
        - cp.shift(3) / cp.shift(6)
    )

    regime = pd.Series('', index=periods, dtype='object')

    regime.loc[
        (acceleration > 0) & (inflation < 0)
    ] = 'R1 crescita+ inflazione-'

    regime.loc[
        (acceleration > 0) & (inflation > 0)
    ] = 'R2 crescita+ inflazione+'

    regime.loc[
        (acceleration < 0) & (inflation < 0)
    ] = 'R3 crescita- inflazione-'

    regime.loc[
        (acceleration < 0) & (inflation > 0)
    ] = 'R4 crescita- inflazione+'

    return regime


def observations(px, macro_regimes):
    dates = px.index

    signal_dates = px.groupby(
        px.index.to_period('W-FRI')
    ).apply(lambda g: g.index[-1])

    records = []

    for dt in signal_dates:

        if dt < pd.Timestamp(FIRST_SIGNAL):
            continue

        pos = dates.get_loc(dt)

        macro_month = dt.to_period('M') - 3
        regime = macro_regimes.get(macro_month, '')

        if not regime:
            continue

        for ticker in TICKERS:

            if ticker == 'SPY' or ticker not in px:
                continue

            prev20 = pos - 20

            if prev20 < 0:
                continue

            current = px.at[dt, ticker]
            spy_now = px.at[dt, 'SPY']

            if not (
                np.isfinite(current)
                and np.isfinite(spy_now)
                and current > 0
                and spy_now > 0
            ):
                continue

            def relative_past(days):
                a, b = px.iloc[pos - days][
                    [ticker, 'SPY']
                ]

                if not (
                    np.isfinite(a)
                    and np.isfinite(b)
                    and a > 0
                    and b > 0
                ):
                    return np.nan

                return (
                    (current / a - 1)
                    - (spy_now / b - 1)
                )

            mom5 = relative_past(5)
            mom20 = relative_past(20)

            for h in HORIZONS:

                endpos = pos + h

                if endpos >= len(px):
                    continue

                future = px.iloc[endpos]

                asset_end = future[ticker]
                spy_end = future['SPY']

                if not (
                    np.isfinite(asset_end)
                    and np.isfinite(spy_end)
                ):
                    continue

                ret = asset_end / current - 1
                spyret = spy_end / spy_now - 1

                records.append(dict(
                    data_segnale=dt.date().isoformat(),
                    mese_macro_usato=str(macro_month),
                    regime=regime,
                    asset=ticker,
                    orizzonte_sedute=h,
                    data_uscita=dates[endpos].date().isoformat(),
                    rendimento=ret,
                    rendimento_spy=spyret,
                    excess_return=ret - spyret,
                    campione=(
                        'SVILUPPO'
                        if dt < pd.Timestamp(SPLIT)
                        else 'TEST'
                    ),
                    momentum_rel_5=mom5,
                    momentum_rel_20=mom20
                ))

    if not records:
        raise RuntimeError('Nessuna osservazione elaborata')

    return pd.DataFrame.from_records(records)


def historical_strategies(df):
    """
    Confronto di tre strategie:
    MOMENTUM, MACRO e MACRO_MOMENTUM.

    Utilizza soltanto risultati storici gia'
    conclusi alla data del segnale.
    """

    data = df.copy()

    data['date'] = pd.to_datetime(
        data.data_segnale
    )

    data['exit_date'] = pd.to_datetime(
        data.data_uscita
    )

    results = []

    unique_dates = sorted(data['date'].unique())

    for h in HORIZONS:

        src = data[
            data.orizzonte_sedute == h
        ]

        for dt in unique_dates:

            snapshot = src[
                src.date == dt
            ].dropna(subset=['momentum_rel_20'])

            if len(snapshot) < 5:
                continue

            cutoff = dt - pd.Timedelta(
                weeks=TRAIN_WEEKS
            )

            prior = src[
                (src.exit_date < dt)
                & (src.date >= cutoff)
            ]

            current_regime = snapshot.regime.iloc[0]

            cond = prior[
                prior.regime == current_regime
            ]

            hist = cond.groupby('asset').agg(
                mean_excess=('excess_return', 'mean'),
                n=('excess_return', 'size')
            )

            hist = hist[
                hist.n >= MIN_REGIME_TRAIN
            ]

            mom = snapshot.set_index('asset')[
                'momentum_rel_20'
            ]

            picks = {
                'MOMENTUM': mom.idxmax()
            }

            available = hist.index.intersection(
                mom.index
            )

            if len(available) >= 3:

                picks['MACRO'] = hist.loc[
                    available, 'mean_excess'
                ].idxmax()

                macro_rank = hist.loc[
                    available, 'mean_excess'
                ].rank(pct=True)

                mom_rank = mom.loc[
                    available
                ].rank(pct=True)

                picks['MACRO_MOMENTUM'] = (
                    macro_rank + mom_rank
                ).idxmax()

            for name, ticker in picks.items():

                observation = snapshot[
                    snapshot.asset == ticker
                ].iloc[0]

                results.append(dict(
                    data_segnale=observation.data_segnale,
                    data_uscita=observation.data_uscita,
                    campione=observation.campione,
                    regime=current_regime,
                    orizzonte_sedute=h,
                    strategia=name,
                    asset=ticker,
                    rendimento=observation.rendimento,
                    rendimento_spy=observation.rendimento_spy,
                    excess_return=observation.excess_return
                ))

    return pd.DataFrame(results)


def summarize(raw, by):
    rows = []

    for keys, g in raw.groupby(by):

        if not isinstance(keys, tuple):
            keys = (keys,)

        ex = g.excess_return

        row = dict(zip(by, keys))

        row.update(
            n=len(g),
            rendimento_medio=g.rendimento.mean(),
            excess_medio=ex.mean(),
            excess_mediano=ex.median(),
            prob_sovraperformance=(ex > 0).mean(),
            prob_rendimento_positivo=(
                g.rendimento > 0
            ).mean(),
            dev_std_excess=ex.std(ddof=1),
            guadagno_medio=g.loc[
                g.rendimento > 0,
                'rendimento'
            ].mean(),
            perdita_media=g.loc[
                g.rendimento < 0,
                'rendimento'
            ].mean(),
            excess_vincente=g.loc[
                ex > 0,
                'excess_return'
            ].mean(),
            excess_perdente=g.loc[
                ex < 0,
                'excess_return'
            ].mean()
        )

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    print('Download prezzi ETF e macro revisionata...')

    px = download_prices()

    regime = regime_table(
        fred('INDPRO'),
        fred('PCEPILFE')
    )

    obs = observations(px, regime)

    obs.to_csv(
        OUT / 'osservazioni.csv',
        index=False,
        float_format='%.8f'
    )

    summ = summarize(
        obs,
        ['campione', 'regime', 'orizzonte_sedute', 'asset']
    )

    summ.to_csv(
        OUT / 'risultati.csv',
        index=False,
        float_format='%.8f'
    )

    strat = historical_strategies(obs)

    strat.to_csv(
        OUT / 'strategie.csv',
        index=False,
        float_format='%.8f'
    )

    if not strat.empty:
        summarize(
            strat,
            ['campione', 'orizzonte_sedute', 'strategia']
        ).to_csv(
            OUT / 'confronto_strategie.csv',
            index=False,
            float_format='%.8f'
        )

    print(
        f'Date uniche: {obs.data_segnale.nunique()}, '
        f'osservazioni: {len(obs)}, '
        f'scelte: {len(strat)}'
    )

    print(
        'ATTENZIONE: dati macro FRED revisionati; '
        'backtest NON point-in-time.'
    )

    print(
        'ATTENZIONE: rendimenti sovrapposti, '
        'prezzi close-to-close e costi esclusi.'
    )


if __name__ == '__main__':
    main()
