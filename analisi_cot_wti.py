"""Ciclo 15: CFTC WTI futures-only, PIT prudenziale, rendimenti Daily forward.
Eseguire in ambiente con Internet: python analisi_cot_wti.py
"""

import io
import zipfile
import requests
import pandas as pd
import numpy as np
import pathlib
import json

ROOT = pathlib.Path(__file__).resolve().parent
PRICE = ROOT / 'TVC_USOIL, 1D.zip'
OUT = ROOT / 'risultati'
OUT.mkdir(exist_ok=True)


def price():
    with zipfile.ZipFile(PRICE) as z:
        f = next(
            n for n in z.namelist()
            if n.endswith('.csv') and not n.startswith('__')
        )
        d = pd.read_csv(z.open(f))

    d['date'] = (
        pd.to_datetime(d.time, unit='s', utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
    )

    d = d[
        (d.date >= '2009-01-01') &
        (d.date <= '2026-11-10') &
        (d.close > 0)
    ].sort_values('date').drop_duplicates('date', keep='last')

    for h in [5, 10]:
        d[f'fwd_{h}d_pct'] = 100 * (
            d.close.shift(-h) / d.close - 1
        )

    return d


def load_cot():
    frames = []

    for year in range(2009, 2027):
        urls = [
            f'https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip',
            f'https://www.cftc.gov/files/dea/history/fut_disagg_txt_hist_{year}.zip'
        ]

        data = None

        for url in urls:
            try:
                r = requests.get(
                    url,
                    timeout=35,
                    headers={
                        'User-Agent': 'MacroSwingLab-research/1.0 research project'
                    }
                )

                r.raise_for_status()

                z = zipfile.ZipFile(io.BytesIO(r.content))

                name = max(
                    (
                        x for x in z.namelist()
                        if x.lower().endswith(('.txt', '.csv'))
                    ),
                    key=lambda x: z.getinfo(x).file_size
                )

                data = pd.read_csv(
                    z.open(name),
                    low_memory=False
                )

                break

            except Exception:
                continue

        if data is None:
            print('MISSING', year)
            continue

        data.columns = [
            str(x).strip()
            for x in data.columns
        ]

        frames.append(data)

    if not frames:
        raise RuntimeError(
            'Nessun archivio CFTC scaricato. '
            'Verificare accesso Internet.'
        )

    return pd.concat(frames, ignore_index=True)


def col(d, *parts):
    """
    Cerca prima il nome esatto della colonna.

    Correzione:
    Open_Interest_All non deve essere confuso con
    Change_in_Open_Interest_All o
    Pct_of_Open_Interest_All.
    """

    requested = '_'.join(parts)
    names = list(d.columns)

    exact = [
        c for c in names
        if c.casefold() == requested.casefold()
    ]

    if len(exact) == 1:
        return exact[0]

    matches = [
        c for c in names
        if all(
            p.casefold() in c.casefold()
            for p in parts
        )
    ]

    if len(matches) != 1:
        raise ValueError(
            f'Colonna ambigua/mancante {parts}: '
            f'{matches}; colonne: {names}'
        )

    return matches[0]


def main():

    d = load_cot()

    market = (
        col(d, 'Market_and_Exchange_Names')
        if 'Market_and_Exchange_Names' in d
        else col(d, 'Market', 'Exchange')
    )

    subset = d[
        d[market].astype(str).str.contains(
            'CRUDE OIL, LIGHT SWEET',
            case=False,
            regex=False
        )
    ].copy()

    if subset.empty:
        raise ValueError(
            'Nessun contratto NYMEX light sweet crude oil; '
            'verificare codifica CFTC'
        )

    print(
        'Mercati:',
        subset[market].value_counts().to_dict()
    )

    # Disaggregated Futures Only: posizioni Managed Money.

    date = (
        col(subset, 'Report_Date_as_YYYY-MM-DD')
        if 'Report_Date_as_YYYY-MM-DD' in subset
        else col(subset, 'Report_Date')
    )

    long = col(
        subset,
        'M_Money_Positions_Long_All'
    )

    short = col(
        subset,
        'M_Money_Positions_Short_All'
    )

    oi = col(
        subset,
        'Open_Interest_All'
    )

    subset['report_date'] = (
        pd.to_datetime(
            subset[date],
            errors='coerce'
        ).dt.normalize()
    )

    for c in [long, short, oi]:
        subset[c] = pd.to_numeric(
            subset[c],
            errors='coerce'
        )

    subset = (
        subset
        .dropna(
            subset=[
                'report_date',
                long,
                short,
                oi
            ]
        )
        .sort_values('report_date')
        .drop_duplicates('report_date')
    )

    subset['net_mm'] = (
        subset[long] - subset[short]
    )

    subset['net_pct_oi'] = (
        subset.net_mm / subset[oi]
    )

    subset['delta_4w'] = (
        subset.net_pct_oi -
        subset.net_pct_oi.shift(4)
    )

    # Report normalmente riferito al martedì,
    # pubblicato il venerdì.
    # Prima disponibilità prudenziale:
    # lunedì successivo (martedì + 6 giorni).
    #
    # I ritardi straordinari richiedono
    # una verifica delle date effettive.

    subset['usable_from'] = (
        subset.report_date +
        pd.Timedelta(days=6)
    )

    subset[
        [
            'report_date',
            'usable_from',
            'net_mm',
            'net_pct_oi',
            'delta_4w'
        ]
    ].to_csv(
        OUT / 'cot_wti_settimanale.csv',
        index=False
    )

    px = price()

    joined = pd.merge_asof(
        px.sort_values('date'),
        subset[
            [
                'usable_from',
                'report_date',
                'net_pct_oi',
                'delta_4w'
            ]
        ].sort_values('usable_from'),
        left_on='date',
        right_on='usable_from',
        direction='backward'
    )

    joined['cot_positive_delta'] = (
        joined.delta_4w > 0
    )

    joined['cot_negative_delta'] = (
        joined.delta_4w < 0
    )

    joined['year'] = joined.date.dt.year

    joined.to_csv(
        OUT / 'wti_daily_cot_pit_prudenziale.csv',
        index=False
    )

    rows = []

    for year, g in joined.groupby('year'):

        groups = [
            (
                'all',
                g.delta_4w.notna()
            ),
            (
                'cot_delta_positive',
                g.cot_positive_delta
            ),
            (
                'cot_delta_negative',
                g.cot_negative_delta
            )
        ]

        for label, mask in groups:

            for h in [5, 10]:

                v = g.loc[
                    mask,
                    f'fwd_{h}d_pct'
                ].dropna()

                rows.append(
                    {
                        'year': year,
                        'group': label,
                        'horizon_days': h,
                        'n': len(v),
                        'mean_pct': v.mean(),
                        'median_pct': v.median(),
                        'positive_share': (v > 0).mean()
                    }
                )

    pd.DataFrame(rows).to_csv(
        OUT / 'risultati_annuali.csv',
        index=False
    )

    audit = {
        'records_cot': len(subset),
        'records_price': len(px),
        'first_cot': str(
            subset.report_date.min()
        ),
        'last_cot': str(
            subset.report_date.max()
        ),
        'release_assumption': (
            'Tuesday + 6 calendar days, '
            'not verified exceptional releases'
        ),
        'warning': (
            'Descriptive forward returns overlap; '
            'not a tradable backtest, '
            'no macro causal attribution, '
            'not certified PIT.'
        )
    }

    (
        OUT / 'audit.json'
    ).write_text(
        json.dumps(
            audit,
            indent=2
        )
    )

    print('OK:', OUT)


if __name__ == '__main__':
    main()
