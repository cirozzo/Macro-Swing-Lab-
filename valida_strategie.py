#!/usr/bin/env python3
"""Macro Swing Lab 3.1: confronto appaiato e bootstrap a blocchi.
Usa output/strategie.csv generato dal motore 3.0.
Non corregge revisioni FRED, look-ahead macro o prezzi close-to-close.
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / 'output'
OUT.mkdir(exist_ok=True)

SEED = 314159
N_BOOT = 2000

STRATEGIES = [
    'MACRO',
    'MOMENTUM',
    'MACRO_MOMENTUM'
]


def block_ci(arr, horizon, rng):
    """Intervallo di confidenza tramite bootstrap a blocchi."""

    x = np.asarray(arr, dtype=float)
    x = x[np.isfinite(x)]

    n = len(x)

    if n < 8:
        return (np.nan, np.nan)

    # Blocchi consecutivi di settimane.
    block = min(
        n,
        max(2, int(np.ceil(horizon / 5)))
    )

    starts = np.arange(n - block + 1)
    boot = np.empty(N_BOOT)

    for j in range(N_BOOT):

        out = []

        while len(out) < n:
            a = rng.choice(starts)
            out.extend(x[a:a + block])

        boot[j] = np.mean(out[:n])

    return tuple(
        np.quantile(boot, [0.025, 0.975])
    )


def evaluate(frame, sample, horizon, regime_label, rng):

    wide = frame.pivot_table(
        index='data_segnale',
        columns='strategia',
        values='excess_return',
        aggfunc='first'
    )

    if not set(STRATEGIES).issubset(wide.columns):
        return []

    wide = wide[STRATEGIES].dropna().sort_index()

    if wide.empty:
        return []

    out = []

    mom = wide['MOMENTUM'].to_numpy()

    for name in STRATEGIES:

        vals = wide[name].to_numpy()

        diff = vals - mom

        ci = block_ci(vals, horizon, rng)

        if name != 'MOMENTUM':
            ci_diff = block_ci(diff, horizon, rng)
        else:
            ci_diff = (0.0, 0.0)

        out.append(dict(

            campione=sample,

            orizzonte_sedute=horizon,

            regime=regime_label,

            strategia=name,

            date_comuni=len(wide),

            excess_medio=vals.mean(),

            excess_mediano=np.median(vals),

            prob_batte_spy=(vals > 0).mean(),

            excess_negativo_medio=(
                vals[vals < 0].mean()
                if (vals < 0).any()
                else np.nan
            ),

            ci95_excess_basso=ci[0],

            ci95_excess_alto=ci[1],

            differenza_vs_momentum=diff.mean(),

            ci95_diff_basso=ci_diff[0],

            ci95_diff_alto=ci_diff[1]

        ))

    return out


def main():

    path = OUT / 'strategie.csv'

    if not path.exists():
        raise FileNotFoundError(
            f'Manca {path}; esegui prima backtest_macro.py'
        )

    df = pd.read_csv(path)

    required = {
        'data_segnale',
        'campione',
        'orizzonte_sedute',
        'regime',
        'strategia',
        'excess_return'
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f'Colonne mancanti: {sorted(missing)}'
        )

    df['excess_return'] = pd.to_numeric(
        df['excess_return'],
        errors='coerce'
    )

    df['data_segnale'] = pd.to_datetime(
        df['data_segnale'],
        errors='coerce'
    )

    df = df.dropna(
        subset=['excess_return', 'data_segnale']
    )

    if df.duplicated([
        'campione',
        'orizzonte_sedute',
        'data_segnale',
        'strategia'
    ]).any():

        raise ValueError(
            'Segnali duplicati per strategia/data/orizzonte'
        )

    rng = np.random.default_rng(SEED)

    rows = []

    for (sample, horizon), g in df.groupby([
        'campione',
        'orizzonte_sedute'
    ]):

        horizon = int(horizon)

        rows.extend(
            evaluate(
                g,
                sample,
                horizon,
                'TUTTI',
                rng
            )
        )

        for regime, subset in g.groupby('regime'):

            rows.extend(
                evaluate(
                    subset,
                    sample,
                    horizon,
                    regime,
                    rng
                )
            )

    if not rows:
        raise RuntimeError(
            'Nessuna data comune fra le tre strategie'
        )

    result = pd.DataFrame(rows)

    file = OUT / 'validazione_31.csv'

    result.to_csv(
        file,
        index=False,
        float_format='%.8f'
    )

    print(
        f'Creato {file}: {len(result)} righe'
    )

    print(
        'ATTENZIONE: intervalli di confidenza '
        'esplorativi; dati macro revisionati; '
        'costi operativi esclusi.'
    )

    print(
        'I periodi gia analizzati non sono '
        'test indipendenti incontaminati.'
    )


if __name__ == '__main__':
    main()
