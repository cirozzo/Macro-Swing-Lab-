#!/usr/bin/env python3
"""Macro Swing Lab 5.1: independent ALFRED vintage audit; no trading claims."""
import argparse
import json
import os
import random
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

URL = 'https://api.stlouisfed.org/fred/series/observations'
SERIES = {'INDPRO': ('growth_change', 'growth_last_observation'),
          'PCEPILFE': ('inflation_change', 'inflation_last_observation')}
LABELS = ('R1_CRESCITA_DISINFLAZIONE', 'R2_CRESCITA_INFLAZIONE',
          'R3_RALLENTAMENTO_INFLAZIONE', 'R4_RALLENTAMENTO_DISINFLAZIONE')


def fetch_one(series, vintage, key):
    params = {'series_id': series, 'api_key': key, 'file_type': 'json',
              'output_type': 1, 'realtime_start': vintage, 'realtime_end': vintage,
              'observation_start': '2005-01-01', 'observation_end': vintage,
              'limit': 100000}
    url = URL + '?' + urlencode(params)
    for attempt in range(7):
        try:
            with urlopen(Request(url, headers={'User-Agent': 'MacroSwingLab-Audit/5.1'}), timeout=90) as r:
                payload = json.load(r)
            obs = payload.get('observations')
            if not isinstance(obs, list):
                raise ValueError(f'ALFRED invalid response for {series} {vintage}')
            if int(payload.get('count', len(obs))) > len(obs):
                raise ValueError(f'ALFRED truncated response for {series} {vintage}')
            return obs
        except HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == 6:
                raise RuntimeError(f'HTTP {e.code}: {series} {vintage}') from e
            wait = min(120, 5 * 2**attempt)
            if e.headers.get('Retry-After'):
                try:
                    wait = max(wait, min(180, float(e.headers['Retry-After'])))
                except ValueError:
                    pass
        except (URLError, TimeoutError) as e:
            if attempt == 6:
                raise RuntimeError(f'Network failure {series} {vintage}: {e}') from e
            wait = min(120, 5 * 2**attempt)
        time.sleep(wait + random.uniform(0, 1))
    raise AssertionError('unreachable')


def independent_feature(observations):
    if not observations:
        return np.nan, None
    table = pd.DataFrame(observations)
    table['date'] = pd.to_datetime(table['date'])
    table['value'] = pd.to_numeric(table['value'], errors='coerce')
    table = table.dropna(subset=['value']).sort_values('date')
    if table.empty:
        return np.nan, None
    if table.date.duplicated().any():
        raise ValueError('Duplicate observation dates returned by ALFRED')
    values = pd.Series(table.value.to_numpy(), index=table.date)
    yoy = (values / values.shift(12) - 1) * 100
    delta = (yoy - yoy.shift(3)).dropna()
    if delta.empty:
        return np.nan, None
    return float(delta.iloc[-1]), delta.index[-1].strftime('%Y-%m-%d')


def choose_indices(frame, sample_size):
    n = len(frame)
    selected = {0, n - 1}
    # Include regime boundaries, and both sides of the transition.
    change = frame['regime_pit_49'].ne(frame['regime_pit_49'].shift())
    boundaries = np.flatnonzero(change.to_numpy())[1:]
    for i in boundaries:
        selected.add(int(i))
        selected.add(int(i - 1))
    # Stratify evenly by calendar year and regime to cover the entire period.
    for _, group in frame.groupby(frame.data_segnale.dt.year):
        selected.add(int(group.index[len(group) // 2]))
    for _, group in frame.groupby('regime_pit_49'):
        selected.add(int(group.index[len(group) // 2]))
    rng = random.Random(51049)
    if len(selected) > sample_size:
        mandatory = {0, n - 1}
        pool = sorted(selected - mandatory)
        selected = mandatory | set(rng.sample(pool, sample_size - len(mandatory)))
    else:
        pool = sorted(set(range(n)) - selected)
        selected |= set(rng.sample(pool, min(sample_size - len(selected), len(pool))))
    return sorted(selected)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='input/regimi_pit_49.csv')
    p.add_argument('--out', default='output/audit_pit_51')
    p.add_argument('--sample-size', type=int, default=40)
    args = p.parse_args()
    key = os.getenv('FRED_API_KEY', '').strip()
    if not key:
        raise SystemExit('FRED_API_KEY missing (must be GitHub Actions secret)')
    frame = pd.read_csv(args.input)
    required = ['data_segnale', 'vintage_asof', 'regime_pit_49',
                'growth_change', 'inflation_change',
                'growth_last_observation', 'inflation_last_observation']
    missing = set(required) - set(frame.columns)
    if missing:
        raise SystemExit(f'Missing columns: {sorted(missing)}')
    frame['data_segnale'] = pd.to_datetime(frame['data_segnale'], errors='raise')
    frame['vintage_asof'] = pd.to_datetime(frame['vintage_asof'], errors='raise')
    frame = frame.sort_values('data_segnale').reset_index(drop=True)
    errors = []
    if frame.data_segnale.duplicated().any(): errors.append('Duplicate signal dates')
    if not (frame.vintage_asof < frame.data_segnale).all(): errors.append('Vintage not earlier than signal')
    if not (frame.data_segnale.dt.weekday == 4).all(): errors.append('Not all signals are Fridays')
    if not ((frame.data_segnale - frame.vintage_asof).dt.days == 1).all(): errors.append('Vintage not Thursday')
    if not (frame.data_segnale.diff().dropna().dt.days == 7).all(): errors.append('Nonweekly cadence')
    if not frame.regime_pit_49.isin(LABELS).all(): errors.append('Unknown regime labels')
    g = frame.growth_change
    i = frame.inflation_change
    expected = np.select([(g >= 0) & (i < 0), (g >= 0) & (i >= 0),
                          (g < 0) & (i >= 0), (g < 0) & (i < 0)], LABELS, default='SCONOSCIUTO')
    if (expected != frame.regime_pit_49).any(): errors.append('Regime labels inconsistent with features')
    for col in ('growth_last_observation', 'inflation_last_observation'):
        dates = pd.to_datetime(frame[col], errors='coerce')
        if dates.isna().any(): errors.append(f'Missing {col}')
        if (dates > frame.vintage_asof).any(): errors.append(f'Future observations in {col}')
    if errors:
        raise SystemExit('STRUCTURAL FAIL: ' + '; '.join(errors))

    sample = choose_indices(frame, max(2, min(args.sample_size, len(frame))))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    audit = []
    for position, idx in enumerate(sample, 1):
        row = frame.iloc[idx]
        vintage = row.vintage_asof.strftime('%Y-%m-%d')
        for series, (col, last_col) in SERIES.items():
            obs = fetch_one(series, vintage, key)
            value, last_date = independent_feature(obs)
            stored = float(row[col])
            abs_error = abs(value - stored)
            passed = bool(np.isfinite(value) and np.isclose(value, stored, rtol=1e-8, atol=1e-7)
                          and last_date == row[last_col])
            audit.append({'data_segnale': row.data_segnale.strftime('%Y-%m-%d'),
                          'vintage_asof': vintage, 'series': series,
                          'stored_feature': stored, 'alfred_feature': value,
                          'absolute_error': abs_error, 'stored_last_observation': row[last_col],
                          'alfred_last_observation': last_date, 'match': passed})
            time.sleep(1.0)
        print(f'Audited {position}/{len(sample)} vintage dates', flush=True)
    results = pd.DataFrame(audit)
    results.to_csv(out / 'confronto_vintage_51.csv', index=False)
    # Conservative executable trading alignment: signal Friday AFTER Thursday as-of;
    # any ETF trade must use next trading session open or later, never same Friday open.
    schedule = frame[['data_segnale', 'vintage_asof', 'regime_pit_49']].copy()
    schedule['earliest_candidate_entry'] = schedule.data_segnale + pd.offsets.BDay(1)
    schedule['entry_note'] = 'Next business day is indicative; verify US exchange holidays and actual ETF price timestamps.'
    schedule.to_csv(out / 'calendario_segnali_51.csv', index=False)
    failed = results.loc[~results['match']]
    report = {'status': 'PASS_CAMPIONE' if failed.empty else 'FAIL_CAMPIONE',
              'dataset_rows': len(frame), 'sampled_vintages': len(sample),
              'independent_api_comparisons': len(results), 'mismatches': len(failed),
              'sample_method': 'Fixed-seed 51049; boundary/year/regime coverage + random',
              'limitations': ['Sample audit does not certify all historical vintages',
                              'No intraday release-time verification',
                              'US exchange holiday adjustment not yet applied',
                              'No ETF price data or out-of-sample predictive test',
                              'ALFRED API data may be corrected after the audit date']}
    (out / 'report_51.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    if not failed.empty:
        raise SystemExit(f'PIT AUDIT FAILED: {len(failed)} discrepancies; see artifact')


if __name__ == '__main__':
    main()
