#!/usr/bin/env python3
"""Macro Swing Lab 4.9 — ALFRED point-in-time weekly regime snapshots.

Batch exact Thursday vintages with ALFRED output_type=2. Never substitute
revised/current data for unavailable historical vintages.
"""
import json
import os
import random
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

START = '2007-01-01'
SERIES = ('INDPRO', 'PCEPILFE')
BATCH_SIZE = 20  # conservative URL/response size; ~104 calls instead of ~2066
OUT = Path('output')
API_URL = 'https://api.stlouisfed.org/fred/series/observations'


def fetch_batch(series, dates, key):
    params = {
        'series_id': series, 'api_key': key, 'file_type': 'json',
        'output_type': 2, 'vintage_dates': ','.join(dates),
        'observation_start': '2005-01-01',
        'observation_end': dates[-1], 'limit': 100000,
    }
    url = API_URL + '?' + urlencode(params)
    for attempt in range(9):
        try:
            with urlopen(Request(url, headers={'User-Agent': 'MacroSwingLab-PIT/4.9'}), timeout=90) as response:
                payload = json.load(response)
            if 'observations' not in payload:
                raise RuntimeError(f'Invalid ALFRED payload for {series}')
            if int(payload.get('count', 0)) > len(payload['observations']):
                raise RuntimeError(f'Truncated ALFRED response for {series}: reduce BATCH_SIZE')
            return payload['observations']
        except HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 8:
                raise RuntimeError(f'ALFRED HTTP {exc.code} for {series}, batch {dates[0]}..{dates[-1]}') from None
            retry_after = exc.headers.get('Retry-After')
            try:
                delay = min(180, float(retry_after)) if retry_after else min(180, 8 * 2**attempt)
            except ValueError:
                delay = min(180, 8 * 2**attempt)
        except (TimeoutError, URLError) as exc:
            if attempt == 8:
                raise RuntimeError(f'ALFRED connection failure for {series}: {exc}') from None
            delay = min(180, 8 * 2**attempt)
        print(f'ALFRED retry {attempt+1}/9 for {series}; sleeping {delay:.0f}s', flush=True)
        time.sleep(delay + random.uniform(0, 2))
    raise RuntimeError('Unexpected retry exhaustion')


def decode_batch(observations, dates, series):
    """Output type 2 is a wide vintage table: date + one column per vintage."""
    table = pd.DataFrame(observations)
    if table.empty:
        return {v: pd.Series(dtype='float64') for v in dates}
    if 'date' not in table:
        raise RuntimeError(f'Missing observation date in {series} batch')
    table['date'] = pd.to_datetime(table['date'])
    result = {}
    for vintage in dates:
        if vintage not in table.columns:
            raise RuntimeError(f'ALFRED did not return vintage {vintage} for {series}; columns={list(table.columns)[:8]}')
        values = pd.to_numeric(table[vintage], errors='coerce')
        result[vintage] = pd.Series(values.to_numpy(), index=table['date']).sort_index().dropna()
    return result


def feature(values):
    if values.empty:
        return None, None
    # Same definition as original 4.9: 3-month change in 12-month YoY rate.
    yoy = values.pct_change(12, fill_method=None) * 100
    change = yoy.diff(3).dropna()
    if change.empty:
        return None, None
    return float(change.iloc[-1]), change.index[-1].strftime('%Y-%m-%d')


def classify(g, i):
    if g is None or i is None:
        return 'SCONOSCIUTO'
    if g >= 0 and i < 0:
        return 'R1_CRESCITA_DISINFLAZIONE'
    if g >= 0 and i >= 0:
        return 'R2_CRESCITA_INFLAZIONE'
    if g < 0 and i >= 0:
        return 'R3_RALLENTAMENTO_INFLAZIONE'
    return 'R4_RALLENTAMENTO_DISINFLAZIONE'


def main():
    key = os.getenv('FRED_API_KEY', '').strip()
    if not key:
        raise RuntimeError('Missing FRED_API_KEY GitHub Actions secret')
    OUT.mkdir(exist_ok=True)
    snapshots = pd.date_range(START, pd.Timestamp.now(tz='UTC').date(), freq='W-FRI')
    pairs = [(day.strftime('%Y-%m-%d'), (day - pd.Timedelta(days=1)).strftime('%Y-%m-%d')) for day in snapshots]
    vintages = [v for _, v in pairs]
    snapshots_by_series = {s: {} for s in SERIES}
    calls = 0
    for start in range(0, len(vintages), BATCH_SIZE):
        chunk = vintages[start:start+BATCH_SIZE]
        for series in SERIES:
            raw = fetch_batch(series, chunk, key)
            snapshots_by_series[series].update(decode_batch(raw, chunk, series))
            calls += 1
            time.sleep(1.2)  # additional pacing between successful API calls
        print(f'Fetched {min(start+BATCH_SIZE,len(vintages))}/{len(vintages)} weekly vintages ({calls} requests)', flush=True)
    rows = []
    for decision, vintage in pairs:
        growth, gd = feature(snapshots_by_series['INDPRO'][vintage])
        inflation, idate = feature(snapshots_by_series['PCEPILFE'][vintage])
        rows.append({
            'data_segnale': decision, 'vintage_asof': vintage,
            'growth_change': growth, 'inflation_change': inflation,
            'growth_last_observation': gd, 'inflation_last_observation': idate,
            'regime_pit_49': classify(growth, inflation),
        })
    result = pd.DataFrame(rows)
    assert result.data_segnale.is_unique
    assert (result.vintage_asof < result.data_segnale).all()
    result.to_csv(OUT / 'regimi_pit_49.csv', index=False)
    print(f'Saved {len(result)} weekly PIT snapshots with {calls} API requests')
    print(result.regime_pit_49.value_counts(dropna=False).to_string())
    print('Diagnostic only: not yet joined to ETF outcomes; daily vintage does not prove intraday availability.')


if __name__ == '__main__':
    main()
