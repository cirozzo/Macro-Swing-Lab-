#!/usr/bin/env python3
"""Macro Swing Lab 4.9: historical ALFRED vintage-based monthly regime snapshots.
Outputs diagnostic snapshots, deliberately separate from validated 4.8 outputs.
"""
import os
import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import pandas as pd

KEY = os.environ.get('FRED_API_KEY', '').strip()
if not KEY:
    raise RuntimeError('FRED_API_KEY missing from GitHub Actions environment')
OUT = Path('output')
OUT.mkdir(exist_ok=True)
START = '2007-01-01'
END = pd.Timestamp.utcnow().date().isoformat()
SERIES = ('INDPRO', 'PCEPILFE')


def request(series, asof):
    params = {'series_id': series, 'api_key': KEY, 'file_type': 'json',
              'realtime_start': asof, 'realtime_end': asof,
              'observation_start': '2005-01-01', 'observation_end': asof,
              'limit': 100000}
    url = 'https://api.stlouisfed.org/fred/series/observations?' + urlencode(params)
    for attempt in range(4):
        try:
            with urlopen(Request(url, headers={'User-Agent': 'MacroSwingLab/4.9'}), timeout=35) as r:
                payload = json.load(r)
            if 'observations' not in payload:
                raise RuntimeError(f'Unexpected response for {series} at {asof}')
            frame = pd.DataFrame(payload['observations'])
            if frame.empty:
                return pd.Series(dtype=float)
            frame['value'] = pd.to_numeric(frame['value'], errors='coerce')
            return frame.set_index(pd.to_datetime(frame['date']))['value'].sort_index().dropna()
        except HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise RuntimeError(f'FRED HTTP {e.code} for {series}, snapshot {asof}') from None
        except (TimeoutError, URLError) as e:
            if attempt == 3:
                raise RuntimeError(f'FRED connection error {series}, snapshot {asof}: {e}') from None
        time.sleep(2 ** attempt)
    raise RuntimeError('Unreachable')


def feature(series):
    yoy = series.pct_change(12, fill_method=None) * 100
    change = yoy.diff(3)
    valid = change.dropna()
    if valid.empty:
        return (None, None)
    last_period = valid.index[-1]
    return (float(valid.iloc[-1]), last_period.strftime('%Y-%m-%d'))


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
    # Weekly decision snapshots on Fridays, after the previous day's releases.
    # Daily-only release vintages do not establish exact intraday publication time.
    snapshots = pd.date_range(START, END, freq='W-FRI')
    rows = []
    for idx, decision in enumerate(snapshots, 1):
        vintage = (decision - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        a = request('INDPRO', vintage)
        b = request('PCEPILFE', vintage)
        g, gd = feature(a)
        i, idate = feature(b)
        rows.append({'data_segnale': decision.date().isoformat(),
                     'vintage_asof': vintage,
                     'growth_change': g, 'inflation_change': i,
                     'growth_last_observation': gd, 'inflation_last_observation': idate,
                     'regime_pit_49': classify(g, i)})
        if idx % 25 == 0:
            print(f'Progress: {idx}/{len(snapshots)} weekly snapshots', flush=True)
    result = pd.DataFrame(rows)
    assert result['data_segnale'].is_unique
    assert (result['vintage_asof'] < result['data_segnale']).all()
    result.to_csv(OUT / 'regimi_pit_49.csv', index=False)
    print('Saved', len(result), 'PIT weekly snapshots to output/regimi_pit_49.csv')
    print(result['regime_pit_49'].value_counts(dropna=False).to_string())
    print('WARNING: exploratory PIT snapshot only; not yet integrated with 4.8 asset outcomes')

if __name__ == '__main__':
    main()
