#!/usr/bin/env python3
"""Macro Swing Lab 6.0 phase 2: conservative ALFRED vintage snapshots.

Each Friday signal uses a Thursday ALFRED *date* vintage. Never silently
fall back to latest revised observations. Outputs are AUDIT CANDIDATES, not
intraday release-time certified or approved model features.
"""
import argparse
import csv
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

URL = 'https://api.stlouisfed.org/fred/series/observations'
FREQ_LAGS = {'quotidiana': (5, 20, 60), 'settimanale': (1, 4, 13),
             'mensile': (1, 3, 12), 'trimestrale': (1, 2, 4)}


def fetch(series, vintages, key, obs_start, max_retries=7):
    args = dict(series_id=series, api_key=key, file_type='json', output_type=2,
                vintage_dates=','.join(vintages), observation_start=obs_start,
                observation_end=vintages[-1], limit=100000)
    url = URL + '?' + urlencode(args)
    for attempt in range(max_retries):
        try:
            with urlopen(Request(url, headers={'User-Agent': 'MacroSwingLab/6.0-PIT'}), timeout=90) as resp:
                obj = json.load(resp)
            if 'observations' not in obj:
                raise ValueError(f'{series}: missing observations key')
            if int(obj.get('count', len(obj['observations']))) > len(obj['observations']):
                raise ValueError(f'{series}: truncated response (reduce vintage batch)')
            return obj['observations']
        except HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == max_retries-1:
                raise RuntimeError(f'{series}: HTTP {exc.code}: {exc.reason}') from None
            retry = exc.headers.get('Retry-After')
            try:
                delay = min(120, float(retry)) if retry else min(120, 5 * 2 ** attempt)
            except (ValueError, TypeError):
                delay = min(120, 5 * 2 ** attempt)
        except (URLError, TimeoutError) as exc:
            if attempt == max_retries-1:
                raise RuntimeError(f'{series}: network error {exc}') from None
            delay = min(120, 5 * 2 ** attempt)
        print(f'Retry {series} {attempt+1}/{max_retries}: {delay:.1f}s', flush=True)
        time.sleep(delay + random.uniform(0, 1))
    raise RuntimeError('unreachable')


def unpack(observations, series, dates):
    table = pd.DataFrame(observations)
    if 'date' not in table.columns:
        raise ValueError(f'{series}: no date column (empty/incompatible vintage response)')
    idx = pd.to_datetime(table['date'], errors='raise')
    if idx.duplicated().any():
        raise ValueError(f'{series}: duplicate observation dates')
    out = {}
    for vintage in dates:
        options = [f'{series}_{vintage.replace("-", "")}', vintage, vintage.replace('-', '')]
        col = next((c for c in options if c in table.columns), None)
        if col is None:
            raise ValueError(f'{series}: vintage {vintage} missing, columns={table.columns[:6].tolist()}')
        s = pd.Series(pd.to_numeric(table[col], errors='coerce').to_numpy(), index=idx).dropna().sort_index()
        s = s.loc[s.index <= pd.Timestamp(vintage)]
        out[vintage] = s
    return out


def compute(series_id, series, freq, signal, vintage, values):
    lags = FREQ_LAGS.get(freq, (1, 3, 12))
    row = {'id': series_id, 'serie': series, 'frequenza_prevista': freq,
           'data_segnale': signal, 'vintage_asof': vintage, 'valore': None,
           'ultima_osservazione': None, 'n_osservazioni_asof': len(values),
           'differenza_breve': None, 'differenza_media': None,
           'differenza_lunga': None, 'stato': 'NESSUN_DATO_ASOF'}
    if values.empty:
        return row
    row.update(valore=float(values.iloc[-1]), ultima_osservazione=values.index[-1].strftime('%Y-%m-%d'),
               stato='SNAPSHOT_PIT_DA_VERIFICARE')
    for col, lag in zip(('differenza_breve', 'differenza_media', 'differenza_lunga'), lags):
        if len(values) > lag:
            row[col] = float(values.iloc[-1] - values.iloc[-lag-1])
    return row


def read_catalog(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    fred = [r for r in rows if r.get('fonte') == 'FRED']
    if not fred:
        raise ValueError('No FRED series in catalog')
    ids = [r['serie'] for r in fred]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate FRED series in catalog')
    return rows, fred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--catalog', default='catalogo_fonti_60.csv')
    parser.add_argument('--out', default='output/storico_pit_60')
    parser.add_argument('--start', default='2007-01-01')
    parser.add_argument('--max-series', type=int, default=12, help='0 means all 37 (including unavailable source)')
    parser.add_argument('--max-signals', type=int, default=24, help='0 means all Fridays; >0 evenly spread over full history')
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--pacing', type=float, default=1.5)
    parser.add_argument('--offline-test', action='store_true')
    a = parser.parse_args()
    if a.offline_test:
        test_dates = ['2020-01-02', '2020-02-06']
        fake = [dict(date='2019-12-01', TEST_20200102='2', TEST_20200206='3'),
                dict(date='2020-01-01', TEST_20200102='.', TEST_20200206='5')]
        decoded = unpack(fake, 'TEST', test_dates)
        assert len(decoded[test_dates[0]]) == 1 and decoded[test_dates[0]].iloc[-1] == 2
        assert len(decoded[test_dates[1]]) == 2 and decoded[test_dates[1]].iloc[-1] == 5
        row = compute('TEST', 'TEST', 'mensile', '2020-02-07', test_dates[1], decoded[test_dates[1]])
        assert row['valore'] == 5 and row['differenza_breve'] == 2
        print('PASS: vintage decoding, missing values, strictly historical observation dates')
        return

    key = os.getenv('FRED_API_KEY', '').strip()
    if not key:
        raise RuntimeError('FRED_API_KEY missing; set GitHub Actions secret')
    if a.max_series < 0 or a.max_signals < 0 or not 1 <= a.batch_size <= 20 or a.pacing < 0:
        raise ValueError('Invalid limits / batch size / pacing')
    _, fred = read_catalog(a.catalog)
    # FRED discontinued: retain in audit as unavailable, no invented substitute.
    bad_ids = {'GOLDAMGBD228NLBM'}
    valid = [r for r in fred if r['serie'] not in bad_ids]
    selected = valid if a.max_series == 0 else valid[:a.max_series]
    today = pd.Timestamp.now(tz='UTC').date()
    weekly = pd.date_range(a.start, today, freq='W-FRI')
    # Remove current-week signal until Friday has finished everywhere (conservative).
    weekly = weekly[weekly.date <= (pd.Timestamp(today) - pd.Timedelta(days=2)).date()]
    if a.max_signals and len(weekly) > a.max_signals:
        positions = sorted(set(round(i * (len(weekly)-1) / (a.max_signals-1)) for i in range(a.max_signals))) if a.max_signals > 1 else [len(weekly)-1]
        weekly = weekly[positions]
    signals = [(d.strftime('%Y-%m-%d'), (d-pd.Timedelta(days=1)).strftime('%Y-%m-%d')) for d in weekly]
    if not signals:
        raise ValueError('No historical signals in range')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results, audits = [], []
    for ix, entry in enumerate(selected, 1):
        sid = entry['serie']
        status, err, success_batches = 'OK', '', 0
        decoded = {}
        for j in range(0, len(signals), a.batch_size):
            dates = [v for _, v in signals[j:j+a.batch_size]]
            try:
                raw = fetch(sid, dates, key, '2000-01-01')
                decoded.update(unpack(raw, sid, dates))
                success_batches += 1
            except Exception as exc:
                status, err = 'ERRORE_SORGENTE', str(exc)[:350]
                print(f'WARNING {sid}: {err}', flush=True)
                break
            time.sleep(a.pacing)
        if status == 'OK':
            for signal, vintage in signals:
                # This is a conservative daily vintage, NOT verified timestamp of release.
                results.append(compute(entry['id'], sid, entry['frequenza_prevista'], signal,
                                       vintage, decoded[vintage]))
        audits.append({'id': entry['id'], 'serie': sid, 'stato': status,
                       'segnali_attesi': len(signals), 'segnali_ottenuti': len(signals) if status == 'OK' else 0,
                       'batch_ok': success_batches, 'errore': err})
        print(f'[{ix}/{len(selected)}] {sid}: {status}', flush=True)
    audits.append({'id': 'GOLD', 'serie': 'GOLDAMGBD228NLBM', 'stato': 'FONTE_NON_DISPONIBILE',
                   'segnali_attesi': len(signals), 'segnali_ottenuti': 0,
                   'batch_ok': 0, 'errore': 'FRED series unavailable; no substitution'})
    cols = ['id','serie','frequenza_prevista','data_segnale','vintage_asof','valore',
            'ultima_osservazione','n_osservazioni_asof','differenza_breve','differenza_media',
            'differenza_lunga','stato']
    df = pd.DataFrame(results, columns=cols)
    if not df.empty:
        assert (pd.to_datetime(df['vintage_asof']) < pd.to_datetime(df['data_segnale'])).all()
        valid_dates = df['ultima_osservazione'].notna()
        assert (pd.to_datetime(df.loc[valid_dates, 'ultima_osservazione']) <= pd.to_datetime(df.loc[valid_dates, 'vintage_asof'])).all()
        assert not df.duplicated(['serie','data_segnale']).any()
    df.to_csv(out / 'snapshot_pit_60.csv', index=False)
    pd.DataFrame(audits).to_csv(out / 'audit_acquisizione_60.csv', index=False)
    report = {'versione':'6.0_fase2', 'generato_utc':datetime.now(timezone.utc).isoformat(),
              'serie_richieste':len(selected), 'serie_ok':sum(x['stato']=='OK' for x in audits),
              'serie_con_errori':sum(x['stato']=='ERRORE_SORGENTE' for x in audits),
              'segnali_per_serie':len(signals), 'righe_snapshot':len(df),
              'start':signals[0][0], 'end':signals[-1][0],
              'dati_ammessi_al_backtest':0,
              'qualifica':'PILOT_ALFRED_VINTAGE_DATE_NON_VALIDATO_TIMESTAMP',
              'limitazioni':['ALFRED vintage date is not precise publication time',
                             'Derived changes are differences over observed periods, not percent returns',
                             'Only selected dates/series are sampled when limits set',
                             'No join to prices, no model training, no event-news sources yet']}
    (out / 'report_pit_60.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if not df.empty and df['stato'].eq('SNAPSHOT_PIT_DA_VERIFICARE').sum() == 0:
        raise RuntimeError('No usable PIT snapshots downloaded')
    if sum(x['stato']=='OK' for x in audits)==0:
        raise RuntimeError('No source completed successfully')


if __name__ == '__main__':
    main()
