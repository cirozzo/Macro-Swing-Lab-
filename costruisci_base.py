#!/usr/bin/env python3
"""Macro Swing Lab fase 26: audit e base WTI/COT, senza certificazione PIT implicita.

Esecuzione: python costruisci_base.py
Dati esterni (EIA, term structure, macro, geopolitica) NON inclusi.
"""
from __future__ import annotations
import json
import sys
import traceback
from pathlib import Path
from zipfile import ZipFile, BadZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'risultati'
OUT.mkdir(exist_ok=True)
LAGS = (14, 21, 35)
EXPECTED_COT_ROWS = 879  # fingerprint della serie audit precedente, NON requisito universale


def verify(condition, message):
    if not bool(condition):
        raise ValueError(message)


def read_csv_from_zip(filename, member):
    path = ROOT / filename
    verify(path.is_file() and path.stat().st_size > 0, f'Input mancante o vuoto: {filename}')
    try:
        with ZipFile(path) as z:
            verify(member in z.namelist(), f'{filename}: manca {member}; presenti: {z.namelist()[:10]}')
            verify(z.testzip() is None, f'{filename}: CRC danneggiato')
            with z.open(member) as f:
                return pd.read_csv(f)
    except (BadZipFile, OSError) as e:
        raise ValueError(f'ZIP non leggibile {filename}: {e}') from e


def load_prices(filename, member, label):
    df = read_csv_from_zip(filename, member)
    required = {'time', 'open', 'high', 'low', 'close'}
    verify(required.issubset(df.columns), f'{label}: colonne mancanti: {sorted(required - set(df.columns))}')
    df = df[list(required)].copy()
    for field in ('time', 'open', 'high', 'low', 'close'):
        df[field] = pd.to_numeric(df[field], errors='coerce')
    verify(df[list(required)].notna().all().all(), f'{label}: valori null/non numerici')
    verify(np.isfinite(df[list(required)].to_numpy(dtype=float)).all(), f'{label}: valori non finiti')
    df['timestamp'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    verify(df['timestamp'].is_unique, f'{label}: timestamp duplicati')
    verify((df.high >= df[['open','close','low']].max(axis=1)).all(), f'{label}: high incoerenti')
    verify((df.low <= df[['open','close','high']].min(axis=1)).all(), f'{label}: low incoerenti')
    verify((df[['open','close']] > 0).all().all(), f'{label}: prezzo non positivo: proxy non gestibile con rendimenti percentuali')
    df = df[df.timestamp >= pd.Timestamp('2010-01-01', tz='UTC')].copy().reset_index(drop=True)
    verify(len(df) > 100, f'{label}: numero candele insufficiente')
    return df


def load_cot():
    df = read_csv_from_zip('cot-wti-audit-unificato.zip','serie_verificata_origini.csv')
    columns = {'report_date','M_Money_Positions_Long_All','M_Money_Positions_Short_All',
               'Open_Interest_All','net_mm','net_pct_oi','delta_4w','source_url','source_sha256'}
    verify(columns.issubset(df.columns), f'COT: colonne mancanti: {sorted(columns-set(df.columns))}')
    df['report_date'] = pd.to_datetime(df['report_date'], utc=True, errors='coerce')
    verify(df.report_date.notna().all(), 'COT: date non valide')
    df = df.sort_values('report_date').reset_index(drop=True)
    verify(df.report_date.is_unique, 'COT: date duplicate')
    verify(len(df) == EXPECTED_COT_ROWS, f'COT: attese {EXPECTED_COT_ROWS} righe, trovate {len(df)}')
    for c in ('M_Money_Positions_Long_All','M_Money_Positions_Short_All','Open_Interest_All','net_mm','net_pct_oi','delta_4w'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    verify((df.Open_Interest_All>0).all(), 'COT: open interest nullo/negativo')
    verify(np.allclose(df.net_mm, df.M_Money_Positions_Long_All-df.M_Money_Positions_Short_All), 'COT: net_mm incoerente')
    verify(np.allclose(df.net_pct_oi, df.net_mm/df.Open_Interest_All), 'COT: net_pct_oi incoerente')
    verify(np.allclose(df.delta_4w.iloc[4:], df.net_pct_oi.diff(4).iloc[4:], equal_nan=True), 'COT: delta_4w incoerente')
    verify(df.source_url.notna().all() and df.source_sha256.notna().all(), 'COT: provenienza mancante')
    return df


def daily_features(df):
    d=df.copy()
    # Feature calcolate solo con barre COMPLETATE PRIMA della barra di riferimento:
    # fruibili all'apertura della barra corrente, senza conoscere high/low/close attuali.
    prior_close=d.close.shift(1)
    d['ret_5_pre']=prior_close/d.close.shift(6)-1
    d['ret_20_pre']=prior_close/d.close.shift(21)-1
    tr=pd.concat([d.high-d.low,(d.high-d.close.shift(1)).abs(),(d.low-d.close.shift(1)).abs()],axis=1).max(axis=1)
    d['atr14_pre']=tr.rolling(14,min_periods=14).mean().shift(1)
    d['atr_pct_pre']=d.atr14_pre/prior_close
    daily_return=d.close.pct_change(fill_method=None)
    d['rv20_pre']=daily_return.rolling(20,min_periods=20).std().shift(1)
    rv_median=d['rv20_pre'].rolling(252,min_periods=126).median()
    d['compression_pre']=d.rv20_pre/rv_median.shift(1)
    # Le etichette sono ANCORATE ALL'OPEN della barra corrente.
    # I successivi n BARRE del feed TradingView non equivalgono necessariamente a n sedute di borsa.
    for n in (5,10):
        future_high=pd.concat([d.high.shift(-i) for i in range(0,n)],axis=1).max(axis=1)
        future_low=pd.concat([d.low.shift(-i) for i in range(0,n)],axis=1).min(axis=1)
        complete=d.close.shift(-(n-1)).notna() & d.atr14_pre.notna() & (d.atr14_pre>0)
        d[f'future_return_{n}bars_open_to_close']=np.where(complete,d.close.shift(-(n-1))/d.open-1,np.nan)
        d[f'mfe_atr_{n}bars']=np.where(complete,(future_high-d.open)/d.atr14_pre,np.nan)
        d[f'mae_atr_{n}bars']=np.where(complete,(future_low-d.open)/d.atr14_pre,np.nan)
        for threshold in (2,3):
            d[f'up{threshold}atr_{n}bars']=np.where(complete,(d[f'mfe_atr_{n}bars']>=threshold).astype(float),np.nan)
    return d


def h4_features(df):
    h=df.copy()
    # Segnale noto SOLO alla chiusura della barra H4 corrente;
    # eventuale esecuzione nella barra H4 SUCCESSIVA, mai nella stessa.
    h['ema20_at_close']=h.close.ewm(span=20,adjust=False).mean()
    h['ema20_rising_at_close']=h.ema20_at_close>h.ema20_at_close.shift(1)
    h['breakout6_at_close']=h.close>h.high.shift(1).rolling(6,min_periods=6).max()
    h['signal_bull_at_close']=(h.breakout6_at_close & h.ema20_rising_at_close)
    h['next_bar_open']=h.open.shift(-1)
    h['next_bar_timestamp']=h.timestamp.shift(-1)
    return h


def main():
    cot=load_cot()
    day=load_prices('TVC_USOIL, 1D.zip','TVC_USOIL, 1D.csv','WTI D1')
    h4=load_prices('TVC_USOIL, 240.zip','TVC_USOIL, 240.csv','WTI H4')
    day=daily_features(day)
    h4=h4_features(h4)
    h4[['timestamp','open','high','low','close','ema20_at_close','ema20_rising_at_close',
        'breakout6_at_close','signal_bull_at_close','next_bar_timestamp','next_bar_open']].to_csv(OUT/'h4_features.csv',index=False)
    for lag in LAGS:
        c=cot[['report_date','net_mm','net_pct_oi','delta_4w','Open_Interest_All','source_url','source_sha256']].copy()
        # Timestamp sintetico: mai presentato come data effettiva CFTC.
        # +14/21/35 giorni dalla data di rilevazione alle 00:00 UTC.
        c['cot_available_assumed_utc']=c.report_date+pd.Timedelta(days=lag)
        d=pd.merge_asof(day,c.sort_values('cot_available_assumed_utc'),
            left_on='timestamp',right_on='cot_available_assumed_utc',direction='backward')
        d['cot_lag_assumed_days']=lag
        d['cot_pit_certified']=False
        d['cot_time_status']='ASSUMED_LAG_NOT_VERIFIED'
        d['cot_report_age_days']=(d.timestamp-d.report_date).dt.total_seconds()/86400
        # 2025 e report 2025 riutilizzati nel 2026: non usare per validazione predittiva.
        d['cot_exception_2025_risk']=(d.timestamp.dt.year.eq(2025) | d.report_date.dt.year.eq(2025))
        d['cot_model_eligible_excluding_2025']=d.report_date.notna() & ~d.cot_exception_2025_risk
        d.to_csv(OUT/f'base_giornaliera_lag{lag}.csv',index=False)
    all_missing=['EIA crude/Cushing/products, SPR and surprise vs consensus',
      'WTI futures M1-M2/M1-M6 term structure with historical roll',
      'OPEC+ decision event times and surprise vs expectations',
      'US oil production, oil rigs and historical revisions',
      'IEA/EIA/OPEC supply-demand forecast vintages',
      'USD, nominal/real yields and rate expectations',
      'PMI/ISM, industrial output and China crude import release vintages',
      'Geopolitical event release times, not hindsight labels',
      'A tradable WTI contract/CFD with spread, swaps and continuous-roll methodology']
    report={'status':'PASSED_DATASET_BUILD_NOT_PIT_CERTIFIED_NOT_MULTIFACTOR_COMPLETE',
      'cot':{'rows':len(cot),'first':str(cot.report_date.min()),'last':str(cot.report_date.max()),'source':'CFTC structural source-audited series; timestamps of actual release unavailable'},
      'wti_daily_proxy':{'rows':len(day),'first':str(day.timestamp.min()),'last':str(day.timestamp.max()),'source':'TradingView USOIL; contract roll and tradeability NOT verified'},
      'wti_h4_proxy':{'rows':len(h4),'first':str(h4.timestamp.min()),'last':str(h4.timestamp.max()),'source':'TradingView USOIL; price series starts in 2013'},
      'cot_lags_assumed_days':list(LAGS),'cot_pit_certified':False,
      'methodology':{'daily_features':'past completed daily bars only, available at current daily bar open',
                     'daily_future_labels':'future 5/10 observed daily feed bars FROM current bar OPEN; outcomes only, never predictive inputs',
                     'h4_signal':'current H4 completed bar close; execution at next H4 bar open',
                     'trading_days':'5/10 feed bars are not assumed 5/10 exchange sessions',
                     '2025':'flag 2025 signal dates and 2025 report dates; exclude them from predictive validation'},
      'missing_external_families':all_missing,
      'limits':['Only COT/price/volatility covered: other factor families NOT downloaded',
                'Scenario delays 14/21/35d do not certify actual publication or guarantee no exceptional leakage',
                'Future-bar high/low can trigger target and stop in unknown order',
                'WTI proxy data and roll transitions may distort ATR/returns',
                'No honest untouched out-of-sample history remaining for already selected rules',
                'No trading system, profit forecast or statistical edge certified'],
      'outputs':[p.name for p in sorted(OUT.glob('*.csv'))]}
    (OUT/'audit_copertura.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        (OUT/'ERRORE.txt').write_text(traceback.format_exc(),encoding='utf-8')
        print('FALLIMENTO FASE 26:',exc,file=sys.stderr)
        raise
