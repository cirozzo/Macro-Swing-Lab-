#!/usr/bin/env python3
"""Macro Swing Lab - CFTC COT feature extraction v2 (NO backtest).

Fixes TFF field names, rolling PIT features, deterministic API pagination,
contract-code grouping, disclosure of historical publication-date uncertainty,
and the officially announced 2025/26 CFTC backlog release timetable.

Requires: pandas, numpy, requests.
Usage: python cot_pit_etf.py --etf-zip etf-pit-52.zip --out cot_results
Offline: python cot_pit_etf.py --etf-zip etf-pit-52.zip --cot-dir fixtures --out test_results
"""
from __future__ import annotations
import argparse
import bisect
from collections import deque
from datetime import datetime, timedelta
import json
from pathlib import Path
import time
import zipfile
import numpy as np
import pandas as pd
import requests

BASE='https://publicreporting.cftc.gov/resource/'
DATASETS={'TFF':'gpe5-46if','DISAGG':'72hh-3qpy'}  # Futures only, NOT combined.
COMMON=['id','report_date_as_yyyy_mm_dd','market_and_exchange_names','cftc_contract_market_code','open_interest_all']
FIELDS={
 'TFF':{
  'asset_manager': [('asset_mgr_positions_long','asset_mgr_positions_long_all'),('asset_mgr_positions_short','asset_mgr_positions_short_all')],
  'leveraged': [('lev_money_positions_long','lev_money_positions_long_all'),('lev_money_positions_short','lev_money_positions_short_all')],
 },
 'DISAGG':{
  'managed_money':[('m_money_positions_long_all','m_money_positions_long'),('m_money_positions_short_all','m_money_positions_short')],
 },
}
# Official CFTC announcement 2025-11-18, report dates -> revised publication dates.
# Source: https://www.cftc.gov/PressRoom/PressReleases/9138-25
BACKLOG_RELEASE={
 '2025-09-30':'2025-11-19','2025-10-07':'2025-11-21',
 '2025-10-14':'2025-11-25','2025-10-21':'2025-12-02',
 '2025-10-28':'2025-12-05','2025-11-04':'2025-12-09',
 '2025-11-10':'2025-12-12','2025-11-18':'2025-12-16',
 '2025-11-25':'2025-12-19','2025-12-02':'2025-12-23',
 '2025-12-09':'2025-12-30','2025-12-16':'2026-01-06',
 '2025-12-23':'2026-01-09','2025-12-30':'2026-01-13',
 '2026-01-06':'2026-01-16','2026-01-13':'2026-01-20',
 '2026-01-20':'2026-01-23',
}
# CFTC 2026 tentative release schedule, see
# https://www.cftc.gov/MarketReports/CommitmentsofTraders/ReleaseSchedule/index.htm
# Scheduled dates are NOT proof of actual timestamp or initial release version.
RELEASES_2026={
 1:[23,30],2:[6,13,20,27],3:[6,13,20,27],4:[3,10,17,24],
 5:[1,8,15,22,29],6:[5,12,22,26],7:[6,10,17,24,31],
 8:[7,14,21,28],9:[4,11,18,25],10:[2,9,16,23,30],
 11:[6,16,20,30],12:[4,11,18,28],
}
RELEASE_SCHEDULE_2026={}
for _month,_days in RELEASES_2026.items():
 for _day in _days:
  _release=pd.Timestamp(2026,_month,_day)
  # Most recent Tuesday strictly BEFORE scheduled release.
  _diff=(_release.weekday()-1)%7 or 7
  _report=_release-pd.Timedelta(days=_diff)
  _report_key=_report.strftime('%Y-%m-%d')
  if _report_key in RELEASE_SCHEDULE_2026:raise RuntimeError('2026 release schedule ambiguous')
  RELEASE_SCHEDULE_2026[_report_key]=_release.strftime('%Y-%m-%d')
# Additional user-supplied release dates may override proxies, but without
# documentary verification are NOT classified as certified PIT.


def first_weekday_after(d):
 d=pd.Timestamp(d).normalize()+pd.Timedelta(days=1)
 while d.weekday()>=5:d+=pd.Timedelta(days=1)
 return d

def lookup_release_date(d, override):
 day=pd.Timestamp(d).strftime('%Y-%m-%d')
 if day in override:
  return pd.Timestamp(override[day]),'CALENDARIO_UTENTE_NON_VERIFICATO'
 if day in BACKLOG_RELEASE:
  return pd.Timestamp(BACKLOG_RELEASE[day]),'CFTC_CALENDARIO_PUBBLICATO_2025_11_18'
 if day in RELEASE_SCHEDULE_2026:
  return pd.Timestamp(RELEASE_SCHEDULE_2026[day]),'CFTC_CALENDARIO_2026_PROGRAMMATO'
 # Friday of the same week for the usual Tuesday report (proxy only).
 return pd.Timestamp(d).normalize()+pd.Timedelta(days=4-pd.Timestamp(d).weekday()),'PROXY_VENERDI_NON_CERTIFICATO'

def read_calendar(path):
 if path is None:return {}
 d=pd.read_csv(path,dtype=str)
 if not {'report_date','release_date'}<=set(d):
  raise ValueError('Calendario richiede report_date,release_date')
 if d.report_date.duplicated().any():raise ValueError('Date report duplicate nel calendario')
 for c in ('report_date','release_date'):
  d[c]=pd.to_datetime(d[c],errors='raise').dt.strftime('%Y-%m-%d')
 if (d.release_date<d.report_date).any():raise ValueError('Pubblicazione prima del report')
 return dict(zip(d.report_date,d.release_date))

def fetch(kind, start, session, page_size=20000):
 dataset=DATASETS[kind]
 # Official Socrata field names of TFF differ from display labels ending in _All.
 sel=COMMON+([c[0] for pair in FIELDS[kind].values() for c in pair] )
 rows=[];offset=0
 while True:
  params={'$select':','.join(dict.fromkeys(sel)), '$where':f"report_date_as_yyyy_mm_dd >= '{start}'",
          '$order':'report_date_as_yyyy_mm_dd ASC,id ASC','$limit':page_size,'$offset':offset}
  for attempt in range(6):
   try:
    resp=session.get(f'{BASE}{dataset}.json',params=params,timeout=90)
    resp.raise_for_status()
    chunk=resp.json()
    if not isinstance(chunk,list):raise ValueError(f'{kind}: invalid API response {str(chunk)[:200]}')
    break
   except (requests.RequestException,ValueError) as e:
    if attempt==5:raise RuntimeError(f'{kind} download offset {offset}: {e}') from e
    time.sleep(min(2**attempt,16))
  rows.extend(chunk);offset+=len(chunk)
  if len(chunk)<page_size:break
 print(f'{kind}: fetched {len(rows)} raw rows, {offset//page_size+1} pages',flush=True)
 return pd.DataFrame.from_records(rows)

def resolve(df,alternatives,kind,label):
 for n in alternatives:
  if n in df.columns:return n
 raise ValueError(f'{kind}/{label}: nessuna colonna tra {alternatives}; disponibili: {list(df.columns)[:35]}')

def percentile_prior_only(values, min_periods=52, window=156):
 """Percentile of current value vs immediately preceding <=156 observations only."""
 answer=np.full(len(values),np.nan,dtype=float)
 previous=deque();ordered=[]
 for i,v in enumerate(values):
  if np.isfinite(v) and len(previous)>=min_periods:
   answer[i]=bisect.bisect_right(ordered,v)/len(ordered)
  if np.isfinite(v):
   bisect.insort(ordered,float(v));previous.append(float(v))
   if len(previous)>window:
    old=previous.popleft();ordered.pop(bisect.bisect_left(ordered,old))
 return answer

def features_from_raw(df,kind,overrides=None):
 overrides=overrides or {}
 if df.empty:raise ValueError(f'{kind}: dataset senza righe')
 date=resolve(df,['report_date_as_yyyy_mm_dd','report_date'],kind,'date')
 market=resolve(df,['market_and_exchange_names','market_and_exchange'],kind,'market')
 code=resolve(df,['cftc_contract_market_code'],kind,'contract code')
 oi=resolve(df,['open_interest_all'],kind,'open interest')
 base=pd.DataFrame({
  'report_date':pd.to_datetime(df[date],errors='coerce').dt.normalize(),
  'market':df[market].astype(str).str.strip(),
  'contract_code':df[code].astype(str).str.strip(),
  'open_interest':pd.to_numeric(df[oi],errors='coerce')})
 if base.report_date.isna().any():raise ValueError(f'{kind}: report dates invalid')
 if base.contract_code.eq('').any() or base.contract_code.eq('nan').any():raise ValueError(f'{kind}: missing contract code')
 if base.duplicated(['contract_code','report_date']).any():
  raise ValueError(f'{kind}: contract/date duplicates; no aggregation permitted')
 if (base.open_interest<=0).any() or base.open_interest.isna().any():
  raise ValueError(f'{kind}: OI <= 0 or missing')
 out=[]
 for category,pairs in FIELDS[kind].items():
  left=resolve(df,pairs[0],kind,category+'_long')
  right=resolve(df,pairs[1],kind,category+'_short')
  x=base.copy()
  l=pd.to_numeric(df[left],errors='coerce')
  s=pd.to_numeric(df[right],errors='coerce')
  if l.isna().any() or s.isna().any():raise ValueError(f'{kind}/{category}: missing or invalid positions')
  x['category']=category
  x['net']=l-s
  x['net_oi']=x['net']/x['open_interest']
  x=x.sort_values(['contract_code','report_date']).reset_index(drop=True)
  g=x.groupby('contract_code',sort=False)['net_oi']
  x['change_net_oi']=g.diff()
  x['prior_mean']=g.transform(lambda t:t.shift(1).rolling(156,min_periods=52).mean())
  x['prior_sd']=g.transform(lambda t:t.shift(1).rolling(156,min_periods=52).std())
  x['z_position']=(x.net_oi-x.prior_mean)/x.prior_sd.replace(0,np.nan)
  x['prior_percentile']=g.transform(lambda t:pd.Series(percentile_prior_only(t.to_numpy(dtype=float)),index=t.index))
  x['kind']=kind
  # Explicitly label unverified historical release dates. Never infer release date
  # from observation timestamp, and never use post-report data retroactively.
  unique_dates=x.report_date.drop_duplicates().tolist()
  metadata={d:lookup_release_date(d,overrides) for d in unique_dates}
  x['release_date_assumed']=x.report_date.map(lambda d:metadata[d][0])
  x['release_source']=x.report_date.map(lambda d:metadata[d][1])
  x['usable_from']=x.release_date_assumed.map(first_weekday_after)
  x['release_schedule_documented']=x.release_source.isin(['CFTC_CALENDARIO_PUBBLICATO_2025_11_18','CFTC_CALENDARIO_2026_PROGRAMMATO'])
  x['temporal_quality']=np.where(x.release_schedule_documented,'CALENDARIO_CFTC_ANNUNCIATO_NON_VERIFICA_PUBBLICAZIONE','PROXY_RELEASE_NON_VERIFICATA')
  x['strict_pit_eligible']=False  # Verify actual historical release timestamp before strict backtesting.
  if (x.usable_from<=x.report_date).any():raise ValueError('COT chronology violation')
  out.append(x)
 return pd.concat(out,ignore_index=True)

def load_etf(zip_path):
 with zipfile.ZipFile(zip_path) as z:
  bad=z.testzip()
  if bad:raise ValueError(f'ZIP damaged: {bad}')
  candidates=[n for n in z.namelist() if n.endswith('osservazioni_pit_52.csv')]
  if len(candidates)!=1:raise ValueError('Expected exactly one osservazioni_pit_52.csv in ZIP')
  with z.open(candidates[0]) as f: etf=pd.read_csv(f)
 required={'asset','data_segnale','data_ingresso','data_uscita','sedute','rendimento','excess_return'}
 if not required<=set(etf):raise ValueError('ETF columns missing: '+str(required-set(etf)))
 if etf.empty:raise ValueError('ETF observations empty')
 for c in ['data_segnale','data_ingresso','data_uscita']:
  etf[c]=pd.to_datetime(etf[c],errors='raise').dt.normalize()
 if not ((etf.data_segnale<etf.data_ingresso)&(etf.data_ingresso<=etf.data_uscita)).all():
  raise ValueError('ETF chronology failure')
 if etf.duplicated(['asset','data_segnale','sedute']).any():raise ValueError('Duplicate ETF observations')
 return etf

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--etf-zip',required=True)
 parser.add_argument('--out',default='cot_results')
 parser.add_argument('--cot-dir',default=None,help='Offline TFF.csv and DISAGG.csv')
 parser.add_argument('--release-calendar',default=None,help='Optional report_date,release_date CSV (unverified unless separately audited)')
 parser.add_argument('--start',default='2007-01-01')
 a=parser.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 overrides=read_calendar(a.release_calendar)
 etf=load_etf(a.etf_zip)
 session=requests.Session();all_features=[]
 report={'pipeline_version':'2.0','datasets':DATASETS,'start':a.start,
         'source':'CFTC official Socrata futures-only','status':'PREPARAZIONE_IN_CORSO',
         'historical_release_date_caution':'The 2025/26 backlog and 2026 scheduled release dates come from CFTC announcements (NOT proof of actual timestamp). Other historical rows use NONCERTIFIED Friday proxy. No historical PIT backtest is certified.',
         'source_release_exception':'https://www.cftc.gov/PressRoom/PressReleases/9138-25',
         'source_release_schedule_2026':'https://www.cftc.gov/MarketReports/CommitmentsofTraders/ReleaseSchedule/index.htm'}
 try:
  for kind in DATASETS:
   frame=pd.read_csv(Path(a.cot_dir)/f'{kind}.csv',low_memory=False) if a.cot_dir else fetch(kind,a.start,session)
   report[kind+'_raw_rows']=len(frame)
   f=features_from_raw(frame,kind,overrides)
   if f.empty or f.z_position.notna().sum()==0:raise ValueError(f'{kind}: no valid 52-week z-score observations')
   report[kind+'_feature_rows']=len(f)
   report[kind+'_z_valid']=int(f.z_position.notna().sum())
   report[kind+'_percentile_valid']=int(f.prior_percentile.notna().sum())
   report[kind+'_contracts']=int(f.contract_code.nunique())
   all_features.append(f)
  cot=pd.concat(all_features,ignore_index=True)
  if cot.duplicated(['kind','contract_code','category','report_date']).any():
   raise ValueError('COT unique key collision')
  cot.to_csv(out/'cot_features_pit_proxy.csv',index=False)
  etf[['asset','data_segnale','data_ingresso','data_uscita','sedute','rendimento','excess_return']].to_csv(out/'etf_target_dates.csv',index=False)
  report['etf_rows']=len(etf);report['etf_assets']=sorted(etf.asset.unique().tolist())
  report['cot_markets']=int(cot[['kind','contract_code']].drop_duplicates().shape[0])
  report['release_schedule_documented_rows']=int(cot.release_schedule_documented.sum())
  report['release_schedule_unverified_rows']=int((~cot.release_schedule_documented).sum())
  report['strict_pit_eligible_rows']=int(cot.strict_pit_eligible.sum())
  report['min_report_date']=str(cot.report_date.min().date());report['max_report_date']=str(cot.report_date.max().date())
  report['status']='FEATURES_READY_NOT_BACKTESTED'
  report['next_required']='Map contract codes to ETFs, verify historical release calendars, temporal asof joins, OOS tests and costs.'
 except Exception as e:
  report['status']='FAILED';report['error']=f'{type(e).__name__}: {e}'
  (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
  raise
 (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
 print(json.dumps(report,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
