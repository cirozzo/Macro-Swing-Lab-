#!/usr/bin/env python3
"""Macro Swing Lab | Fase 28 unica: import, acquisizione, release calendar, QA, outcomes.

IMPORTANT: EIA API historical values are latest-revised, not immutable release vintages.
Official WPSR archive proves report release *date*, not individual historical value.
This script never silently promotes assumptions or latest revisions to PIT-certified.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
import re
import sys
import traceback
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

ARCHIVE = 'https://www.eia.gov/petroleum/supply/weekly/archive/'
SCHEDULE = 'https://www.eia.gov/petroleum/supply/weekly/schedule.php'
API = 'https://api.eia.gov/v2/seriesid/'
# Legacy ids are a crosswalk into official v2 API; an invalid ID is reported, never guessed.
EIA = {
 'crude_commercial':'PET.WCESTUS1.W',
 'cushing':'PET.W_EPC0_SAX_YCUOK_MBBL.W',
 'gasoline':'PET.WGTSTUS1.W',
 'distillates':'PET.WDISTUS1.W',
 'spr':'PET.WCSSTUS1.W',
 'crude_production':'PET.WCRFPUS2.W',
 'futures_m1':'PET.RCLC1.D',
 'futures_m2':'PET.RCLC2.D',
 'futures_m6':'PET.RCLC6.D',
}
CATALOG = {
 'eia_weekly':list(EIA)[:6],
 'futures_curve':['futures_m1','futures_m2','futures_m6','wti_m1_m2','wti_m1_m6'],
 'eia_consensus':['crude_inventory_surprise','gasoline_inventory_surprise'],
 'supply_demand':['opec_supply_revision','iea_demand_revision','china_oil_demand'],
 'geopolitics':['geopolitical_shock','supply_disruption'],
}
TZ = ZoneInfo('America/New_York')
HEADERS={'User-Agent':'MacroSwingLab-Research/1.0 (academic data quality audit)'}

def utc(s): return pd.to_datetime(s,utc=True,errors='coerce')
def digest(b): return hashlib.sha256(b).hexdigest()
def write_csv(df,p):
 Path(p).parent.mkdir(parents=True,exist_ok=True)
 df.to_csv(p,index=False)
def fetch(session,url,**kwargs):
 r=session.get(url,timeout=35,headers=HEADERS,**kwargs)
 r.raise_for_status()
 return r

def load_input(z,name):
 if name not in z.namelist(): raise ValueError('Input Fase27 privo di '+name)
 return pd.read_csv(io.BytesIO(z.read(name)),low_memory=False)

def clean_time(df):
 if 'timestamp' not in df.columns: raise ValueError('Colonna timestamp assente')
 x=df.copy();x['timestamp']=utc(x.timestamp)
 if x.timestamp.isna().any() or x.timestamp.duplicated().any():raise ValueError('Timestamp duplicati/non validi')
 return x.sort_values('timestamp').reset_index(drop=True)

def validate_ohlc(d):
 for c in ('open','high','low','close'):
  d[c]=pd.to_numeric(d[c],errors='coerce')
 good=d[['open','high','low','close']].notna().all(axis=1)&(d.high>=d[['open','close','low']].max(axis=1))&(d.low<=d[['open','close','high']].min(axis=1))
 if not bool(good.all()):raise ValueError('OHLC non validi: '+str(int((~good).sum())))
 return d

def calendar_from_html(html):
 """Parse year/month pairs in official WPSR archive tables, including dual-year columns.

 Each month label may appear only in first row of a block, and year(s) in header.
 Returns date pairs; if table structure is unfamiliar returns none instead of invented dates.
 """
 soup=BeautifulSoup(html,'html.parser');items=[]
 for table in soup.find_all('table'):
  rows=table.find_all('tr')
  years=[]
  for row in rows[:5]:
   txt=' '.join(row.stripped_strings)
   years=[int(y) for y in re.findall(r'\b20(?:0\d|1\d|2\d)\b',txt)]
   if years:break
  years=[y for y in years if 2000<=y<=2035]
  years=list(dict.fromkeys(years))
  if not years or len(years)>2:continue
  if not ('release date' in table.get_text(' ',strip=True).lower() and 'data ending' in table.get_text(' ',strip=True).lower()):continue
  month=None
  for row in rows:
   cells=row.find_all(['th','td'],recursive=False)
   if not cells:continue
   vals=[c.get_text(' ',strip=True).replace('\xa0',' ').strip() for c in cells]
   for t in vals:
    m=re.search(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b',t,re.I)
    if m:month=datetime.strptime(m.group(1)[:3],'%b').month;break
   if month is None:continue
   # Match release-date link anchors to their year via cell ordering. Tables may include rowspan month column.
   slots=[]
   for c in cells:
    a=c.find('a',href=True)
    if a:
     day=a.get_text(' ',strip=True)
     if re.fullmatch(r'\d{1,2}',day):slots.append((c,day))
   if not slots:continue
   # Stronger: identify year using 4 data cells (release/end for each year), excluding month cell.
   work=[c for c in cells if not re.search(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b',c.get_text(' ',strip=True),re.I)]
   if len(years)==1:
    for c,day in slots:
     try:
      dd=pd.Timestamp(year=years[0],month=month,day=int(day))
      # data week-ending is in cell following release cell
      k=work.index(c) if c in work else -1
      end=work[k+1].get_text(' ',strip=True) if k>=0 and k+1<len(work) else ''
      wd=parse_week_ending(end,dd)
      if wd is not None:items.append((wd,dd))
     except (ValueError,TypeError):pass
   else:
    for c,day in slots:
     try:
      k=work.index(c)
      # table usually release, ending, release, ending
      yi=0 if k<2 else 1
      if k not in (0,2):continue
      dd=pd.Timestamp(year=years[yi],month=month,day=int(day))
      end=work[k+1].get_text(' ',strip=True) if k+1<len(work) else ''
      wd=parse_week_ending(end,dd)
      if wd is not None:items.append((wd,dd))
     except (ValueError,TypeError,IndexError):pass
 data=pd.DataFrame(items,columns=['observation_date','release_date'])
 if data.empty:return data
 data=data.drop_duplicates()
 bad=(data.release_date<data.observation_date)|(data.release_date>data.observation_date+pd.Timedelta(days=28))
 data=data.loc[~bad].sort_values('observation_date').reset_index(drop=True)
 # distinct release dates can be shared by multiple reference weeks, e.g. June 29 2022.
 if data.observation_date.duplicated().any():
  data=data.drop_duplicates('observation_date',keep=False)
 return data

def parse_week_ending(v,release):
 v=re.sub(r'\s+','',str(v))
 if not re.fullmatch(r'(?:\d{1,2}/)?\d{1,2}',v):return None
 month,day=map(int,v.split('/')) if '/' in v else (release.month,int(v))
 # explicit month is same/previous year as necessary, no explicit means same or preceding month
 yr=release.year
 candidates=[]
 for yy in (yr,yr-1):
  for mm in ([month] if '/' in v else (release.month, release.month-1 if release.month>1 else 12)):
   try:
    dt=pd.Timestamp(year=yy,month=mm,day=day)
    diff=(release-dt).days
    if 1<=diff<=28:candidates.append((diff,dt))
   except ValueError:pass
 return min(candidates,key=lambda x:x[0])[1] if candidates else None

def get_eia_calendar(session,log):
 try:
  resp=fetch(session,ARCHIVE)
  df=calendar_from_html(resp.text)
  if len(df)<300:raise ValueError('Archivio non riconosciuto (righe '+str(len(df))+')')
  df['source_url']=ARCHIVE
  df['availability_utc_next_day']=[pd.Timestamp((x+pd.Timedelta(days=1)).date(),tz=TZ).tz_convert('UTC') for x in df.release_date]
  df['temporal_quality']='ARCHIVE_RELEASE_DATE_NEXT_DAY_NOT_VALUE_VINTAGE'
  log.append({'source':'EIA_WPSR_ARCHIVE','status':'OK','records':len(df),'html_sha256':digest(resp.content)})
  return df
 except Exception as exc:
  log.append({'source':'EIA_WPSR_ARCHIVE','status':'UNAVAILABLE','error':str(exc)[:300]})
  return pd.DataFrame(columns=['observation_date','release_date','source_url','availability_utc_next_day','temporal_quality'])

def eia_series(session,key,series_id,log):
 url=API+series_id
 if not key:
  log.append({'source':series_id,'status':'SKIPPED_NO_EIA_API_KEY'})
  return pd.DataFrame()
 try:
  out=[];offset=0;limit=5000
  for _ in range(6):
   params={'api_key':key,'start':'2009-01-01','length':limit,'offset':offset}
   raw=fetch(session,url,params=params).json()
   payload=raw.get('response',{})
   data=payload.get('data',[])
   if not isinstance(data,list):raise ValueError('Schema API inatteso: '+str(raw)[:160])
   out.extend(data)
   if len(data)<limit:break
   offset+=limit
  if not out:raise ValueError('API non ha restituito osservazioni')
  result=[]
  for v in out:
   date=v.get('period')
   value=v.get('value')
   if date is None or value is None:continue
   date=utc(str(date))
   value=pd.to_numeric(value,errors='coerce')
   if pd.isna(date) or not np.isfinite(value):continue
   result.append({'observation_date':date.tz_localize(None).normalize(),'value':float(value)})
  d=pd.DataFrame(result).drop_duplicates('observation_date',keep='last').sort_values('observation_date')
  if d.empty:raise ValueError('Nessuna coppia data-valore valida')
  log.append({'source':series_id,'status':'DOWNLOADED_CURRENT_VINTAGE','rows':len(d),'url':url})
  return d
 except Exception as exc:
  log.append({'source':series_id,'status':'UNAVAILABLE','error':str(exc)[:300],'url':url})
  return pd.DataFrame()

def build_eia(session,key,calendar,log):
 data=[]
 for var,sid in EIA.items():
  d=eia_series(session,key,sid,log)
  if d.empty:continue
  d['variable']=var;d['source_url']=API+sid
  d['value_quality']='CURRENT_VINTAGE_REVISIONS_POSSIBLE'
  if var.startswith('futures_'):
   # A daily settlement dated D isn't necessarily available before an H4 timestamp on D.
   d['available_at_utc']=pd.to_datetime(d.observation_date+pd.Timedelta(days=1)).dt.tz_localize(TZ).dt.tz_convert('UTC')
   d['temporal_quality']='SETTLEMENT_PLUS_1_DAY_ASSUMED_NOT_PIT_CERTIFIED'
   d['release_date']=pd.NaT
  else:
   d=d.merge(calendar[['observation_date','release_date','availability_utc_next_day']],on='observation_date',how='left')
   d['available_at_utc']=utc(d['availability_utc_next_day'])
   d['temporal_quality']=np.where(d.available_at_utc.notna(),'EIA_ARCHIVE_DATE_VALUE_LATEST_UNCERTIFIED','NO_RELEASE_MATCH')
   d=d.drop(columns=['availability_utc_next_day'])
  data.append(d)
 return pd.concat(data,ignore_index=True) if data else pd.DataFrame(columns=['observation_date','value','variable','source_url','value_quality','available_at_utc','temporal_quality','release_date'])

def external_import(folder,log):
 result=[]
 for p in sorted(Path(folder).glob('*.csv')):
  try:
   d=pd.read_csv(p,low_memory=False)
   required={'variable','observation_date','available_at_utc','value','source_url'}
   if not required.issubset(d):raise ValueError('Mancano colonne '+str(sorted(required-set(d))))
   valid=set(sum(CATALOG.values(),[]))
   if not d.variable.isin(valid).all():raise ValueError('Variabili fuori catalogo')
   d['observation_date']=utc(d.observation_date).dt.tz_localize(None).dt.normalize()
   d['available_at_utc']=utc(d.available_at_utc)
   d['value']=pd.to_numeric(d.value,errors='coerce')
   d=d[d.value.notna()&d.observation_date.notna()&d.available_at_utc.notna()].copy()
   d=d[d.available_at_utc>=d.observation_date.dt.tz_localize('UTC')]
   d['source_url']=d.source_url.astype(str)
   d=d[d.source_url.str.startswith('https://')]
   d['value_quality']='USER_DOCUMENTED_UNVERIFIED'
   d['temporal_quality']='USER_SUPPLIED_TIMESTAMP_UNVERIFIED'
   d['release_date']=pd.NaT
   result.append(d[['variable','observation_date','available_at_utc','value','source_url','value_quality','temporal_quality','release_date']])
   log.append({'source':str(p),'status':'IMPORTED_UNVERIFIED','rows':len(d)})
  except Exception as exc:
   log.append({'source':str(p),'status':'INVALID','error':str(exc)[:250]})
 return pd.concat(result,ignore_index=True) if result else pd.DataFrame()

def make_futures_spreads(long,log):
 d=long[long.variable.isin(['futures_m1','futures_m2','futures_m6'])].copy()
 if d.empty:return long
 p=d.pivot_table(index='observation_date',columns='variable',values='value',aggfunc='last')
 extras=[]
 for a,b,tar in [('futures_m1','futures_m2','wti_m1_m2'),('futures_m1','futures_m6','wti_m1_m6')]:
  if a not in p or b not in p:continue
  v=(p[a]-p[b]).dropna()
  for dt,x in v.items():
   t=pd.Timestamp(dt).tz_localize(TZ)+pd.Timedelta(days=1)
   extras.append({'variable':tar,'observation_date':dt,'value':float(x),
    'source_url':'https://www.eia.gov/opendata/browser/petroleum/pri/fut',
    'available_at_utc':t.tz_convert('UTC'),
    'value_quality':'EIA_CLOSED_2024_04_05_REVISED_SERIES',
    'temporal_quality':'NEXT_DAY_FUTURES_ASSUMPTION_NOT_CERTIFIED','release_date':pd.NaT})
  log.append({'source':tar,'status':'DERIVED_FUTURES_SPREAD','rows':len(v)})
 return pd.concat([long,pd.DataFrame(extras)],ignore_index=True) if extras else long

def merge_features(base,external,cols):
 out=clean_time(base)
 # Preserve all original fields, only append named external features.
 for name,subset in external.groupby('variable',sort=True):
  g=subset.copy();g['available_at_utc']=utc(g.available_at_utc)
  g=g[g.available_at_utc.notna()].sort_values('available_at_utc').drop_duplicates('available_at_utc',keep='last')
  if g.empty:continue
  col='exploratory_'+name
  g=g[['available_at_utc','value']].rename(columns={'available_at_utc':'timestamp','value':col})
  out=pd.merge_asof(out.sort_values('timestamp'),g,on='timestamp',direction='backward')
  cols.append(col)
 # All external fields are tagged exploratory unless independent archived original value evidence exists.
 out['external_features_pit_certified']=False
 out['external_features_exploratory_only']=bool(len(cols))
 for k in ('macro_effective_utc','macro_vintage_max','cot_available_assumed_utc'):
  if k in out:
   t=utc(out[k]);ok=t.isna() | (out.timestamp>=t)
   if not bool(ok.all()):raise ValueError('LOOK-AHEAD verificato per '+k)
 if 'macro_effective_utc' in out:
  out['macro_snapshot_age_days_actual']=(out.timestamp-utc(out.macro_effective_utc)).dt.total_seconds()/86400
  out['macro_snapshot_stale_gt10d']=out.macro_snapshot_age_days_actual>10
 return out

def h4_outcomes(h4,daily):
 """Entry next H4 open; exit after N subsequent D1 timestamps; no overlapping trades asserted.

 If the same H4 bar crosses stop and target mark ambiguous; never impute sequence.
 """
 h4=validate_ohlc(clean_time(h4));daily=validate_ohlc(clean_time(daily))
 tr=pd.concat([h4.high-h4.low,(h4.high-h4.close.shift()).abs(),(h4.low-h4.close.shift()).abs()],axis=1).max(axis=1)
 h4['atr14_h4_at_close']=tr.rolling(14,min_periods=14).mean()
 sig=h4.signal_bull_at_close.astype(str).str.lower().isin(('true','1'))
 # Daily UTC candle starts; use session bars by observed D1 index. 5/10 = observed D1 bars (not exchange calendar).
 dates=daily.timestamp.to_numpy(dtype='datetime64[ns]')
 times=h4.timestamp.to_numpy(dtype='datetime64[ns]')
 records=[]
 for i in np.flatnonzero(sig):
  if i+1>=len(h4):continue
  entry=h4.iloc[i+1];price=float(entry.open);atr=h4.iloc[i].atr14_h4_at_close
  start=int(np.searchsorted(dates,times[i+1],side='right')-1)
  if start<0:continue
  r={'signal_timestamp':h4.iloc[i].timestamp.isoformat(),'entry_timestamp':entry.timestamp.isoformat(),'entry_open':price,'atr14_h4_signal':atr}
  for n in (5,10):
   end=start+n
   # end is the first D1 bar beyond our N daily sessions
   if end>=len(daily):
    r.update({f'complete_{n}d':False,f'return_{n}d_pct':np.nan,f'mfe_{n}d_pct':np.nan,f'mae_{n}d_pct':np.nan,f'first_touch_{n}d':None})
    continue
   end_time=dates[end]
   j=np.searchsorted(times,end_time,side='left')
   window=h4.iloc[i+1:j]
   if window.empty:
    r[f'complete_{n}d']=False;continue
   high=float(window.high.max());low=float(window.low.min());last=window.iloc[-1]
   r.update({f'complete_{n}d':True,f'exit_{n}d_timestamp':last.timestamp.isoformat(),
    f'return_{n}d_pct':100*(float(last.close)/price-1),
    f'mfe_{n}d_pct':100*(high/price-1),f'mae_{n}d_pct':100*(low/price-1),
    f'up_2atr_h4_{n}d':bool(high>=price+2*atr) if pd.notna(atr) else None,
    f'up_3atr_h4_{n}d':bool(high>=price+3*atr) if pd.notna(atr) else None})
   touch='NO_HIT'
   if pd.notna(atr):
    stop=price-1.5*atr;target=price+2*atr
    for candle in window.itertuples():
     s=candle.low<=stop;t=candle.high>=target
     if s and t:touch='AMBIGUOUS_SAME_BAR';break
     if s:touch='STOP_FIRST';break
     if t:touch='TARGET_FIRST';break
   r[f'first_touch_{n}d']=touch
  records.append(r)
 return pd.DataFrame(records)

def optional_urls_check(session,log):
 sources={'OPEC_MOMR':'https://www.opec.org/monthly-oil-market-report.html',
 'IEA_OMR':'https://www.iea.org/reports/oil-market-report-november-2026',
 'EIA_STEO':'https://www.eia.gov/outlooks/steo/',
 'EIA_WPSR_SCHEDULE':SCHEDULE,
 'CME_DATA_MINE':'https://www.cmegroup.com/market-data/datamine-historical-data.html'}
 for code,url in sources.items():
  try:
   resp=fetch(session,url)
   log.append({'source':code,'status':'SOURCE_PAGE_REACHABLE_ONLY_NOT_DATA_INGESTED','url':resp.url,'http':resp.status_code})
  except Exception as exc:log.append({'source':code,'status':'SOURCE_PAGE_UNAVAILABLE','url':url,'error':str(exc)[:150]})

def execute(args,session=None):
 out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 log=[];audit={'phase':'28_UNICA','status':'IN_PROGRESS','limitations':[], 'source_acquisition':log}
 session=session or requests.Session()
 inpath=Path(args.input)
 if not inpath.is_file():raise FileNotFoundError('Manca ZIP Fase27: '+str(inpath))
 audit['input_sha256']=digest(inpath.read_bytes())
 try:
  calendar=get_eia_calendar(session,log) if not args.offline else pd.DataFrame(columns=['observation_date','release_date','source_url','availability_utc_next_day','temporal_quality'])
  write_csv(calendar,out/'eia_wpsr_calendario_ufficiale.csv')
  fundamental=build_eia(session,os.environ.get('EIA_API_KEY','').strip() if not args.offline else '',calendar,log)
  fundamental=make_futures_spreads(fundamental,log)
  user=external_import(args.extras,log)
  if not user.empty:fundamental=pd.concat([fundamental,user],ignore_index=True)
  write_csv(fundamental,out/'fondamentali_con_provenienza.csv')
  if not args.offline and not args.skip_catalog:optional_urls_check(session,log)
  audit['eia_archive_release_dates']=len(calendar)
  audit['fundamental_observations']=len(fundamental)
  audit['fundamental_coverage']={v:int((fundamental.variable==v).sum()) for v in sorted(set(sum(CATALOG.values(),[])))}
  with zipfile.ZipFile(inpath) as z:
   d1={}
   for lag in (14,21,35):
    raw=load_input(z,f'features_d1_cot_lag{lag}_macro_pit.csv')
    cols=[];x=merge_features(validate_ohlc(clean_time(raw)),fundamental,cols)
    write_csv(x,out/f'features_d1_lag{lag}_fase28.csv');d1[lag]=x
    audit[f'd1_lag{lag}_rows']=len(x)
    audit[f'd1_lag{lag}_features_added']=cols
   raw_h=load_input(z,'features_h4_macro_pit.csv')
   cols=[];h=merge_features(validate_ohlc(clean_time(raw_h)),fundamental,cols)
   write_csv(h,out/'features_h4_fase28.csv')
   e=h4_outcomes(h,d1[14]);write_csv(e,out/'outcomes_h4_SOLO_TEST.csv')
   audit['h4_bars']=len(h);audit['h4_events']=len(e)
   audit['h4_complete_10d']=int(e.complete_10d.sum()) if not e.empty else 0
   audit['h4_same_bar_ambiguous']=int((e.first_touch_10d=='AMBIGUOUS_SAME_BAR').sum()) if not e.empty else 0
   audit['macro_stale_gt10d_h4']=int(h.macro_snapshot_stale_gt10d.sum())
  audit['limitations']=[
   'EIA archive gives report release date; downloaded API values may be revised: NOT PIT-CERTIFIED.',
   'Archive page may cover only certain years. Missing dates are not replaced by guessed Wednesdays.',
   'Historical futures series EIA stops 2024-04-05; M6 may be unavailable. Spreads are never fabricated.',
   'OPEC/IEA/China/consensus/geopolitics are cataloged, not automatically parsed into reliable original numeric vintages.',
   'Snapshot macro weekly can be stale. No refreshed daily ALFRED vintages have been invented.',
   'WTI TVC_USOIL is a proxy, not a verified executable futures roll-adjusted settlement.',
   'COT lags 14/21/35 are assumptions; exceptional historical delays not fully certified.',
   'H4 OHLC cannot order stop and target within one bar; such rows marked ambiguous.',
   'D1 bar count is not an official exchange session calendar.',
   'Other families may require paid feeds or authenticated licensed original-report snapshots.'
  ]
  audit['status']='COMPLETE_WITH_DOCUMENTED_GAPS'
 except Exception as exc:
  audit['status']='FAILED';audit['fatal_error']=repr(exc);audit['traceback']=traceback.format_exc()
  raise
 finally:
  (out/'audit_fase28.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
  pd.DataFrame(log).to_csv(out/'log_acquisizione.csv',index=False)
  (out/'LEGGIMI.txt').write_text('Fase 28 unica. Leggere audit_fase28.json: i dati EIA API possono essere revisionati e NON sono PIT certificati. Outcomes H4 sono SOLO_TEST. Fonti non disponibili non vengono fabbricate.\n',encoding='utf-8')
 return audit

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--input',default='macro-swing-lab-fase27-pit-risultati.zip')
 p.add_argument('--out',default='fase28_risultati')
 p.add_argument('--extras',default='fase28_input')
 p.add_argument('--offline',action='store_true',help='Test locale senza chiamate esterne')
 p.add_argument('--skip-catalog',action='store_true')
 args=p.parse_args()
 audit=execute(args)
 print(json.dumps({k:v for k,v in audit.items() if k in ('status','h4_events','fundamental_observations','eia_archive_release_dates')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
