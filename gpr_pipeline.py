#!/usr/bin/env python3
"""GPR vintage-aware downloader and weekly signal builder. No trading signal generated."""
import argparse, hashlib, io, json, os, re, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import requests

SITE='https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls'
API='https://api.github.com/repos/iacoviel/iacoviel.github.io/contents/gpr_archive_files'
HEADERS={'User-Agent':'MacroSwingLab-GPR-Research/1.0','Accept':'application/vnd.github+json'}

def fetch(url, session):
    for attempt in range(4):
        try:
            r=session.get(url,timeout=70);r.raise_for_status();return r.content
        except requests.RequestException:
            if attempt==3:raise
            time.sleep(2**attempt)

def parse_xls(blob):
    book=pd.ExcelFile(io.BytesIO(blob))
    for sheet in book.sheet_names:
        d=pd.read_excel(book,sheet_name=sheet)
        d.columns=[str(c).strip() for c in d.columns]
        datecols=[c for c in d if re.search(r'date|day|time',c,re.I)]
        if not datecols: datecols=[d.columns[0]]
        for dc in datecols:
            raw=d[dc]
            dt=pd.to_datetime(raw,errors='coerce')
            if pd.api.types.is_numeric_dtype(raw):
                excel=pd.to_datetime(raw,unit='D',origin='1899-12-30',errors='coerce')
                if excel.notna().sum()>dt.notna().sum()/2:dt=excel
            if dt.notna().sum()<50:continue
            nums=[c for c in d.columns if c!=dc and pd.to_numeric(d[c],errors='coerce').notna().sum()>=50]
            prefer=[c for c in nums if re.search(r'^gpr(d|_d|daily)?$|^gpr$',c,re.I)]
            if not prefer:prefer=[c for c in nums if 'gpr' in c.lower()]
            if not prefer:continue
            col=prefer[0]
            out=pd.DataFrame({'observation_date':dt.dt.normalize(),'gpr':pd.to_numeric(d[col],errors='coerce')}).dropna()
            out=out[(out.observation_date>='1985-01-01')&(out.observation_date<=pd.Timestamp.today()+pd.Timedelta(days=1))]
            out=out.drop_duplicates('observation_date',keep='last').sort_values('observation_date')
            if len(out)>500:return out,{'sheet':sheet,'date_column':dc,'value_column':col}
    raise ValueError('Impossibile individuare colonne data/GPR: verificare formato Excel fonte')

def archive_list(session):
    result=[]
    for page in range(1,30):
        url=API+f'?per_page=100&page={page}'
        r=session.get(url,headers=HEADERS,timeout=60)
        if r.status_code==403:raise RuntimeError('GitHub API rate limit; configurare GITHUB_TOKEN')
        r.raise_for_status(); batch=r.json()
        if not isinstance(batch,list):raise ValueError('GitHub contents API non ha restituito lista')
        for x in batch:
            name=x['name'];m=re.fullmatch(r'data_gpr_daily_recent_(\d{8})\.xls',name,re.I)
            if m:result.append((pd.Timestamp(m.group(1)),x['download_url'],name))
        if len(batch)<100:break
    return sorted(result)

def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    sess=requests.Session();sess.headers.update(HEADERS)
    if os.getenv('GITHUB_TOKEN'):sess.headers['Authorization']='Bearer '+os.getenv('GITHUB_TOKEN')
    manifest=[]; observations=[]
    archive=archive_list(sess) if not args.latest_only else []
    if args.max_vintages:archive=archive[-args.max_vintages:]
    if not archive and not args.latest_only:print('AVVISO: nessuna vintage storica individuata')
    sources=[(d,url,n,'archived') for d,url,n in archive]
    # Latest is never backdated: snapshot is known only on download day, even if it contains old observations.
    sources.append((pd.Timestamp(datetime.now(timezone.utc).date()),SITE,'data_gpr_daily_recent.xls','download_snapshot'))
    for vintage,url,name,kind in sources:
        try:
            blob=fetch(url,sess);frame,metadata=parse_xls(blob)
            sha=hashlib.sha256(blob).hexdigest()
            # Archive filename date is an assumed *not-before* date, not verified release timestamp.
            # Add one calendar day to ensure conservative next-day use.
            available=vintage+pd.Timedelta(days=1)
            frame=frame[frame.observation_date<=vintage]
            frame['vintage_date']=vintage;frame['available_from']=available
            frame['source_type']=kind;frame['source_file']=name
            observations.append(frame)
            manifest.append(dict(filename=name,vintage_date=str(vintage.date()),available_from=str(available.date()),source_type=kind,sha256=sha,rows=len(frame),source_url=url,**metadata))
            print('OK',name,len(frame))
        except Exception as e:
            manifest.append(dict(filename=name,vintage_date=str(vintage.date()),source_type=kind,error=str(e)))
            print('ERRORE',name,str(e))
    pd.DataFrame(manifest).to_csv(out/'GPR_SOURCE_MANIFEST.csv',index=False)
    if not observations:raise RuntimeError('Nessun dato GPR importato: consultare manifest')
    allv=pd.concat(observations,ignore_index=True)
    allv.to_csv(out/'GPR_VINTAGES_LONG.csv',index=False)
    signals=pd.read_csv(args.signals,usecols=['asset','data_segnale']) if args.signals else None
    if signals is not None:
        signals['data_segnale']=pd.to_datetime(signals['data_segnale'])
        signals=signals.drop_duplicates().sort_values('data_segnale')
        # latest known vintage for each signal; do not use latest historical revisions in old weeks.
        avail=allv[['vintage_date','available_from']].drop_duplicates().sort_values('available_from')
        decisions=pd.merge_asof(signals[['data_segnale']].drop_duplicates().sort_values('data_segnale'),avail,left_on='data_segnale',right_on='available_from',direction='backward')
        by_vintage={k:g.sort_values('observation_date').set_index('observation_date')['gpr'] for k,g in allv.groupby('vintage_date')}
        rows=[]
        for _,r in decisions.iterrows():
            date=r['data_segnale'];v=r['vintage_date']
            row={'data_segnale':date,'gpr_vintage_date':v,'gpr_available_from':r['available_from'],'gpr_pit_quality':'NO_HISTORIC_VINTAGE'}
            if pd.notna(v):
                series=by_vintage[v];series=series[series.index<date]
                if len(series):
                    recent=series.iloc[-1];row.update(gpr_value=recent,gpr_observation_date=series.index[-1],gpr_age_days=(date-series.index[-1]).days,gpr_pit_quality='ARCHIVE_FILENAME_DATE_PROXY' if v!=sources[-1][0] else 'LIVE_SNAPSHOT')
                    if len(series)>=8:row['gpr_delta_7obs']=recent-series.iloc[-8]
                    if len(series)>=31:row['gpr_delta_30obs']=recent-series.iloc[-31]
                    if len(series)>=253 and series.iloc[-253:].std()>0:row['gpr_z252_past']=(recent-series.iloc[-253:-1].mean())/series.iloc[-253:-1].std()
            rows.append(row)
        weekly=pd.DataFrame(rows)
        panel=signals.merge(weekly,on='data_segnale',how='left',validate='many_to_one')
        if ((panel['gpr_available_from'].notna())&(panel['gpr_available_from']>panel['data_segnale'])).any():raise AssertionError('Look-ahead detected')
        panel.to_csv(out/'GPR_ETF_WEEKLY_PANEL.csv',index=False)
        weekly.to_csv(out/'GPR_WEEKLY_UNIQUE.csv',index=False)
        print('ETF panel rows:',len(panel),'coverage:',panel.gpr_value.notna().sum(),'/',len(panel))
    report={'vintages_downloaded':len(observations),'vintages_failed':sum('error' in x for x in manifest),'first_archive_date':str(archive[0][0].date()) if archive else None,'last_archive_date':str(archive[-1][0].date()) if archive else None,'historical_pit_note':'Vintage archive filename is a proxy for public release, not independently timestamp certified. No backfilling pre-archive history.'}
    (out/'GPR_REPORT.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='gpr_output');p.add_argument('--signals',default='');p.add_argument('--latest-only',action='store_true');p.add_argument('--max-vintages',type=int,default=0,help='0 = all')
    run(p.parse_args())
