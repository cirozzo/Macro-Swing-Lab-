#!/usr/bin/env python3
"""COT historical TFF (financial) and disaggregated (physical): conservative weekly PIT ETF join.
Usage: python cot_pit_etf.py --etf-zip etf-pit-52.zip --out output
Uses public CFTC Socrata datasets; requires requests, pandas, numpy.
This is exploratory, NOT an event-time-certified historical release calendar.
"""
import argparse, io, json, os, pathlib, zipfile
import numpy as np
import pandas as pd
import requests
DATASETS={'TFF':'gpe5-46if','DISAGG':'72hh-3qpy'}
BASE='https://publicreporting.cftc.gov/resource/'
def fetch(dataset, start='2007-01-01'):
    rows=[]; offset=0
    while True:
        r=requests.get(f'{BASE}{dataset}.json',params={'$limit':50000,'$offset':offset,'$where':f'report_date_as_yyyy_mm_dd >= \'{start}\''},timeout=90)
        r.raise_for_status(); part=r.json()
        if not part: break
        rows.extend(part); offset+=len(part)
        if len(part)<50000:break
    return pd.DataFrame(rows)
def col(df,*names):
    for n in names:
        if n in df.columns:return n
    return None
def make_features(df,kind):
    date=col(df,'report_date_as_yyyy_mm_dd','report_date_as_yyyy_mm_dd_','report_date')
    market=col(df,'market_and_exchange_names','market_and_exchange_names_','market_and_exchange')
    if not date or not market:raise ValueError(f'{kind}: colonne data/mercato assenti: {list(df.columns)[:30]}')
    if kind=='TFF':
        pairs=[('asset_manager','asset_mgr_positions_long_all','asset_mgr_positions_short_all'),('leveraged','lev_money_positions_long_all','lev_money_positions_short_all')]
    else:
        pairs=[('managed_money','m_money_positions_long_all','m_money_positions_short_all')]
    oi=col(df,'open_interest_all','open_interest_all_')
    out=[]
    for label,l,s in pairs:
        if l not in df or s not in df:continue
        x=pd.DataFrame({'report_date':pd.to_datetime(df[date],errors='coerce').dt.normalize(), 'market':df[market].astype(str), 'category':label,
          'net':pd.to_numeric(df[l],errors='coerce')-pd.to_numeric(df[s],errors='coerce'),
          'open_interest':pd.to_numeric(df[oi],errors='coerce') if oi else np.nan})
        x['net_oi']=x['net']/x['open_interest'].replace(0,np.nan)
        x=x.dropna(subset=['report_date','net_oi']).sort_values(['market','report_date'])
        x['change_net_oi']=x.groupby('market')['net_oi'].diff()
        # Min 52 previous weekly observations, percentile uses PAST ONLY; do not use current week's own position in threshold.
        x['prior_percentile']=x.groupby('market')['net_oi'].transform(lambda s:s.shift(1).rolling(156,min_periods=52).apply(lambda w: np.mean(w<=s.loc[w.index[-1]]) if False else np.nan,raw=False)) if False else np.nan
        # More interpretable: trailing prior mean and standard deviation.
        g=x.groupby('market')['net_oi']
        x['prior_mean']=g.transform(lambda s:s.shift(1).rolling(156,min_periods=52).mean())
        x['prior_sd']=g.transform(lambda s:s.shift(1).rolling(156,min_periods=52).std())
        x['z_position']=(x['net_oi']-x['prior_mean'])/x['prior_sd'].replace(0,np.nan)
        # COT report as of Tuesday, typically released Friday 15:30 ET. Conservative use NEXT MONDAY;
        # holiday-delayed release exceptions must be checked against official release schedule.
        x['usable_from']=x['report_date']+pd.to_timedelta((7-x['report_date'].dt.weekday)%7,unit='D')
        x['kind']=kind
        out.append(x)
    return pd.concat(out,ignore_index=True) if out else pd.DataFrame()
def main():
    p=argparse.ArgumentParser();p.add_argument('--etf-zip',required=True);p.add_argument('--out',default='output');p.add_argument('--cot-dir',default=None,help='Optional local TFF.csv and DISAGG.csv instead of downloads');a=p.parse_args()
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
    features=[];report={'source':'CFTC official Socrata','warning':'Conservative Monday availability is a proxy; check delayed historical releases before certification.'}
    for k,id in DATASETS.items():
        df=pd.read_csv(pathlib.Path(a.cot_dir)/f'{k}.csv',low_memory=False) if a.cot_dir else fetch(id)
        report[k+'_raw_rows']=len(df)
        f=make_features(df,k)
        report[k+'_feature_rows']=len(f)
        features.append(f)
    cot=pd.concat(features,ignore_index=True)
    cot.to_csv(out/'cot_features_pit_proxy.csv',index=False)
    with zipfile.ZipFile(a.etf_zip) as z: etf=pd.read_csv(z.open('osservazioni_pit_52.csv'))
    etf['data_ingresso']=pd.to_datetime(etf['data_ingresso']);etf['data_segnale']=pd.to_datetime(etf['data_segnale'])
    # No retroactive attachment to Friday signals: report Friday 15:30 ET is too late for old weekly Friday signal.
    # A COT available Monday may be used only for ETF entries Monday OR LATER, never for the prior Friday signal.
    # Output ready for explicit ETF/futures mapping, NOT cartesian join (which would create spurious tests).
    etf[['asset','data_segnale','data_ingresso','data_uscita','sedute','rendimento','excess_return']].to_csv(out/'etf_target_dates.csv',index=False)
    report['etf_rows']=len(etf);report['etf_assets']=sorted(etf.asset.unique().tolist());report['cot_markets']=int(cot.market.nunique());report['cot_date_min']=str(cot.report_date.min().date());report['cot_date_max']=str(cot.report_date.max().date())
    report['status']='FEATURES_READY_NOT_BACKTESTED';report['next_required']='Validated COT contract -> ETF exposure mapping; historical release exception calendar; temporal join with next executable ETF entry.'
    (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps(report,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
