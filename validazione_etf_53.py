#!/usr/bin/env python3
"""Macro Swing Lab 5.3 — exploratory statistical audit, NOT trading signals.
Uses only 5.2 artifact; no fresh price downloads or hidden tuning.
Bootstrap resamples calendar quarters jointly across regimes for each asset/horizon.
"""
import argparse, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

SEED=20261024
B=1200
COST_BPS=10  # assumed total round-trip cost, sensitivity only
OUT=Path('output/validazione_etf_53')
REQUIRED={'asset','regime','data_segnale','data_ingresso','data_uscita','sedute','rendimento','rendimento_spy','sigma_h','periodo'}

def bh(p):
    p=np.asarray(p,dtype=float); order=np.argsort(p); out=np.empty(len(p)); n=len(p)
    out[order]=np.minimum.accumulate((p[order]*n/(np.arange(n)+1))[::-1])[::-1].clip(0,1)
    return out

def load(path):
    if str(path).endswith('.zip'):
        with zipfile.ZipFile(path) as z:
            names=[x for x in z.namelist() if x.endswith('osservazioni_pit_52.csv')]
            if len(names)!=1: raise ValueError('ZIP must contain exactly one osservazioni_pit_52.csv')
            return pd.read_csv(z.open(names[0]))
    return pd.read_csv(path)

def run(df, boot=B):
    missing=REQUIRED-set(df.columns)
    if missing: raise ValueError(f'Missing columns: {sorted(missing)}')
    for col in ['data_segnale','data_ingresso','data_uscita']:
        df[col]=pd.to_datetime(df[col],errors='raise')
    for col in ['rendimento','rendimento_spy','sigma_h']:
        df[col]=pd.to_numeric(df[col],errors='raise')
    if df[list(REQUIRED)].isna().any().any(): raise ValueError('Missing required data')
    if df.duplicated(['asset','sedute','data_segnale']).any(): raise ValueError('Duplicate asset-horizon-signal')
    if not ((df.data_segnale<df.data_ingresso)&(df.data_ingresso<=df.data_uscita)).all(): raise ValueError('Invalid chronology')
    if not set(df.periodo.unique())<= {'RICERCA_PRE_2022','VERIFICA_DAL_2022'}: raise ValueError('Unexpected period labels')
    df['quarter']=df.data_segnale.dt.to_period('Q').astype(str)
    df['year']=df.data_segnale.dt.year
    df['excess_spy']=df.rendimento-df.rendimento_spy
    df['net_10bps']=df.rendimento-COST_BPS/10000
    df['vol_norm']=np.where(df.sigma_h>0,df.rendimento/df.sigma_h,np.nan)
    rng=np.random.default_rng(SEED)
    records=[]
    for (asset,horizon,period), g in df.groupby(['asset','sedute','periodo'],sort=True):
        # Quarter-level paired bootstrap preserves cross-regime co-movement in a quarter.
        qlist=sorted(g.quarter.unique()); qi={q:i for i,q in enumerate(qlist)}; Q=len(qlist)
        regimes=sorted(g.regime.unique()); R=len(regimes)
        counts=np.zeros((Q,R)); sums=np.zeros((Q,R)); spy_sums=np.zeros((Q,R))
        for j,r in enumerate(regimes):
            sub=g[g.regime==r]
            for q, part in sub.groupby('quarter'):
                i=qi[q]; counts[i,j]=len(part); sums[i,j]=part.rendimento.sum();spy_sums[i,j]=part.excess_spy.sum()
        draws=rng.integers(0,Q,size=(boot,Q))
        # Count multiplicities to reduce memory; bootstrap means for regime and all-regime baseline.
        weights=np.zeros((boot,Q),dtype=np.int16)
        for k in range(boot): weights[k]=np.bincount(draws[k],minlength=Q)
        bc=weights@counts; bs=weights@sums
        total_c=bc.sum(axis=1); total_s=bs.sum(axis=1)
        base=total_s/np.maximum(total_c,1)
        all_mean=sums.sum()/counts.sum()
        for j,r in enumerate(regimes):
            sub=g[g.regime==r].sort_values('data_segnale')
            n=len(sub); raw=sub.rendimento.to_numpy(); delta=raw.mean()-all_mean
            ok=bc[:,j]>0
            boot_delta=bs[ok,j]/bc[ok,j]-base[ok]
            if len(boot_delta)<boot*.95: raise ValueError(f'Insufficient bootstrap regime coverage: {asset} {r}')
            lo,hi=np.quantile(boot_delta,[.025,.975]);
            # Centered bootstrap null: one-sided H0 delta <= 0; exploratory, not causal.
            p=(1+np.sum((boot_delta-delta)>=delta))/(len(boot_delta)+1)
            # Cluster robustness: leave-one-year-out changes in delta vs contemporaneous baseline.
            loo=[]
            for yr in sorted(g.year.unique()):
                remain=g[g.year!=yr]; rs=remain[remain.regime==r]
                if len(rs): loo.append(rs.rendimento.mean()-remain.rendimento.mean())
            rec=dict(asset=asset,sedute=int(horizon),periodo=period,regime=r,n=n,quarters=int(sub.quarter.nunique()),years=int(sub.year.nunique()),
                rendimento_medio_pct=100*raw.mean(),baseline_medio_pct=100*all_mean,delta_baseline_pp=100*delta,
                ci95_delta_low_pp=100*lo,ci95_delta_high_pp=100*hi,p_one_sided_boot=p,
                prob_positiva=float((raw>0).mean()),excess_spy_medio_pct=100*sub.excess_spy.mean(),
                netto_10bps_medio_pct=100*sub.net_10bps.mean(),vol_norm_medio=float(sub.vol_norm.mean()),
                worst_pct=100*raw.min(),best_pct=100*raw.max(),
                loo_year_min_delta_pp=100*min(loo) if loo else np.nan,
                loo_year_max_delta_pp=100*max(loo) if loo else np.nan,
                segnale_solo_esplorativo=True)
            records.append(rec)
    res=pd.DataFrame(records)
    for period in res.periodo.unique():
        m=res.periodo==period
        res.loc[m,'q_bh_120_famiglia']=bh(res.loc[m,'p_one_sided_boot'])
    res['etichetta']='NON_CONFERMATO'
    res.loc[(res.periodo=='RICERCA_PRE_2022')&(res.n>=30)&(res.quarters>=12)&(res.ci95_delta_low_pp>0)&(res.q_bh_120_famiglia<.05),'etichetta']='EVIDENZA_ESPLORATIVA_PRE_2022'
    # Never promote 2022+ to truly independent holdout: previously inspected.
    return res.sort_values(['periodo','sedute','regime','delta_baseline_pp'],ascending=[True,True,True,False])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--bootstrap',type=int,default=B);args=ap.parse_args()
    if args.bootstrap<200: raise ValueError('bootstrap must be >= 200')
    df=load(args.input); OUT.mkdir(parents=True,exist_ok=True)
    res=run(df,args.bootstrap)
    res.to_csv(OUT/'test_120_combinazioni_53.csv',index=False,float_format='%.10g')
    pivot=res.pivot_table(index=['asset','sedute','regime'],columns='periodo',values=['delta_baseline_pp','n','ci95_delta_low_pp'],aggfunc='first')
    pivot.columns=['__'.join(map(str,x)) for x in pivot.columns]
    pivot.reset_index().to_csv(OUT/'confronto_periodi_53.csv',index=False)
    summary={'status':'AUDIT_ESPLORATIVO_NON_OPERATIVO','seed':SEED,'bootstrap':args.bootstrap,'resample_unit':'calendar_quarter',
        'baseline':'mean of same ETF and horizon in same period, all regimes',
        'multiple_testing':'Benjamini-Hochberg within period across all ETF x regime x horizon tests',
        'cost_assumption':'10 basis points total round-trip, hypothetical',
        'input_rows':len(df),'tests':len(res),'tests_by_period':res.groupby('periodo').size().to_dict(),
        'warnings':['All regimes share dates and macro signals; no independent ETF trials.',
          '2022+ was inspected in version 5.2; NOT untouched out-of-sample.',
          'Quarter bootstrap is approximate, not a proof of independence or causal effect.',
          'Yahoo adjusted OHLC are not historical PIT price vintages.',
          'Prices, dividends, spreads, survivorship and execution need separate audit.',
          'No live-trading recommendation; exploratory labels only.']}
    (OUT/'report_53.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps(summary,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
