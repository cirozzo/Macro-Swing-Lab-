#!/usr/bin/env python3
"""Macro Swing Lab 5.4: exploratory regime-value tests and prequential forecast audit.
No external data downloads. No trading recommendations.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

REGIMES = ('R1','R2','R3','R4')

def load(path):
    df = pd.read_csv(path)
    needed = {'asset','regime','data_segnale','data_ingresso','data_uscita','sedute','rendimento','rendimento_spy','sigma_h'}
    if missing := needed - set(df):
        raise ValueError(f'Missing columns: {sorted(missing)}')
    for col in ('data_segnale','data_ingresso','data_uscita'):
        df[col] = pd.to_datetime(df[col],errors='raise')
    df['regime_id'] = df['regime'].str.extract(r'^(R[1-4])',expand=False)
    if df.regime_id.isna().any(): raise ValueError('Unknown regimes')
    for col in ('rendimento','rendimento_spy','sigma_h'):
        df[col] = pd.to_numeric(df[col],errors='raise')
    if not np.isfinite(df[['rendimento','rendimento_spy','sigma_h']].to_numpy()).all():
        raise ValueError('Non-finite returns or sigma')
    if df.duplicated(['asset','sedute','data_segnale']).any():
        raise ValueError('Duplicate signal/asset/horizon')
    if (df.data_ingresso <= df.data_segnale).any() or (df.data_uscita < df.data_ingresso).any():
        raise ValueError('Invalid execution chronology')
    return df.sort_values(['asset','sedute','data_segnale']).reset_index(drop=True)

def nonoverlap(group):
    rows=[]; last_end=pd.Timestamp.min
    for row in group.sort_values('data_ingresso').itertuples():
        if row.data_ingresso > last_end:
            rows.append(row.Index); last_end=row.data_uscita
    return group.loc[rows].copy()

def permutation_test(df, nperm, seed, min_shift=13):
    """Global circular regime shift; preserves regime runs and cross-ETF dependence.
    Null is temporal alignment, NOT causal predictive content. Max-stat controls family-wise error.
    """
    rng=np.random.default_rng(seed)
    rows=[]; nulls=[]; groups=[]
    for (asset,horizon), g in df.groupby(['asset','sedute'],sort=True):
        g=g.sort_values('data_segnale')
        if len(g)<70: continue
        y=(g.rendimento-g.rendimento_spy).to_numpy(dtype=float)
        sig=g.sigma_h.to_numpy(dtype=float)
        y=y/np.maximum(sig,1e-8)
        # use calendar-wide regime vector, not asset-specific, to keep common shifts
        groups.append((asset,int(horizon),g.data_segnale.to_numpy(),y,g.regime_id.to_numpy()))
    calendar=df[['data_segnale','regime_id']].drop_duplicates()
    if calendar.duplicated('data_segnale').any(): raise ValueError('Conflicting regimes on same signal date')
    calendar=calendar.sort_values('data_segnale')
    dates=calendar.data_segnale.to_numpy(); reg=calendar.regime_id.to_numpy()
    n=len(reg)
    if n < 2*min_shift+20: raise ValueError('Insufficient history for circular shifts')
    idx={pd.Timestamp(d):i for i,d in enumerate(dates)}
    mapped=[]
    for asset,horizon,gdates,y,gr in groups:
        ii=np.array([idx[pd.Timestamp(d)] for d in gdates])
        if not np.array_equal(reg[ii],gr): raise ValueError('Regime alignment mismatch')
        mapped.append((asset,horizon,ii,y))
    def calc(regime_vec):
        out=[]
        for asset,horizon,ii,y in mapped:
            rr=regime_vec[ii]
            for r in REGIMES:
                mask=rr==r
                if mask.sum()<15 or (~mask).sum()<15:
                    out.append(np.nan)
                else:
                    out.append(float(y[mask].mean()-y[~mask].mean()))
        return np.array(out)
    obs=calc(reg)
    shifts=rng.integers(min_shift,n-min_shift,size=nperm)
    perm=np.stack([calc(np.roll(reg,int(s))) for s in shifts])
    max_abs=np.nanmax(np.abs(perm),axis=1)
    labels=[(a,h,r) for a,h,_,_ in mapped for r in REGIMES]
    for j,(a,h,r) in enumerate(labels):
        valid=np.isfinite(perm[:,j]); o=obs[j]
        if not np.isfinite(o) or not valid.any():
            p=np.nan; pf=np.nan
        else:
            p=(1+int(np.sum(np.abs(perm[valid,j])>=abs(o))))/(1+int(valid.sum()))
            pf=(1+int(np.sum(max_abs>=abs(o))))/(1+len(max_abs))
        rows.append({'asset':a,'sedute':h,'regime':r,'delta_excess_sigma_vs_other_regimes':o,
                     'p_shift_nominale':p,'p_shift_FWER_maxT':pf,
                     'n_regime':int(np.sum(reg[ii]==r)) if False else int(np.sum(reg[[m[2] for m in mapped if m[0]==a and m[1]==h][0]]==r))})
    return pd.DataFrame(rows),{'permutations':nperm,'min_shift_weeks':min_shift,'seed':seed,
                               'null':'circularly shift weekly regime sequence, preserving regime run lengths and ETF date alignment',
                               'test':'absolute regime vs other regimes excess-return / contemporaneous sigma_h; global maxT FWER'}

def prequential(df, min_train=156):
    """At each signal, train only on trades whose EXIT was strictly before signal.
    Forecast each ETF return via expanding unconditional mean or regime mean.
    This is a predictive scoring comparison, not a simulated trading PnL.
    """
    out=[]; summary=[]
    for (asset,h),g in df.groupby(['asset','sedute']):
        g=g.sort_values('data_segnale').reset_index(drop=True)
        y=g.rendimento.to_numpy(); r=g.regime_id.to_numpy()
        exits=g.data_uscita.to_numpy(); dates=g.data_segnale.to_numpy()
        for i in range(len(g)):
            eligible=np.flatnonzero(exits[:i]<dates[i])
            if len(eligible)<min_train: continue
            base=float(y[eligible].mean()); sub=eligible[r[eligible]==r[i]]
            # shrink regime estimate to unconditional mean, to limit noisy small cells
            regime=float((y[sub].sum()+30*base)/(len(sub)+30))
            out.append({'asset':asset,'sedute':int(h),'data_segnale':str(pd.Timestamp(dates[i]).date()),
                        'regime':r[i],'periodo':'POST_2022' if dates[i]>=np.datetime64('2022-01-01') else 'PRE_2022',
                        'actual':float(y[i]),'forecast_baseline':base,'forecast_regime':regime,
                        'se_baseline':(y[i]-base)**2,'se_regime':(y[i]-regime)**2,
                        'n_training':len(eligible),'n_regime_training':len(sub)})
    pred=pd.DataFrame(out)
    if pred.empty: raise ValueError('No predictions generated')
    for (asset,h,period),g in pred.groupby(['asset','sedute','periodo']):
        mse0=g.se_baseline.mean(); mse1=g.se_regime.mean()
        summary.append({'asset':asset,'sedute':int(h),'periodo':period,'n':len(g),
                        'mse_baseline':mse0,'mse_regime':mse1,
                        'oos_r2_vs_baseline':1-mse1/mse0 if mse0>0 else np.nan,
                        'delta_mse':mse0-mse1})
    return pred,pd.DataFrame(summary)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True)
    ap.add_argument('--output',default='output/valore_predittivo_54')
    ap.add_argument('--permutations',type=int,default=999)
    ap.add_argument('--seed',type=int,default=20261025)
    args=ap.parse_args()
    if args.permutations<99: raise ValueError('At least 99 permutations')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    df=load(args.input)
    tests,method=permutation_test(df,args.permutations,args.seed)
    tests.to_csv(out/'test_shift_regimi_54.csv',index=False)
    pred,score=prequential(df)
    score.to_csv(out/'previsioni_walk_forward_54.csv',index=False)
    pred.to_csv(out/'dettaglio_previsioni_54.csv',index=False)
    non=[]
    for (a,h),g in df.groupby(['asset','sedute']):
        sub=nonoverlap(g)
        for r in REGIMES:
            rr=sub[sub.regime_id==r];other=sub[sub.regime_id!=r]
            if len(rr)<5 or len(other)<5:continue
            non.append({'asset':a,'sedute':int(h),'regime':r,'n_regime':len(rr),'n_other':len(other),
                        'delta_excess_vs_other':float((rr.rendimento-rr.rendimento_spy).mean()-(other.rendimento-other.rendimento_spy).mean())})
    pd.DataFrame(non).to_csv(out/'sensibilita_non_overlap_54.csv',index=False)
    report={'status':'ESPLORATIVO_NON_VALIDAZIONE_INDIPENDENTE','rows':len(df),'assets':df.asset.nunique(),
            'period': [str(df.data_segnale.min().date()),str(df.data_segnale.max().date())],
            'method':method,'n_tested':len(tests),'fwer_p_lt_005':int((tests.p_shift_FWER_maxT<.05).sum()),
            'prequential':'expanding historical means, min 156 settled historical signals, 30-observation shrinkage',
            'caveats':['Post-2022 previously inspected; not independent holdout',
                       'Circular shifts test alignment under strong stationarity assumptions; macro/market secular trends may invalidate null',
                       'Volatility denominator sigma_h from input must be verified as pre-entry',
                       'Prequential MSE tests predictive point forecasts, not trading profitability',
                       'Overlapping return windows and adjusted current-vintage Yahoo data',
                       'No momentum or volatility baseline beyond the volatility-normalized permutation statistic',
                       'No multiple-testing adjusted significance on walk-forward forecast scores']}
    (out/'report_54.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({'n_tests':len(tests),'FWER discoveries':report['fwer_p_lt_005'],'predictions':len(pred)},indent=2))
if __name__=='__main__':main()
