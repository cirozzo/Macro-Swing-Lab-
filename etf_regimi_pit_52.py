#!/usr/bin/env python3
"""Macro Swing Lab 5.2: ricerca descrittiva ETF x regime PIT, NON segnali di trading.

Segnale venerdi' post-close; dati macro vintage al giovedi' precedente.
Ingresso open della PRIMA seduta SPY successiva; uscita close della 5a/10a
seduta inclusa quella d'ingresso. OHLC adjusted Yahoo: dati correntemente
rettificati per dividendi/split, NON vintage storiche dei prezzi.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

ASSETS = ['QQQ','SPY','IWM','SMH','XLE','XLF','XLV','XLP','GLD','SLV','USO','TLT','IEF','UUP','EEM']
HORIZONS = [5, 10]
SPLIT = pd.Timestamp('2022-01-01')
REQUIRED = ['data_segnale','vintage_asof','regime_pit_49']


def load_macro(path):
    df = pd.read_csv(path)
    missing = set(REQUIRED) - set(df.columns)
    if missing: raise ValueError(f'Dataset PIT incompleto: {sorted(missing)}')
    for c in ['data_segnale','vintage_asof']:
        df[c] = pd.to_datetime(df[c], errors='raise').dt.normalize()
    if df['data_segnale'].duplicated().any(): raise ValueError('Date PIT duplicate')
    if not (df.vintage_asof < df.data_segnale).all(): raise ValueError('Violazione vintage PIT')
    if not df.regime_pit_49.str.match(r'^R[1-4]_').all(): raise ValueError('Regimi sconosciuti')
    return df.sort_values('data_segnale').reset_index(drop=True)


def download_one(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, auto_adjust=True,
                       actions=False, progress=False, threads=False, timeout=45)
    if data is None or data.empty: raise RuntimeError('download vuoto')
    if isinstance(data.columns, pd.MultiIndex):
        # yfinance 0.2+ usually returns ('Open', 'TICKER')
        if ticker in data.columns.get_level_values(-1):
            data = data.xs(ticker, axis=1, level=-1)
        elif len(data.columns.get_level_values(-1).unique()) == 1:
            data.columns = data.columns.get_level_values(0)
        else: raise RuntimeError('MultiIndex non riconosciuto')
    needed = ['Open','High','Low','Close']
    if not set(needed).issubset(data.columns): raise RuntimeError('OHLC mancanti')
    data = data[needed].copy()
    idx = pd.DatetimeIndex(data.index)
    if idx.tz is not None: idx = idx.tz_localize(None)
    data.index = idx.normalize()
    data = data[~data.index.duplicated(keep='last')].sort_index()
    data = data.apply(pd.to_numeric, errors='coerce').dropna()
    data = data[(data[needed] > 0).all(axis=1)]
    if len(data) < 100: raise RuntimeError(f'Sedute insufficienti: {len(data)}')
    return data


def make_observations(macro, market):
    spy = market['SPY']
    common_index = spy.index
    rows, coverage = [], []
    for asset, bars in market.items():
        b = bars.loc[bars.index.intersection(common_index)].copy()
        b = b.sort_index()
        # important: use signal-time data only, never entry/exit in vol estimate
        sigma = b.Close.pct_change(fill_method=None).rolling(60, min_periods=60).std(ddof=1)
        count = 0
        for s in macro.itertuples(index=False):
            signal, vintage = s.data_segnale, s.vintage_asof
            # Vintage precedes Friday; entry only after Friday.
            signal_idx = b.index.searchsorted(signal, side='right') - 1
            if signal_idx < 60: continue
            prev_session = b.index[signal_idx]
            # Friday signal must refer to same week and no close earlier than Thu.
            if prev_session < signal - pd.Timedelta(days=1): continue
            # Require next SPY session also traded by asset (no shifted entry)
            spy_next = common_index.searchsorted(signal, side='right')
            if spy_next >= len(common_index): continue
            entry_date = common_index[spy_next]
            if entry_date not in b.index: continue
            entry_idx = b.index.get_loc(entry_date)
            vol = float(sigma.iloc[signal_idx])
            if not np.isfinite(vol) or vol <= 0: continue
            for h in HORIZONS:
                # h trading sessions inclusive, so final is index+ h-1
                exit_spy_idx = spy_next + h - 1
                if exit_spy_idx >= len(common_index): continue
                exit_date = common_index[exit_spy_idx]
                if entry_idx + h - 1 >= len(b): continue
                if b.index[entry_idx + h - 1] != exit_date: continue
                entry_open = float(b.at[entry_date,'Open'])
                exit_close = float(b.at[exit_date,'Close'])
                spy_entry = float(spy.at[entry_date,'Open'])
                spy_exit = float(spy.at[exit_date,'Close'])
                # MFE and MAE use high/low during the holding interval.
                window = b.iloc[entry_idx:entry_idx + h]
                ret = exit_close / entry_open - 1
                ret_spy = spy_exit / spy_entry - 1
                mfe = float(window.High.max()/entry_open - 1)
                mae = float(window.Low.min()/entry_open - 1)
                rows.append(dict(asset=asset, regime=s.regime_pit_49,
                    data_segnale=signal, vintage_asof=vintage,
                    data_ingresso=entry_date, data_uscita=exit_date,
                    sedute=h, entry_open=entry_open, exit_close=exit_close,
                    rendimento=ret, rendimento_spy=ret_spy, excess_return=ret-ret_spy,
                    mfe=mfe, mae=mae, sigma_daily_60=vol,
                    sigma_h=vol*np.sqrt(h), mfe_sigma=mfe/(vol*np.sqrt(h)),
                    mae_sigma=mae/(vol*np.sqrt(h)),
                    periodo='RICERCA_PRE_2022' if signal < SPLIT else 'VERIFICA_DAL_2022'))
                count += 1
        coverage.append({'asset':asset,'sedute_download':len(b),'osservazioni':count,
                         'prima_seduta':str(b.index.min().date()),'ultima_seduta':str(b.index.max().date())})
    if not rows: raise RuntimeError('Nessuna osservazione creata')
    result = pd.DataFrame(rows).sort_values(['asset','sedute','data_ingresso'])
    if result.duplicated(['asset','data_segnale','sedute']).any(): raise ValueError('Duplicati osservazioni')
    if not ((result.data_segnale < result.data_ingresso)&(result.data_ingresso <= result.data_uscita)).all():
        raise ValueError('Ordine temporale errato')
    if (result.data_uscita > pd.Timestamp.now(tz='UTC').tz_localize(None).normalize()).any():
        raise ValueError('Rendimenti futuri non ancora disponibili')
    return result.reset_index(drop=True), pd.DataFrame(coverage)


def non_overlap(df):
    # Select independently of regime. Intervals [entry, exit] never overlap.
    indexes = []
    for _, group in df.groupby(['asset','sedute','periodo'],sort=True):
        end = pd.Timestamp.min
        for i, row in group.sort_values('data_ingresso').iterrows():
            if row.data_ingresso > end:
                indexes.append(i)
                end = row.data_uscita
    return df.loc[indexes].copy()


def summary(df):
    result=[]
    groupby = ['periodo','asset','sedute']
    for key, g in df.groupby(groupby, sort=True):
        for regime, subset in [('TUTTI', g), *list(g.groupby('regime'))]:
            if isinstance(regime, tuple): raise AssertionError('invalid group')
            ret=subset.rendimento
            result.append(dict(periodo=key[0],asset=key[1],sedute=key[2],regime=regime,
                n=len(subset),anni_distinti=subset.data_segnale.dt.year.nunique(),
                prob_positiva=float((ret>0).mean()),
                rendimento_medio_pct=float(100*ret.mean()),rendimento_mediano_pct=float(100*ret.median()),
                delta_spy_medio_pct=float(100*subset.excess_return.mean()),
                prob_batte_spy=float((subset.excess_return>0).mean()),
                mfe_mediano_sigma=float(subset.mfe_sigma.median()),
                mae_mediano_sigma=float(subset.mae_sigma.median()),
                prob_mfe_1_5sigma=float((subset.mfe_sigma >= 1.5).mean()),
                peggiore_rendimento_pct=float(100*ret.min())))
    out=pd.DataFrame(result)
    base=out[out.regime=='TUTTI'][groupby+['rendimento_medio_pct','prob_positiva','prob_mfe_1_5sigma']]
    base=base.rename(columns={c:c+'_baseline' for c in base if c not in groupby})
    out=out.merge(base,on=groupby,validate='many_to_one')
    for col in ['rendimento_medio_pct','prob_positiva','prob_mfe_1_5sigma']:
        out['delta_'+col]=out[col]-out[col+'_baseline']
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='input/regimi_pit_49.csv')
    p.add_argument('--outdir',default='output/etf_pit_52')
    args=p.parse_args()
    output=Path(args.outdir); output.mkdir(parents=True,exist_ok=True)
    macro=load_macro(args.input)
    end=(pd.Timestamp.now(tz='UTC').tz_localize(None)+pd.Timedelta(days=2)).strftime('%Y-%m-%d')
    start=(macro.data_segnale.min()-pd.Timedelta(days=150)).strftime('%Y-%m-%d')
    market={}; failures=[]
    for asset in ASSETS:
        try:
            market[asset]=download_one(asset,start,end)
            print(f'{asset}: {len(market[asset])} sedute',flush=True)
        except Exception as exc:
            failures.append({'asset':asset,'errore':str(exc)})
            print(f'DOWNLOAD FALLITO {asset}: {exc}',flush=True)
    pd.DataFrame(failures,columns=['asset','errore']).to_csv(output/'errori_download_52.csv',index=False)
    if 'SPY' not in market: raise RuntimeError('SPY indispensabile per benchmark e calendario')
    if len(market) != len(ASSETS):
        raise RuntimeError(f'Prezzi mancanti per {len(ASSETS)-len(market)} ETF. Nessun ranking parziale ammesso.')
    obs,cov=make_observations(macro,market)
    cov.to_csv(output/'copertura_52.csv',index=False)
    # Limit exposure: exclude tickers lacking a sufficiently broad sample.
    if (cov.osservazioni < 200).any(): raise RuntimeError('Copertura ETF insufficiente, controllare copertura_52.csv')
    independent=non_overlap(obs)
    stats=summary(independent)
    stats['campione_limitato']=(stats.n<20)|(stats.anni_distinti<5)
    stats['solo_descrittivo']=True
    annual=(independent.assign(anno=independent.data_segnale.dt.year)
            .groupby(['periodo','asset','regime','sedute','anno'])
            .agg(n=('rendimento','size'),rendimento_medio_pct=('rendimento',lambda x:100*x.mean()),
                 prob_positiva=('rendimento',lambda x:(x>0).mean()))
            .reset_index())
    obs.to_csv(output/'osservazioni_pit_52.csv',index=False)
    independent.to_csv(output/'episodi_non_overlap_52.csv',index=False)
    stats.to_csv(output/'statistiche_descrittive_52.csv',index=False)
    annual.to_csv(output/'stabilita_annuale_52.csv',index=False)
    meta={'status':'RICERCA_DESCRITTIVA_NON_VALIDATA','macro_settimane':len(macro),
          'etf':len(market),'osservazioni':len(obs),'episodi_non_overlap':len(independent),
          'prezzi':'Yahoo adjusted OHLC scaricati oggi, non point-in-time vintage',
          'ingresso':'open prima seduta successiva al venerdi del segnale',
          'uscita':'close della quinta/decima seduta inclusa entrata',
          'avvertenza':'Non usare per trading live; costi, spread, execution, testing multiplo non inclusi. Selezione ETF predefinita; survivorship bias possibile.'}
    (output/'report_52.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False))
    print(json.dumps(meta,indent=2,ensure_ascii=False),flush=True)

if __name__=='__main__': main()
