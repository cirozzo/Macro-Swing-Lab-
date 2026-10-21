#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Macro Swing Lab 4.7 - Ricerca descrittiva, NON point-in-time.

Legge le osservazioni 4.6 e stima la volatilita' usando SOLO i 60
rendimenti giornalieri noti alla data del segnale. Non crea trade.
La graduatoria di ricerca e' costruita SOLO sui dati prima del 2022;
i risultati dal 2022 sono una verifica descrittiva separata.
"""
from pathlib import Path
import math
import numpy as np
import pandas as pd
import yfinance as yf
from macro_swing_40 import ASSETS, START

BASE = Path(__file__).resolve().parent
OUT = BASE / 'output'
OUT.mkdir(exist_ok=True)
INPUT = OUT / 'osservazioni_46.csv'
SPLIT = pd.Timestamp('2022-01-01')
HORIZONS = {1: 5, 2: 10, 3: 15, 4: 20, 6: 30, 8: 40}
MULTIPLIERS = (1.0, 1.5, 2.0)
MIN_TRAIN = 20


def load_observations():
    if not INPUT.is_file():
        raise FileNotFoundError('Eseguire prima analisi_settimanale_46.py')
    df = pd.read_csv(INPUT, parse_dates=['data_segnale','data_ingresso','data_uscita'])
    required = {'data_segnale','data_ingresso','data_uscita','asset','regime',
                'settimane','mfe','mae','rendimento','excess_return'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Colonne mancanti in 4.6: {sorted(missing)}')
    df['settimane'] = pd.to_numeric(df['settimane'], errors='raise').astype(int)
    df = df[df['settimane'].isin(HORIZONS)].copy()
    df = df.drop_duplicates(['asset','data_segnale','settimane'], keep=False)
    if df.empty:
        raise ValueError('Nessuna osservazione 4.6 valida')
    df['periodo'] = np.where(df['data_segnale'] < SPLIT,
                            'RICERCA_PRE_2022', 'VERIFICA_DAL_2022')
    df['anno'] = df['data_segnale'].dt.year
    return df


def get_volatility(asset):
    data = yf.download(asset, start=START, auto_adjust=True,
                       progress=False, threads=False)
    if data.empty:
        raise RuntimeError(f'Nessun prezzo per {asset}')
    close = data['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors='coerce').dropna()
    idx = pd.DatetimeIndex(close.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    close.index = idx.normalize()
    close = close[~close.index.duplicated(keep='last')].sort_index()
    # Valori di dispersione giornaliera con 60 rendimenti precedenti.
    return close.pct_change(fill_method=None).rolling(60, min_periods=60).std(ddof=1)


def add_normalized_excursions(df):
    pieces = []
    for asset, part in df.groupby('asset', sort=True):
        if asset not in ASSETS:
            print(f'SALTO asset sconosciuto: {asset}')
            continue
        print(f'Volatilita storica: {asset}', flush=True)
        try:
            sigma = get_volatility(asset)
        except Exception as err:
            raise RuntimeError(f'Download volatilita fallito per {asset}: {err}') from err
        p = part.copy()
        # Reindex su data segnale; NON sulla data ingresso o di uscita.
        p['sigma_daily_60'] = sigma.reindex(pd.DatetimeIndex(p['data_segnale'])).to_numpy()
        p['sedute'] = p['settimane'].map(HORIZONS)
        p['sigma_orizzonte'] = p['sigma_daily_60'] * np.sqrt(p['sedute'])
        p['mfe_sigma'] = p['mfe'] / p['sigma_orizzonte']
        p['mae_sigma'] = p['mae'] / p['sigma_orizzonte']
        for mult in MULTIPLIERS:
            label = str(mult).replace('.', '_')
            p[f'mfe_ge_{label}sigma'] = (p['mfe_sigma'] >= mult).astype(int)
        pieces.append(p)
    if not pieces:
        raise ValueError('Nessun asset analizzato')
    result = pd.concat(pieces, ignore_index=True)
    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.dropna(subset=['sigma_orizzonte','mfe_sigma','mae_sigma'])
    result = result[result['sigma_orizzonte'] > 0].copy()
    if result.empty:
        raise ValueError('Volatilita non disponibile per le date delle osservazioni')
    return result.sort_values(['asset','settimane','data_segnale']).reset_index(drop=True)


def non_overlapping(df):
    # Selezione deterministica PRIMA di distinguere i regimi.
    # Non rende gli episodi indipendenti tra asset.
    chosen = []
    for _, group in df.groupby(['asset','settimane'], sort=True):
        last_end = pd.Timestamp.min
        for i, row in group.sort_values('data_ingresso').iterrows():
            if row['data_ingresso'] > last_end:
                chosen.append(i)
                last_end = row['data_uscita']
    return df.loc[chosen].copy()


def wilson_lower(successes, n, z=1.96):
    if n <= 0:
        return np.nan
    p = successes / n
    den = 1 + z*z/n
    mid = p + z*z/(2*n)
    spread = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return (mid - spread)/den


def summarize(group, regime):
    n = len(group)
    hit1 = group['mfe_ge_1_0sigma']
    hit15 = group['mfe_ge_1_5sigma']
    hit2 = group['mfe_ge_2_0sigma']
    return {
        'regime': regime, 'n': n, 'anni_distinti': group['anno'].nunique(),
        'prob_rendimento_positivo': (group['rendimento'] > 0).mean(),
        'prob_batte_spy': (group['excess_return'] > 0).mean(),
        'rendimento_mediano_pct': 100*group['rendimento'].median(),
        'mfe_mediano_pct': 100*group['mfe'].median(),
        'mae_mediano_pct': 100*group['mae'].median(),
        'mfe_mediano_sigma': group['mfe_sigma'].median(),
        'mae_mediano_sigma': group['mae_sigma'].median(),
        'prob_mfe_1sigma': hit1.mean(),
        'prob_mfe_1_5sigma': hit15.mean(),
        'prob_mfe_2sigma': hit2.mean(),
        'wilson95_inf_1_5sigma': wilson_lower(int(hit15.sum()), n),
        'wilson95_inf_2sigma': wilson_lower(int(hit2.sum()), n),
    }


def aggregate(df):
    rows = []
    for (period, asset, weeks), subset in df.groupby(['periodo','asset','settimane']):
        baseline = summarize(subset, 'TUTTI')
        baseline.update(periodo=period, asset=asset, settimane=weeks)
        rows.append(baseline)
        for regime, selected in subset.groupby('regime'):
            record = summarize(selected, regime)
            record.update(periodo=period, asset=asset, settimane=weeks)
            rows.append(record)
    result = pd.DataFrame(rows)
    base = result[result.regime.eq('TUTTI')][
        ['periodo','asset','settimane','prob_rendimento_positivo',
         'prob_mfe_1sigma','prob_mfe_1_5sigma','prob_mfe_2sigma',
         'mfe_mediano_sigma','prob_batte_spy']
    ].copy()
    values = list(base.columns[3:])
    base = base.rename(columns={c: c+'_baseline' for c in values})
    result = result.merge(base, on=['periodo','asset','settimane'], how='left', validate='many_to_one')
    for metric in values:
        result['delta_'+metric] = result[metric]-result[metric+'_baseline']
    result['campione_limitato'] = result['n'] < MIN_TRAIN
    return result


def yearly_stability(df):
    rows=[]
    for (period,asset,weeks,year), grp in df.groupby(['periodo','asset','settimane','anno']):
        baseline = grp['mfe_ge_1_5sigma'].mean()
        for regime, g in grp.groupby('regime'):
            rows.append(dict(periodo=period, asset=asset, settimane=weeks, anno=year,
                             regime=regime, n=len(g), prob_mfe_1_5sigma=g['mfe_ge_1_5sigma'].mean(),
                             baseline_annuale=baseline,
                             delta_annuale=g['mfe_ge_1_5sigma'].mean()-baseline))
    return pd.DataFrame(rows)


def rank_training(stats, annual):
    d = stats[(stats['periodo']=='RICERCA_PRE_2022') & (stats['regime']!='TUTTI')].copy()
    s = annual[annual['periodo']=='RICERCA_PRE_2022'].copy()
    if s.empty or d.empty:
        raise RuntimeError('Assenza dati del periodo storico di ricerca')
    yr = (s.assign(positivo=lambda x: x['delta_annuale'] > 0)
          .groupby(['asset','regime','settimane'],as_index=False)
          .agg(anni_osservati=('anno','nunique'),
               anni_favorevoli=('positivo','sum'),
               delta_annuo_mediano=('delta_annuale','median')))
    d = d.merge(yr, on=['asset','regime','settimane'],how='left',validate='one_to_one')
    # Punteggio descrittivo a priori, focalizzato su differenze rispetto alla
    # baseline e robustezza annuale. Nessun tuning sul periodo 2022+.
    d['score_descrittivo'] = 100*(
        0.40*d['delta_prob_mfe_1_5sigma']+
        0.25*d['delta_prob_mfe_2sigma']+
        0.15*d['delta_prob_rendimento_positivo']+
        0.10*d['delta_prob_batte_spy']+
        0.10*d['delta_annuo_mediano'].fillna(0))
    d['qualita_campione'] = np.select(
        [(d.n>=40)&(d.anni_distinti>=7),
         (d.n>=20)&(d.anni_distinti>=5)],
        ['PIU_STABILE','DA_VERIFICARE'],default='INSUFFICIENTE')
    # Vietato promuovere episodi insufficienti ai primi posti.
    d['idoneo_a_ranking'] = d['qualita_campione']!='INSUFFICIENTE'
    d = d.sort_values(['settimane','regime','idoneo_a_ranking','score_descrittivo','n'],
                      ascending=[True,True,False,False,False])
    d['posizione_nel_regime'] = d.groupby(['settimane','regime']).cumcount()+1
    # verifica separata: non altera il ranking storico
    test = stats[(stats['periodo']=='VERIFICA_DAL_2022') & (stats['regime']!='TUTTI')][
        ['asset','regime','settimane','n','prob_mfe_1_5sigma',
         'delta_prob_mfe_1_5sigma','prob_mfe_2sigma',
         'delta_prob_mfe_2sigma']].copy()
    test = test.rename(columns={c:c+'_verifica' for c in test.columns if c not in ('asset','regime','settimane')})
    return d.merge(test,on=['asset','regime','settimane'],how='left',validate='one_to_one')


def main():
    print('MACRO SWING LAB 4.7 — NON POINT-IN-TIME',flush=True)
    raw = load_observations()
    full = add_normalized_excursions(raw)
    subset = non_overlapping(full)
    stats = aggregate(subset)
    annual = yearly_stability(subset)
    ranking = rank_training(stats, annual)
    # I file intermedi sono conservati per verifiche e audit.
    full.to_csv(OUT/'osservazioni_volatilita_47.csv',index=False)
    subset.to_csv(OUT/'episodi_non_overlap_47.csv',index=False)
    stats.to_csv(OUT/'statistiche_regimi_47.csv',index=False)
    annual.to_csv(OUT/'stabilita_annuale_47.csv',index=False)
    ranking.to_csv(OUT/'ranking_ricerca_47.csv',index=False)
    print('Osservazioni:',len(full),'Episodi non sovrapposti:',len(subset),
          'Combinazioni classificate:',len(ranking))
    print('I risultati sono esplorativi, NON point-in-time e NON segnali di acquisto.')

if __name__ == '__main__':
    main()
