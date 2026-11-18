#!/usr/bin/env python3
"""Macro Swing Lab Fase 27: join di vintage FRED/ALFRED PIT gia' verificate.

Non riscarica FRED. Non certifica intraday ne' pubblicazioni COT.
La macro ha decisioni originariamente settimanali: NESSUN valore viene anticipato.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile
import pandas as pd
import numpy as np

REQUIRED_MACRO = {'serie','data_segnale','vintage_asof','ultima_osservazione',
                  'valore_snapshot','valore_alfred','stato','certificato_intraday','incluso_backtest'}
SCENARIOS = [14, 21, 35]
OUTCOME_PREFIX = ('future_', 'mfe_', 'mae_', 'up2atr_', 'up3atr_')


def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def read_phase26(source, name):
    p=Path(source)
    if p.is_dir():
        f=p/name
        if not f.exists():raise FileNotFoundError(f'{name} assente in {p}')
        return pd.read_csv(f,low_memory=False)
    with zipfile.ZipFile(p) as z:
        matches=[s for s in z.namelist() if Path(s).name==name and not s.startswith('__MACOSX/')]
        if len(matches)!=1:raise ValueError(f'{name}: atteso un file, trovati {len(matches)}')
        return pd.read_csv(io.BytesIO(z.read(matches[0])),low_memory=False)


def macro_validate(path):
    d=pd.read_csv(path,low_memory=False)
    missing=REQUIRED_MACRO-set(d.columns)
    if missing:raise ValueError('Macro colonne assenti: '+', '.join(sorted(missing)))
    for c in ('data_segnale','vintage_asof','ultima_osservazione'):
        d[c]=pd.to_datetime(d[c],utc=True,errors='coerce')
    if d[['data_segnale','vintage_asof','ultima_osservazione']].isna().any().any():
        raise ValueError('Date macro non leggibili')
    for c in ('valore_snapshot','valore_alfred'):
        d[c]=pd.to_numeric(d[c],errors='coerce')
    if d[['valore_snapshot','valore_alfred']].isna().any().any():
        raise ValueError('Valori macro assenti/non numerici: non possibile join completo')
    if d.duplicated(['serie','data_segnale']).any():
        raise ValueError('Doppioni macro (serie,data_segnale)')
    # Only documented records conforming with the source certification gates.
    bad_status=~d['stato'].eq('PIT_DAILY_CONFORME')
    bad_bt=~d['incluso_backtest'].astype(str).str.upper().eq('SI')
    if bad_status.any() or bad_bt.any():
        raise ValueError(f'Record non conformi: stato={int(bad_status.sum())}, incluso_backtest={int(bad_bt.sum())}')
    if not np.allclose(d['valore_snapshot'],d['valore_alfred'],rtol=0,atol=1e-9):
        raise ValueError('Mismatch valore_snapshot / valore_alfred')
    if (d['vintage_asof']>=d['data_segnale']).any():
        raise ValueError('Vintage non precede data_segnale')
    if (d['ultima_osservazione']>d['vintage_asof']).any():
        raise ValueError('Osservazione successiva al vintage_asof')
    d['serie']=d['serie'].astype(str).str.strip().str.upper()
    if not d['serie'].str.fullmatch(r'[A-Z0-9_]+').all():
        raise ValueError('Nomi serie fuori standard')
    # Snapshot date is Friday in source. Daily conservative first-use from next UTC day,
    # because intraday release timing is NOT certified.
    d['macro_effective_utc']=d['data_segnale']+pd.Timedelta(days=1)
    return d.sort_values(['data_segnale','serie']).reset_index(drop=True)


def build_calendar(m):
    w=m.pivot(index='data_segnale',columns='serie',values='valore_snapshot').sort_index()
    w.columns=['macro_'+str(x) for x in w.columns]
    if w.columns.duplicated().any():raise ValueError('Nomi macro duplicati')
    extra=m.groupby('data_segnale',sort=True).agg(
        macro_vintage_max=('vintage_asof','max'),
        macro_vintage_min=('vintage_asof','min'),
        macro_coverage_count=('serie','count'),
        macro_snapshot_intraday_certified=('certificato_intraday',lambda a: bool(a.astype(str).str.upper().eq('SI').all())))
    w=w.join(extra)
    w['macro_signal_date']=w.index
    w['macro_effective_utc']=w.index+pd.Timedelta(days=1)
    w=w.reset_index(drop=True).sort_values('macro_effective_utc')
    if w['macro_effective_utc'].duplicated().any():raise ValueError('Snapshot date duplicate')
    return w


def join_asof(d,w,source_label):
    if 'timestamp' not in d:raise ValueError(f'{source_label}: timestamp assente')
    d=d.copy()
    d['timestamp']=pd.to_datetime(d['timestamp'],utc=True,errors='coerce')
    if d['timestamp'].isna().any() or d['timestamp'].duplicated().any():
        raise ValueError(f'{source_label}: timestamp invalidi/duplicati')
    d=d.sort_values('timestamp').reset_index(drop=True)
    labels=[c for c in d if c.startswith(OUTCOME_PREFIX)]
    # make explicit separate predictors and forward outcomes
    y=d[['timestamp']+labels].copy()
    x=d.drop(columns=labels)
    if any(str(c).startswith('macro_') for c in x.columns):
        raise ValueError(f'{source_label}: macro_ columns already present, avoid duplicate merge')
    x=pd.merge_asof(x,w,left_on='timestamp',right_on='macro_effective_utc',direction='backward')
    selected=x['macro_signal_date'].notna()
    for c in ('macro_signal_date','macro_vintage_max','macro_vintage_min','macro_effective_utc'):
        if c in x.columns:
            x[c]=pd.to_datetime(x[c],utc=True)
    if (x.loc[selected,'macro_effective_utc']>x.loc[selected,'timestamp']).any():
        raise ValueError('LOOKAHEAD: macro effective > timestamp')
    if (x.loc[selected,'macro_vintage_max']>x.loc[selected,'macro_signal_date']).any():
        raise ValueError('LOOKAHEAD: vintage > decision date')
    x['macro_snapshot_age_days']=(x['timestamp']-x['macro_signal_date']).dt.total_seconds()/86400
    x['macro_pit_status']=np.where(selected,'PIT_DAILY_SNAPSHOT_WEEKLY_NOT_INTRADAY','NO_PIT_SNAPSHOT_YET')
    # Preserve ALL original macro series; weekly re-release first observed in signal snapshots.
    meta={'rows':len(x),'macro_attached':int(selected.sum()),'macro_missing_before_first_snapshot':int((~selected).sum()),
          'first_timestamp':str(x['timestamp'].min()),'last_timestamp':str(x['timestamp'].max()),
          'macro_coverage_counts':x['macro_coverage_count'].value_counts(dropna=False).to_dict()}
    return x,y,meta


def write_csv(d,p):
    d.to_csv(p,index=False,date_format='%Y-%m-%dT%H:%M:%SZ')


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--fase26',required=True,help='ZIP artifact Fase26 oppure directory')
    a.add_argument('--macro-pit',required=True,help='CSV macro_pit_consolidato.csv gia verificato')
    a.add_argument('--out',default='fase27_risultati')
    z=a.parse_args()
    out=Path(z.out);out.mkdir(parents=True,exist_ok=True)
    audit={'status':'STARTED','phase26_source':str(z.fase26),'macro_pit_source':str(z.macro_pit),
           'source_sha256':{'macro_pit':sha256(z.macro_pit)}}
    try:
        m=macro_validate(z.macro_pit)
        w=build_calendar(m)
        write_csv(m,out/'macro_pit_originale_validato.csv')
        write_csv(w,out/'macro_snapshot_weekly.csv')
        audit.update(macro_records=int(len(m)),macro_series=int(m['serie'].nunique()),
                     macro_snapshots=int(m['data_segnale'].nunique()),
                     macro_first_signal=str(m['data_segnale'].min()),macro_last_signal=str(m['data_segnale'].max()),
                     macro_all_intraday_certified=bool(m['certificato_intraday'].astype(str).str.upper().eq('SI').all()),
                     macro_series_names=sorted(m['serie'].unique().tolist()),joined={})
        for lag in SCENARIOS:
            name=f'base_giornaliera_lag{lag}.csv'
            d=read_phase26(z.fase26,name)
            if 'cot_lag_assumed_days' not in d or not (pd.to_numeric(d['cot_lag_assumed_days'],errors='coerce')==lag).all():
                raise ValueError('COT lag mismatch per '+name)
            x,y,stats=join_asof(d,w,name)
            write_csv(x,out/f'features_d1_cot_lag{lag}_macro_pit.csv')
            write_csv(y,out/f'outcomes_d1_cot_lag{lag}_SOLO_TEST.csv')
            audit['joined'][name]=stats
        h=read_phase26(z.fase26,'h4_features.csv')
        xh,yh,hm=join_asof(h,w,'h4_features.csv')
        write_csv(xh,out/'features_h4_macro_pit.csv')
        write_csv(yh,out/'outcomes_h4_SOLO_TEST.csv')
        audit['joined']['h4_features.csv']=hm
        audit['status']='COMPLETED_MACRO_PIT_DAILY_JOIN_NO_INTRADAY_CERTIFICATION'
        audit['missing_families']=['EIA_RELEASE_VINTAGES_AND_CONSENSUS','WTI_FUTURES_TERM_STRUCTURE',
          'OPEC_IEA_FORECAST_VINTAGES','GEOPOLITICAL_EVENT_DATA','CHINA_RELEASE_VINTAGES']
        audit['limits']=['PIT giornaliero nelle date del calendario settimanale originale, NON orari intraday certificati.',
         'I dati FRED/ALFRED esistenti NON sono riscaricati o rivisti.',
         'Tutte le 36 serie mantenute. Alcune hanno copertura storica incompleta: i valori restano NaN.',
         'COT lag 14/21/35 sono scenari ipotetici, non date effettive verificate.',
         'Fase26 contiene proxy USOIL, non serie futures negoziabile verificata.',
         'Non e\' effettuato nessun test predittivo o backtest in questa fase.']
        (out/'README_RISULTATI.txt').write_text('Fase27: 36 macro PIT storiche + Fase26, zero nuovi download FRED.\n'
          'features_*: predittori separati dalle outcomes_*_SOLO_TEST.\n'
          'Ogni macro snapshot e\' disponibile dal giorno UTC successivo alla data_segnale della ricerca originale.\n'
          'Non equivale a una certificazione intraday ne\' al rilascio effettivo del COT.\n'
          'Per output e copertura vedi audit_fase27.json.\n',encoding='utf-8')
    except Exception as e:
        audit['status']='FAILED'
        audit['error']=f'{type(e).__name__}: {e}'
        raise
    finally:
        (out/'audit_fase27.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    print(json.dumps({'status':audit['status'],'series':audit['macro_series'],'rows':audit['macro_records'],
                      'joined':{k:v['macro_attached'] for k,v in audit['joined'].items()}},indent=2))

if __name__=='__main__':main()
