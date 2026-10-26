#!/usr/bin/env python3
"""Merge and validate independently produced year-range partitions."""
import json
from pathlib import Path
import pandas as pd
root=Path('partizioni'); out=Path('output/storico_pit_63_completo');out.mkdir(parents=True,exist_ok=True)
parts=sorted(root.glob('*/snapshot_pit_63.csv'))
if len(parts)!=4:raise RuntimeError(f'Expected 4 partitions, found {len(parts)}: {parts}')
frames=[pd.read_csv(p) for p in parts]
df=pd.concat(frames,ignore_index=True)
if df.duplicated(['serie','data_segnale']).any():raise RuntimeError('Duplicate series/date')
if not (pd.to_datetime(df.vintage_asof)<pd.to_datetime(df.data_segnale)).all():raise RuntimeError('Future vintage')
valid=df.ultima_osservazione.notna()
if not (pd.to_datetime(df.loc[valid,'ultima_osservazione'])<=pd.to_datetime(df.loc[valid,'vintage_asof'])).all():raise RuntimeError('Future observation')
checks=[]
for p in sorted(root.glob('*/audit_indipendente_63.csv')):
    checks.append(pd.read_csv(p))
if len(checks)!=4:raise RuntimeError('Missing independent audits')
audit=pd.concat(checks,ignore_index=True)
if audit.empty or not audit.esito.eq('PASS').all():raise RuntimeError('Independent vintage audit failed')
df.sort_values(['data_segnale','serie']).to_csv(out/'snapshot_pit_63_completo.csv',index=False)
audit.to_csv(out/'audit_indipendente_63_completo.csv',index=False)
coverage=df.groupby('serie').agg(snapshot=('data_segnale','size'),disponibili=('valore','count'),derivati=('metodo_acquisizione',lambda s:(s=='DERIVATO_COMPONENTI_ASOF').sum()),errori=('stato',lambda s:(s=='ERRORE_ACQUISIZIONE').sum())).reset_index()
coverage['copertura_pct']=(100*coverage.disponibili/coverage.snapshot).round(2)
coverage.to_csv(out/'copertura_63.csv',index=False)
report={'versione':'6.0_fase2.3','partizioni':len(parts),'serie':int(df.serie.nunique()),'segnali':int(df.data_segnale.nunique()),'snapshot':len(df),'snapshot_con_dati':int(df.valore.notna().sum()),'derivati_non_certificati':int((df.metodo_acquisizione=='DERIVATO_COMPONENTI_ASOF').sum()),'audit_indipendenti_pass':len(audit),'dati_ammessi_al_backtest':0,'note':['Audits are sampled and verify values, not original publication timestamps.','Historical derived breakevens remain unvalidated.','Political and geopolitical data are not yet integrated.']}
(out/'report_63.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
