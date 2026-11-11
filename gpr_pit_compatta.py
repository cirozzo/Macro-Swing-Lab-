#!/usr/bin/env python3
"""GPR PIT weekly extraction from huge CSV; bounded-memory chunk processing."""
import argparse, csv, json, pathlib, zipfile, tempfile, shutil
from collections import Counter
import numpy as np
import pandas as pd

COLS=['observation_date','gpr','vintage_date','available_from','source_type','source_file']

def locate(root):
    direct=list(root.rglob('GPR_VINTAGES_LONG.csv'))
    if direct:return direct[0],None
    for zpath in root.rglob('*.zip'):
        with zipfile.ZipFile(zpath) as z:
            matches=[x for x in z.namelist() if x.endswith('GPR_VINTAGES_LONG.csv')]
            if matches:return zpath,matches[0]
    raise FileNotFoundError('GPR_VINTAGES_LONG.csv non trovato in artifact o ZIP annidati')

def run(args):
    root=pathlib.Path(args.root);out=pathlib.Path(args.output);out.mkdir(parents=True,exist_ok=True)
    path,member=locate(root)
    print('Sorgente:',path,'membro:',member,flush=True)
    handle=zipfile.ZipFile(path).open(member) if member else open(path,'rb')
    # Keep only recent observation dates in each vintage; no full dataset in RAM.
    snapshots={};counts=Counter(); scanned=0; cutoff='2021-12-01'
    try:
      for chunk in pd.read_csv(handle,dtype=str,chunksize=180000,usecols=COLS,low_memory=False):
        scanned+=len(chunk)
        chunk=chunk[(chunk.observation_date>=cutoff)&(chunk.observation_date<='2026-11-11')].copy()
        chunk=chunk[chunk.gpr.notna() & chunk.available_from.notna() & chunk.vintage_date.notna()]
        if not chunk.empty:
          chunk['gpr']=pd.to_numeric(chunk.gpr,errors='coerce');chunk=chunk[np.isfinite(chunk.gpr)]
          # Date-only availability is conservative: use vintage only from the NEXT calendar day.
          chunk['key']=chunk.vintage_date+'|'+chunk.available_from+'|'+chunk.source_type.fillna('')+'|'+chunk.source_file.fillna('')
          for key,g in chunk.groupby('key',sort=False):
            vals=g[['observation_date','gpr']]
            if key in snapshots:vals=pd.concat([snapshots[key],vals],ignore_index=True)
            snapshots[key]=vals.drop_duplicates('observation_date',keep='last').sort_values('observation_date').tail(50).reset_index(drop=True)
        if scanned%3600000<180000:print('Righe esaminate:',scanned,'snapshot:',len(snapshots),flush=True)
    finally:
      handle.close()
    print('Scansione completata:',scanned,'snapshot:',len(snapshots),flush=True)
    index=[]
    for key,frame in snapshots.items():
      vintage,available,typ,src=key.split('|',3)
      # Source available_from may be blank or older than vintage; prevent future vintage leakage.
      valid=max(available,vintage)
      index.append((valid,vintage,typ,src,frame))
    index.sort(key=lambda x:(x[0],x[1],x[2],x[3]))
    signals=pd.date_range('2022-03-04','2026-11-06',freq='W-FRI')
    rows=[];audit=[]
    for ts in signals:
      day=ts.strftime('%Y-%m-%d')
      eligible=[x for x in index if x[0]<day] # day-only release -> next-day eligibility, conservative
      if not eligible:
        rows.append({'data_segnale':day,'status':'NO_VINTAGE'});continue
      # Prefer latest available vintage, then archived sources over alternatives.
      best=max(eligible,key=lambda x:(x[1],x[0],int('archiv' in x[2].lower()),x[3]))
      avail,vintage,typ,src,frame=best
      hist=frame[frame.observation_date<=day].sort_values('observation_date')
      if hist.empty:
        rows.append({'data_segnale':day,'status':'NO_OBSERVATION'});continue
      last=hist.iloc[-1];obs=last.observation_date;v=float(last.gpr)
      def prior(days):
        limit=(ts-pd.Timedelta(days=days)).strftime('%Y-%m-%d')
        p=hist[hist.observation_date<=limit]
        return float(p.iloc[-1].gpr) if len(p) else np.nan
      v7,v30=prior(7),prior(30)
      rows.append(dict(data_segnale=day,gpr_pit=v,data_osservazione=obs,vintage_utilizzato=vintage,available_from=avail,source_type=typ,source_file=src,variazione_7g=v-v7 if np.isfinite(v7) else np.nan,variazione_30g=v-v30 if np.isfinite(v30) else np.nan,delta_pct_7g=100*(v/v7-1) if np.isfinite(v7) and v7 else np.nan,status='OK'))
      audit.append(dict(data_segnale=day,vintage_utilizzato=vintage,available_from=avail,observation_date=obs,eta_giorni=(ts-pd.Timestamp(obs)).days,osservazioni_nella_vintage=len(hist),source_type=typ))
    df=pd.DataFrame(rows);df['percentile_storico_pit']=np.nan;df['shock_gpr']=False
    # Rolling history calculated from preceding *weekly PIT observations only*.
    ok=df.status.eq('OK')
    for i in df.index[ok]:
      prev=df.loc[:i-1];prev=prev[prev.status.eq('OK')].tail(104)
      if len(prev)>=26:
        x=prev.gpr_pit.astype(float)
        df.at[i,'percentile_storico_pit']=float((x<=df.at[i,'gpr_pit']).mean())
        change=prev.variazione_7g.dropna().astype(float)
        if len(change)>=20 and change.std()>0:
          df.at[i,'shock_gpr']=bool(df.at[i,'variazione_7g']>change.mean()+2*change.std())
    df.to_csv(out/'GPR_PIT_SETTIMANALE.csv',index=False)
    df[df.shock_gpr].to_csv(out/'GPR_SHOCK_STORICI.csv',index=False)
    pd.DataFrame(audit).to_csv(out/'GPR_QUALITA_DATI.csv',index=False)
    report={'righe_sorgente_lette':scanned,'snapshot_distinti':len(index),'settimane':len(df),'settimane_valide':int(ok.sum()),'settimane_senza_dati':int((~ok).sum()),'shock':int(df.shock_gpr.sum()),'nota':'Date di sola disponibilita giornaliera: vintage ammissibile soltanto se available_from e vintage_date sono entrambe precedenti alla data segnale. Non certifica l’ora di pubblicazione. Percentile basato sulle 104 settimane PIT precedenti, minimo 26. Shock: variazione settimanale > media + 2 dev std delle precedenti 104 settimane, minimo 20. Dataset di ricerca, non segnali operativi.'}
    (out/'GPR_METODOLOGIA.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'LEGGIMI.txt').write_text('FASE 38.1 - GPR PIT SETTIMANALE\nI quattro CSV/JSON sono compatti. Verificare GPR_QUALITA_DATI prima di utilizzare i dati. Nessun dato di prezzo o COT incluso.\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default='input_artifact');p.add_argument('--output',default='output_gpr');run(p.parse_args())
