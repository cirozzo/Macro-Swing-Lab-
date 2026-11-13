"""Ciclo 18: ricostruzione rigorosa COT futures-only WTI NYMEX (067651).
Eseguire dalla root del repository: python certifica_cot_wti.py
"""
import io, json, re, zipfile
from pathlib import Path
import requests
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'risultati_cot_certificazione';OUT.mkdir(exist_ok=True)
YEARS=range(2009,2027)
CODE='067651'

def norm(x): return re.sub(r'[^a-z0-9]','',str(x).lower())
def field(df,*aliases):
    lookup={norm(c):c for c in df.columns}
    for a in aliases:
        if norm(a) in lookup:return lookup[norm(a)]
    raise KeyError(f'Colonna assente {aliases}. Disponibili: {list(df.columns)}')

def download(year):
    urls=[f'https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip',f'https://www.cftc.gov/files/dea/history/fut_disagg_txt_hist_{year}.zip']
    errors=[]
    for url in urls:
        try:
            r=requests.get(url,timeout=65,headers={'User-Agent':'Mozilla/5.0 MacroSwingLabResearch/1.0'})
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                members=[m for m in z.namelist() if m.lower().endswith(('.txt','.csv')) and not m.startswith('__')]
                if not members:raise ValueError('ZIP senza CSV/TXT')
                member=max(members,key=lambda x:z.getinfo(x).file_size)
                frame=pd.read_csv(z.open(member),low_memory=False,dtype={'CFTC_Contract_Market_Code':str})
            return frame,url
        except Exception as e:errors.append(f'{url}: {e}')
    raise RuntimeError('; '.join(errors))

def main():
    frames=[]; logs=[]
    for year in YEARS:
        try:
            raw,url=download(year)
            code=field(raw,'CFTC_Contract_Market_Code')
            market=field(raw,'Market_and_Exchange_Names')
            date=field(raw,'Report_Date_as_YYYY-MM-DD','Report_Date_as_YYYY_MM_DD','Report_Date')
            longs=field(raw,'M_Money_Positions_Long_All')
            shorts=field(raw,'M_Money_Positions_Short_All')
            oi=field(raw,'Open_Interest_All')
            selected=raw.loc[raw[code].astype(str).str.strip().str.zfill(6).eq(CODE)].copy()
            markets=selected[market].astype(str).value_counts().to_dict()
            if selected.empty:raise ValueError('Nessuna riga con codice 067651')
            # Safety: selected rows must refer to NYMEX crude; code alone is necessary, not sufficient.
            if not selected[market].astype(str).str.contains('CRUDE OIL|WTI',case=False,regex=True).all():
                raise ValueError('Mercati imprevisti per codice 067651: '+str(markets))
            part=pd.DataFrame({'report_date':pd.to_datetime(selected[date],errors='coerce'),
                               'market':selected[market].astype(str),'code':selected[code].astype(str),
                               'mm_long':pd.to_numeric(selected[longs],errors='coerce'),
                               'mm_short':pd.to_numeric(selected[shorts],errors='coerce'),
                               'open_interest':pd.to_numeric(selected[oi],errors='coerce'),
                               'source_year':year,'source_url':url})
            part['report_date']=part['report_date'].dt.normalize()
            frames.append(part)
            logs.append({'year':year,'status':'ok','rows':len(part),'markets':str(markets),'url':url})
            print(year,'OK',len(part),markets,flush=True)
        except Exception as e:
            logs.append({'year':year,'status':'ERROR','rows':0,'error':str(e)})
            print(year,'ERROR',str(e),flush=True)
    pd.DataFrame(logs).to_csv(OUT/'audit_download.csv',index=False)
    if not frames:raise RuntimeError('Nessun dato CFTC disponibile')
    data=pd.concat(frames,ignore_index=True).sort_values('report_date')
    data.to_csv(OUT/'cot_wti_raw_codice_067651.csv',index=False)
    duplicates=data[data.duplicated('report_date',keep=False)]
    duplicates.to_csv(OUT/'date_duplicate_da_verificare.csv',index=False)
    invalid=data[data[['report_date','mm_long','mm_short','open_interest']].isna().any(axis=1) | (data.open_interest<=0)]
    invalid.to_csv(OUT/'righe_invalide.csv',index=False)
    # Never silently drop duplicate dates: the prior bug was exactly a blind drop_duplicates.
    if not duplicates.empty or not invalid.empty:
        raise RuntimeError(f'Audit bloccante: {len(duplicates)} righe con date duplicate; {len(invalid)} invalide. Vedere CSV.')
    data['net_mm']=data.mm_long-data.mm_short
    data['net_pct_oi']=data.net_mm/data.open_interest
    data['delta_4w']=data.net_pct_oi-data.net_pct_oi.shift(4)
    data['gap_days']=data.report_date.diff().dt.days
    data['weekly_jump_pct_points']=data.net_pct_oi.diff()*100
    data['usable_from_assumed']=data.report_date+pd.Timedelta(days=6)
    data.to_csv(OUT/'cot_wti_067651_futures_only.csv',index=False)
    gaps=data[data.gap_days.gt(8) | data.gap_days.lt(6)]
    jumps=data[data.weekly_jump_pct_points.abs().gt(5)]
    gaps.to_csv(OUT/'settimane_mancanti_o_irregolari.csv',index=False)
    jumps.to_csv(OUT/'salti_superiori_5pp.csv',index=False)
    old=ROOT/'cot_wti_settimanale.csv'
    if old.exists():
        prev=pd.read_csv(old,parse_dates=['report_date'])
        comp=data.merge(prev[['report_date','net_mm','net_pct_oi']],on='report_date',how='outer',suffixes=('_new','_old'),indicator=True)
        comp['delta_net_mm']=comp.net_mm_new-comp.net_mm_old
        comp['delta_net_pct_oi']=comp.net_pct_oi_new-comp.net_pct_oi_old
        comp.to_csv(OUT/'confronto_serie_vecchia_nuova.csv',index=False)
    audit={'code':CODE,'dataset':'CFTC disaggregated futures-only','years_requested':list(YEARS),
           'years_downloaded':int(sum(x['status']=='ok' for x in logs)),
           'records':len(data),'first':str(data.report_date.min()),'last':str(data.report_date.max()),
           'duplicates':len(duplicates),'invalid':len(invalid),'irregular_gaps':len(gaps),'jumps_over_5pp':len(jumps),
           'PIT_status':'NON CERTIFICATO: usable_from_assumed=martedi+6 giorni; verificare eccezioni CFTC',
           'note':'Non un backtest. I salti anomali sono segnalati, non rimossi.'}
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False))
    print(json.dumps(audit,indent=2,ensure_ascii=False))
    if any(x['status']=='ERROR' for x in logs):
        raise RuntimeError('Alcuni anni non scaricati: audit parziale, non certificabile')

if __name__=='__main__':main()
