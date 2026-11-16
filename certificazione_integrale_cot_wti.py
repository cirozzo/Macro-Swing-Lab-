"""Audit WTI CFTC 067651. PIT certificato SOLO con calendario pubblicazioni verificabile."""
import csv, hashlib, io, json, pathlib, sys, zipfile
from datetime import datetime, timezone, timedelta
import pandas as pd
import requests

ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT/'certificazione_cot_wti'; OUT.mkdir(exist_ok=True)
YEARS=range(2010,2027)
CODE='067651'
COLS=['M_Money_Positions_Long_All','M_Money_Positions_Short_All','Open_Interest_All']
LOG=[]

def save(name,df): df.to_csv(OUT/name,index=False)
def fail(msg): raise ValueError(msg)
def date_parse(df):
    candidates=[('Report_Date_as_YYYY-MM-DD',['%Y-%m-%d','%Y/%m/%d']),('Report_Date_as_YYYY_MM_DD',['%Y_%m_%d','%Y-%m-%d']),('Report_Date_as_MM_DD_YYYY',['%m/%d/%Y','%m-%d-%Y','%m_%d_%Y']),('As_of_Date_In_Form_YYMMDD',['%y%m%d'])]
    for name,formats in candidates:
        if name not in df: continue
        for fmt in formats:
            dt=pd.to_datetime(df[name].str.strip(),format=fmt,errors='coerce')
            if dt.notna().all(): return dt.dt.normalize()
    fail('Nessuna colonna data interamente valida')

def fetch(y,s):
    urls=[f'https://www.cftc.gov/files/dea/history/fut_disagg_txt_{y}.zip',f'https://www.cftc.gov/files/dea/history/fut_disagg_txt_hist_{y}.zip']
    errors=[]
    for url in urls:
        for attempt in range(3):
            try:
                r=s.get(url,timeout=(15,90));r.raise_for_status(); raw=r.content
                with zipfile.ZipFile(io.BytesIO(raw)) as z:
                    names=[n for n in z.namelist() if n.lower().endswith(('.txt','.csv')) and not n.startswith('__MACOSX/')]
                    if len(names)!=1: fail(f'Numero file tabellari: {names}')
                    with z.open(names[0]) as f: d=pd.read_csv(f,dtype=str,low_memory=False,encoding='utf-8-sig')
                d.columns=[c.strip() for c in d.columns]
                if len(d.columns)!=len(set(d.columns)): fail('Colonne duplicate')
                return d,url,hashlib.sha256(raw).hexdigest()
            except Exception as e: errors.append(f'{type(e).__name__}: {str(e)[:130]}')
    fail('Download non riuscito: '+' | '.join(errors[-3:]))

def main():
    s=requests.Session();s.headers.update({'User-Agent':'MacroSwingLab/1.0 CFTC historical verification'})
    frames=[]
    for y in YEARS:
        item={'year':y,'status':'ERROR','rows':0,'source_url':'','sha256_zip':'','error':''}
        try:
            d,url,digest=fetch(y,s);item.update(source_url=url,sha256_zip=digest)
            needed=['CFTC_Contract_Market_Code',*COLS]
            if any(c not in d for c in needed):fail('Campi obbligatori mancanti: '+str([c for c in needed if c not in d]))
            code=d.CFTC_Contract_Market_Code.str.strip().str.replace(r'\.0$','',regex=True).str.zfill(6)
            d=d.loc[code.eq(CODE).fillna(False)].copy()
            if d.empty:fail('Contratto 067651 assente')
            if 'FutOnly_or_Combined' in d:
                kinds=d.FutOnly_or_Combined.str.strip().str.lower().dropna().unique().tolist()
                if not kinds or any('fut' not in k or 'comb' in k for k in kinds):fail('Tipologia non futures-only: '+str(kinds))
            d['report_date']=date_parse(d)
            if not d.report_date.dt.year.eq(y).all():fail('Date fuori anno')
            if d.report_date.duplicated().any():fail('Date duplicate nel file originale')
            for c in COLS:d[c]=pd.to_numeric(d[c].str.replace(',','',regex=False),errors='coerce')
            if d[COLS].isna().any().any() or (d[COLS]<0).any().any() or (d[COLS[2]]<=0).any():fail('Posizioni numeriche invalide')
            if (d[COLS[0]]>d[COLS[2]]).any() or (d[COLS[1]]>d[COLS[2]]).any():fail('Posizioni oltre open interest')
            d['market']=d['Market_and_Exchange_Names'].str.strip() if 'Market_and_Exchange_Names' in d else ''
            d['source_year']=y;d['source_url']=url;d['source_sha256']=digest
            frames.append(d[['report_date','market',*COLS,'source_year','source_url','source_sha256']]);item.update(status='OK',rows=len(d))
        except Exception as e:item['error']=str(e)
        LOG.append(item);save('download_audit.csv',pd.DataFrame(LOG));print(y,item['status'],item['rows'],item['error'][:150],flush=True)
    summary={'generated_utc':datetime.now(timezone.utc).isoformat(),'code':CODE,'years':'2010-2026','structural_status':'FAILED','pit_status':'NOT_CERTIFIED','warnings':[]}
    if not frames:
        summary['warnings'].append('Nessun archivio disponibile');finish(summary);return
    d=pd.concat(frames,ignore_index=True).sort_values('report_date').reset_index(drop=True)
    d['net_mm']=d[COLS[0]]-d[COLS[1]];d['net_pct_oi']=d.net_mm/d[COLS[2]];d['delta_4w']=d.net_pct_oi.diff(4)
    d['gap_days']=d.report_date.diff().dt.days;d['weekly_change_pp']=d.net_pct_oi.diff()*100
    save('serie_verificata_origini.csv',d)
    dup=d[d.report_date.duplicated(keep=False)].copy();save('duplicati.csv',dup)
    gaps=d[d.gap_days.notna() & d.gap_days.ne(7)].copy();save('intervalli_non_7_giorni.csv',gaps)
    spikes=d[d.weekly_change_pp.abs()>5].copy();save('variazioni_oltre_5pp.csv',spikes)
    missing=[x['year'] for x in LOG if x['status']!='OK']
    counts=d.groupby(d.report_date.dt.year).size()
    # Intervalli di 6/8 giorni non sono errori se presenti nel report originale,
    # ma non provano da soli che le pubblicazioni siano state puntuali.
    irregular_other=gaps[~gaps.gap_days.isin([6,8])]
    structural=(not missing and dup.empty and irregular_other.empty and d.net_pct_oi.notna().all() and len(d)>800 and all(counts.get(y,0)>=50 for y in range(2010,2026)) and counts.get(2026,0)>=1)
    summary.update(records=len(d),first_report=str(d.report_date.min().date()),last_report=str(d.report_date.max().date()),missing_years=missing,duplicates=len(dup),non_7_day_intervals=len(gaps),unexpected_intervals=len(irregular_other),spikes_5pp=len(spikes),structural_status='PASSED_SOURCE_RECONCILED' if structural else 'FAILED')
    if len(gaps):summary['warnings'].append('Intervalli 6/8 giorni presenti nei file CFTC: conservati e documentati')
    if len(spikes):summary['warnings'].append('Variazioni >5pp presenti nei file CFTC: conservate e documentate')
    # Calendario verificato fornito dall'utente: report_date,release_datetime_utc,source_url.
    # NON assumere che martedi+6 sia una data reale di pubblicazione.
    cal=ROOT/'calendario_rilasci_cot_verificato.csv'
    if cal.exists():
        try:
            c=pd.read_csv(cal,dtype=str)
            required={'report_date','release_datetime_utc','source_url'}
            if not required.issubset(c):fail('Colonne calendario mancanti')
            c['report_date']=pd.to_datetime(c.report_date,errors='raise').dt.normalize()
            c['release_datetime_utc']=pd.to_datetime(c.release_datetime_utc,utc=True,errors='raise')
            if c.report_date.duplicated().any():fail('Date duplicate nel calendario')
            if c.source_url.isna().any() or c.source_url.str.strip().eq('').any():fail('Fonte calendario mancante')
            j=d[['report_date']].merge(c,on='report_date',how='left',validate='one_to_one')
            if j.release_datetime_utc.isna().any():fail(f'Date rilascio mancanti: {int(j.release_datetime_utc.isna().sum())}')
            report_utc=j.report_date.dt.tz_localize('UTC')
            if (j.release_datetime_utc<report_utc).any():fail('Rilascio prima della data di rilevazione')
            if (j.release_datetime_utc>report_utc+pd.Timedelta(days=180)).any():fail('Rilascio eccessivamente ritardato: controllare fonte')
            j['first_usable_date_utc']=(j.release_datetime_utc+pd.Timedelta(days=1)).dt.normalize()
            save('calendario_pit_associato.csv',j)
            summary['pit_status']='CALENDAR_COMPLETE_REQUIRES_SOURCE_AUTHENTICATION'
            summary['warnings'].append('Un calendario CSV non prova da solo la veridicità dei link: autenticare manualmente le fonti prima di certificare PIT')
        except Exception as e:summary['warnings'].append('Calendario fornito non valido: '+str(e))
    else:summary['warnings'].append('Calendario storico ufficiale delle date di rilascio non fornito: PIT NON CERTIFICATO; vietato usare martedì+6 come prova')
    finish(summary)

def finish(summary):
    (OUT/'audit_finale.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['# Macro Swing Lab — certificazione COT WTI','',f"**Strutturale:** {summary['structural_status']}",f"**Point-in-time:** {summary['pit_status']}",'','## Limitazioni']+['- '+x for x in summary['warnings']]
    (OUT/'LEGGIMI_RISULTATI.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    if summary['structural_status']!='PASSED_SOURCE_RECONCILED':sys.exit(1)

if __name__=='__main__':main()
