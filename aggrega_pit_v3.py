#!/usr/bin/env python3
import argparse,csv,json,pathlib,collections
p=argparse.ArgumentParser();p.add_argument('--input',default='input_serie');p.add_argument('--out',default='report_pit_v3');a=p.parse_args()
root=pathlib.Path(a.input);out=pathlib.Path(a.out);out.mkdir(exist_ok=True,parents=True)
files=sorted(root.glob('*/esiti_snapshot.csv'))
rows=[];issues=[];reports=[]
for f in files:
    with f.open(encoding='utf-8-sig',newline='') as h:rows+=list(csv.DictReader(h))
    e=f.parent/'errori_acquisizione.csv'
    if e.exists():
        with e.open(encoding='utf-8-sig',newline='') as h:issues+=list(csv.DictReader(h))
    r=f.parent/'report.json'
    if r.exists():reports.append(json.loads(r.read_text()))
def write(name,data,columns):
    with (out/name).open('w',encoding='utf-8',newline='') as h:
        w=csv.DictWriter(h,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(data)
cols=['serie','data_segnale','vintage_asof','ultima_osservazione','valore_snapshot','valore_alfred','stato','motivo','ammesso_strict','ammesso_vintage_daily']
write('esiti_snapshot.csv',rows,cols)
write('vintage_giornaliera_verificata.csv',[r for r in rows if r.get('ammesso_vintage_daily')=='SI'],cols)
write('strict.csv',[],cols)
write('errori_acquisizione.csv',issues,['serie','errore'])
expected=set(['ANFCI', 'BAA10Y', 'BAMLH0A0HYM2', 'CPIAUCSL', 'CPILFESL', 'DCOILWTICO', 'DFEDTARU', 'DFII10', 'DGS10', 'DGS2', 'DGS3MO', 'DTWEXBGS', 'FEDFUNDS', 'GDP', 'ICSA', 'INDPRO', 'JTSJOL', 'M2SL', 'NFCI', 'PAYEMS', 'PCEPI', 'PCEPILFE', 'PPIACO', 'RRPONTSYD', 'RSXFS', 'SOFR', 'STLFSI4', 'T10Y2Y', 'T10YIE', 'T5YIE', 'UMCSENT', 'UNRATE', 'USEPUINDXD', 'VIXCLS', 'WALCL', 'WRESBAL'])
seen=set(r['serie'] for r in rows)
summary={'serie_attese':len(expected),'serie_completate':len(seen),'serie_mancanti':sorted(expected-seen),'snapshot_elaborati':len(rows),'stati':dict(collections.Counter(r['stato'] for r in rows)),'richieste_api':sum(r.get('richieste_api',0) for r in reports),'cache_hits':sum(r.get('cache_hits',0) for r in reports),'nota':'STRICT sempre 0 senza evidenze di timestamp; validazione vintage giornaliera non equivale a certificazione release.'}
(out/'report.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(summary,indent=2,ensure_ascii=False))
