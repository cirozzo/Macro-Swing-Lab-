#!/usr/bin/env python3
"""Inventario ALFRED: copertura, numero vintage, disponibilità PIT su tre date-sonda.
Non certifica automaticamente 1.034 snapshot né timestamp intraday.
"""
import os, csv, json, time, random, urllib.request, urllib.error, urllib.parse, pathlib, datetime, sys

SERIES=['ANFCI','BAA10Y','BAMLH0A0HYM2','CPIAUCSL','CPILFESL','DCOILWTICO','DFEDTARU','DFII10','DGS10','DGS2','DGS3MO','DTWEXBGS','FEDFUNDS','GDP','ICSA','INDPRO','JTSJOL','M2SL','NFCI','PAYEMS','PCEPI','PCEPILFE','PPIACO','RRPONTSYD','RSXFS','SOFR','STLFSI4','T10Y2Y','T10YIE','T5YIE','UMCSENT','UNRATE','USEPUINDXD','VIXCLS','WALCL','WRESBAL']
ASOF=['2007-01-05','2015-01-09','2023-01-06']
KEY=os.environ.get('FRED_API_KEY','').strip()
if not KEY: sys.exit('ERRORE: secret FRED_API_KEY mancante')
ROOT=pathlib.Path('ricognizione_alfred'); ROOT.mkdir(exist_ok=True)

def fetch(path,params):
    args={**params,'api_key':KEY,'file_type':'json'}
    url='https://api.stlouisfed.org/fred/'+path+'?'+urllib.parse.urlencode(args)
    for attempt in range(7):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'MacroSwingLabInventory/1.0'})
            with urllib.request.urlopen(req,timeout=40) as response: out=json.load(response)
            if 'error_code' in out: raise ValueError(str(out.get('error_message',out)))
            time.sleep(0.6) # esecuzione sequenziale per evitare burst
            return out
        except urllib.error.HTTPError as exc:
            body=exc.read(500).decode('utf-8','replace')
            if exc.code not in (429,500,502,503,504): raise RuntimeError(f'HTTP {exc.code}: {body}') from exc
            retry=exc.headers.get('Retry-After')
            try: delay=min(120,max(1,float(retry))) if retry else min(90,3*(2**attempt))
            except ValueError: delay=min(90,3*(2**attempt))
            print(f'HTTP {exc.code}, attesa {delay}s',file=sys.stderr);time.sleep(delay+random.random())
        except (urllib.error.URLError,TimeoutError) as exc:
            time.sleep(min(60,2**attempt)+random.random())
    raise RuntimeError('Retry esauriti: '+path)

def pages(path,args,key,limit):
    rows=[];offset=0;total=None
    while total is None or offset<total:
        data=fetch(path,{**args,'limit':limit,'offset':offset})
        total=int(data.get('count',0)); part=data.get(key,[])
        if not isinstance(part,list):raise RuntimeError('Risposta malformata')
        if not part and offset<total:raise RuntimeError('Paginazione incompleta')
        rows.extend(part);offset+=len(part)
    if len(rows)!=total:raise RuntimeError(f'Paginazione parziale {len(rows)}/{total}')
    return rows

def inventory(s):
    row={'serie':s,'stato':'OK'}
    try:
        meta=fetch('series',{'series_id':s}).get('seriess',[])
        if len(meta)!=1:raise RuntimeError('Serie assente o ambigua')
        m=meta[0]
        row.update(titolo=m.get('title',''),frequenza=m.get('frequency_short',''),inizio_serie=m.get('observation_start',''),fine_serie=m.get('observation_end',''))
        vintages=pages('series/vintagedates',{'series_id':s},'vintage_dates',10000)
        row.update(vintage_totali=len(vintages),prima_vintage=vintages[0] if vintages else '',ultima_vintage=vintages[-1] if vintages else '',vintage_dal_2006=sum(v>='2006-01-01' for v in vintages))
        current=pages('series/observations',{'series_id':s,'observation_start':'2006-01-01','realtime_start':'2026-10-29','realtime_end':'2026-10-29'},'observations',100000)
        row['osservazioni_2006_2026']=len(current)
        row['valori_non_mancanti_2006_2026']=sum(o.get('value') not in ('.','','NaN',None) for o in current)
        for day in ASOF:
            data=pages('series/observations',{'series_id':s,'realtime_start':day,'realtime_end':day,'observation_start':'2006-01-01','observation_end':day,'sort_order':'desc','limit':100000},'observations',100000)
            valid=[x for x in data if x.get('value') not in ('.','','NaN',None)]
            row['pit_'+day]='SI' if valid else 'NO'
            row['ultima_osservazione_'+day]=max(x['date'] for x in valid) if valid else ''
        row['sonde_pit_positive']=sum(row['pit_'+d]=='SI' for d in ASOF)
    except Exception as e:
        row['stato']='ERRORE';row['errore']=str(e)[:500]
    return row

rows=[]
for i,s in enumerate(SERIES,1):
    r=inventory(s);rows.append(r)
    print(f'[{i}/{len(SERIES)}] {s}: {r["stato"]} vintage={r.get("vintage_totali","?")} sonde={r.get("sonde_pit_positive","?")}',flush=True)
    # salva checkpoint anche in caso di errore
    (ROOT/'checkpoint.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
fields=list(dict.fromkeys(k for r in rows for k in r))
with (ROOT/'inventario_36_serie.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
report={'serie_totali':len(rows),'serie_ok':sum(r['stato']=='OK' for r in rows),'serie_con_tre_sonde_positive':sum(r.get('sonde_pit_positive')==3 for r in rows),'errori':{r['serie']:r.get('errore') for r in rows if r['stato']!='OK'},'nota':'Sonde indicative e inventario vintage: NON certificano ogni snapshot settimanale né l’orario intraday. Le osservazioni correnti sono richieste alla data 2026-10-29.'}
(ROOT/'riepilogo.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
