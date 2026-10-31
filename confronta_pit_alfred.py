#!/usr/bin/env python3
"""Confronto PIT ALFRED: una richiesta per coppia (vintage_asof, ultima_osservazione).
Non confonde valori recuperabili con certificazione intraday; non certifica derivati.
"""
import argparse,csv,io,zipfile,datetime as dt,decimal,json,os,pathlib,time,urllib.parse,urllib.request,urllib.error,random,collections,hashlib
SERIES=['ANFCI','BAA10Y','BAMLH0A0HYM2','CPIAUCSL','CPILFESL','DCOILWTICO','DFEDTARU','DFII10','DGS10','DGS2','DGS3MO','DTWEXBGS','FEDFUNDS','GDP','ICSA','INDPRO','JTSJOL','M2SL','NFCI','PAYEMS','PCEPI','PCEPILFE','PPIACO','RRPONTSYD','RSXFS','SOFR','STLFSI4','T10Y2Y','T10YIE','T5YIE','UMCSENT','UNRATE','USEPUINDXD','VIXCLS','WALCL','WRESBAL']
API='https://api.stlouisfed.org/fred/series/observations'
def read_csv_zip(p,suffix):
 with zipfile.ZipFile(p) as z:
  names=[x for x in z.namelist() if x.endswith(suffix)]
  if len(names)!=1: raise ValueError(f'In {p} atteso un {suffix}, trovati {len(names)}')
  return list(csv.DictReader(io.StringIO(z.read(names[0]).decode('utf-8-sig'))))
def csvout(path,rows,fields):
 with open(path,'w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def eq(a,b):
 try:return abs(decimal.Decimal(str(a))-decimal.Decimal(str(b)))<=decimal.Decimal('0.000001')
 except (decimal.InvalidOperation,ValueError):return False
class ALFRED:
 def __init__(self,key,cache):self.key=key;self.cache=pathlib.Path(cache);self.cache.mkdir(parents=True,exist_ok=True);self.calls=0;self.cached=0
 def check(self,serie,asof,obs):
  ident=hashlib.sha256(f'{serie}|{asof}|{obs}'.encode()).hexdigest()
  p=self.cache/(ident+'.json')
  if p.exists():self.cached+=1;return json.loads(p.read_text())
  opts={'series_id':serie,'api_key':self.key,'file_type':'json','realtime_start':asof,'realtime_end':asof,'observation_start':obs,'observation_end':obs,'output_type':1,'limit':1000}
  u=API+'?'+urllib.parse.urlencode(opts)
  for attempt in range(8):
   try:
    with urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'MacroSwingLab-PIT-verify/1.0'}),timeout=45) as r:d=json.load(r)
    if 'error_code' in d:raise ValueError(str(d.get('error_message',d)))
    records=d.get('observations')
    if not isinstance(records,list):raise ValueError('Risposta priva di observations')
    exact=[x for x in records if x.get('date')==obs]
    result={'result':'OK' if len(exact)==1 and exact[0].get('value') not in (None,'.','') else ('NESSUN_VALORE' if len(exact)<=1 else 'AMBIGUO'), 'value':exact[0]['value'] if len(exact)==1 else '', 'count':len(exact)}
    p.write_text(json.dumps(result));self.calls+=1;time.sleep(0.7);return result
   except urllib.error.HTTPError as e:
    body=e.read(400).decode('utf8','replace')
    if e.code not in (429,500,502,503,504): return {'result':'HTTP_'+str(e.code),'detail':body[:200],'value':''}
    delay=min(90,2**attempt)+random.random()*2
    ra=e.headers.get('Retry-After')
    if ra and ra.isdigit():delay=max(delay,min(int(ra),120))
   except (OSError,TimeoutError) as e:delay=min(90,2**attempt)+random.random()*2
   if attempt==7:return {'result':'ERRORE_RETRY','detail':str(e)[:200] if 'e' in locals() else '', 'value':''}
   time.sleep(delay)
  return {'result':'ERRORE','value':''}
def main():
 a=argparse.ArgumentParser();a.add_argument('--serie',required=True);a.add_argument('--source',default='storico-pit-64-completo.zip');a.add_argument('--association',default='macro-swing-lab-associazione-pit.zip');a.add_argument('--out',default='report_confronto_pit');a.add_argument('--limit',type=int,default=0);args=a.parse_args()
 if not os.getenv('FRED_API_KEY'):raise SystemExit('Secret FRED_API_KEY mancante')
 if args.serie not in SERIES:raise SystemExit('Serie non riconosciuta')
 rows=[r for r in read_csv_zip(args.source,'snapshot_pit_64_completo.csv') if r.get('serie')==args.serie]
 if not rows:raise SystemExit('Nessuna riga snapshot per la serie')
 assoc=read_csv_zip(args.association,'associazioni_snapshot.csv')
 keys={(r['serie'],r['data_segnale']) for r in assoc}
 if len(keys)!=len(assoc):raise SystemExit('Associazioni duplicate: interrompo')
 if args.limit:rows=rows[:args.limit]
 out=pathlib.Path(args.out);out.mkdir(parents=True,exist_ok=True)
 api=ALFRED(os.getenv('FRED_API_KEY'),pathlib.Path('.cache_pit_asof')/args.serie)
 results=[];totals=collections.Counter();seen=set()
 for i,r in enumerate(rows,1):
  serie=r['serie'];date=r['data_segnale'];asof=r.get('vintage_asof','');obs=r.get('ultima_osservazione','');value=r.get('valore','');deriv=r.get('metodo_acquisizione','').startswith('DERIVATO')
  official='';detail=''
  try:
   date_d=dt.date.fromisoformat(date);a_d=dt.date.fromisoformat(asof);o_d=dt.date.fromisoformat(obs) if obs else None
   if a_d>=date_d:status='DATA_NON_CONSERVATIVA'
   elif o_d and o_d>a_d:status='OSSERVAZIONE_FUTURA'
   elif (serie,date) not in keys or (serie,date) in seen:status='ASSOCIAZIONE_ERRATA'
   elif not value or value=='.':status='VALORE_SNAPSHOT_ASSENTE'
   elif deriv:status='DERIVATO_NON_CERTIFICATO'
   elif not o_d:status='DATA_OSSERVAZIONE_ASSENTE'
   else:
    ans=api.check(serie,asof,obs);official=ans.get('value','');detail=ans.get('detail','')
    status=('PIT_DAILY_CONFORME' if eq(value,official) else 'VALORE_DIVERSO') if ans['result']=='OK' else ans['result']
  except (ValueError,KeyError) as e:status='INPUT_NON_VALIDO';detail=str(e)
  seen.add((serie,date));totals[status]+=1
  results.append({'serie':serie,'data_segnale':date,'vintage_asof':asof,'ultima_osservazione':obs,'valore_snapshot':value,'valore_alfred':official,'stato':status,'dettaglio':detail,'certificato_intraday':'NO'})
  if i%100==0:print(serie,i,'/',len(rows),dict(totals),flush=True)
 fields=list(results[0]);csvout(out/(serie+'_confronto.csv'),results,fields)
 report={'serie':serie,'righe':len(rows),'stati':dict(totals),'richieste_api':api.calls,'cache_hits':api.cached,'tipo_certificazione':'Solo compatibilita PIT giornaliera: non prova timestamp intraday e non certifica derivati','completo':args.limit==0}
 (out/(serie+'_riepilogo.json')).write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
 print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
