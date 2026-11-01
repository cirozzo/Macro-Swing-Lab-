#!/usr/bin/env python3
"""Audit indipendente 738 derivati T10YIE/T5YIE, senza promuovere automaticamente dati a certificati."""
import argparse,csv,io,zipfile,os,urllib.parse,urllib.request,urllib.error,json,decimal,time,random,hashlib,pathlib,collections,datetime
API='https://api.stlouisfed.org/fred/series/observations'
D=decimal.Decimal
COMP={'T10YIE':('DGS10','DFII10'),'T5YIE':('DGS5','DFII5')}

def number(x):
 try:
  if str(x).strip() in ('','.','None'):return None
  return D(str(x).strip())
 except decimal.InvalidOperation:return None

class Fred:
 def __init__(self,key,cache):
  self.key=key;self.path=pathlib.Path(cache);self.path.mkdir(exist_ok=True,parents=True)
  self.calls=0;self.hits=0
 def get(self,serie,asof,obs):
  ident=hashlib.sha256(f'{serie}|{asof}|{obs}'.encode()).hexdigest();p=self.path/(ident+'.json')
  if p.exists():self.hits+=1;return json.loads(p.read_text())
  opts={'series_id':serie,'api_key':self.key,'file_type':'json','realtime_start':asof,'realtime_end':asof,'observation_start':obs,'observation_end':obs,'output_type':1,'limit':1000}
  url=API+'?'+urllib.parse.urlencode(opts)
  for attempt in range(7):
   try:
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'MacroSwingLab-Derived-PIT/1.0'}),timeout=45) as resp: d=json.load(resp)
    rec=[x for x in d.get('observations',[]) if x.get('date')==obs]
    if len(rec)!=1:ans={'state':'NO_OBSERVATION' if not rec else 'AMBIGUOUS','value':''}
    else:ans={'state':'OK' if number(rec[0].get('value')) is not None else 'MISSING','value':rec[0].get('value','')}
    p.write_text(json.dumps(ans));self.calls+=1;time.sleep(.55);return ans
   except urllib.error.HTTPError as e:
    detail=e.read(300).decode('utf8','replace')
    if e.code not in (429,500,502,503,504):return {'state':f'HTTP_{e.code}','detail':detail,'value':''}
    delay=min(80,2**attempt+random.random()*2)
    ra=e.headers.get('Retry-After')
    if ra and ra.isdigit():delay=max(delay,min(120,int(ra)))
   except (OSError,ValueError) as e:delay=min(80,2**attempt+random.random()*2);detail=str(e)
   if attempt<6:time.sleep(delay)
  return {'state':'RETRY_EXHAUSTED','detail':detail,'value':''}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',default='macro-swing-lab-confronto-pit-unico.zip');ap.add_argument('--out',default='audit_derivati');ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
 key=os.getenv('FRED_API_KEY');
 if not key:raise SystemExit('FRED_API_KEY mancante')
 output=pathlib.Path(a.out);output.mkdir(exist_ok=True,parents=True)
 rows=[]
 with zipfile.ZipFile(a.input) as z:
  for name in z.namelist():
   if name.endswith('_confronto.csv'):
    rs=list(csv.DictReader(io.StringIO(z.read(name).decode('utf-8-sig'))))
    rows.extend(r for r in rs if r.get('stato')=='DERIVATO_NON_CERTIFICATO')
 print('Derivati trovati:',len(rows),collections.Counter(r['serie'] for r in rows),flush=True)
 if not rows:raise SystemExit('Nessuna riga DERIVATO_NON_CERTIFICATO trovata: controllare archivio input')
 if a.limit:rows=rows[:a.limit]
 fred=Fred(key,output/'cache');done=[]
 for i,r in enumerate(rows,1):
  serie=r['serie'];date=r['data_segnale'];asof=r['vintage_asof'];obs=r['ultima_osservazione'];actual=number(r['valore_snapshot'])
  row={'serie':serie,'data_segnale':date,'vintage_asof':asof,'ultima_osservazione':obs,'valore_originale':r['valore_snapshot'],'valore_alfred_diretto':'','stato_diretto':'','nominale_serie':'','reale_serie':'','valore_nominale':'','valore_reale':'','valore_ricostruito':'','delta_diretto':'','delta_ricostruito':'','stato_finale':'','nota':''}
  try:
   if serie not in COMP:row['stato_finale']='DERIVATO_NON_MAPPATO'
   elif not (datetime.date.fromisoformat(obs)<=datetime.date.fromisoformat(asof)<datetime.date.fromisoformat(date)):
    row['stato_finale']='DATE_NON_VALIDE'
   elif actual is None:row['stato_finale']='VALORE_ORIGINALE_MANCANTE'
   else:
    direct=fred.get(serie,asof,obs);row['valore_alfred_diretto']=direct['value'];row['stato_diretto']=direct['state']
    v=number(direct['value']);
    if v is not None:row['delta_diretto']=str(actual-v)
    nominal,real=COMP[serie];row['nominale_serie']=nominal;row['reale_serie']=real
    n=fred.get(nominal,asof,obs);q=fred.get(real,asof,obs)
    row['valore_nominale']=n['value'];row['valore_reale']=q['value']
    nv=number(n['value']);rv=number(q['value']);reconstructed=nv-rv if nv is not None and rv is not None else None
    if reconstructed is not None:row['valore_ricostruito']=str(reconstructed);row['delta_ricostruito']=str(actual-reconstructed)
    if v is not None and abs(actual-v)<=D('0.000001'):
     row['stato_finale']='DIRETTO_PIT_CONFORME'
     if reconstructed is not None and abs(actual-reconstructed)>D('0.015'):
      row['nota']='Differenza con componenti: indagare metodologia/arrotondamento'
    elif v is not None:row['stato_finale']='DIRETTO_DISCREPANTE'
    elif reconstructed is not None and abs(actual-reconstructed)<=D('0.015'):
     row['stato_finale']='SOLO_COMPONENTI_COMPATIBILI';row['nota']='Non è certificazione della serie ufficiale: differenze di arrotondamento possibili'
    elif reconstructed is not None:row['stato_finale']='COMPONENTI_DISCREPANTI'
    else:row['stato_finale']='DATI_COMPONENTI_ASSENTI';row['nota']=f'{nominal}:{n["state"]} {real}:{q["state"]}; diretto:{direct["state"]}'
  except (ValueError,KeyError,TypeError) as e:row['stato_finale']='ERRORE_INPUT';row['nota']=str(e)
  done.append(row)
  if i%50==0:print(f'{i}/{len(rows)} - {dict(collections.Counter(x["stato_finale"] for x in done))}',flush=True)
 path=output/'derivati_738_audit.csv'
 with path.open('w',newline='',encoding='utf-8-sig') as f:
  wr=csv.DictWriter(f,fieldnames=list(done[0]));wr.writeheader();wr.writerows(done)
 summary={'righe_input':len(rows),'per_serie':dict(collections.Counter(x['serie'] for x in done)),'stati':dict(collections.Counter(x['stato_finale'] for x in done)),'richieste_api':fred.calls,'cache_hits':fred.hits,'completo':a.limit==0 and len(rows)==738,'criterio':'PIT giornaliero; non certifica gli orari intraday; componenti come controllo indipendente, non sostitutivo della serie ufficiale'}
 (output/'riepilogo_derivati.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf8')
 print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
