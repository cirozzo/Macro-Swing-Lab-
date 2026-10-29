#!/usr/bin/env python3
"""ALFRED value/vintage verification; conservative day-lag, never STRICT from daily vintages."""
import os,csv,io,json,zipfile,pathlib,urllib.request,urllib.parse,time,datetime,collections,decimal,hashlib
ROOT=pathlib.Path('.');OUT=ROOT/'certificazione_pit_finale';OUT.mkdir(exist_ok=True)
CACHE=ROOT/'.pit_finale_cache';CACHE.mkdir(exist_ok=True)
def readzip(path,suffix):
 with zipfile.ZipFile(path) as z:
  names=[n for n in z.namelist() if n.endswith(suffix)]
  if len(names)!=1:raise ValueError(f"Expected {suffix} in {path}, got {names}")
  return list(csv.DictReader(io.StringIO(z.read(names[0]).decode('utf-8-sig'))))
def csvout(name,rows,columns):
 with (OUT/name).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)
def fetch(series,api_key):
 """Fetch all ALFRED vintage intervals (output_type=2) in pages. Cache by series/page."""
 records=[];offset=0
 while True:
  cache=CACHE/(hashlib.sha256(f'{series}:{offset}:v1'.encode()).hexdigest()+'.json')
  if cache.exists():data=json.loads(cache.read_text())
  else:
   params={'series_id':series,'api_key':api_key,'file_type':'json','output_type':2,
           'realtime_start':'1776-07-04','realtime_end':'9999-12-31','limit':100000,'offset':offset}
   url='https://api.stlouisfed.org/fred/series/observations?'+urllib.parse.urlencode(params)
   for attempt in range(5):
    try:
     with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'MSL-PIT-Research/1.0'}),timeout=120) as r:data=json.load(r)
     if 'error_code' in data:raise RuntimeError(data.get('error_message',str(data)))
     cache.write_text(json.dumps(data),encoding='utf-8');time.sleep(.3);break
    except Exception:
     if attempt==4:raise
     time.sleep(min(2**attempt,16))
  part=data.get('observations',[]);records.extend(part);offset+=len(part)
  if not part or offset>=int(data.get('count',offset)):break
 return records
def valid_value(a,b):
 try:
  return abs(decimal.Decimal(str(a))-decimal.Decimal(str(b)))<=decimal.Decimal('0.000001')
 except (decimal.InvalidOperation,ValueError):return False
def check(row,byobs,api_ok):
 if not row['valore']:return 'NON_DISPONIBILE','VALORE_ASSENTE',''
 if row.get('metodo_acquisizione','').upper().find('DERIV')>=0 or row.get('stato','').upper().find('DERIV')>=0:
  return 'NON_VERIFICABILE','DERIVATO_COMPONENTI_NON_VERIFICATE',''
 if not api_ok:return 'NON_VERIFICABILE','API_ALFRED_NON_DISPONIBILE',''
 try:
  signal=datetime.date.fromisoformat(row['data_segnale'])
  vintage=datetime.date.fromisoformat(row['vintage_asof'])
  obs=datetime.date.fromisoformat(row['ultima_osservazione'])
 except ValueError:return 'ERRORE_DATI','DATE_INVALID',''
 if obs>signal or vintage>signal:return 'ERRORE_DATI','FUTURE_INFORMATION',''
 # Select the interval that contains the exact as-of date.
 candidates=[r for r in byobs.get(row['ultima_osservazione'],[])
  if r.get('realtime_start','9999')<=row['vintage_asof']<=r.get('realtime_end','0000')]
 if len(candidates)!=1:return 'NON_VERIFICABILE','INTERVALLO_VINTAGE_ASSENTE_O_AMBIGUO',str(len(candidates))
 r=candidates[0]
 if not valid_value(row['valore'],r.get('value','')):return 'ERRORE_DATI','VALORE_NON_COINCIDE',r.get('value','')
 # Require >= 1 full day between vintage as-of and signal, conservatively
 # avoid claiming intraday release timing from ALFRED daily dates.
 if (signal-vintage).days<1:return 'NON_VERIFICABILE','NESSUN_MARGINE_TEMPORALE',r.get('value','')
 return 'VALIDATO_GIORNALIERO','ALFRED_INTERVAL_MATCH_CON_1_GIORNO_DI_MARGINE',r.get('value','')
def main():
 original=readzip(ROOT/'storico-pit-64-completo.zip','snapshot_pit_64_completo.csv')
 assoc=readzip(ROOT/'macro-swing-lab-associazione-pit.zip','associazioni_snapshot.csv')
 keys={(r['serie'],r['data_segnale']) for r in assoc}
 key=os.getenv('FRED_API_KEY','').strip()
 output=[];issues=[];counts=collections.Counter()
 for series in sorted(set(r['serie'] for r in original)):
  group=[r for r in original if r['serie']==series]
  ok=False;byobs=collections.defaultdict(list);err=''
  if key:
   try:
    records=fetch(series,key)
    for r in records:byobs[r.get('date','')].append(r)
    ok=True
   except Exception as e:err=str(e)
  else:err='FRED_API_KEY mancante'
  if err:issues.append({'serie':series,'errore':err})
  for r in group:
   status,reason,official=check(r,byobs,ok)
   if (series,r['data_segnale']) not in keys:
    status,reason='ERRORE_DATI','ASSOCIAZIONE_MANCANTE'
   counts[status]+=1
   output.append({'serie':series,'data_segnale':r['data_segnale'],'vintage_asof':r['vintage_asof'],
    'ultima_osservazione':r['ultima_osservazione'],'valore_snapshot':r['valore'],
    'valore_alfred':official,'stato':status,'motivo':reason,
    'ammesso_strict':'NO','ammesso_daily':'SI' if status=='VALIDATO_GIORNALIERO' else 'NO'})
 cols=list(output[0]);csvout('esiti_completi.csv',output,cols)
 csvout('dataset_daily_conservativo.csv',[r for r in output if r['ammesso_daily']=='SI'],cols)
 csvout('dataset_strict.csv',[],cols)
 csvout('errori_api.csv',issues,['serie','errore'])
 report={'totale':len(output),'serie':len(set(r['serie'] for r in original)),
         'esiti':dict(counts),'errori_api':len(issues),
         'strict':0,'daily_conservativo':counts['VALIDATO_GIORNALIERO'],
         'nota':'ALFRED vintage giornaliera non prova timestamp intraday o pubblicazione originale. Il dataset daily richiede decisione successiva di almeno un giorno alla vintage e verifica del valore, non equivale a certificazione STRICT. Serie derivate escluse.'}
 (OUT/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
 print(json.dumps(report,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
