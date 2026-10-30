import os,sys,json,zipfile,csv,io,urllib.request,urllib.parse
files={'storico-pit-64-completo.zip':'snapshot_pit_64_completo.csv','macro-swing-lab-associazione-pit.zip':'associazioni_snapshot.csv'}
for name,suffix in files.items():
 with zipfile.ZipFile(name) as z:
  matches=[x for x in z.namelist() if x.endswith(suffix)]
  if len(matches)!=1:sys.exit(f'Input invalido: {name} / {suffix}')
  rows=list(csv.DictReader(io.StringIO(z.read(matches[0]).decode('utf-8-sig'))))
  print(name,len(rows),flush=True)
  if name.startswith('storico-'):series=sorted({r['serie'] for r in rows})
key=os.getenv('FRED_API_KEY','')
if not key:sys.exit('Configura GitHub Secret FRED_API_KEY')
params=urllib.parse.urlencode({'series_id':'GDP','api_key':key,'file_type':'json','limit':1})
with urllib.request.urlopen('https://api.stlouisfed.org/fred/series/observations?'+params,timeout=20) as r:
 data=json.load(r)
if 'observations' not in data:sys.exit('API FRED non valida: '+str(data)[:250])
with open('serie_pit_v3.json','w') as f:json.dump({'series':series},f)
print('Preflight OK:',len(series),'serie')
