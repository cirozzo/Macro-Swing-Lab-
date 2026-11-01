#!/usr/bin/env python3
"""Audit PIT indipendente di 738 derivati: non promuove righe senza prove API."""
import argparse,csv,io,zipfile,os,json,datetime,decimal,urllib.parse,urllib.request,urllib.error,time,pathlib,collections,hashlib
D=decimal.Decimal
MAP={'T10YIE':('DGS10','DFII10'),'T5YIE':('DGS5','DFII5')}
def val(s):
 try:return D(str(s)) if str(s).strip() not in ('','.','None') else None
 except decimal.InvalidOperation:return None
def load(zpath,suffix):
 with zipfile.ZipFile(zpath) as z:
  for name in z.namelist():
   if name.endswith(suffix):
    yield from csv.DictReader(io.StringIO(z.read(name).decode('utf-8-sig')))
class Api:
 def __init__(self,key,folder):self.key=key;self.folder=pathlib.Path(folder);self.folder.mkdir(parents=True,exist_ok=True);self.calls=0
 def fetch(self,series,asof,obs):
  args={'series_id':series,'api_key':self.key,'file_type':'json','realtime_start':asof,'realtime_end':asof,'observation_start':obs,'observation_end':obs}
  url='https://api.stlouisfed.org/fred/series/observations?'+urllib.parse.urlencode(args)
  digest=hashlib.sha256(f'{series}|{asof}|{obs}'.encode()).hexdigest();p=self.folder/(digest+'.json')
  if p.exists():return json.loads(p.read_text())
  for retry in range(6):
   try:
    with urllib.request.urlopen(url,timeout=40) as f: data=json.load(f)
    matches=[r for r in data.get('observations',[]) if r.get('date')==obs]
    result={'status':'OK' if len(matches)==1 and val(matches[0].get('value')) is not None else 'MISSING_OR_AMBIGUOUS','value':matches[0]['value'] if len(matches)==1 else '', 'observation_date':obs,'asof':asof,'series':series,'realtime_start':matches[0].get('realtime_start','') if len(matches)==1 else '', 'realtime_end':matches[0].get('realtime_end','') if len(matches)==1 else ''}
    p.write_text(json.dumps(result));self.calls+=1;time.sleep(.25);return result
   except urllib.error.HTTPError as e:
    detail=e.read(500).decode('utf-8','replace')
    if e.code not in (429,500,502,503,504):return {'status':f'HTTP_{e.code}','detail':detail,'value':''}
   except Exception as e:detail=str(e)
   time.sleep(min(30,2**retry))
  return {'status':'RETRY_EXHAUSTED','detail':detail,'value':''}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--audit',default='macro-swing-lab-audit-derivati-unico.zip');ap.add_argument('--confronto',default='macro-swing-lab-confronto-pit-unico.zip');ap.add_argument('--out',default='certificazione_componenti');ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
 out=pathlib.Path(a.out);out.mkdir(exist_ok=True,parents=True)
 audit=list(load(a.audit,'derivati_738_audit.csv')); original=list(load(a.confronto,'_confronto.csv'))
 originals={(r.get('serie'),r.get('data_segnale')):r for r in original if r.get('serie') in MAP}
 refs={(r.get('serie'),r.get('data_segnale')):r for r in original if r.get('serie') in ('DGS10','DFII10')}
 if not audit:raise SystemExit('Audit vuoto')
 if a.limit:audit=audit[:a.limit]
 key=os.getenv('FRED_API_KEY');
 if not key:raise SystemExit('FRED_API_KEY mancante; audit indipendente impossibile')
 api=Api(key,out/'evidenze_api');results=[]
 for idx,r in enumerate(audit,1):
  series=r['serie'];date=r['data_segnale'];asof=r['vintage_asof'];obs=r['ultima_osservazione'];components=MAP.get(series)
  item={'serie':series,'data_segnale':date,'vintage_asof':asof,'data_osservazione':obs,'valore_originale':r['valore_originale'],'componente_nominale':components[0] if components else '', 'componente_reale':components[1] if components else '', 'stato_nominale':'','stato_reale':'','valore_nominale_api':'','valore_reale_api':'','data_nominale_api':'','data_reale_api':'','valore_ricostruito_api':'','delta':'','confronto_archivio':'','esito':'NON_CERTIFICATO','nota':''}
  try:
   if not components:raise ValueError('Serie non mappata')
   if not (datetime.date.fromisoformat(obs)<=datetime.date.fromisoformat(asof)<datetime.date.fromisoformat(date)):raise ValueError('Date non ordinate')
   orig=originals.get((series,date))
   if not orig or val(orig.get('valore_snapshot'))!=val(r['valore_originale']):raise ValueError('Snapshot non coincide con archivio originale')
   n=api.fetch(components[0],asof,obs);q=api.fetch(components[1],asof,obs)
   item.update(stato_nominale=n['status'],stato_reale=q['status'],valore_nominale_api=n.get('value',''),valore_reale_api=q.get('value',''),data_nominale_api=n.get('observation_date',''),data_reale_api=q.get('observation_date',''))
   nv=val(n.get('value'));rv=val(q.get('value'))
   if nv is None or rv is None:raise ValueError('Componente API non disponibile: '+n['status']+' / '+q['status'])
   if n.get('observation_date')!=obs or q.get('observation_date')!=obs:raise ValueError('Date osservazione non allineate')
   calc=nv-rv;delta=val(r['valore_originale'])-calc;item['valore_ricostruito_api']=str(calc);item['delta']=str(delta)
   item['confronto_archivio']='COINCIDE' if nv==val(r['valore_nominale']) and rv==val(r['valore_reale']) else 'DIVERSO'
   if abs(delta)<=D('0.015') and item['confronto_archivio']=='COINCIDE':item['esito']='PIT_COMPONENTI_CONFORME_GIORNALIERO'
   else:item['esito']='DA_INDAGARE';item['nota']='Scostamento valori, tolleranza o componente'
  except Exception as e:item['nota']=str(e)
  results.append(item)
  if idx%50==0:print(idx,'/',len(audit),collections.Counter(x['esito'] for x in results),flush=True)
 with (out/'certificazione_738.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
 summary={'totale':len(results),'esiti':dict(collections.Counter(x['esito'] for x in results)),'chiamate_api':api.calls,'certificazione_completa':len(results)==738 and all(x['esito']=='PIT_COMPONENTI_CONFORME_GIORNALIERO' for x in results),'criterio':'Componenti ALFRED richieste as-of e sulla medesima data osservazione; nessuna certificazione intraday; tolleranza 0.015 punti percentuali'}
 (out/'riepilogo_certificazione.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False));print(json.dumps(summary,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
