#!/usr/bin/env python3
"""Associazione calendari-vintage e audit PIT fail-closed. Python stdlib."""
import csv,io,zipfile,json,collections,datetime,bisect,pathlib,sys
ROOT=pathlib.Path('.')
OUT=ROOT/'associazione_pit_risultati';OUT.mkdir(exist_ok=True)
def load(path,suffix):
 with zipfile.ZipFile(path) as z:
  names=[n for n in z.namelist() if n.endswith(suffix)]
  if len(names)!=1:raise ValueError(f'Atteso {suffix} in {path}, trovati {names}')
  return list(csv.DictReader(io.StringIO(z.read(names[0]).decode('utf-8-sig'))))
def save(name,rows,fields):
 with (OUT/name).open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def day(v):
 try:return datetime.date.fromisoformat(v)
 except (ValueError,TypeError):return None
def main():
 snaps=load(ROOT/'storico-pit-64-completo.zip','snapshot_pit_64_completo.csv')
 cal=load(ROOT/'macro-swing-lab-audit-pit-integrato.zip','calendari_ufficiali.csv')
 reg=load(ROOT/'Macro_Swing_Lab_6_0_Fase_2_9_Registro_Fonti.zip','registro_fonti_69.csv')
 audit=load(ROOT/'storico-pit-64-completo.zip','audit_indipendente_64_completo.csv')
 auditset={(x['serie'],x['vintage_asof']) for x in audit if x['esito']=='PASS'}
 registered={x['serie'] for x in reg}
 dates=collections.defaultdict(set); releases=collections.defaultdict(set)
 for r in cal:
  d=day(r['data'])
  if d:dates[r['serie']].add(d);releases[r['serie']].add(r['release_id'])
 sorted_dates={s:sorted(v) for s,v in dates.items()}
 outputs=[];issues=[];stats=collections.defaultdict(collections.Counter);seen=set()
 for r in snaps:
  s=r['serie'];sig=day(r['data_segnale']);vint=day(r['vintage_asof']);obs=day(r['ultima_osservazione'])
  ds=sorted_dates.get(s,[])
  index=bisect.bisect_right(ds,vint) if vint else 0
  prev=ds[index-1] if index else None
  nextd=ds[index] if index<len(ds) else None
  same=vint in dates.get(s,set()) if vint else False
  isderived='DERIV' in r['stato'].upper() or 'DERIV' in r['metodo_acquisizione'].upper()
  errors=[];key=(s,r['data_segnale'])
  if key in seen:errors.append('DUPLICATO')
  seen.add(key)
  if s not in registered:errors.append('REGISTRO_MANCANTE')
  if not sig or not vint:errors.append('DATA_NON_VALIDA')
  if sig and vint and vint>sig:errors.append('VINTAGE_FUTURA')
  if sig and obs and obs>sig:errors.append('OSSERVAZIONE_FUTURA')
  if errors:state='ERRORE_DATI'
  elif not r['valore']:state='NON_DISPONIBILE'
  else:state='NON_CERTIFICATO'
  # Association is only calendar proximity, not proof that the specific value was published.
  if state=='NON_CERTIFICATO':
   if isderived:reason='DERIVATO_COMPONENTI_NON_VERIFICATE'
   elif same:reason='CALENDARIO_STESSA_DATA_NON_PROVA_VALORE'
   elif prev:reason='RELEASE_PRECEDENTE_NON_PROVA_VALORE'
   else:reason='NESSUNA_RELEASE_PRECEDENTE'
  else:reason=';'.join(errors) if errors else 'VALORE_NON_DISPONIBILE'
  age=(vint-prev).days if vint and prev else ''
  # A successful spot-check confirms API consistency, not release chronology.
  checked=(s,r['vintage_asof']) in auditset
  result={'serie':s,'data_segnale':r['data_segnale'],'vintage_asof':r['vintage_asof'],
   'ultima_osservazione':r['ultima_osservazione'],'valore':r['valore'],
   'release_ultima_precedente':prev.isoformat() if prev else '',
   'release_successiva':nextd.isoformat() if nextd else '',
   'giorni_da_release_precedente':age,'release_nella_data_vintage':'SI' if same else 'NO',
   'audit_api_spotcheck':'PASS' if checked else 'NON_CAMPIONATO',
   'derivato':'SI' if isderived else 'NO','stato':state,'motivazione':reason,
   'ammesso_backtest_strict':'NO','ammesso_backtest_daily':'NO'}
  outputs.append(result);stats[s][state]+=1
  if errors:issues.append({'serie':s,'data_segnale':r['data_segnale'],'errori':';'.join(errors)})
 save('associazioni_snapshot.csv',outputs,list(outputs[0]))
 save('anomalie.csv',issues,['serie','data_segnale','errori'])
 per=[{'serie':s,'snapshot':sum(c.values()),'release_eventi_unici':len(dates.get(s,set())),
       'release_id':len(releases.get(s,set())),**{k:c[k] for k in ('NON_CERTIFICATO','NON_DISPONIBILE','ERRORE_DATI')}} for s,c in sorted(stats.items())]
 save('riepilogo_serie.csv',per,['serie','snapshot','release_eventi_unici','release_id','NON_CERTIFICATO','NON_DISPONIBILE','ERRORE_DATI'])
 counts=collections.Counter(x['stato'] for x in outputs)
 report={'snapshot':len(outputs),'serie':len(stats),'eventi_calendario_input':len(cal),
 'date_release_uniche_per_serie':sum(map(len,dates.values())),
 'spotcheck_api_pass':sum(x['audit_api_spotcheck']=='PASS' for x in outputs),
 'esiti':dict(counts),'certificati_strict':0,'ammessi_backtest':0,
 'spiegazione':'Associazione release-vintage effettuata. Calendari FRED non provano il valore, la sua vintage effettiva, l’orario di disponibilità o le componenti derivate. Nessun dato ammesso senza evidenze puntuali indipendenti.'}
 (OUT/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
 (OUT/'LEGGIMI.txt').write_text('I risultati sono un audit di associazione, NON una certificazione PIT. Per certificare occorre fonte archivistica che colleghi osservazione e valore alla pubblicazione, con orario o regola daily prudenziale, revisioni e componenti per derivati.\n',encoding='utf-8')
 print(json.dumps(report,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
