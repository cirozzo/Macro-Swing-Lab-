#!/usr/bin/env python3
"""Consolida solo snapshot con stato PIT_DAILY_CONFORME; non certifica release-time."""
import argparse,csv,io,json,zipfile,collections,pathlib
p=argparse.ArgumentParser();p.add_argument('zip_input');p.add_argument('--output',default='macro_pit_consolidato');a=p.parse_args()
out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=True)
records=[]; counts=collections.Counter();byseries=collections.defaultdict(collections.Counter); duplicates=[];keys=set()
with zipfile.ZipFile(a.zip_input) as z:
 names=[n for n in z.namelist() if n.endswith('_confronto.csv')]
 for name in names:
  for row in csv.DictReader(io.StringIO(z.read(name).decode('utf-8-sig'))):
   status=row['stato'];counts[status]+=1;byseries[row['serie']][status]+=1
   if status!='PIT_DAILY_CONFORME':continue
   key=(row['serie'],row['data_segnale'])
   if key in keys: duplicates.append(key);continue
   keys.add(key)
   if not row['valore_snapshot'] or not row['valore_alfred'] or not row['vintage_asof'] or not row['ultima_osservazione']:
    raise ValueError('Campi obbligatori assenti: '+str(key))
   if row['vintage_asof']>=row['data_segnale'] or row['ultima_osservazione']>row['vintage_asof']:
    raise ValueError('Anomalia cronologica: '+str(key))
   records.append({**row,'incluso_backtest':'SI','qualita':'PIT_GIORNALIERO_VERIFICATO_NON_INTRADAY'})
if duplicates:raise ValueError(f'Duplicati: {duplicates[:5]}')
records.sort(key=lambda r:(r['data_segnale'],r['serie']))
if records:
 with open(out/'macro_pit_consolidato.csv','w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=records[0].keys());w.writeheader();w.writerows(records)
summary={'input_files':len(names),'totale_righe':sum(counts.values()),'stati':dict(counts),'inclusi':len(records),'esclusi':sum(counts.values())-len(records),'serie':{k:dict(v) for k,v in sorted(byseries.items())},'avvertenze':['PIT giornaliero non equivale a verifica del timestamp intraday','Derivati non conformi esclusi','Dati mancanti non imputati','Non certifica regimi macro né prezzi ETF né assenza di lookahead nel backtest']}
(out/'riepilogo_consolidamento.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='serie'},indent=2,ensure_ascii=False))
