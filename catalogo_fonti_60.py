#!/usr/bin/env python3
"""Macro Swing Lab 6.0, fase 1: catalogo e audit copertura, NON backtest PIT."""
import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import random
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = 'https://api.stlouisfed.org/fred'
FIELDS = ['id','categoria','fonte','serie','frequenza_prevista','stato_ingestione','nota_pit']

def load_catalog(path):
    with open(path, newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        if list(reader.fieldnames or []) != FIELDS:
            raise ValueError('Intestazione catalogo inattesa')
        rows = list(reader)
    ids = [r['id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('ID ripetuti nel catalogo')
    for row in rows:
        if row['stato_ingestione'] not in ('attiva','in_attesa'):
            raise ValueError('Stato non valido: '+row['id'])
        if row['stato_ingestione']=='attiva' and (row['fonte']!='FRED' or not row['serie']):
            raise ValueError('La fonte attiva deve essere serie FRED: '+row['id'])
    return rows

def request_fred(path, series, key, retries=5, pause=1.5):
    query = urllib.parse.urlencode({'series_id':series, 'api_key':key, 'file_type':'json'})
    url = f'{BASE}/{path}?{query}'
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent':'MacroSwingLab-CatalogAudit/6.0'})
            with urllib.request.urlopen(req, timeout=25) as response:
                payload = json.load(response)
            if 'error_code' in payload:
                raise RuntimeError('Errore API FRED '+str(payload['error_code']))
            return payload
        except urllib.error.HTTPError as exc:
            if exc.code not in (429,500,502,503,504) or attempt == retries-1:
                raise RuntimeError(f'HTTP {exc.code} serie {series}') from exc
            delay = min(50, pause*(2**attempt)) + random.uniform(0,.3)
            time.sleep(delay)
        except (TimeoutError, urllib.error.URLError) as exc:
            if attempt==retries-1:
                raise RuntimeError(f'Rete non disponibile per {series}: {type(exc).__name__}') from exc
            time.sleep(min(50, pause*(2**attempt)))
    raise RuntimeError('Richiesta fallita')

def write_csv(path, rows, fields):
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def analyze(catalog, key, limit, out, sleep_seconds=0.35, offline=False):
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'catalogo_completo_60.csv',catalog,FIELDS)
    current = dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')
    status=[]
    active = [r for r in catalog if r['stato_ingestione']=='attiva']
    # Round-robin per categoria: un lotto piccolo copre domini diversi.
    groups = {}
    for item in active:
        groups.setdefault(item['categoria'], []).append(item)
    ordered = []
    while any(groups.values()):
        for group in groups.values():
            if group: ordered.append(group.pop(0))
    selected = {r['id'] for r in ordered[:limit]}
    for row in catalog:
        record=dict(row)
        record.update({'audit_utc':current,'api_status':'NON_INTERROGATA',
            'titolo_fred':'','inizio_serie':'','fine_serie':'','frequenza_fred':'',
            'vintage_date_count':'','pit_status':'NON_AMMESSA_BACKTEST', 'errore':''})
        if row['id'] in selected:
            if offline:
                record['api_status']='OFFLINE_TEST'
            else:
                try:
                    meta=request_fred('series',row['serie'],key)['seriess'][0]
                    record['titolo_fred']=meta.get('title','')
                    record['inizio_serie']=meta.get('observation_start','')
                    record['fine_serie']=meta.get('observation_end','')
                    record['frequenza_fred']=meta.get('frequency','')
                    record['api_status']='METADATA_OK'
                except (RuntimeError,ValueError,KeyError,IndexError,TypeError) as exc:
                    record['api_status']='ERRORE';record['errore']=str(exc)[:250]
                time.sleep(sleep_seconds)
        status.append(record)
    status_fields=FIELDS+['audit_utc','api_status','titolo_fred','inizio_serie',
        'fine_serie','frequenza_fred','vintage_date_count','pit_status','errore']
    write_csv(out/'audit_fonti_60.csv',status,status_fields)
    summary={'versione':'6.0_fase1','timestamp_utc':current,
             'catalogo_totale':len(catalog),'fred_catalogate':len(active),
             'fonti_in_attesa':len(catalog)-len(active),
             'metadata_interrogate':len(selected) if not offline else 0,
             'metadata_ok':sum(x['api_status']=='METADATA_OK' for x in status),
             'metadata_errori':sum(x['api_status']=='ERRORE' for x in status),
             'dati_ammessi_al_backtest':0,
             'qualifica':'CATALOGO_E_AUDIT_METADATI_NON_VALIDAZIONE_PIT',
             'limitazione':'Data inizio serie non equivale a prima disponibilita storica; nessun dato aggiunto al modello.',
             'esito':'TEST_OFFLINE' if offline else ('OK' if any(x['api_status']=='METADATA_OK' for x in status) else 'NESSUNA_SERIE_VERIFICATA')}
    (out/'report_60.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    return summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--catalog',default='catalogo_fonti_60.csv')
    p.add_argument('--out',default='output/catalogo_60')
    p.add_argument('--limit',type=int,default=20,help='Numero massimo serie FRED da interrogare in questo run')
    p.add_argument('--sleep',type=float,default=.35)
    p.add_argument('--offline',action='store_true',help='Verifica CSV e pipeline senza chiamare API')
    args=p.parse_args()
    if args.limit<0 or args.sleep<0: p.error('limit e sleep non negativi')
    catalog=load_catalog(args.catalog)
    key=os.environ.get('FRED_API_KEY','')
    if not args.offline and not key: p.error('FRED_API_KEY mancante nei segreti GitHub')
    report=analyze(catalog,key,args.limit,Path(args.out),args.sleep,args.offline)
    if not args.offline and args.limit and report['metadata_ok']==0:
        raise SystemExit('Audit fallito: nessuna serie verificata. Report comunque salvato.')

if __name__=='__main__': main()
