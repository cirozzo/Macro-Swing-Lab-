#!/usr/bin/env python3
"""PIT audit integrato con raccolta ufficiale FRED, cache e gate fail-closed."""
import argparse,csv,io,zipfile,json,collections,datetime,pathlib,os,time,urllib.parse,urllib.request,hashlib
UTC=datetime.timezone.utc
def zipcsv(path,ending):
    with zipfile.ZipFile(path) as z:
        names=[n for n in z.namelist() if n.endswith(ending)]
        if len(names)!=1:raise ValueError(f"{path}: atteso unico {ending}; trovati {names}")
        return list(csv.DictReader(io.StringIO(z.read(names[0]).decode('utf-8-sig'))))
def date(s):
    try:return datetime.date.fromisoformat(s)
    except (ValueError,TypeError):return None
def writecsv(path,rows,cols):
    with open(path,'w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)
def api(endpoint,params,key,cache,delay):
    args={**params,'api_key':key,'file_type':'json'}
    token=hashlib.sha256(json.dumps([endpoint,sorted(args.items())],sort_keys=True).encode()).hexdigest()
    target=cache/(token+'.json')
    if target.exists():return json.loads(target.read_text())
    url='https://api.stlouisfed.org/fred/'+endpoint+'?'+urllib.parse.urlencode(args)
    err=None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'MacroSwingLab-PIT-Audit/1.0'}),timeout=35) as resp:
                data=json.load(resp)
            if 'error_code' in data:raise ValueError(data.get('error_message',str(data)))
            target.write_text(json.dumps(data),encoding='utf-8')
            time.sleep(delay)
            return data
        except Exception as e:
            err=str(e);time.sleep(min(2**attempt,8))
    raise RuntimeError(err)
def collect_release_dates(series,key,cache,delay):
    """Release-date calendar is corroboration, not evidence of observation-specific timestamps."""
    releases=api('series/release',{'series_id':series},key,cache,delay).get('releases',[])
    ids=[str(x['id']) for x in releases]
    result=[]
    for rid in ids:
        offset=0
        while True:
            response=api('release/dates',{'release_id':rid,'include_release_dates_with_no_data':'true','limit':1000,'offset':offset},key,cache,delay)
            dates=response.get('release_dates',[])
            result.extend({'serie':series,'release_id':rid,'data':x.get('date',''),
                           'release_name':next((v.get('name','') for v in releases if str(v['id'])==rid),''),
                           'fonte':f'https://api.stlouisfed.org/fred/release/dates?release_id={rid}'}
                          for x in dates)
            offset+=len(dates)
            if not dates or offset>=int(response.get('count',offset)):break
    return result
def assess(row,registered,calendar):
    signal=date(row.get('data_segnale')); vintage=date(row.get('vintage_asof'))
    observation=date(row.get('ultima_osservazione'))
    value=row.get('valore','')
    errors=[]
    if not signal or not vintage:errors.append('DATA_INVALIDA')
    if signal and vintage and vintage>signal:errors.append('VINTAGE_FUTURA')
    if observation and signal and observation>signal:errors.append('OSSERVAZIONE_FUTURA')
    if row.get('serie') not in registered:errors.append('SERIE_NON_REGISTRATA')
    if not value:status='NON_DISPONIBILE'
    elif errors:status='ERRORE_DATI'
    else:status='NON_VERIFICABILE'
    # Calendar corroboration alone does NOT establish observation publication,
    # intraday timestamp, ALFRED vintage or numerical version.
    has_calendar=bool(calendar)
    return status,errors,has_calendar
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='.')
    p.add_argument('--out',default='audit_pit_completo')
    p.add_argument('--delay',type=float,default=.12)
    a=p.parse_args()
    root=pathlib.Path(a.root);out=root/a.out;out.mkdir(exist_ok=True,parents=True)
    rows=zipcsv(root/'storico-pit-64-completo.zip','snapshot_pit_64_completo.csv')
    reg=zipcsv(root/'Macro_Swing_Lab_6_0_Fase_2_9_Registro_Fonti.zip','registro_fonti_69.csv')
    registered={x['serie']:x for x in reg}
    key=os.environ.get('FRED_API_KEY','').strip()
    cache=root/'.cache_fred';cache.mkdir(exist_ok=True)
    calendars=[];fetch_errors=[];per=collections.defaultdict(collections.Counter)
    for series in sorted({r['serie'] for r in rows}):
        if not key:
            fetch_errors.append({'serie':series,'errore':'FRED_API_KEY assente: nessuna verifica esterna'})
            continue
        try:calendars.extend(collect_release_dates(series,key,cache,a.delay))
        except Exception as e:fetch_errors.append({'serie':series,'errore':str(e)})
    byseries=collections.defaultdict(list)
    for r in calendars:byseries[r['serie']].append(r)
    seen=set();results=[];anomalies=[]
    for row in rows:
        series=row['serie'];signal=row['data_segnale'];pair=(series,signal)
        status,issues,has_calendar=assess(row,registered,byseries[series])
        if pair in seen:issues.append('DUPLICATO');status='ERRORE_DATI'
        seen.add(pair)
        if issues:anomalies.append({'serie':series,'data_segnale':signal,'errore':';'.join(issues)})
        per[series][status]+=1
        method=row.get('metodo_acquisizione','')
        results.append({'serie':series,'data_segnale':signal,'vintage_asof':row['vintage_asof'],
            'ultima_osservazione':row['ultima_osservazione'],'valore':row['valore'],
            'metodo':method,'calendario_release_recuperato':'SI' if has_calendar else 'NO',
            'stato_pit':status,'ammesso_strict':'NO','ammesso_daily':'NO',
            'motivo':';'.join(issues) if issues else
             ('VALORE_ASSENTE' if not row['valore'] else
              'CALENDARIO_NON_PROVA_VALORE_E_ORARIO' if has_calendar else 'EVIDENZE_INSUFFICIENTI')})
    writecsv(out/'snapshot_gate.csv',results,list(results[0]))
    writecsv(out/'anomalie.csv',anomalies,['serie','data_segnale','errore'])
    writecsv(out/'calendari_ufficiali.csv',calendars,['serie','release_id','data','release_name','fonte'])
    writecsv(out/'errori_fonti.csv',fetch_errors,['serie','errore'])
    stats=[{'serie':s,'totale':sum(v.values()),'con_valore':sum(v.values())-v['NON_DISPONIBILE'],
            'calendario_recuperato':'SI' if byseries[s] else 'NO',
            **{k:v[k] for k in ('NON_DISPONIBILE','NON_VERIFICABILE','ERRORE_DATI')}} for s,v in sorted(per.items())]
    writecsv(out/'copertura.csv',stats,['serie','totale','con_valore','calendario_recuperato','NON_DISPONIBILE','NON_VERIFICABILE','ERRORE_DATI'])
    counts=collections.Counter(r['stato_pit'] for r in results)
    report={'serie':len(per),'snapshot':len(results),'esiti':dict(counts),
            'calendari_eventi':len(calendars),'serie_con_calendario':sum(bool(v) for v in byseries.values()),
            'errori_fonti':len(fetch_errors),'anomalie':len(anomalies),
            'strict_certificati':0,
            'limite':'Calendari FRED e vintage_asof NON certificano da soli timestamp di pubblicazione, valore iniziale e revisioni. Gate fail-closed.'}
    (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    with (out/'RIEPILOGO.md').open('w') as f:
        f.write('# Audit PIT integrato\n\n'+json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(report,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
