#!/usr/bin/env python3
"""Audit PIT ALFRED per snapshot: acquisizione adattiva, cache e gate conservativo.
Nessun valore è STRICT senza timestamp indipendente di pubblicazione.
"""
import csv, io, zipfile, json, os, time, hashlib, pathlib, datetime, decimal, urllib.request, urllib.parse, urllib.error, collections, argparse, bisect

API='https://api.stlouisfed.org/fred/series/observations'
START=datetime.date(1776,7,4); END=datetime.date(9999,12,31)

def csv_zip(path, ending):
    with zipfile.ZipFile(path) as z:
        names=[n for n in z.namelist() if n.endswith(ending)]
        if len(names)!=1: raise ValueError(f'{path}: expected one {ending}; found {names}')
        return list(csv.DictReader(io.StringIO(z.read(names[0]).decode('utf-8-sig'))))

def csv_write(path,rows,cols):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)

def iso(s):return datetime.date.fromisoformat(s)

def match(a,b):
    try:return abs(decimal.Decimal(str(a))-decimal.Decimal(str(b)))<=decimal.Decimal('0.000001')
    except (ValueError,decimal.InvalidOperation):return False

class Client:
    def __init__(self,key,cache):self.key=key;self.cache=cache;cache.mkdir(parents=True,exist_ok=True);self.requests=0;self.hits=0
    def query(self,series,first,last,offset=0,limit=10000):
        args={'series_id':series,'realtime_start':'1776-07-04','realtime_end':'9999-12-31',
              'observation_start':first.isoformat(),'observation_end':last.isoformat(),
              'output_type':2,'file_type':'json','limit':limit,'offset':offset}
        token=hashlib.sha256(json.dumps(args,sort_keys=True).encode()).hexdigest()
        path=self.cache/(token+'.json')
        if path.exists():self.hits+=1;return json.loads(path.read_text())
        if not self.key:raise RuntimeError('FRED_API_KEY assente')
        url=API+'?'+urllib.parse.urlencode({**args,'api_key':self.key})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'MacroSwingLabPIT/2.0'}),timeout=90) as resp:
                    obj=json.load(resp)
                if 'error_code' in obj:raise RuntimeError(str(obj.get('error_message',obj)))
                path.write_text(json.dumps(obj),encoding='utf-8');self.requests+=1;time.sleep(.12)
                return obj
            except (urllib.error.HTTPError,urllib.error.URLError,TimeoutError,OSError,RuntimeError) as e:
                if attempt==4:raise RuntimeError(f'{series} {first}:{last}: {e}') from e
                time.sleep(min(2**attempt,12))

    def range(self,series,start,end):
        """If a broad request fails, bisect observation date interval until feasible.
        Complete pages required; never interpret truncated/failed pages as success.
        """
        try:
            all_rows=[];offset=0
            while True:
                data=self.query(series,start,end,offset)
                part=data.get('observations',[])
                if not isinstance(part,list):raise ValueError('observations missing')
                all_rows.extend(part);offset+=len(part)
                total=int(data.get('count',offset))
                if offset>=total:break
                if not part:raise RuntimeError('empty page before count reached')
                if offset>150000:raise RuntimeError('large range: split')
            return all_rows
        except Exception:
            if start>=end:raise
            midpoint=start+datetime.timedelta(days=(end-start).days//2)
            return self.range(series,start,midpoint)+self.range(series,midpoint+datetime.timedelta(days=1),end)

def classify(r,records,source_error=''):
    if not r['valore']:return ('NON_DISPONIBILE','VALORE_ASSENTE','')
    try:
        decision=iso(r['data_segnale']);asof=iso(r['vintage_asof']);obs=iso(r['ultima_osservazione'])
    except (ValueError,TypeError):return ('ERRORE_DATI','DATE_INVALID','')
    if asof>decision or obs>decision:return ('ERRORE_DATI','FUTURE_DATE','')
    if r.get('metodo_acquisizione','').startswith('DERIVATO'):
        return ('NON_VERIFICABILE','DERIVATO_RICHIEDE_COMPONENTI_PIT','')
    if source_error:return ('NON_VERIFICABILE','ERRORE_ACQUISIZIONE','')
    candidates=[x for x in records if x.get('date')==r['ultima_osservazione'] and
        x.get('realtime_start','9999-12-31')<=r['vintage_asof']<=x.get('realtime_end','0000-01-01')]
    if len(candidates)!=1:return ('NON_VERIFICABILE','VINTAGE_NON_UNIVOCA_'+str(len(candidates)),'')
    official=candidates[0].get('value','')
    if not match(r['valore'],official):return ('ERRORE_DATI','VALORE_DIFFERENTE',official)
    # ALFRED vintage date is a day-level fact. Do not assert intraday release time.
    if asof>=decision:return ('NON_VERIFICABILE','MARGINE_TEMPORALE_INSUFFICIENTE',official)
    return ('VALIDATO_VINTAGE_GIORNALIERO','VALORE_E_VINTAGE_CONFERMATI_ALFRED',official)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--out',default='risultati_certificazione_pit_v2');p.add_argument('--no-network',action='store_true');a=p.parse_args()
    root=pathlib.Path(a.root);out=root/a.out;out.mkdir(parents=True,exist_ok=True)
    snapshots=csv_zip(root/'storico-pit-64-completo.zip','snapshot_pit_64_completo.csv')
    assoc=csv_zip(root/'macro-swing-lab-associazione-pit.zip','associazioni_snapshot.csv')
    assoc_keys={(x['serie'],x['data_segnale']) for x in assoc}
    if len(assoc_keys)!=len(assoc):raise ValueError('Duplicati associazioni')
    groups=collections.defaultdict(list)
    for r in snapshots:groups[r['serie']].append(r)
    client=Client('' if a.no_network else os.getenv('FRED_API_KEY',''),root/'.pit_cache_v2')
    results=[];issues=[];counts=collections.Counter();seen=set()
    for series,group in sorted(groups.items()):
        # Fetch only observation dates that appear in snapshots, grouped in annual windows.
        observation_dates=[iso(r['ultima_osservazione']) for r in group if r['valore'] and r['ultima_osservazione'] and not r['metodo_acquisizione'].startswith('DERIVATO')]
        source_error='';failed_years=set();records=[]
        if observation_dates:
            years=sorted(set(d.year for d in observation_dates))
            for year in years:
                ds=[d for d in observation_dates if d.year==year]
                try:records.extend(client.range(series,min(ds),max(ds)))
                except Exception as e:
                    failed_years.add(year);source_error+=f'year {year}: {str(e)[:180]}; '
        if source_error:issues.append({'serie':series,'errore':source_error})
        index=collections.defaultdict(list)
        for x in records:index[x.get('date','')].append(x)
        for r in group:
            key=(series,r['data_segnale'])
            row_year=int(r['ultima_osservazione'][:4]) if r['ultima_osservazione'] else -1
            status,reason,official=classify(r,index.get(r['ultima_osservazione'],[]),'ERRORE_ACQUISIZIONE' if row_year in failed_years else '')
            if key in seen or key not in assoc_keys:status,reason='ERRORE_DATI','DUPLICATO_O_ASSOCIAZIONE_ASSENTE'
            seen.add(key);counts[status]+=1
            results.append({'serie':series,'data_segnale':r['data_segnale'],'vintage_asof':r['vintage_asof'],
                'ultima_osservazione':r['ultima_osservazione'],'valore_snapshot':r['valore'],'valore_alfred':official,
                'stato':status,'motivo':reason,'ammesso_strict':'NO',
                'ammesso_vintage_daily':'SI' if status=='VALIDATO_VINTAGE_GIORNALIERO' else 'NO'})
        print(f'{series}: {len(group)} snapshots, records={len(records)}, error={bool(source_error)}',flush=True)
    cols=list(results[0]);csv_write(out/'esiti_snapshot.csv',results,cols)
    csv_write(out/'vintage_giornaliera_verificata.csv',[x for x in results if x['ammesso_vintage_daily']=='SI'],cols)
    csv_write(out/'strict.csv',[],cols)
    csv_write(out/'errori_acquisizione.csv',issues,['serie','errore'])
    report={'snapshot':len(results),'serie':len(groups),'stati':dict(counts),'errori_serie':len(issues),
      'richieste_api':client.requests,'cache_hits':client.hits,
      'caveat':'VALIDATO_VINTAGE_GIORNALIERO non dimostra timestamp reale di rilascio; non e certificazione STRICT. Per decisioni intraday serve prova separata. I derivati richiedono verifica componenti.'}
    (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False));print(json.dumps(report,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
