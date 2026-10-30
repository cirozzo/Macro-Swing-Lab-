#!/usr/bin/env python3
"""Sonde PIT dinamiche: partenza dalla prima vintage disponibile della serie.
Non prova l'orario di pubblicazione e non certifica valori locali senza confronto.
"""
import argparse,csv,datetime as dt,json,os,pathlib,random,time,urllib.error,urllib.parse,urllib.request

ALL='ANFCI BAA10Y BAMLH0A0HYM2 CPIAUCSL CPILFESL DCOILWTICO DFEDTARU DFII10 DGS10 DGS2 DGS3MO DTWEXBGS FEDFUNDS GDP ICSA INDPRO JTSJOL M2SL NFCI PAYEMS PCEPI PCEPILFE PPIACO RRPONTSYD RSXFS SOFR STLFSI4 T10Y2Y T10YIE T5YIE UMCSENT UNRATE USEPUINDXD VIXCLS WALCL WRESBAL'.split()
KEY=os.getenv('FRED_API_KEY','').strip()
BASE='https://api.stlouisfed.org/fred/'

def api(endpoint,params):
    if not KEY: raise RuntimeError('Secret FRED_API_KEY mancante')
    url=BASE+endpoint+'?'+urllib.parse.urlencode({**params,'api_key':KEY,'file_type':'json'})
    for attempt in range(7):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'MacroSwingLab-DynamicPIT/1.0'}),timeout=45) as resp:
                data=json.load(resp)
            time.sleep(.35)
            return data
        except urllib.error.HTTPError as e:
            body=e.read(700).decode('utf-8','replace')
            if e.code not in (429,500,502,503,504): raise RuntimeError(f'HTTP {e.code}: {body}')
            delay=min(90,2**(attempt+1))+random.random()
            if e.headers.get('Retry-After'):
                try: delay=min(120,max(delay,float(e.headers['Retry-After'])))
                except ValueError: pass
            print(f'HTTP {e.code}: retry tra {delay:.1f}s',flush=True)
            time.sleep(delay)
        except (urllib.error.URLError,TimeoutError) as e:
            if attempt==6: raise RuntimeError(f'Errore rete: {e}')
            time.sleep(min(90,2**(attempt+1))+random.random())
    raise RuntimeError('Retry esauriti')

def weekdays_friday(start,end,step):
    day=start+dt.timedelta(days=(4-start.weekday())%7)
    while day<=end:
        yield day
        day+=dt.timedelta(days=7*step)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--serie',default='ALL')
    p.add_argument('--passo-settimane',type=int,default=4,help='1 = ogni venerdi; 4 = una sonda ogni quattro settimane')
    p.add_argument('--inizio-analisi',default='2006-01-01',help='limite minimo backtest; origine effettiva=max(prima vintage, limite)')
    p.add_argument('--fine-analisi',default=dt.date.today().isoformat())
    a=p.parse_args()
    if a.passo_settimane<1: p.error('passo-settimane deve essere >= 1')
    series=ALL if a.serie=='ALL' else [a.serie]
    if any(s not in ALL for s in series):p.error('serie sconosciuta')
    end=min(dt.date.fromisoformat(a.fine_analisi),dt.date.today())
    floor=dt.date.fromisoformat(a.inizio_analisi)
    out=pathlib.Path('verifica_alfred_dinamica');out.mkdir(exist_ok=True)
    summary=[]
    for name in series:
        record={'serie':name,'prima_vintage':'','ultima_vintage':'','vintage_totali':'','data_inizio_test':'','test_totali':0,'valore_presente':0,'senza_valore':0,'errore_http':0,'non_testati':0,'stato':'NON_ESEGUITO'}
        samples=[]
        try:
            v=api('series/vintagedates',{'series_id':name,'limit':10000})
            # vintage_dates endpoint supports 10000 results; fail closed if truncated
            total=int(v.get('count',len(v.get('vintage_dates',[]))))
            dates=v.get('vintage_dates',[])
            if total>len(dates): raise RuntimeError(f'Paginazione vintage incompleta: {len(dates)}/{total}')
            if not dates: raise RuntimeError('Nessuna vintage disponibile')
            first,last=dt.date.fromisoformat(dates[0]),dt.date.fromisoformat(dates[-1])
            start=max(first,floor)
            record.update(prima_vintage=str(first),ultima_vintage=str(last),vintage_totali=len(dates),data_inizio_test=str(start))
            for day in weekdays_friday(start,end,a.passo_settimane):
                date=str(day)
                row={'serie':name,'data_segnale':date,'stato':'','ultima_osservazione':'','valore':'','errore':''}
                try:
                    # Richiesta as-of giornaliera, esplicitamente nessun dato futuro
                    info=api('series/observations',{'series_id':name,'realtime_start':date,'realtime_end':date,'observation_end':date,'sort_order':'desc','limit':1000})
                    found=next((x for x in info.get('observations',[]) if x.get('value') not in (None,'','.','NaN') and x['date']<=date),None)
                    if found:
                        row.update(stato='PRESENTE',ultima_osservazione=found['date'],valore=found['value'])
                        record['valore_presente']+=1
                    else:
                        row['stato']='SENZA_VALORE';record['senza_valore']+=1
                except Exception as exc:
                    row.update(stato='ERRORE_API',errore=str(exc)[:300]);record['errore_http']+=1
                record['test_totali']+=1
                samples.append(row)
                # checkpoint per sopravvivere a interruzioni
                if len(samples)%30==0:
                    (out/f'{name}_checkpoint.json').write_text(json.dumps({'riepilogo':record,'sonde':samples},ensure_ascii=False),encoding='utf-8')
            record['stato']='COMPLETATO' if record['errore_http']==0 else 'COMPLETATO_CON_ERRORI'
        except Exception as exc:
            record.update(stato='ERRORE_INVENTARIO',errore=str(exc)[:500])
        with (out/f'{name}_sonde.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=['serie','data_segnale','stato','ultima_osservazione','valore','errore']);w.writeheader();w.writerows(samples)
        (out/f'{name}_riepilogo.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
        summary.append(record)
        print(json.dumps(record,ensure_ascii=False),flush=True)
    with (out/'riepilogo_complessivo.csv').open('w',newline='',encoding='utf-8-sig') as f:
        cols=list(dict.fromkeys(k for r in summary for k in r));w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(summary)
    if any(r['stato']=='ERRORE_INVENTARIO' for r in summary):raise SystemExit(2)

if __name__=='__main__':main()
