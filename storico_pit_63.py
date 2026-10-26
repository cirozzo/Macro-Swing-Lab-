#!/usr/bin/env python3
"""Macro Swing Lab 6.0, fase 2.3: ALFRED PIT with explicit as-of fallbacks.

Output is NOT automatically admissible to a live or historical trade backtest:
ALFRED day-level vintages do not establish exact publication timestamps.
"""
import argparse
import csv
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import pandas as pd

VINTAGE_URL = 'https://api.stlouisfed.org/fred/series/vintagedates'
COMPONENTS = {'T10YIE': ('DGS10','DFII10'), 'T5YIE': ('DGS5','DFII5')}
URL = 'https://api.stlouisfed.org/fred/series/observations'
FREQ_LAGS = {'quotidiana':(5,20,60),'settimanale':(1,4,13),
             'mensile':(1,3,12),'trimestrale':(1,2,4)}
UNAVAILABLE = {'GOLDAMGBD228NLBM'}


def api(series, key, params, attempts=7):
    args = dict(series_id=series, api_key=key, file_type='json',
                observation_start='2000-01-01', limit=100000)
    args.update(params)
    if params.get('_vintage_endpoint'):
        args.pop('observation_start',None)
    url = (VINTAGE_URL if params.get('_vintage_endpoint') else URL) + '?' + urlencode({k:v for k,v in args.items() if not k.startswith('_')})
    for attempt in range(attempts):
        try:
            with urlopen(Request(url, headers={'User-Agent':'MacroSwingLab/6.0-Fase2.2'}), timeout=90) as response:
                payload = json.load(response)
            if params.get('_vintage_endpoint'):
                return payload.get('vintage_dates', [])
            observations = payload.get('observations')
            if observations is None:
                raise ValueError(f'{series}: invalid API response (no observations)')
            if int(payload.get('count',len(observations))) > len(observations):
                raise ValueError(f'{series}: API pagination detected; decrease batch size')
            return observations
        except HTTPError as exc:
            if exc.code not in (429,500,502,503,504) or attempt == attempts-1:
                body=exc.read(1600).decode('utf-8','replace')
                try:
                    detail=json.loads(body).get('error_message',json.loads(body).get('message',body))
                except (ValueError,TypeError):
                    detail=body
                raise RuntimeError(f'{series}: HTTP {exc.code}: {str(detail)[:350]}') from None
            retry = exc.headers.get('Retry-After')
            try:
                delay = min(120,float(retry)) if retry else min(120,5*2**attempt)
            except (ValueError,TypeError):
                delay = min(120,5*2**attempt)
        except (URLError,TimeoutError) as exc:
            if attempt == attempts-1:
                raise RuntimeError(f'{series}: network {exc}') from None
            delay = min(120,5*2**attempt)
        print(f'Retry {series}: {attempt+1}/{attempts}, {delay:.1f}s',flush=True)
        time.sleep(delay + random.uniform(0,1))
    raise AssertionError('unreachable')


def parse_values(observations, series, col, asof):
    if not observations:
        return pd.Series(dtype=float)
    df = pd.DataFrame(observations)
    if 'date' not in df.columns or col not in df.columns:
        raise ValueError(f'{series}: missing date/value field for vintage {asof}')
    dates = pd.to_datetime(df['date'],errors='raise')
    if dates.duplicated().any():
        raise ValueError(f'{series}: duplicated observation dates')
    numeric = pd.to_numeric(df[col],errors='coerce')
    values = pd.Series(numeric.to_numpy(),index=dates).dropna().sort_index()
    # Defensive bound: an observation dated in the future is not permitted.
    return values.loc[values.index <= pd.Timestamp(asof)]


def batch_unpack(observations, series, vintage_dates):
    if not observations:
        return {},list(vintage_dates)
    df = pd.DataFrame(observations)
    if 'date' not in df.columns:
        raise ValueError(f'{series}: no observation date')
    got,missing = {},[]
    for vintage in vintage_dates:
        variants = (f'{series}_{vintage.replace("-", "")}',vintage,vintage.replace('-',''))
        col = next((x for x in variants if x in df.columns),None)
        if col is None:
            missing.append(vintage)
        else:
            got[vintage] = parse_values(observations,series,col,vintage)
    return got,missing


def fetch_exact(series, vintage, key):
    """Explicit point-in-time request, NEVER the latest revised series."""
    rows = api(series,key,dict(output_type=1,realtime_start=vintage,
                               realtime_end=vintage,observation_end=vintage))
    return parse_values(rows,series,'value',vintage)


def first_vintage(series, key):
    """First historically recorded update, not an observation-start date."""
    dates=api(series,key,{'_vintage_endpoint':True, 'limit':1,
                          'sort_order':'asc'})
    return dates[0] if dates else None


def derived_breakeven(series, vintage, key):
    """Return a derived snapshot only when BOTH components are available as-of.

    The components are acquired independently and never backfilled with current data.
    If a component is inaccessible, the derived value remains unavailable.
    """
    positive, negative = COMPONENTS[series]
    a=fetch_exact(positive,vintage,key)
    b=fetch_exact(negative,vintage,key)
    common=a.index.intersection(b.index)
    if common.empty:
        raise ValueError(f'{series}: no contemporaneous nominal and real yields as of {vintage}')
    return (a.loc[common]-b.loc[common]).sort_index()


def snapshot(entry, signal, vintage, series):
    freq=entry['frequenza_prevista']
    result={'id':entry['id'],'serie':entry['serie'],'frequenza_prevista':freq,
            'data_segnale':signal,'vintage_asof':vintage,'valore':None,
            'ultima_osservazione':None,'n_osservazioni_asof':len(series),
            'differenza_breve':None,'differenza_media':None,'differenza_lunga':None,
            'stato':'NESSUN_DATO_ASOF'}
    if series.empty:
        return result
    result.update(valore=float(series.iloc[-1]),
                  ultima_osservazione=series.index[-1].strftime('%Y-%m-%d'),
                  stato='SNAPSHOT_PIT_DA_VERIFICARE')
    for name,lag in zip(('differenza_breve','differenza_media','differenza_lunga'),
                        FREQ_LAGS.get(freq,(1,3,12))):
        if len(series)>lag:
            result[name]=float(series.iloc[-1]-series.iloc[-lag-1])
    return result


def signals_between(start, end, count):
    fridays = pd.date_range(start,end,freq='W-FRI')
    # Exclude current/very recent signals conservatively.
    fridays = fridays[fridays <= pd.Timestamp.now(tz='UTC').tz_localize(None)-pd.Timedelta(days=2)]
    if count and len(fridays)>count:
        ids = sorted(set(round(i*(len(fridays)-1)/(count-1)) for i in range(count))) if count>1 else [len(fridays)-1]
        fridays=fridays[ids]
    return [(d.strftime('%Y-%m-%d'),(d-pd.Timedelta(days=1)).strftime('%Y-%m-%d')) for d in fridays]


def offline_test():
    vintages=['2007-01-04','2014-10-02']
    obs=[{'date':'2006-12-01','T10YIE_20141002':'2.1'},
         {'date':'2014-09-30','T10YIE_20141002':'2.2'}]
    got,missing=batch_unpack(obs,'T10YIE',vintages)
    assert missing==['2007-01-04']
    assert len(got['2014-10-02'])==2
    exact=parse_values([{'date':'2006-12-01','value':'1.7'},
                        {'date':'2007-01-05','value':'9.9'}],'T10YIE','value','2007-01-04')
    assert len(exact)==1 and exact.iloc[-1]==1.7
    assert snapshot({'id':'A','serie':'T10YIE','frequenza_prevista':'quotidiana'},
                    '2007-01-05','2007-01-04',exact)['valore']==1.7
    assert len(parse_values([],'T10YIE','value','2007-01-04'))==0
    print('PASS: batch missing vintage, exact as-of parsing, date guard, no-data',flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--catalog',default='catalogo_fonti_60.csv')
    p.add_argument('--out',default='output/storico_pit_63')
    p.add_argument('--start',default='2007-01-01')
    p.add_argument('--end',default=datetime.now(timezone.utc).strftime('%Y-%m-%d'))
    p.add_argument('--max-series',type=int,default=12)
    p.add_argument('--max-signals',type=int,default=24)
    p.add_argument('--batch-size',type=int,default=12)
    p.add_argument('--pacing',type=float,default=1.5)
    p.add_argument('--offline-test',action='store_true')
    p.add_argument('--audit-samples',type=int,default=2,help='Independent exact-asof checks per series')
    p.add_argument('--derive-breakeven',action='store_true',help='Try PIT nominal minus real yields when breakeven PIT unavailable')
    a=p.parse_args()
    if a.offline_test:
        offline_test();return
    if a.max_series<0 or a.max_signals<0 or a.pacing<0 or not 1<=a.batch_size<=20:
        raise ValueError('Invalid arguments')
    key=os.getenv('FRED_API_KEY','').strip()
    if not key: raise RuntimeError('FRED_API_KEY secret missing')
    with open(a.catalog,encoding='utf-8-sig',newline='') as file:
        catalog=list(csv.DictReader(file))
    fred=[r for r in catalog if r.get('fonte')=='FRED']
    assert len({r['serie'] for r in fred})==len(fred),'Repeated FRED series IDs'
    active=[r for r in fred if r['serie'] not in UNAVAILABLE]
    selected=active[:a.max_series] if a.max_series else active
    dates=signals_between(a.start,a.end,a.max_signals)
    if not dates: raise ValueError('No signal dates')
    output=Path(a.out);output.mkdir(parents=True,exist_ok=True)
    rows=[];audit=[];fallbacks=[];metadata=[];independent=[]
    vintage_bounds={}
    for sid in COMPONENTS:
        if sid in {e['serie'] for e in selected}:
            try:
                vintage_bounds[sid]=first_vintage(sid,key)
                metadata.append({'serie':sid,'prima_vintage_registrata':vintage_bounds[sid], 'errore':''})
            except Exception as exc:
                vintage_bounds[sid]=None
                metadata.append({'serie':sid,'prima_vintage_registrata':'','errore':str(exc)[:350]})
    for ix,entry in enumerate(selected,1):
        sid=entry['serie']; values={};methods={};errors={};batches=0
        for start in range(0,len(dates),a.batch_size):
            vintages=[v for _,v in dates[start:start+a.batch_size]]
            try:
                observations=api(sid,key,dict(output_type=2,vintage_dates=','.join(vintages),
                                               observation_end=vintages[-1]))
                got,missing=batch_unpack(observations,sid,vintages)
                values.update(got)
                methods.update({d:'BATCH_ASOF' for d in got})
                batches+=1
            except Exception as exc:
                print(f'WARNING {sid} batch: {str(exc)[:160]}',flush=True)
                missing=vintages
            # A batch without a vintage column is NOT license to use a later vintage.
            for vintage in missing:
                bound=vintage_bounds.get(sid)
                if bound and vintage < bound:
                    methods[vintage]='PRECEDENTE_PRIMA_VINTAGE'
                    errors[vintage]=f'First registered ALFRED vintage: {bound}'
                    if a.derive_breakeven and sid in COMPONENTS:
                        try:
                            values[vintage]=derived_breakeven(sid,vintage,key)
                            methods[vintage]='DERIVATO_COMPONENTI_ASOF'
                            errors.pop(vintage,None)
                        except Exception as exc:
                            errors[vintage]+=f'; derived sources unavailable: {str(exc)[:240]}'
                    fallbacks.append({'serie':sid,'vintage_asof':vintage,'stato':methods[vintage],
                                      'n_osservazioni':len(values[vintage]) if vintage in values else 0,
                                      'errore':errors.get(vintage,'')})
                    time.sleep(a.pacing)
                    continue
                try:
                    vals=fetch_exact(sid,vintage,key)
                    values[vintage]=vals
                    methods[vintage]='EXACT_ASOF' if not vals.empty else 'EXACT_ASOF_EMPTY'
                    fallbacks.append({'serie':sid,'vintage_asof':vintage,
                                      'stato':methods[vintage],'n_osservazioni':len(vals),'errore':''})
                except Exception as exc:
                    methods[vintage]='ERRORE_VINTAGE'
                    errors[vintage]=str(exc)[:350]
                    if a.derive_breakeven and sid in COMPONENTS:
                        try:
                            values[vintage]=derived_breakeven(sid,vintage,key)
                            methods[vintage]='DERIVATO_COMPONENTI_ASOF'
                            errors.pop(vintage,None)
                        except Exception as derived_exc:
                            errors[vintage]+=f'; derived sources unavailable: {str(derived_exc)[:220]}'
                    fallbacks.append({'serie':sid,'vintage_asof':vintage,
                                      'stato':methods[vintage],'n_osservazioni':len(values[vintage]) if vintage in values else 0,'errore':errors.get(vintage,'')})
                time.sleep(a.pacing)
            time.sleep(a.pacing)
        # Independent exact-asof API calls, separate from batched output_type=2.
        # Compare the latest observation and its date, not just the numeric value.
        valid_vintages=[v for _,v in dates if v in values and not values[v].empty and methods.get(v)=='BATCH_ASOF']
        sample_indices=sorted(set(round(i*(len(valid_vintages)-1)/max(1,a.audit_samples-1))
                                  for i in range(min(a.audit_samples,len(valid_vintages))))) if valid_vintages else []
        for sample_index in sample_indices:
            vintage=valid_vintages[sample_index]
            original=values[vintage]
            try:
                direct=fetch_exact(sid,vintage,key)
                ok=(not direct.empty and direct.index[-1]==original.index[-1] and
                    abs(float(direct.iloc[-1])-float(original.iloc[-1]))<=1e-8)
                independent.append({'serie':sid,'vintage_asof':vintage,'esito':'PASS' if ok else 'MISMATCH',
                    'data_batch':original.index[-1].strftime('%Y-%m-%d'),
                    'data_diretta':direct.index[-1].strftime('%Y-%m-%d') if not direct.empty else '',
                    'valore_batch':float(original.iloc[-1]),
                    'valore_diretto':float(direct.iloc[-1]) if not direct.empty else None,'errore':''})
            except Exception as exc:
                independent.append({'serie':sid,'vintage_asof':vintage,'esito':'ERROR',
                    'data_batch':original.index[-1].strftime('%Y-%m-%d'),'data_diretta':'',
                    'valore_batch':float(original.iloc[-1]),'valore_diretto':None,'errore':str(exc)[:350]})
            time.sleep(a.pacing)
        for signal,vintage in dates:
            if vintage not in values:
                row=snapshot(entry,signal,vintage,pd.Series(dtype=float))
                row['stato']='ERRORE_ACQUISIZIONE'
            else:
                row=snapshot(entry,signal,vintage,values[vintage])
            row['metodo_acquisizione']=methods.get(vintage,'ERRORE_VINTAGE')
            if row['metodo_acquisizione']=='DERIVATO_COMPONENTI_ASOF' and row['valore'] is not None:
                row['stato']='DERIVATO_PIT_DA_VERIFICARE'
            if row['metodo_acquisizione']=='PRECEDENTE_PRIMA_VINTAGE':
                row['stato']='NON_DISPONIBILE_PRIMA_VINTAGE'
            row['errore']=errors.get(vintage,'')
            rows.append(row)
        n_ok=sum(v in values and not values[v].empty for _,v in dates)
        n_empty=sum(v in values and values[v].empty for _,v in dates)
        n_fail=len(dates)-n_ok-n_empty
        audit.append({'id':entry['id'],'serie':sid,
                      'stato':'OK' if n_fail==0 and n_empty==0 else 'PARZIALE' if n_ok else 'NESSUN_DATO',
                      'segnali_attesi':len(dates),'snapshot_con_dati':n_ok,
                      'snapshot_vuoti':n_empty,'snapshot_errori':n_fail,
                      'fallback_asof':sum(methods.get(v,'').startswith('EXACT') for _,v in dates),
                      'batch_ok':batches,'copertura_pct':round(100*n_ok/len(dates),2)})
        print(f'[{ix}/{len(selected)}] {sid}: {n_ok}/{len(dates)} snapshots, {n_fail} errors',flush=True)
    for sid in sorted(UNAVAILABLE.intersection({r['serie'] for r in fred})):
        audit.append({'id':'GOLD','serie':sid,'stato':'FONTE_NON_DISPONIBILE',
                      'segnali_attesi':len(dates),'snapshot_con_dati':0,
                      'snapshot_vuoti':0,'snapshot_errori':0,'fallback_asof':0,
                      'batch_ok':0,'copertura_pct':0})
    frame=pd.DataFrame(rows)
    if not frame.empty:
        assert not frame.duplicated(['serie','data_segnale']).any()
        assert (pd.to_datetime(frame['vintage_asof'])<pd.to_datetime(frame['data_segnale'])).all()
        valid=frame['ultima_osservazione'].notna()
        assert (pd.to_datetime(frame.loc[valid,'ultima_osservazione']) <=
                pd.to_datetime(frame.loc[valid,'vintage_asof'])).all()
    frame.to_csv(output/'snapshot_pit_62.csv',index=False)
    pd.DataFrame(audit).to_csv(output/'audit_acquisizione_62.csv',index=False)
    pd.DataFrame(metadata,columns=['serie','prima_vintage_registrata','errore']).to_csv(output/'prima_vintage_62.csv',index=False)
    pd.DataFrame(fallbacks,columns=['serie','vintage_asof','stato','n_osservazioni','errore']).to_csv(
        output/'dettaglio_fallback_62.csv',index=False)
    report={'versione':'6.0_fase2.3','generato_utc':datetime.now(timezone.utc).isoformat(),
            'serie_selezionate':len(selected),'segnali_per_serie':len(dates),'snapshot':len(frame),
            'snapshot_con_dati':int(frame['valore'].notna().sum()),
            'snapshot_errori':int(frame['stato'].eq('ERRORE_ACQUISIZIONE').sum()),
            'snapshot_prima_vintage':int(frame['stato'].eq('NON_DISPONIBILE_PRIMA_VINTAGE').sum()),
            'snapshot_derivati':int(frame['stato'].eq('DERIVATO_PIT_DA_VERIFICARE').sum()),
            'snapshot_vuoti':int(frame['stato'].eq('NESSUN_DATO_ASOF').sum()),
            'fallback_tentati':len(fallbacks),'dati_ammessi_al_backtest':0,
            'audit_indipendente_controlli':len(independent),
            'audit_indipendente_pass':sum(x['esito']=='PASS' for x in independent),
            'audit_indipendente_mismatch':sum(x['esito']=='MISMATCH' for x in independent),
            'audit_indipendente_error':sum(x['esito']=='ERROR' for x in independent),
            'avvertenze':['The first vintage registered in ALFRED is NOT proof of the first real-world publication date',
                          'Derived breakevens are proxies and not identical to the quoted breakeven series',
                          'No historical completeness guaranteed for the four component series',
                          'ALFRED day-level as-of is not an intraday timestamp audit',
                          'Fallback uses exact realtime_start=realtime_end: no future revision',
                          'Empty vintages are retained; no forward fills from future values',
                          'No PIT independent verification of each indicator performed',
                          'Differences are level differences, not necessarily percentage changes']}
    (output/'report_pit_62.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)
    if any(x['esito']!='PASS' for x in independent):
        raise RuntimeError('Independent ALFRED checks failed; inspect audit_indipendente_63.csv')
    if frame.empty or not frame['valore'].notna().any():
        raise RuntimeError('No snapshots with values were acquired')

if __name__=='__main__':main()
