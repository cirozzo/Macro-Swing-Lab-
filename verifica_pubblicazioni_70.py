#!/usr/bin/env python3
"""Macro Swing Lab 6.0 Phase 3.0: conservative publication availability gate.
No live data or dates are invented. Missing release evidence => blocked.
"""
import argparse, csv, io, json, pathlib, zipfile, collections, datetime

SNAP='snapshot_pit_64_completo.csv'
REG='registro_fonti_69.csv'

def load_csv(path, filename):
    p=pathlib.Path(path)
    if p.suffix.lower()=='.zip':
        with zipfile.ZipFile(p) as z:
            candidates=[n for n in z.namelist() if n.endswith('/'+filename) or n==filename]
            if len(candidates)!=1: raise ValueError(f'Expected one {filename}, found {candidates}')
            with z.open(candidates[0]) as f:
                return list(csv.DictReader(io.TextIOWrapper(f, encoding='utf-8-sig')))
    if p.is_dir(): p=p/filename
    with p.open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))

def parse_date(s):
    try: return datetime.date.fromisoformat(s)
    except (ValueError,TypeError): return None

def classify(r, reg):
    signal=parse_date(r.get('data_segnale'))
    asof=parse_date(r.get('vintage_asof'))
    obs=parse_date(r.get('ultima_osservazione'))
    val=(r.get('valore') or '').strip()
    state=(r.get('stato') or '').upper()
    method=(r.get('metodo_acquisizione') or '').upper()
    if not signal or not asof or asof>=signal or (obs and obs>asof):
        return 'BLOCCATO_ANOMALIA_TEMPORALE','Date invalid or look-ahead in recorded dates'
    if not val or val=='.':
        return 'BLOCCATO_MANCANTE','No numerical observation'
    try: float(val)
    except ValueError: return 'BLOCCATO_NON_NUMERICO','Invalid numerical value'
    if 'DERIV' in state or 'RICOSTR' in state or 'DERIV' in method or 'RICOSTR' in method:
        return 'BLOCCATO_DERIVATO','Derived data requires component-level independent PIT and matching observation dates'
    if reg is None:
        return 'BLOCCATO_FONTE_SCONOSCIUTA','Series absent from provenance register'
    # Date-only vintage is insufficient for a guaranteed as-of market-close timestamp.
    # Official release timestamp evidence is absent for individual observations.
    return 'IN_ATTESA_PROVA_PUBBLICAZIONE','ALFRED date-only as-of, historical release timestamp not individually evidenced'

def write_csv(path, rows, fields):
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--snapshot',required=True,help='Phase 2.4 full ZIP or extracted CSV directory')
    p.add_argument('--registro',required=True,help='Phase 2.9 ZIP or extracted CSV directory')
    p.add_argument('--out',default='output/verifica_pubblicazioni_70')
    args=p.parse_args()
    rows=load_csv(args.snapshot,SNAP); reg_rows=load_csv(args.registro,REG)
    reg={r['serie']:r for r in reg_rows}
    out=pathlib.Path(args.out);out.mkdir(parents=True,exist_ok=True)
    counts=collections.Counter();by_series=collections.defaultdict(collections.Counter)
    seen=set();records=[]
    for r in rows:
        sid=r.get('serie','');sig=r.get('data_segnale','')
        key=(sid,sig)
        status,reason=classify(r,reg.get(sid))
        if key in seen: status,reason='BLOCCATO_DUPLICATO','Duplicate series/signal pair'
        seen.add(key)
        counts[status]+=1;by_series[sid][status]+=1
        records.append({'serie':sid,'data_segnale':sig,'vintage_asof':r.get('vintage_asof',''),
                        'ultima_osservazione':r.get('ultima_osservazione',''),'stato_origine':r.get('stato',''),
                        'esito_gate':status,'motivazione':reason,'ammesso_backtest':'NO'})
    write_csv(out/'gate_snapshot_70.csv',records,list(records[0]) if records else ['serie','data_segnale','esito_gate'])
    diag=[]
    for sid,c in sorted(by_series.items()):
        diag.append({'serie':sid,'totale':sum(c.values()),'ammessi':0,
                     'in_attesa_prova_pubblicazione':c['IN_ATTESA_PROVA_PUBBLICAZIONE'],
                     'mancanti':c['BLOCCATO_MANCANTE'],
                     'derivati':c['BLOCCATO_DERIVATO'],
                     'altri_blocchi':sum(v for k,v in c.items() if k not in ('IN_ATTESA_PROVA_PUBBLICAZIONE','BLOCCATO_MANCANTE','BLOCCATO_DERIVATO'))})
    write_csv(out/'diagnostica_serie_70.csv',diag,list(diag[0]) if diag else ['serie'])
    report={'version':'3.0','snapshot_totali':len(rows),'serie':len(by_series),
            'ammessi_backtest':0,'classificazioni':dict(sorted(counts.items())),
            'principio':'Fail closed: ALFRED date-only as-of is not timestamp-level release certification',
            'evidenze_esterne_acquisite_durante_esecuzione':0,
            'prossima_azione':'Join with historically archived official release events and independently audit derived components'}
    (out/'report_70.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
