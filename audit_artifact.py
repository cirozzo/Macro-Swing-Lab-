#!/usr/bin/env python3
"""Inventario streaming di un artifact GitHub, anche con ZIP interni.
Non associa dati PIT, non elimina i grezzi: produce un campione e un catalogo.
"""
import argparse, csv, io, json, os, pathlib, random, shutil, tempfile, zipfile, traceback
from collections import Counter
from datetime import datetime, timezone

KEYWORDS = ('date','time','timestamp','asof','vintage','release','publication','published','period','observation','effective','signal','asset','ticker','symbol','regime','cot','gpr','geo','close','price','return')

def safe_text(v, lim=150):
    s = str(v).replace('\x00','')
    return s[:lim] + ('…' if len(s)>lim else '')

def json_dump(obj, path):
    with open(path,'w',encoding='utf-8') as f: json.dump(obj,f,ensure_ascii=False,indent=2,default=str)

def classify(name, cols):
    s=(' '.join([name]+list(cols))).lower()
    groups=[]
    for label,words in [('geopolitica',('gpr','gdelt','geopolit','conflict','war_')),('cot',('cot','commitment','noncommercial','managed_money')),('macro',('macro','fred','cpi','nfp','gdp','pmi','inflation','unemployment','regime')),('mercati',('ohlc','open','high','low','close','ticker','etf','price','rendiment','returns'))]:
        if any(w in s for w in words):groups.append(label)
    return ','.join(groups) or 'non_classificato'

def read_csv_stream(handle, name, max_sample, max_rows):
    rows=0; sample=[]; nulls=Counter(); seen=Counter(); errors=0; headers=[]
    text=io.TextIOWrapper(handle,encoding='utf-8-sig',errors='replace',newline='')
    try:
        reader=csv.reader(text)
        try:headers=[safe_text(x,150) for x in next(reader)]
        except StopIteration:return dict(rows=0,columns=[],sample=[],note='CSV vuoto')
        focus=[i for i,c in enumerate(headers) if any(k in c.lower() for k in KEYWORDS)][:24]
        focus_extrema={str(headers[i]):{'first':None,'last':None,'min_lex':None,'max_lex':None} for i in focus}
        rng=random.Random(20261111)
        for row in reader:
            rows+=1
            if len(row)!=len(headers):errors+=1
            if rows<=max_sample:sample.append({headers[i]:safe_text(row[i],300) for i in range(min(len(row),len(headers)))})
            elif max_sample and rng.randrange(rows)<max_sample:
                sample[rng.randrange(max_sample)]={headers[i]:safe_text(row[i],300) for i in range(min(len(row),len(headers)))}
            for i in focus:
                key=headers[i]
                v=row[i].strip() if i<len(row) else ''
                if not v:nulls[key]+=1;continue
                seen[key]+=1
                a=focus_extrema[key]
                if a['first'] is None:a['first']=safe_text(v)
                a['last']=safe_text(v)
                # Min/max lessicale sono SOLO diagnostici: non equivalgono a min/max cronologici.
                if a['min_lex'] is None or v<a['min_lex']:a['min_lex']=safe_text(v)
                if a['max_lex'] is None or v>a['max_lex']:a['max_lex']=safe_text(v)
            if max_rows and rows>=max_rows:break
    finally:text.detach()
    return dict(rows=rows,columns=headers,selected_columns=focus_extrema,missing_selected=dict(nulls),nonempty_selected=dict(seen),malformed_rows=errors,sample=sample,limited=bool(max_rows and rows>=max_rows))

def audit_regular(path, name, results, samples, max_sample, max_rows, depth=0):
    suffix=pathlib.Path(name).suffix.lower()
    size=path.stat().st_size
    if suffix=='.zip' and depth<3:
        try:
            with zipfile.ZipFile(path) as z:
                for entry in z.infolist():
                    if entry.is_dir():continue
                    ename=name+'!/'+entry.filename
                    if entry.file_size>12_000_000_000:
                        results.append(dict(file=ename,bytes=entry.file_size,status='skip_oversize'));continue
                    if entry.filename.lower().endswith('.zip') and depth<2:
                        with z.open(entry) as src, tempfile.NamedTemporaryFile(suffix='.zip') as tmp:
                            shutil.copyfileobj(src,tmp,1024*1024);tmp.flush()
                            audit_regular(pathlib.Path(tmp.name),ename,results,samples,max_sample,max_rows,depth+1)
                    elif entry.filename.lower().endswith(('.csv','.tsv')):
                        try:
                            with z.open(entry) as f: handle_csv(f,ename,entry.file_size,results,samples,max_sample,max_rows)
                        except Exception as e:results.append(dict(file=ename,bytes=entry.file_size,status='error',error=str(e)))
                    else:results.append(dict(file=ename,bytes=entry.file_size,status='not_csv'))
        except Exception as e:results.append(dict(file=name,bytes=size,status='invalid_zip',error=str(e)))
    elif suffix in ('.csv','.tsv'):
        with open(path,'rb') as f:handle_csv(f,name,size,results,samples,max_sample,max_rows)
    else:results.append(dict(file=name,bytes=size,status='not_csv'))

def handle_csv(f,name,size,results,samples,max_sample,max_rows):
    print('ANALISI',name,'bytes',size,flush=True)
    stats=read_csv_stream(f,name,max_sample,max_rows)
    sample=stats.pop('sample')
    samples.append({'file':name,'records':sample})
    results.append(dict(file=name,bytes=size,status='ok',category=classify(name,stats['columns']),**stats))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--samples',type=int,default=12)
    ap.add_argument('--max-rows-per-file',type=int,default=0,help='0 = scansione integrale dei CSV')
    args=ap.parse_args();root=pathlib.Path(args.root);out=pathlib.Path(args.output);out.mkdir(parents=True,exist_ok=True)
    results=[];samples=[]
    paths=sorted(p for p in root.rglob('*') if p.is_file())
    for p in paths:
        try:audit_regular(p,str(p.relative_to(root)),results,samples,args.samples,args.max_rows_per_file)
        except Exception as e:results.append(dict(file=str(p.relative_to(root)),bytes=p.stat().st_size,status='error',error=str(e)))
    json_dump({'created_utc':datetime.now(timezone.utc).isoformat(),'source_files':len(paths),'entries':len(results),'files':results},out/'catalogo_schema.json')
    json_dump(samples,out/'campioni.json')
    with open(out/'inventario.csv','w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['file','categoria','dimensione_bytes','righe','n_colonne','colonne','status','errore'])
        for r in results:w.writerow([r.get('file'),r.get('category'),r.get('bytes'),r.get('rows',''),len(r.get('columns',[])),' | '.join(r.get('columns',[])),r['status'],r.get('error','')])
    with open(out/'LEGGIMI.txt','w',encoding='utf-8') as f:
        f.write('MACRO SWING LAB — AUDIT ARTIFACT\n')
        f.write('inventario.csv: elenco sorgenti; catalogo_schema.json: colonne, conteggi, controlli diagnostici; campioni.json: righe d’esempio.\n')
        f.write('Non sono stati calcolati segnali, rendimenti, correlazioni o associazioni PIT.\n')
        f.write('Gli estremi min_lex/max_lex sono lessicali, NON date interpretate.\n')
        f.write('I campioni sono diagnostici e non sono un campionamento statistico dei rendimenti.\n')
        f.write('La scansione è integrale se max-rows-per-file=0; file non CSV e ZIP profondi sono solo inventariati.\n')
    print('COMPLETATO:',len(paths),'file su disco;',len(results),'elementi analizzati',flush=True)

if __name__=='__main__':main()
