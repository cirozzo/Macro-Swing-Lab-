#!/usr/bin/env python3
"""Macro Swing MVP: regimi INDPRO/PCEPILFE e performance successive ETF.
Eseguire: pip install yfinance pandas numpy ; python backtest_macro.py
I risultati NON sono point-in-time: vedere LEGGIMI.
"""
from pathlib import Path
from datetime import datetime, timedelta
import io
import urllib.request
import pandas as pd
import numpy as np
import yfinance as yf

START = '2008-01-01'  # warmup prima dell'inizio dello studio
FIRST_SIGNAL = '2010-01-01'
END = (datetime.utcnow().date() + timedelta(days=1)).isoformat()  # end esclusivo
SPLIT = '2022-01-01'
TICKERS = ['SPY','XLK','XLE','XLF','XLI','XLB','XLV','XLP','XLY','XLU','XLRE','XLC']
HORIZONS = (5, 10, 20)
OUT = Path(__file__).resolve().parent / 'output'
OUT.mkdir(exist_ok=True)

def fred(series):
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}'
    with urllib.request.urlopen(url, timeout=40) as res:
        raw = res.read().decode('utf-8-sig')
    df = pd.read_csv(io.StringIO(raw))
    df.columns = ['Date', series]
    df['Date'] = pd.to_datetime(df['Date'])
    df[series] = pd.to_numeric(df[series], errors='coerce')
    return df.set_index('Date')[series].dropna().sort_index()

def prices_download():
    # auto_adjust=True: prezzi rettificati per split e dividendi, se forniti dal provider.
    raw = yf.download(TICKERS, start=START, end=END,
                      auto_adjust=True, progress=False, group_by='column', threads=False)
    if raw.empty:
        raise RuntimeError('Nessun prezzo recuperato: controlla connessione/provider.')
    close = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw[['Close']].rename(columns={'Close':'SPY'})
    if 'SPY' not in close.columns:
        raise RuntimeError('SPY non disponibile nei dati scaricati.')
    close = close.sort_index().loc[:, [t for t in TICKERS if t in close.columns]]
    return close

def signals(indpro, core):
    # Valori mensili: indice IP crescita a/a accelerazione su 3 mesi,
    # inflazione core PCE, confronto della variazione annualizzata degli ultimi 3 mesi
    # con i precedenti 3. Sono proxy, NON i tre indicatori ISM/payroll della fase teorica.
    months = pd.date_range('2009-01-01', pd.Timestamp.today().strftime('%Y-%m-01'), freq='MS')
    ip = indpro.copy()
    cp = core.copy()
    ip.index = ip.index.to_period('M')
    cp.index = cp.index.to_period('M')
    period = pd.period_range('2008-01', pd.Timestamp.today().strftime('%Y-%m'), freq='M')
    ip = ip.groupby(level=0).last().reindex(period)
    cp = cp.groupby(level=0).last().reindex(period)
    yoy = ip / ip.shift(12) - 1
    growth_acc = yoy - yoy.shift(3)
    # Inflazione cumulata sui 3 mesi più recenti rispetto ai 3 precedenti;
    # annualizzazione monotona, dunque basta il confronto diretto dei rapporti.
    inflation_acc = (cp / cp.shift(3)) - (cp.shift(3) / cp.shift(6))
    result = pd.DataFrame({'growth_delta': growth_acc, 'inflation_delta': inflation_acc})
    result['regime'] = np.select([
        (result.growth_delta > 0) & (result.inflation_delta < 0),
        (result.growth_delta > 0) & (result.inflation_delta > 0),
        (result.growth_delta < 0) & (result.inflation_delta < 0),
        (result.growth_delta < 0) & (result.inflation_delta > 0),
    ], ['R1 crescita+ inflazione-', 'R2 crescita+ inflazione+',
        'R3 crescita- inflazione-', 'R4 crescita- inflazione+'], default='')
    return result

def compute():
    print('Scarico INDPRO, PCEPILFE e prezzi ETF...')
    ip = fred('INDPRO')
    pce = fred('PCEPILFE')
    px = prices_download()
    reg = signals(ip, pce)
    rows = []
    # Decisioni: ultimo giorno di negoziazione del mese.
    for month_end, group in px.groupby(px.index.to_period('M')):
        trade_date = group.index[-1]
        if trade_date < pd.Timestamp(FIRST_SIGNAL) or trade_date >= pd.Timestamp(END):
            continue
        # Ritardo cautelativo fisso di DUE mesi rispetto al mese decisionale.
        # NON garantisce coerenza point-in-time in presenza di revisioni.
        macro_month = trade_date.to_period('M') - 2
        if macro_month not in reg.index:
            continue
        regime = reg.at[macro_month, 'regime']
        if not regime:
            continue
        location = px.index.get_loc(trade_date)
        for h in HORIZONS:
            next_idx = location + h
            if next_idx >= len(px):
                continue
            final_date = px.index[next_idx]
            # Non utilizzare prezzi oltre fine periodo studio.
            if final_date >= pd.Timestamp(END):
                continue
            spy0, spy1 = px.at[trade_date, 'SPY'], px.at[final_date, 'SPY']
            if not np.isfinite(spy0) or not np.isfinite(spy1):
                continue
            spyret = float(spy1 / spy0 - 1)
            for ticker in TICKERS:
                if ticker == 'SPY':
                    continue
                p0, p1 = px.at[trade_date, ticker], px.at[final_date, ticker]
                if not np.isfinite(p0) or not np.isfinite(p1) or p0 <= 0:
                    continue
                ret = float(p1 / p0 - 1)
                rows.append({'data_segnale':trade_date.date().isoformat(),
                             'mese_macro_usato': str(macro_month),
                             'regime':regime,'asset':ticker,'orizzonte_sedute':h,
                             'data_uscita':final_date.date().isoformat(),
                             'rendimento':ret,'rendimento_spy':spyret,
                             'excess_return':ret-spyret,
                             'campione':'SVILUPPO' if trade_date < pd.Timestamp(SPLIT) else 'TEST'})
    raw = pd.DataFrame(rows)
    if raw.empty:
        raise RuntimeError('Nessuna osservazione valida: controlla provider/date.')
    raw.to_csv(OUT/'osservazioni.csv', index=False, float_format='%.8f')
    results = []
    for (camp, reg_name, h, asset), group in raw.groupby(['campione','regime','orizzonte_sedute','asset']):
        ex = group.excess_return
        n = len(ex)
        results.append({'campione':camp,'regime':reg_name,'orizzonte_sedute':h,
                        'asset':asset,'n':n,
                        'rendimento_medio':group.rendimento.mean(),
                        'excess_medio':ex.mean(),
                        'excess_mediano':ex.median(),
                        'prob_sovraperformance':(ex>0).mean(),
                        'prob_rendimento_positivo':(group.rendimento>0).mean(),
                        'dev_std_excess':ex.std(ddof=1) if n>1 else np.nan})
    summary = pd.DataFrame(results).sort_values(['campione','regime','orizzonte_sedute','excess_medio'],ascending=[True,True,True,False])
    summary.to_csv(OUT/'risultati.csv', index=False, float_format='%.8f')
    print('Creati:',OUT/'osservazioni.csv', 'e',OUT/'risultati.csv')
    print('Osservazioni:',len(raw),' | gruppi:',len(summary))
    print('ATTENZIONE: macro revisionata ex-post, ETF con storie differenti; NON è un backtest point-in-time validato.')

if __name__ == '__main__':
    compute()
