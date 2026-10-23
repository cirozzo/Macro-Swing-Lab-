#!/usr/bin/env python3
"""Macro Swing Lab 5.0: audit indipendente del CSV PIT 4.9.
Non certifica i valori ALFRED senza confronto con le risposte API originali.
"""
import argparse
import json
from pathlib import Path
import pandas as pd

EXPECTED = {
    'R1_CRESCITA_DISINFLAZIONE': (True, False),
    'R2_CRESCITA_INFLAZIONE': (True, True),
    'R3_RALLENTAMENTO_INFLAZIONE': (False, True),
    'R4_RALLENTAMENTO_DISINFLAZIONE': (False, False),
}
REQUIRED = ['data_segnale','vintage_asof','growth_change','inflation_change',
            'growth_last_observation','inflation_last_observation','regime_pit_49']

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='input/regimi_pit_49.csv')
    p.add_argument('--output', default='output/validazione_pit_50')
    args = p.parse_args()
    dest = Path(args.output); dest.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    missing = set(REQUIRED) - set(df.columns)
    if missing: raise ValueError(f'Colonne mancanti: {sorted(missing)}')
    for col in ['data_segnale','vintage_asof','growth_last_observation','inflation_last_observation']:
        df[col] = pd.to_datetime(df[col], errors='raise')
    for col in ['growth_change','inflation_change']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.sort_values('data_segnale').reset_index(drop=True)
    issues=[]
    def check(label, ok):
        if not ok: issues.append(label)
    check('Date segnale duplicate', df.data_segnale.is_unique)
    check('Date segnale non venerdi', (df.data_segnale.dt.dayofweek == 4).all())
    check('Gap settimanali', (df.data_segnale.diff().dropna().dt.days == 7).all())
    check('Vintage non precedenti al segnale', (df.vintage_asof < df.data_segnale).all())
    check('Vintage non corrispondente al giovedi', (df.vintage_asof == df.data_segnale - pd.Timedelta(days=1)).all())
    check('Osservazione INDPRO futura', (df.growth_last_observation <= df.vintage_asof).all())
    check('Osservazione PCEPILFE futura', (df.inflation_last_observation <= df.vintage_asof).all())
    check('Valori numerici non finiti', df[['growth_change','inflation_change']].notna().all().all())
    check('Regimi sconosciuti', df.regime_pit_49.isin(EXPECTED).all())
    def expected(row):
        g,i = row.growth_change, row.inflation_change
        return ('R1_CRESCITA_DISINFLAZIONE' if g>=0 and i<0 else
                'R2_CRESCITA_INFLAZIONE' if g>=0 and i>=0 else
                'R3_RALLENTAMENTO_INFLAZIONE' if g<0 and i>=0 else
                'R4_RALLENTAMENTO_DISINFLAZIONE')
    check('Classificazioni incoerenti con segni', (df.apply(expected,axis=1) == df.regime_pit_49).all())
    groups = df.regime_pit_49.ne(df.regime_pit_49.shift()).cumsum()
    spells = df.groupby(groups).agg(regime=('regime_pit_49','first'),inizio=('data_segnale','first'),
                                     fine=('data_segnale','last'),settimane=('data_segnale','size')).reset_index(drop=True)
    spells.to_csv(dest/'durata_regimi.csv',index=False)
    counts = df.regime_pit_49.value_counts().rename_axis('regime').reset_index(name='settimane')
    counts['percentuale'] = (counts.settimane/len(df)*100).round(2)
    counts.to_csv(dest/'distribuzione_regimi.csv',index=False)
    # Transizioni settimanali: frequenze, non osservazioni indipendenti
    transitions = pd.crosstab(df.regime_pit_49.shift().iloc[1:],df.regime_pit_49.iloc[1:],
                               rownames=['da'],colnames=['a'])
    transitions.to_csv(dest/'transizioni_regimi.csv')
    summary = {
        'status': 'FAIL' if issues else 'PASS_STRUTTURALE',
        'righe':len(df),'prima_data':df.data_segnale.min().strftime('%Y-%m-%d'),
        'ultima_data':df.data_segnale.max().strftime('%Y-%m-%d'),
        'cambi_regime':int(len(spells)-1),
        'durata_mediana_settimane':float(spells.settimane.median()),
        'issues':issues,
        'limitazioni':['Non verifica i valori contro vintage ALFRED originali',
                       'Non prova disponibilita intraday alla apertura',
                       'Settimane contigue non sono osservazioni statisticamente indipendenti',
                       'Nessuna analisi ETF o performance fuori campione eseguita']
    }
    (dest/'report.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    if issues: raise SystemExit(1)

if __name__=='__main__': main()
