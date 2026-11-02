import pandas as pd, zipfile, io, json, pathlib
ROOT=pathlib.Path(__file__).resolve().parent
macro=pd.read_csv(ROOT/'macro_pit_consolidato.csv',parse_dates=['data_segnale','ultima_osservazione'])
with zipfile.ZipFile(ROOT/'etf-pit-52.zip') as z:
 etf=pd.read_csv(io.BytesIO(z.read('osservazioni_pit_52.csv')))
etf['data_segnale']=pd.to_datetime(etf['data_segnale'])
macro=macro.sort_values(['serie','data_segnale'])
# A new observation is not necessarily a new publication: label strictly as weekly information update.
macro['osservazione_nuova']=macro.groupby('serie')['ultima_osservazione'].transform(lambda s:s.ne(s.shift()))
macro['valore_precedente_disponibile']=macro.groupby('serie')['valore_snapshot'].shift()
macro['delta_valore']=macro['valore_snapshot']-macro['valore_precedente_disponibile']
macro['evento']=pd.NA
valid=macro['osservazione_nuova'] & macro['valore_precedente_disponibile'].notna()
macro.loc[valid & macro['delta_valore'].gt(0),'evento']='AUMENTO'
macro.loc[valid & macro['delta_valore'].lt(0),'evento']='DIMINUZIONE'
macro.loc[valid & macro['delta_valore'].eq(0),'evento']='INVARIATO'
# Explicitly exclude non-new observations and initial records, avoiding fabricated release events.
events=macro.loc[valid, ['serie','data_segnale','vintage_asof','ultima_osservazione','valore_snapshot','valore_precedente_disponibile','delta_valore','evento']]
joined=events.merge(etf,on='data_segnale',how='inner',validate='many_to_many',suffixes=('_macro','_etf'))
joined['evento_id']=joined['serie']+'|'+joined['data_segnale'].astype(str)
joined.to_csv(ROOT/'matrice_eventi_etf.csv',index=False)
summary=(joined.groupby(['serie','evento','asset','sedute'],dropna=False).agg(n=('rendimento','size'),rendimento_medio=('rendimento','mean'),rendimento_mediano=('rendimento','median'),win_rate=('rendimento',lambda x:(x>0).mean()),excess_medio=('excess_return','mean')).reset_index())
summary.to_csv(ROOT/'riepilogo_eventi_etf.csv',index=False)
report={'macro_rows':len(macro),'events_new_observation_with_prior':len(events),'etf_rows':len(etf),'joined_rows':len(joined),'groups':len(summary),'note':'Esplorativo: variazioni tra snapshot settimanali, NON sorprese macro né reazioni intraday. Prezzi ETF ereditati da v5.2, non rivalidati. Finestre sovrapposte; statistiche descrittive non inferenziali.'}
(ROOT/'report_eventi.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
print(json.dumps(report,ensure_ascii=False))
