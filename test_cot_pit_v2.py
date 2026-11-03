import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location('cot',str(Path(__file__).with_name('cot_pit_etf.py')))
cot=importlib.util.module_from_spec(spec);spec.loader.exec_module(cot)

def raw(kind):
    dates=pd.date_range('2024-06-04',periods=78,freq='W-TUE')
    if kind=='TFF':
        cols=('asset_mgr_positions_long','asset_mgr_positions_short','lev_money_positions_long','lev_money_positions_short')
    else:
        cols=('m_money_positions_long_all','m_money_positions_short_all')
    rows=[]
    for j,d in enumerate(dates):
        rec={'id':str(j), 'report_date_as_yyyy_mm_dd':str(d.date()),'market_and_exchange_names':'TEST FUTURE',
             'cftc_contract_market_code':'TEST001','open_interest_all':'10000'}
        for idx,c in enumerate(cols):rec[c]=str(1000+idx*100+j*(idx+2)+int(j*j/(idx+3)))
        rows.append(rec)
    return pd.DataFrame(rows)

def main():
  with TemporaryDirectory() as td:
    td=Path(td);(td/'cot').mkdir()
    for kind in ('TFF','DISAGG'):raw(kind).to_csv(td/'cot'/f'{kind}.csv',index=False)
    with zipfile.ZipFile(td/'etf.zip','w') as z:
      z.writestr('osservazioni_pit_52.csv','asset,data_segnale,data_ingresso,data_uscita,sedute,rendimento,excess_return\nQQQ,2025-06-06,2025-06-09,2025-06-13,5,0.01,0.00\n')
    import subprocess,sys
    cp=subprocess.run([sys.executable,str(Path(__file__).with_name('cot_pit_etf.py')),'--etf-zip',str(td/'etf.zip'),'--cot-dir',str(td/'cot'),'--out',str(td/'out')],capture_output=True,text=True)
    assert cp.returncode==0,cp.stderr
    df=pd.read_csv(td/'out'/'cot_features_pit_proxy.csv')
    rep=json.loads((td/'out'/'report.json').read_text())
    assert rep['TFF_feature_rows']==156,rep
    assert rep['DISAGG_feature_rows']==78,rep
    assert set(df.category)=={'asset_manager','leveraged','managed_money'}
    assert df.prior_percentile.notna().sum()>0
    assert df.z_position.notna().sum()>0
    special=df[df.report_date=='2025-09-30']
    assert len(special)==3
    assert special.release_date_assumed.eq('2025-11-19').all()
    assert special.usable_from.eq('2025-11-20').all()
    assert special.strict_pit_eligible.eq(False).all()
    schedule=cot.lookup_release_date(pd.Timestamp('2026-06-16'),{})
    assert schedule[0]==pd.Timestamp('2026-06-22'),schedule
    assert cot.first_weekday_after(schedule[0])==pd.Timestamp('2026-06-23')
    # Perturb a future value: earlier features should be unchanged (no look-ahead).
    t=raw('TFF'); modified=t.copy();modified.loc[modified.index[-1],'asset_mgr_positions_long']='900000'
    a=cot.features_from_raw(t,'TFF');b=cot.features_from_raw(modified,'TFF')
    c=['prior_mean','prior_sd','z_position','prior_percentile']
    pd.testing.assert_frame_equal(a[a.report_date<a.report_date.max()][c].reset_index(drop=True),b[b.report_date<b.report_date.max()][c].reset_index(drop=True))
    # Unrecognized schema must FAIL, not silently skip category.
    broken=t.drop(columns=['lev_money_positions_long'])
    try:cot.features_from_raw(broken,'TFF')
    except ValueError as e:assert 'leveraged_long' in str(e)
    else:raise AssertionError('Schema mismatch not rejected')
    print('PASS: TFF, DISAGG, percentili PIT, blocco lookahead, calendario 2025, schema')
    print('Feature rows:',len(df),'TFF:',rep['TFF_feature_rows'],'DISAGG:',rep['DISAGG_feature_rows'])

if __name__=='__main__':main()
