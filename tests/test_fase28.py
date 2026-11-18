import io
import sys
import zipfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import fase28_unica as f

class TestFase28(unittest.TestCase):
 def test_calendar_holiday_and_double_release(self):
  html='''<table><thead><tr><th>2022</th><th>2023</th></tr><tr><th>Release date</th><th>Data ending</th><th>Release date</th><th>Data ending</th></tr></thead><tbody>
  <tr><td>June</td><td><a href="a">29</a></td><td>17</td><td><a href="b">28</a></td><td>23</td></tr>
  <tr><td><a href="c">29</a></td><td>24</td><td><a href="d">21</a></td><td>16</td></tr>
  </tbody></table>'''
  df=f.calendar_from_html(html)
  self.assertEqual(len(df),4)
  self.assertEqual(set(df[df.observation_date.dt.year==2022].release_date.dt.strftime('%Y-%m-%d')),{'2022-06-29'})

 def test_crossyear(self):
  html='<table><tr><th>2026</th></tr><tr><th>Release date</th><th>Data ending</th></tr><tr><td>January</td><td><a href="f">7</a></td><td>1/2</td></tr></table>'
  x=f.calendar_from_html(html)
  self.assertEqual(x.observation_date.iloc[0].strftime('%Y-%m-%d'),'2026-01-02')

 def test_no_fake_releases(self):
  self.assertTrue(f.calendar_from_html('<table><tr><td>hello</td></tr></table>').empty)

 def test_parse_dates(self):
  self.assertEqual(f.parse_week_ending('12/29',pd.Timestamp('2024-01-04')).strftime('%Y-%m-%d'),'2023-12-29')

 def test_mock_eia_download_and_calendar_merge(self):
  class R:
   def __init__(self,obj):self._obj=obj
   def json(self):return self._obj
   def raise_for_status(self):pass
  class S:
   def get(self,url,**kwargs):
    self.last=(url,kwargs)
    return R({'response':{'data':[{'period':'2026-11-06','value':'414.4'},{'period':'2026-10-30','value':'409.7'}]}})
  session=S();logs=[]
  df=f.eia_series(session,'TEST_KEY','PET.WCESTUS1.W',logs)
  self.assertEqual(len(df),2)
  self.assertEqual(df.value.max(),414.4)
  self.assertIn('api_key',session.last[1]['params'])
  calendar=pd.DataFrame({'observation_date':[pd.Timestamp('2026-11-06')],
   'release_date':[pd.Timestamp('2026-11-12')],
   'availability_utc_next_day':[pd.Timestamp('2026-11-13T05:00:00Z')]})
  original=f.eia_series
  try:
   f.eia_series=lambda sess,key,sid,log:df if sid=='PET.WCESTUS1.W' else pd.DataFrame()
   merged=f.build_eia(session,'KEY',calendar,logs)
  finally:f.eia_series=original
  yes=merged[(merged.variable=='crude_commercial')&(merged.observation_date==pd.Timestamp('2026-11-06'))]
  self.assertEqual(len(yes),1)
  self.assertEqual(yes.temporal_quality.iloc[0],'EIA_ARCHIVE_DATE_VALUE_LATEST_UNCERTIFIED')
  no=merged[(merged.variable=='crude_commercial')&(merged.observation_date==pd.Timestamp('2026-10-30'))]
  self.assertTrue(pd.isna(no.available_at_utc.iloc[0]))

 def test_spread_not_fabricated_when_leg_missing(self):
  x=pd.DataFrame({'variable':['futures_m1'],'observation_date':[pd.Timestamp('2026-01-01')],
   'value':[70.],'available_at_utc':[pd.Timestamp('2026-01-02T00:00Z')]})
  y=f.make_futures_spreads(x,[])
  self.assertEqual(len(y),1)

 def test_asof_no_lookahead(self):
  df=pd.DataFrame({'timestamp':['2026-11-11T12:00:00Z','2026-11-13T12:00:00Z'],'macro_effective_utc':['2026-11-01T00:00:00Z']*2})
  extra=pd.DataFrame({'variable':['gasoline'],'available_at_utc':['2026-11-12T00:00:00Z'],'value':[5.]})
  cols=[];x=f.merge_features(df,extra,cols)
  self.assertTrue(np.isnan(x.exploratory_gasoline.iloc[0]));self.assertEqual(x.exploratory_gasoline.iloc[1],5.)

 def test_lookahead_rejected(self):
  d=pd.DataFrame({'timestamp':['2026-11-11T12:00:00Z'],'macro_effective_utc':['2026-11-12T00:00:00Z']})
  with self.assertRaises(ValueError):f.merge_features(d,pd.DataFrame(columns=['variable']),[])

 def test_intrabarr_ambiguity(self):
  t=pd.date_range('2026-01-01',periods=18,freq='4h',tz='UTC')
  h=pd.DataFrame({'timestamp':t,'open':100.,'high':101.,'low':99.,'close':100.,'signal_bull_at_close':[False]*14+[True]+[False]*3})
  h.loc[15,'high']=106;h.loc[15,'low']=94
  d=pd.DataFrame({'timestamp':pd.date_range('2026-01-01',periods=8,freq='D',tz='UTC'),'open':100.,'high':101.,'low':99.,'close':100.})
  e=f.h4_outcomes(h,d)
  self.assertEqual(len(e),1)
  self.assertEqual(e.first_touch_5d.iloc[0],'AMBIGUOUS_SAME_BAR')

if __name__=='__main__':unittest.main()
