import numpy as np
from anchor import anchor
from market import Market

def test_anchor_respects_row_identity_permutation():
 m=Market();t,_=m.bounds('2020-01-01','2020-02-01');o=m.observe(t,100000.,np.zeros(5,int),100000.)
 a=anchor(o,'inverse_vol');permutation=np.array([4,2,0,3,1]);other=dict(o);other['stocks']=[o['stocks'][i] for i in permutation]
 assert np.array_equal(anchor(other,'inverse_vol'),a[permutation])

def test_anchor_budget_and_risk_response():
 m=Market();t,_=m.bounds('2020-01-01','2020-02-01');o=m.observe(t,100000.,np.zeros(5,int),100000.)
 for r in o['stocks']:r['vol20']=.2
 uniform=anchor(o,'inverse_vol');o['stocks'][0]['vol20']=1.
 changed=anchor(o,'inverse_vol');assert changed[0]<uniform[0]
 o['equity']=1.;assert np.array_equal(anchor(o,'inverse_vol'),np.ones(5,int))
