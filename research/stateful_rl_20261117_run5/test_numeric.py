import copy
import numpy as np
from market import Market
from numeric_state import numeric_features

def test_numeric_state_shape_finiteness_and_causal_account_effect():
 m=Market();t,_=m.bounds('2023-01-01','2023-06-01');o=m.observe(t,100000.,np.zeros(5,int),100000.)
 x=numeric_features(o,0);assert x.shape==(38,) and np.isfinite(x).all()
 small=m.observe(t,500.,np.zeros(5,int),500.);assert not np.array_equal(x,numeric_features(small,0))
 altered=copy.deepcopy(o);altered['stocks'][0]['vol20']*=2;assert not np.array_equal(x,numeric_features(altered,0))
 m.close[t+1:]*=100;m.index[t+1:]*=100;assert np.array_equal(x,numeric_features(m.observe(t,100000.,np.zeros(5,int),100000.),0))
