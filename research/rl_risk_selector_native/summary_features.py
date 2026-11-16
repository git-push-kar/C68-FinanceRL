import numpy as np
from candidate_market import SCALES

def features(tickers,obs):
    values=obs['summary'][0].detach().cpu().numpy()
    if values.shape!=(33,) or not np.isfinite(values).all():raise ValueError('Expected 33 finite candidate summary features')
    return np.zeros(33,dtype=np.float32)
