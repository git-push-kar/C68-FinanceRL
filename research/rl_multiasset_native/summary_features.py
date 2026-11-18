import numpy as np
from candidate_market import SCALES

def features(tickers,obs):
    values=obs['summary'][0].detach().cpu().numpy()
    if values.shape!=(27,) or not np.isfinite(values).all():raise ValueError('Expected 27 finite candidate summary features')
    return (values/SCALES).astype(np.float32)
