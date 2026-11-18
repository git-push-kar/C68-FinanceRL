"""Exact self-financing cash/stock accounting. Design only, not a live trial."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Execution:
    stock_weights: np.ndarray
    cash_weight: float
    net_return: float
    gross_return: float
    trading_cost: float
    turnover: float
    post_fee_stock_notional: np.ndarray

def execute(stock_weights,cash_weight,target,relatives,fee_bps=10.,cash_relative=1.):
    """Targets are fractions of wealth AFTER fees; costs use actual traded stocks.

    Solve c = k * sum(abs((1-c)*target - old_weights)). Both purchases and
    sales incur k, with no half-turnover convention. Mark the post-fee stock
    and cash holdings through the next market return. No leverage/shorting.
    Targets/cash must be decided using information available before relatives.
    This function handles accounting only, not policy information availability.
    """
    old=np.asarray(stock_weights,dtype=np.float64);target=np.asarray(target,dtype=np.float64)
    relatives=np.asarray(relatives,dtype=np.float64)
    if old.ndim!=1 or len(old)==0 or target.shape!=old.shape or relatives.shape!=old.shape:
        raise ValueError('Matching nonempty one-dimensional asset vectors required')
    if not all(np.isfinite(x).all() for x in (old,target,relatives)):
        raise ValueError('Finite asset inputs required')
    if not np.isfinite([cash_weight,fee_bps,cash_relative]).all():raise ValueError('Finite scalar inputs required')
    if min(old.min(),target.min(),cash_weight)<0 or target.sum()>1.+1e-12:
        raise ValueError('No shorting or leveraged target')
    if abs(old.sum()+cash_weight-1.)>1e-12:raise ValueError('Initial stock and cash weights must sum to one')
    if (relatives<=0).any() or cash_relative<=0 or not 0<=fee_bps<10000:
        raise ValueError('Positive relatives and fee fraction below one required')
    k=fee_bps/10000.;target_cash=max(0.,1.-float(target.sum()))
    def residual(c):return c-k*np.abs((1.-c)*target-old).sum()
    if k==0 or np.array_equal(old,target):cost=0.
    else:
        # Residual is strictly increasing for k<1 and sum(target)<=1.
        lower,upper=0.,1.
        for _ in range(60):
            middle=(lower+upper)/2
            if residual(middle)>0:upper=middle
            else:lower=middle
        cost=(lower+upper)/2
    post_fee=(1.-cost)*target
    turnover=float(np.abs(post_fee-old).sum())
    stock_values=post_fee*relatives
    cash_value=(1.-cost)*target_cash*cash_relative
    growth=float(stock_values.sum()+cash_value)
    gross=float(np.dot(target,relatives-1.)+target_cash*(cash_relative-1.))
    if growth<=0:raise ValueError('Nonpositive wealth')
    return Execution(stock_values/growth,float(cash_value/growth),growth-1.,gross,
                     float(cost),turnover,post_fee)
