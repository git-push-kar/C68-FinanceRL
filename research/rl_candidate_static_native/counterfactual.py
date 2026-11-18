"""All first-action Monte Carlo values from an isolated historical simulator."""
import copy
import numpy as np

def choose(probabilities,u):
    p=np.asarray(probabilities,dtype=float)
    if p.shape!=(4,) or not np.isfinite(p).all() or (p<0).any() or p.sum()<=0:
        raise ValueError('Invalid continuation policy')
    p=p/p.sum();return min(int(np.searchsorted(np.cumsum(p),u)),3)

def action_values(market,targets,continuation,uniforms,gamma=.99):
    """Evaluate each first action; couple subsequent draws across branches.

    Future observations are used only inside these TRAINING rollouts. The
    current actor input remains the original, causal pre-action observation.
    Values are truncated after 1+len(uniforms) five-session decisions.
    """
    values=[]
    for action in range(4):
        branch=copy.copy(market);branch.env=copy.deepcopy(market.env)
        branch.cache=dict(market.cache);branch.infos=[]
        state,reward,done=branch.step(targets[action]);total=reward
        if done:raise ValueError('Counterfactual crossed training boundary')
        for k,u in enumerate(uniforms,1):
            obs,next_targets=state
            next_action=choose(continuation(obs),u)
            state,reward,done=branch.step(next_targets[next_action]);total+=gamma**k*reward
            if done:raise ValueError('Counterfactual crossed training boundary')
        values.append(total)
    return np.asarray(values,dtype=np.float32)
