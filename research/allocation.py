"""Shared action-to-allocation convention for simulation and deployed inference."""
import numpy as np


def allocation_weights(action, max_weight, current_weights=None, rebalance_rate=1.0):
    """Move partway from current holdings toward capped score-based weights."""
    if not np.isfinite(rebalance_rate) or not 0 < rebalance_rate <= 1:
        raise ValueError("Rebalance rate must be in (0, 1]")
    desired = capped_softmax(action, max_weight)
    if current_weights is None:
        return desired
    current = np.array(current_weights, dtype=np.float64, copy=True)
    if current.shape != desired.shape or not np.isfinite(current).all() or (current < 0).any():
        raise ValueError("Invalid current asset weights")
    # Deploy initial cash fully; subsequent decisions limit unnecessary trading.
    if current.sum() <= 1e-12 or rebalance_rate == 1:
        return desired
    current /= current.sum()
    if current.max() > max_weight:
        current = capped_softmax(np.log(np.maximum(current, 1e-30)), max_weight)
    return (1 - rebalance_rate) * current + rebalance_rate * desired


def capped_softmax(action, max_weight):
    scores = np.array(action, dtype=np.float64, copy=True)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Expected a finite vector of stock scores")
    if not np.isfinite(max_weight) or not 0 < max_weight <= 1 or len(scores) * max_weight < 1:
        raise ValueError("Infeasible maximum portfolio weight")
    scores -= scores.max()
    weights = np.zeros(len(scores), dtype=np.float64)
    free = np.ones(len(scores), dtype=bool)
    remaining = 1.0
    for _ in range(len(scores)):
        logits = scores[free]
        mass = np.exp(logits - logits.max())
        weights[free] = remaining * mass / mass.sum()
        over = free & (weights > max_weight)
        if not over.any():
            break
        weights[over] = max_weight
        free[over] = False
        remaining = 1.0 - weights[~free].sum()
        if not free.any():
            break
    return weights
