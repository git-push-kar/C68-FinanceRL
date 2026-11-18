"""Test state-aware InternVL3-2B + LoRA RL adapter in paper trading engine."""

import json
from pathlib import Path
import numpy as np
import pytest
from finance.paper_engine import Panel, PaperAccount
from finance.policy import predict

ROOT = Path(__file__).resolve().parents[1]

def test_state_aware_policy_rebalance():
    p = Panel()
    account = PaperAccount(p, capital=100000.0)
    
    # Run predict with account
    target_weights, receipt = predict(service=None, account=account)
    
    # Verify target weights shape and properties
    assert isinstance(target_weights, np.ndarray)
    assert target_weights.shape == (49,)
    assert np.all(target_weights >= 0.0)
    assert target_weights.sum() <= 1.000001
    assert target_weights.max() <= 0.100001  # 10% maximum position cap
    
    # Verify receipt metadata
    assert receipt['mode'] == 'internvl3_state_rl'
    assert receipt['adapter'] == 'InternVL3-2B-State-RL'
    assert receipt['adapter_sha256'] == '749d98f2420b5ce601b10d1b191dbb2d9985ecac5ce9f290b55defa588c1d340'
    assert 'rationales' in receipt
    assert len(receipt['rationales']) > 0
    
    # Verify rationales content
    for r in receipt['rationales']:
        assert 'ticker' in r
        assert 'features' in r
        assert 'rationale' in r
        assert len(r['rationale']) > 0

def test_paper_account_stateful_step():
    p = Panel()
    account = PaperAccount(p, capital=100000.0)
    
    # Step forward 5 sessions to trigger rebalance
    for _ in range(5):
        has_next = account.step(lambda: predict(service=None, account=account))
        assert has_next
        
    assert len(account.activities) == 1
    act = account.activities[0]
    assert len(act['target_weights']) == 49
    assert len(act['trades']) > 0
    assert 'rationales' in act
    assert len(act['rationales']) > 0
    assert account.equity > 0
    assert account.fees > 0
