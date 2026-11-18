"""Native-LoRA portfolio interface. No model loading or trading side effects on import.

The caller supplies the TextPolicy class from the package's verified source tree.
Feature/candidate construction must use that same tree's causal Market controller.
This interface returns target weights; it neither submits orders nor replaces the
simulator's self-financing transaction-cost accounting.
"""
from datetime import date
import json
from pathlib import Path

import numpy as np
import torch


SCHEMAS = {
    'candidate_static_text_v1': (.30, 4, 27),
    'multiasset_static_text_v1': (.60, 4, 27),
    'multiasset_summary_text_v1': (.60, 4, 27),
    'multiasset_summary_readout_only_v1': (.60, 4, 27),
    'risk_selector_static_text_v1': (.60, 5, 33),
}


class NativeFinanceSpecialist:
    def __init__(self, package, policy_class, shared=None):
        package = Path(package)
        record = json.loads((package/'policy.json').read_text())
        schema = record['prompt_schema']
        if schema not in SCHEMAS:
            raise ValueError('Unsupported native finance execution schema')
        self.cap, self.action_count, self.feature_count = SCHEMAS[schema]
        self.schema = schema
        # TextPolicy verifies the package manifest, pinned base, labels and schema.
        self.policy = policy_class(record['tickers'], package, shared=shared)
        if self.policy.shared.compute_dtype != torch.float32:
            raise ValueError('Finance requires its verified FP32 decoder execution')

    def register(self, router):
        if router.shared is not self.policy.shared:
            raise ValueError('Finance must use the router resident backbone')
        router.register('finance', 'finance', self.predict)

    @torch.inference_mode()
    def predict(self, model, payload):
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if model is not self.policy.shared.model:
            raise ValueError('Different resident backbone supplied')
        if getattr(model, 'training', False):
            raise ValueError('Native finance inference requires evaluation mode')
        if payload.get('tickers') != self.policy.tickers:
            raise ValueError('Exact trained ticker order required')
        step = payload.get('trading_step')
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError('trading_step must be a nonnegative session count from deployment start')
        n = len(self.policy.tickers)
        holdings = np.asarray(payload.get('execution_portfolio'), dtype=np.float64)
        if (holdings.shape != (n+1,) or not np.isfinite(holdings).all()
                or (holdings < 0).any() or not np.isclose(holdings.sum(), 1., atol=1e-8, rtol=0)):
            raise ValueError('Current stock and cash weights must be finite, nonnegative and sum to one')
        # Off-schedule calls must preserve actual drifted holdings, even if they
        # exceed the target cap after price movement. Do not normalize or trade.
        if step % 5:
            return {'weights': holdings[:-1].tolist(), 'cash_weight': float(holdings[-1]),
                    'rebalance_due': False, 'probabilities': None,
                    'execution': 'hold_actual_positions', 'prompt_schema': self.schema}
        lag = payload.get('signal_lag_closes')
        if isinstance(lag, bool) or not isinstance(lag, int) or lag != 1:
            raise ValueError('The complete signal must be delayed one close')
        try:
            signal = date.fromisoformat(payload['signal_close_date'])
            execution = date.fromisoformat(payload['execution_close_date'])
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError('ISO signal/execution close dates required') from exc
        if signal >= execution:
            raise ValueError('Signal close must precede execution close')
        observation = payload.get('observation', {})
        summary = np.asarray(observation.get('summary'), dtype=np.float32)
        targets = np.asarray(observation.get('candidate_targets'), dtype=np.float64)
        if summary.shape != (self.feature_count,) or not np.isfinite(summary).all():
            raise ValueError(f'Expected all {self.feature_count} finite controller summary values')
        rounded = np.asarray([float(f'{float(x):.2f}') for x in summary], dtype=np.float32)
        if not np.array_equal(summary, rounded):
            raise ValueError('Use the verified controller two-decimal summaries')
        if (targets.shape != (self.action_count, n) or not np.isfinite(targets).all()
                or (targets < 0).any() or targets.max() > self.cap + 1e-9
                or not np.allclose(targets.sum(1), 1., atol=1e-9, rtol=0)):
            raise ValueError('Invalid strategy target portfolios')
        device = self.policy.shared.compute_device
        tensors = {'summary': torch.as_tensor(summary, device=device).unsqueeze(0),
                   'candidate_targets': torch.as_tensor(targets, device=device).unsqueeze(0)}
        probabilities = self.policy(tensors)[0].probs[0].detach().cpu().numpy().astype(float)
        if (probabilities.shape != (self.action_count,) or not np.isfinite(probabilities).all()
                or (probabilities < 0).any() or probabilities.sum() <= 0):
            raise ValueError('Invalid native strategy probabilities')
        probabilities /= probabilities.sum()
        weights = probabilities @ targets
        return {'weights': weights.tolist(), 'cash_weight': 0., 'rebalance_due': True,
                'probabilities': probabilities.tolist(), 'execution': 'deterministic_expected_mixture',
                'signal_lag_closes': 1, 'signal_close_date': signal.isoformat(),
                'execution_close_date': execution.isoformat(), 'prompt_schema': self.schema}
