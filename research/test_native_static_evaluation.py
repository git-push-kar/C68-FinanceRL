"""Fail-closed tests for access controls, source integrity, and data handling."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

import native_static_evaluation as evaluation


class EvaluationSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_prompt_must_be_literal_not_input_dependent(self):
        path = self.root/'exposure_prompt.py'
        path.write_text('def prompt(obs, tickers):\n    return "fixed"\n')
        self.assertEqual(evaluation.constant_prompt(self.root), 'fixed')
        path.write_text('def prompt(obs, tickers):\n    return str(obs)\n')
        with self.assertRaises(ValueError):
            evaluation.constant_prompt(self.root)

    def test_incomplete_panel_rejected_instead_of_forward_filled(self):
        dates = pd.bdate_range('2020-01-01', periods=400)
        frame = pd.DataFrame([(str(d.date()), t, 100.) for d in dates for t in ('A', 'B')],
                             columns=['Date', 'Ticker', 'Close'])
        path = self.root/'prices.csv'
        frame.to_csv(path, index=False)
        start, end = str(dates[300].date()), str(dates[-1].date())
        self.assertEqual(evaluation.panel_metadata(path, start, end)['sessions'], 100)
        frame.drop(index=650).to_csv(path, index=False)
        with self.assertRaises(ValueError):
            evaluation.panel_metadata(path, start, end)

    def test_rehearsal_rejects_reserved_dates_before_loading_sources(self):
        path = self.root/'spec.json'
        path.write_text(json.dumps({'purpose': 'development_rehearsal',
                                   'start': '2026-02-02', 'end': '2026-10-30'}))
        with self.assertRaisesRegex(ValueError, 'reserved'):
            evaluation.freeze(path, self.root/'out')
        self.assertFalse((self.root/'out').exists())

    def test_access_receipt_cannot_be_overwritten(self):
        path = self.root/'access.json'
        evaluation.write_new(path, {'seal': 'first'})
        with self.assertRaises(FileExistsError):
            evaluation.write_new(path, {'seal': 'second'})
        self.assertEqual(evaluation.read(path), {'seal': 'first'})

    def test_source_tamper_detected_before_access(self):
        path = self.root/'source.py'
        path.write_text('original')
        contract = {'snapshot_sha256': {'source.py': evaluation.sha(path)}}
        contract['seal_sha256'] = hashlib.sha256(evaluation.canonical(contract)).hexdigest()
        evaluation.write_new(self.root/'contract.json', contract)
        path.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            evaluation.evaluate_sealed(self.root)
        self.assertFalse((self.root/'development_access.json').exists())


if __name__ == '__main__':
    unittest.main()
