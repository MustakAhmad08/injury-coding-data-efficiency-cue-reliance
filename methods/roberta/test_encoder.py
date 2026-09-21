"""Focused checks for leakage guards, the test gate, and comparison metrics."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import run_encoder as runner
import analyze_results as analysis


class EncoderTests(unittest.TestCase):
    def test_test_set_gate_precedes_any_file_read(self):
        with patch.object(runner, "read_json") as read:
            with self.assertRaisesRegex(ValueError, "confirm-locked-test"):
                runner.evaluate({}, False)
            read.assert_not_called()

    def test_case_overlap_is_rejected(self):
        a = pd.DataFrame({"case_id": ["001"], "narrative": ["one"], "diag1": [41]})
        b = pd.DataFrame({"case_id": ["001"], "narrative": ["different"], "diag1": [41]})
        with self.assertRaisesRegex(ValueError, "Case leakage"):
            runner.check_disjoint(a,b)

    def test_normalized_narrative_overlap_is_rejected(self):
        a = pd.DataFrame({"case_id": ["001"], "narrative": ["DX:  FRACTURE"], "diag1": [57]})
        b = pd.DataFrame({"case_id": ["002"], "narrative": [" dx: fracture "], "diag1": [57]})
        with self.assertRaisesRegex(ValueError, "narrative leakage"):
            runner.check_disjoint(a,b)

    def test_split_rejects_future_year(self):
        frame = pd.DataFrame({"case_id": [str(i) for i in range(30)], "year": [2025]*30,
                              "narrative": [f"example {i}" for i in range(30)], "diag1": list(range(30))})
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"split.csv"; frame.to_csv(p,index=False)
            with self.assertRaisesRegex(ValueError, "row count/year"):
                runner.load_split(p, {2024}, 30)

    def test_bootstrap_metric_matches_reference_with_missing_predictions(self):
        rng = np.random.default_rng(42)
        labels = np.array([41, 57, 71])
        for _ in range(10):
            gold = rng.choice(labels, 80)
            pred = rng.choice([-1, 41, 57, 71], 80)
            self.assertAlmostEqual(analysis.score(gold,pred,labels),
                                   f1_score(gold,pred,labels=labels,average="macro",zero_division=0))


if __name__ == "__main__": unittest.main()
