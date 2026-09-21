"""Check subgroup selection, proportion intervals and reported count integrity."""
import unittest
from pathlib import Path
import pandas as pd
from analyze_distinct import subset, wilson, code


class Checks(unittest.TestCase):
    def test_gold_only_subset_excludes_missing_and_repeated(self):
        f=pd.DataFrame({'diag1':[57,57,57,71],'diag2':[None,57,59,57],
                        'prediction':[59,59,57,71]})
        self.assertEqual(subset(f).index.tolist(),[2,3])
    def test_missing_and_labeled_predictions(self):
        self.assertEqual(code('59 (Laceration)'),59)
        self.assertEqual(code('59.0'),59)
        self.assertEqual(code('S52.9'),-1)
        self.assertEqual(code(float('nan')),-1)
    def test_boundary_intervals(self):
        self.assertAlmostEqual(wilson(0,254)[0],0)
        self.assertAlmostEqual(wilson(254,254)[1],1)
        self.assertLess(wilson(127,254)[0],.5)
        self.assertGreater(wilson(127,254)[1],.5)
    def test_saved_counts_and_not_applicable_fields(self):
        f=pd.read_csv(Path(__file__).parent/'results/distinct_254_metrics.csv')
        self.assertEqual(len(f),7)
        self.assertTrue(f.n.eq(254).all())
        for r in f.to_dict('records'):
            self.assertAlmostEqual(r['primary_accuracy'],r['primary_correct']/254)
            if r['secondary_evaluated']:
                self.assertLessEqual(r['both_ordered_correct'],min(r['primary_correct'],r['secondary_correct']))
                self.assertLessEqual(r['exact_swap_count'],254-r['primary_correct'])
            else:
                self.assertTrue(pd.isna(r['secondary_accuracy']))


if __name__=='__main__': unittest.main()
