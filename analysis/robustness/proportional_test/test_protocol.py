import unittest
import numpy as np
import pandas as pd
from experiment import allocate

class AllocationTests(unittest.TestCase):
    def test_exact_total_and_rounding(self):
        counts=pd.Series({41:5092,46:83,57:65595,71:101480})
        q=allocate(counts)
        self.assertEqual(int(q.sum()),1500)
        exact=counts/counts.sum()*1500
        self.assertTrue(((q==np.floor(exact))|(q==np.ceil(exact))).all())

    def test_tie_break_and_order(self):
        a=allocate(pd.Series({3:1,1:1,2:1}),2)
        b=allocate(pd.Series({2:1,3:1,1:1}),2)
        self.assertEqual(a.to_dict(),{1:1,2:1,3:0})
        self.assertEqual(a.to_dict(),b.to_dict())

    def test_no_forced_rare_minimum(self):
        self.assertEqual(allocate(pd.Series({1:99999,2:1}),1500).to_dict(),{1:1500,2:0})

if __name__=='__main__':unittest.main()
