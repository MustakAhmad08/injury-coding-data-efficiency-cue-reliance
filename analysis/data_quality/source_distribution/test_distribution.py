import unittest
import pandas as pd
from build_distribution import eligible_records, merge_recalls


class DistributionTests(unittest.TestCase):
    def test_exclusions_and_denominator_reconcile(self):
        frame=pd.DataFrame({
            'CPSC_Case_Number':[1,2,3,4,4,5,6],
            'Diagnosis':[41,41,70,41,57,41,57],
            'Body_Part':[1,1,1,1,1,None,1],
            'Narrative_1':[None,'  \t ', 'outside','first','duplicate','missing body','valid']})
        result,audit=eligible_records(frame,[41,57])
        self.assertEqual(result.CPSC_Case_Number.tolist(),[4,6])
        self.assertEqual(result.Diagnosis.tolist(),[41,57])
        self.assertEqual(audit['missing_required_fields_removed'],2)
        self.assertEqual(audit['empty_narratives_removed'],1)
        self.assertEqual(audit['duplicate_case_ids_removed'],1)
        self.assertEqual(audit['out_of_inventory_rows_removed'],1)
        self.assertEqual(audit['eligible_rows_30_codes'],2)

    def test_incomplete_recall_table_is_rejected(self):
        reference=pd.DataFrame({'condition':['tfidf_logreg_6000'],'diagnosis_code':[41], 'recall':[1.0],'support':[50]})
        with self.assertRaisesRegex(ValueError,'Invalid balanced-test metrics'):
            merge_recalls(pd.DataFrame({'diagnosis_code':[41]}),reference,pd.DataFrame(),[41])


if __name__=='__main__': unittest.main()
