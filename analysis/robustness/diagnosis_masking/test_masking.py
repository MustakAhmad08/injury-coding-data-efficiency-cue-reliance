import unittest
from run_test import mask

class MaskTests(unittest.TestCase):
    def test_requested_example(self):
        text='A patient fell from a ladder, landed on their hand, and scraped their elbow. DX: wrist fracture; elbow abrasion.'
        self.assertEqual(mask(text)[0], 'A patient fell from a ladder, landed on their hand, and scraped their elbow. DX: wrist [MASK]; elbow [MASK].')

    def test_preserve_context_and_negation(self):
        self.assertEqual(mask('No fracture. Wrist pain after falling; skin scraped.')[0], 'No [MASK]. Wrist pain after falling; skin scraped.')

    def test_all_occurrences_case_and_abbreviations(self):
        self.assertEqual(mask('FX wrist; fracture wrist; DX LAC finger.')[0], '[MASK] wrist; [MASK] wrist; DX [MASK] finger.')

    def test_boundaries(self):
        self.assertEqual(mask('Lactate; strained to reach a shelf; fractured table.')[0], 'Lactate; strained to reach a shelf; fractured table.')

if __name__=='__main__': unittest.main()
