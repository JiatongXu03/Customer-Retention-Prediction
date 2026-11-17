import unittest
from src.targeting import targeting_metrics

class TargetingTests(unittest.TestCase):
    def test_ties_and_rounding(self):
        m=targeting_metrics([1,0,1],[.5,.5,.2],['b','a','c'],.34)
        self.assertEqual((m['selected'],m['hits']),(2,1))
        self.assertEqual(targeting_metrics([1,0],[.5,.5],['b','a'],.5)['hits'],0)
    def test_zero_positive(self):
        m=targeting_metrics([0,0],[.8,.1],['a','b'],.5)
        self.assertIsNone(m['recall'])
        self.assertIsNone(m['lift'])
    def test_invalid(self):
        for budget in [0,1.1]:
            with self.assertRaises(ValueError): targeting_metrics([1],[.2],['a'],budget)
        with self.assertRaises(ValueError): targeting_metrics([1],[float('nan')],['a'],.1)
    def test_full_capacity(self):
        m=targeting_metrics([1,0,1],[.8,.1,.6],['a','b','c'],1)
        self.assertEqual(m['recall'],1)
        self.assertEqual(m['lift'],1)
