import unittest
from core import fit_centroids,classify,fit_line
class CoreTests(unittest.TestCase):
    def test_separable_classes(self):
        m=fit_centroids([[0,0],[0,2],[8,8],[8,10]],['a','a','b','b'])
        self.assertEqual(classify(m,[1,1]),'a')
        self.assertEqual(classify(m,[9,9]),'b')
    def test_tie_is_stable(self):
        self.assertEqual(classify({'b':[2],'a':[0]},[1]),'a')
    def test_exact_regression(self):
        self.assertEqual(fit_line([0,1,2],[3,5,7]),{'slope':2.0,'intercept':3.0})
    def test_constant_rejected(self):
        with self.assertRaises(ValueError): fit_line([1,1],[2,3])
