import unittest
from core import fit_centroids,classify,fit_line
class EdgeTests(unittest.TestCase):
    def test_bad_matrix(self):
        for rows in [[],[[1],[1,2]],[[float('nan')]]]:
            with self.assertRaises(ValueError): fit_centroids(rows,['a']*len(rows))
    def test_dimensions(self):
        with self.assertRaises(ValueError): classify({'a':[1,2]},[1])
    def test_pair_lengths(self):
        with self.assertRaises(ValueError): fit_line([1,2,3],[4,5])
