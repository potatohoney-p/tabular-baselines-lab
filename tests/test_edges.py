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
    def test_nonfinite_centroids_rejected_in_either_order(self):
        from collections import OrderedDict
        for value in (float('nan'), float('inf'), -float('inf')):
            for first in (True, False):
                pairs = [('bad', [value]), ('near', [0.0])]
                if not first:
                    pairs.reverse()
                with self.subTest(value=value, invalid_first=first):
                    with self.assertRaisesRegex(ValueError, 'centroids must be finite'):
                        classify(OrderedDict(pairs), [0.0])
    def test_nonfinite_centroid_in_later_dimension_rejected(self):
        with self.assertRaisesRegex(ValueError, 'centroids must be finite'):
            classify({'bad': [0.0, float('nan')], 'near': [0.0, 0.0]}, [0.0, 0.0])
    def test_finite_model_is_not_mutated(self):
        model = {'b': [2.0, 2.0], 'a': [0.0, 0.0]}
        self.assertEqual(classify(model, [1.0, 1.0]), 'a')
        self.assertEqual(model, {'b': [2.0, 2.0], 'a': [0.0, 0.0]})
