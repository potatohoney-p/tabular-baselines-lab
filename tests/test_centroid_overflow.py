import math
import sys
import unittest
from decimal import Decimal, localcontext

from core import classify, fit_centroids


class CentroidOverflowTests(unittest.TestCase):
    def test_repeated_extreme_values_keep_representable_mean(self):
        for value in (1e308, -1e308, sys.float_info.max, -sys.float_info.max):
            with self.subTest(value=value):
                model = fit_centroids([[value], [value]], ['a', 'a'])
                self.assertEqual(model, {'a': [value]})
                self.assertEqual(classify(model, [value]), 'a')

    def test_overflow_and_cancellation_match_decimal_reference(self):
        fixtures = ([1e308, 1e308, -1e308],
                    [-1e308, -1e308, 1e308],
                    [sys.float_info.max, sys.float_info.max,
                     -sys.float_info.max, -sys.float_info.max, 1.0])
        for values in fixtures:
            # Exact decimal float conversion and sufficient precision for
            # these finite fixtures provide an independent arithmetic oracle.
            with localcontext() as context:
                context.prec = 2000
                expected = float(sum(Decimal.from_float(v) for v in values)
                                 / Decimal(len(values)))
            actual = fit_centroids([[v] for v in values], ['a'] * len(values))['a'][0]
            self.assertTrue(math.isfinite(actual))
            self.assertEqual(actual, expected)

    def test_each_label_and_coordinate_is_averaged_independently(self):
        rows = [[1e308, 2.0], [-1e308, 10.0], [1e308, 4.0], [-1e308, 20.0]]
        model = fit_centroids(rows, ['a', 'b', 'a', 'b'])
        self.assertEqual(model, {'a': [1e308, 3.0], 'b': [-1e308, 15.0]})
        self.assertEqual(rows, [[1e308, 2.0], [-1e308, 10.0],
                                [1e308, 4.0], [-1e308, 20.0]])

    def test_ordinary_sum_behavior_is_preserved(self):
        for values in ([1.0, 2.0, 3.0], [1e16, 1.0, -1e16],
                       [5e-324, 5e-324], [-0.0, -0.0]):
            expected = sum(values) / len(values)
            actual = fit_centroids([[v] for v in values], ['a'] * len(values))['a'][0]
            self.assertEqual(actual, expected)
            self.assertEqual(math.copysign(1, actual), math.copysign(1, expected))
