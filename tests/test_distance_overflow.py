import sys
import unittest
from collections import OrderedDict
from copy import deepcopy
from decimal import Decimal, localcontext

from core import classify, fit_centroids


class DistanceOverflowTests(unittest.TestCase):
    def assert_prediction_in_either_order(self, model, row, expected):
        pairs = list(model.items())
        for ordered in (pairs, list(reversed(pairs))):
            with self.subTest(labels=[label for label, _ in ordered]):
                self.assertEqual(classify(OrderedDict(ordered), row), expected)

    def test_squared_coordinate_overflow(self):
        self.assert_prediction_in_either_order(
            {'a': [2e200], 'b': [1e200]}, [0.0], 'b')

    def test_distance_sum_overflow_is_not_a_tie(self):
        self.assert_prediction_in_either_order(
            {'a': [1e154, 1e154, 1e154], 'b': [1e154, 1e154, 0.0]},
            [0.0, 0.0, 0.0], 'b')

    def test_coordinate_subtraction_overflow_is_not_a_tie(self):
        self.assert_prediction_in_either_order(
            {'a': [-1e308], 'b': [-9e307]}, [1e308], 'b')

    def test_overflowing_candidate_does_not_hide_finite_nearest(self):
        model = fit_centroids([[1e308], [1e308], [2.0], [4.0]],
                              ['a', 'a', 'b', 'b'])
        before = deepcopy(model)
        for row in ([0.0], [3.0]):
            self.assert_prediction_in_either_order(model, row, 'b')
        self.assertEqual(model, before)

    def test_extreme_distance_ties_are_lexical(self):
        fixtures = (
            ({'b': [1e200], 'a': [-1e200]}, [0.0]),
            ({'b': [1e154, 1e154], 'a': [-1e154, -1e154]}, [0.0, 0.0]),
            ({'b': [-1e308, 1e308], 'a': [1e308, -1e308]}, [1e308, 1e308]),
        )
        for model, row in fixtures:
            self.assert_prediction_in_either_order(model, row, 'a')

    def test_small_components_still_break_extreme_distance_ties(self):
        self.assert_prediction_in_either_order(
            {'a': [1e200, 2.0], 'b': [1e200, 1.0]}, [0.0, 0.0], 'b')

    def test_extreme_predictions_match_independent_decimal_reference(self):
        fixtures = (
            ({'a': [sys.float_info.max, -sys.float_info.max],
              'b': [-sys.float_info.max, sys.float_info.max]},
             [sys.float_info.max, -1e308]),
            ({'a': [1e200, 2.0], 'b': [1e200, 1.0]}, [0.0, 0.0]),
            ({'a': [-1e308], 'b': [-9e307]}, [1e308]),
            ({'a': [1e154, 1e154, 1e154], 'b': [1e154, 1e154, 0.0]},
             [0.0, 0.0, 0.0]),
        )
        for model, row in fixtures:
            # Exact input conversion and ample decimal precision are
            # independent of the implementation's rational fallback.
            with localcontext() as context:
                context.prec = 2000
                distances = {
                    label: sum((Decimal.from_float(x) - Decimal.from_float(y)) ** 2
                               for x, y in zip(row, centroid))
                    for label, centroid in model.items()
                }
                expected = min(model, key=lambda label: (distances[label], label))
            self.assert_prediction_in_either_order(model, row, expected)

    def test_ordinary_floating_point_tie_behavior_is_preserved(self):
        # Without overflow, keep the existing rounded-distance behavior.
        self.assert_prediction_in_either_order(
            {'b': [1.0, 0.0], 'a': [1.0, 1e-10]}, [0.0, 0.0], 'a')

    def test_overflow_recomputes_all_candidates_in_the_same_arithmetic(self):
        # An overflowing outlier switches even finite candidates to exact
        # distances, so the rounded tie above must not survive the fallback.
        self.assert_prediction_in_either_order(
            {'a': [1.0, 1e-10], 'b': [1.0, 0.0], 'z': [1e200, 0.0]},
            [0.0, 0.0], 'b')
