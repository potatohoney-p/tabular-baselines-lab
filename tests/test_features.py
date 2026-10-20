import unittest
from features import split_indices,fit_scaler,transform,mse
class FeatureTests(unittest.TestCase):
    def test_split_no_overlap(self):
        train,test=split_indices(20)
        self.assertFalse(set(train)&set(test)); self.assertEqual(sorted(train+test),list(range(20)))
        self.assertEqual((train,test),split_indices(20))
    def test_constant_scaler(self): self.assertEqual(transform([[2],[2]],fit_scaler([[2],[2]])),[[0.0],[0.0]])
    def test_training_only_scaler(self): self.assertEqual(transform([[100]],fit_scaler([[0],[2]])),[[99.0]])
    def test_metric(self): self.assertEqual(mse([1,3],[1,1]),2)
    def test_transform_rejects_scale_dimension_mismatch(self):
        for scales in ([], [1], [1, 1, 1]):
            with self.subTest(scales=scales):
                with self.assertRaises(ValueError):
                    transform([[3, 4]], ([1, 2], scales))
    def test_transform_rejects_mean_dimension_mismatch(self):
        for means in ([], [1], [1, 2, 3]):
            with self.subTest(means=means):
                with self.assertRaises(ValueError):
                    transform([[3, 4]], (means, [1, 1]))
    def test_transform_preserves_all_features(self):
        rows = [[1, 2], [3, 6]]
        self.assertEqual(transform(rows, fit_scaler(rows)),
                         [[-1.0, -1.0], [1.0, 1.0]])
