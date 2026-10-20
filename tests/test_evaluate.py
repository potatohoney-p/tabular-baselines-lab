import unittest
from unittest.mock import patch
import evaluate as evaluation
from features import fit_scaler
class HoldoutTests(unittest.TestCase):
    def test_rare_class_survives_both_partitions(self):
        labels=['common']*18+['rare']*2
        train,test=evaluation.stratified_split(labels)
        self.assertEqual(set(labels[i] for i in train),{'common','rare'})
        self.assertEqual(set(labels[i] for i in test),{'common','rare'})
        self.assertFalse(set(train)&set(test)); self.assertEqual(sorted(train+test),list(range(20)))
    def test_repeatable_split(self):
        labels=['a']*5+['b']*5
        self.assertEqual(evaluation.stratified_split(labels),evaluation.stratified_split(labels))
    def test_unseparable_dataset_can_fail(self):
        report=evaluation.evaluate([[1]]*4,['a','a','b','b'],0.5)
        self.assertEqual(report['accuracy'],0.5)
        self.assertEqual(sum(sum(row.values()) for row in report['confusion'].values()),report['test_size'])
    def test_scaler_sees_training_rows_only(self):
        rows=[[0],[1],[2],[100],[101],[102]]; labels=['a']*3+['b']*3
        with patch.object(evaluation,'fit_scaler',wraps=fit_scaler) as scaler:
            report=evaluation.evaluate(rows,labels)
        self.assertEqual(scaler.call_args[0][0],[rows[i] for i in report['train_indices']])
        self.assertEqual(report['accuracy'],1.0)
    def test_singleton_class_rejected(self):
        with self.assertRaises(ValueError): evaluation.stratified_split(['a','a','b'])
