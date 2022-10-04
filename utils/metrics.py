import numpy as np
import unittest

def prec(k: int, preds: np.ndarray, true: np.ndarray) -> float:
    """Precision function with cutoff (k). Used for MAP@12 metric.

    Args:
        k (int): Cutoff point for prediction array
        preds (np.ndarray): Prediction array
        true (np.ndarray): Ground truth

    Returns:
        float: Precision, i.e. portion of correctly predicted values

    """
    # Assumes that preds and true are 1d arrays ['a','b',...]
    return len(np.intersect1d(preds[:k], true))/k

def rel(k: int, preds: np.ndarray, true: np.ndarray) -> int:
    assert 0 < k <= len(preds), "k must be able to index preds!"
    return int(preds[k-1] in true)

def MAPk(k, preds, true) -> float:
    return np.mean([
        np.sum([prec(i,p,t)*rel(i,p,t) for i in range(1,k+1)])/\
            min(k, len(true))\
                for t, p in zip(true, preds)
    ])

# Tests

class TestMetricFunctions(unittest.TestCase):
    def __init__(self, methodName: str = 'runTest') -> None:
        self.gt = np.array(['a', 'b', 'c', 'd', 'e'])
        self.preds1 = np.array(['b', 'c', 'a', 'd', 'e'])
        self.preds2 = np.array(['a', 'b', 'c', 'd', 'e'])
        self.preds3 = np.array(['f', 'b', 'c', 'd', 'e'])
        self.preds4 = np.array(['a', 'f', 'e', 'g', 'b'])
        self.preds5 = np.array(['a', 'f', 'c', 'g', 'b'])
        self.preds6 = np.array(['d', 'c', 'b', 'a', 'e'])
        super().__init__(methodName)

    def test_prec(self):
        self.assertAlmostEqual(prec(1, self.preds1, self.gt), 1.0)
        self.assertAlmostEqual(prec(1, self.preds2, self.gt), 1.0)
        self.assertAlmostEqual(prec(1, self.preds3, self.gt), 0.0)
        self.assertAlmostEqual(prec(2, self.preds4, self.gt), 0.5)
        self.assertAlmostEqual(prec(3, self.preds5, self.gt), 2/3)
        self.assertAlmostEqual(prec(3, self.preds6, self.gt), 1.0)
    
    def test_rel(self):
        self.assertAlmostEqual(rel(1, self.preds1, self.gt), 1.0)
        self.assertAlmostEqual(rel(1, self.preds2, self.gt), 1.0)
        self.assertAlmostEqual(rel(1, self.preds3, self.gt), 0.0)
        self.assertAlmostEqual(rel(2, self.preds4, self.gt), 0.0)
        self.assertAlmostEqual(rel(3, self.preds5, self.gt), 1.0)
        self.assertAlmostEqual(rel(3, self.preds6, self.gt), 1.0)
    
    def test_mapk(self):
        all_true = np.array([self.gt for i in range(6)])
        all_pred = np.array([self.preds1, self.preds2, self.preds3,\
                            self.preds4, self.preds5, self.preds6])
        self.assertAlmostEqual(MAPk(k=4, preds=all_pred, true=all_true), 0.71875)

if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)