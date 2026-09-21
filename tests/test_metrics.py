import numpy as np

from scasan.metrics import auc_from_scores


def test_auc_handles_order_and_ties():
    assert auc_from_scores(np.array([3.0, 2.0, 1.0, 0.0]), 2) == 1.0
    assert auc_from_scores(np.ones(4), 2) == 0.5
    assert auc_from_scores(np.array([0.0, 1.0, 2.0, 3.0]), 2) == 0.0
