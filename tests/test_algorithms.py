import numpy as np

from scasan.algorithms import ScaSANOperator, sybil_scarepact, sybil_socactnet
from scasan.metrics import auc_from_scores
from scasan.synthetic import aggregated_toy_activity_graph, toy_graphs


def test_scansan_operator_is_stochastic():
    graph, _ = toy_graphs()
    operator = ScaSANOperator(
        graph, np.array([2]), alpha=0.1, beta=0.04,
        gamma=0.425, lambda1=2.0, lambda2=3.0,
    )
    state = operator.initial()
    for _ in range(25):
        state = operator.step(state)
        assert np.all(state >= -1e-15)
        assert np.isclose(state.sum(), 1.0)


def test_scarepact_runs_on_paper_toy():
    graph, _ = toy_graphs()
    result = sybil_scarepact(graph, np.array([2]), epsilon=1e-8)
    assert result.converged
    assert result.scores.shape == (5,)
    assert 0.5 <= auc_from_scores(result.scores, 3) <= 1.0


def test_trapping_node_curves_match_paper_description():
    graph, activity = toy_graphs()
    values = []
    for k in range(6):
        result = sybil_socactnet(activity, np.array([2]), k=k, activity_steps=1)
        values.append(auc_from_scores(result.scores, 3))
    assert values[0] == 1.0
    assert values[3:] == [0.5, 0.5, 0.5]

    aggregated = aggregated_toy_activity_graph(graph)
    result = sybil_socactnet(aggregated, np.array([2]), k=5, activity_steps=1)
    assert auc_from_scores(result.scores, 3) == 0.5
