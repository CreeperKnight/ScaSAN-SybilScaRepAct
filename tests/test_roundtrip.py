import numpy as np

from scasan.graph import ActivityGraph, ScaSANGraph
from scasan.synthetic import toy_graphs


def test_graph_roundtrip(tmp_path):
    graph, activity = toy_graphs()
    graph.save(tmp_path)
    activity.save(tmp_path)
    loaded = ScaSANGraph.load(tmp_path)
    loaded_activity = ActivityGraph.load(tmp_path)
    assert np.array_equal(loaded.create_count, graph.create_count)
    assert (loaded.friendship != graph.friendship).nnz == 0
    assert np.array_equal(loaded_activity.creators, activity.creators)
    assert np.array_equal(loaded_activity.kinds, activity.kinds)
