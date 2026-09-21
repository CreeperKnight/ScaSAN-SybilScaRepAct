"""Small graphs used by the paper's motivating experiments."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from .graph import ActivityGraph, ScaSANGraph, sparse_from_edges


def toy_graphs() -> tuple[ScaSANGraph, ActivityGraph]:
    """Return the weighted ScaSAN (Fig. 4) and original SAN (Fig. 2)."""

    n = 5
    friendship_pairs = [
        (0, 1, 3), (0, 2, 2), (1, 2, 3), (1, 3, 1),
        (1, 4, 2), (2, 3, 1), (2, 4, 2), (3, 4, 3),
    ]
    rows, cols, weights = [], [], []
    for u, v, weight in friendship_pairs:
        rows.extend((u, v))
        cols.extend((v, u))
        weights.extend((weight, weight))
    friendship = sparse_from_edges((n, n), rows, cols, weights)

    def edges(items):
        return sparse_from_edges(
            (n, n),
            [item[0] for item in items],
            [item[1] for item in items],
            [item[2] for item in items],
        )

    scasan = ScaSANGraph(
        user_ids=np.arange(1, n + 1),
        friendship=friendship,
        create_count=np.asarray([2, 0, 2, 1, 0], dtype=float),
        forward_count=np.asarray([0, 0, 1, 0, 1], dtype=float),
        mention_created=edges([(0, 1, 2), (2, 0, 2)]),
        mention_forwarded=edges([(2, 0, 1), (4, 1, 1)]),
        forward_to_created=edges([(2, 0, 1), (4, 3, 1)]),
        forward_to_forwarded=sparse.csr_matrix((n, n)),
        like_created=edges([(2, 0, 1), (4, 3, 1)]),
        like_forwarded=sparse.csr_matrix((n, n)),
        n_honest=3,
    )

    creators = np.asarray([0, 0, 2, 2, 2, 3, 4])
    kinds = np.asarray([0, 0, 1, 0, 0, 0, 1])
    mentions = sparse_from_edges(
        (7, n),
        [0, 1, 6, 2, 3, 4],
        [1, 1, 1, 0, 0, 0],
    )
    follows = sparse_from_edges((7, 7), [2, 6], [1, 5])
    activity = ActivityGraph(
        user_ids=np.arange(1, n + 1),
        friendship=(friendship > 0).astype(float),
        creators=creators,
        kinds=kinds,
        mentions=mentions,
        follows=follows,
        n_honest=3,
    )
    return scasan, activity


def aggregated_toy_activity_graph(graph: ScaSANGraph) -> ActivityGraph:
    """Convert active ScaSAN aggregate nodes to an unweighted SAN (Fig. 10)."""

    created = graph.active_created
    forwarded = graph.active_forwarded
    creators = np.r_[created, forwarded]
    kinds = np.r_[np.zeros(created.size, dtype=np.int8), np.ones(forwarded.size, dtype=np.int8)]
    c_lookup = np.full(graph.n_users, -1, dtype=np.int64)
    f_lookup = np.full(graph.n_users, -1, dtype=np.int64)
    c_lookup[created] = np.arange(created.size)
    f_lookup[forwarded] = created.size + np.arange(forwarded.size)

    mention_rows, mention_cols = [], []
    for source_users, matrix, lookup in (
        (created, graph.mention_created, c_lookup),
        (forwarded, graph.mention_forwarded, f_lookup),
    ):
        coo = (matrix[source_users, :] > 0).tocoo()
        mention_rows.extend(lookup[source_users[coo.row]])
        mention_cols.extend(coo.col)

    follow_rows, follow_cols = [], []
    for matrix, target_lookup in (
        (graph.forward_to_created, c_lookup),
        (graph.forward_to_forwarded, f_lookup),
    ):
        coo = (matrix[forwarded, :] > 0).tocoo()
        valid = target_lookup[coo.col] >= 0
        follow_rows.extend(f_lookup[forwarded[coo.row[valid]]])
        follow_cols.extend(target_lookup[coo.col[valid]])

    a = creators.size
    return ActivityGraph(
        user_ids=graph.user_ids.copy(),
        friendship=(graph.friendship > 0).astype(float),
        creators=creators,
        kinds=kinds,
        mentions=sparse_from_edges((a, graph.n_users), mention_rows, mention_cols),
        follows=sparse_from_edges((a, a), follow_rows, follow_cols),
        n_honest=graph.n_honest,
    )
