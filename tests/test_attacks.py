import numpy as np

from scasan.attacks import inject_attacks
from scasan.synthetic import toy_graphs


def test_attack_injection_is_reproducible_and_non_mutating():
    graph, activity = toy_graphs()
    before = graph.friendship.nnz
    attacked1, act1, summary1 = inject_attacks(
        graph, activity, friendship_attacks=2, xi=0.2, zeta=0.5, rng_seed=7,
    )
    attacked2, act2, summary2 = inject_attacks(
        graph, activity, friendship_attacks=2, xi=0.2, zeta=0.5, rng_seed=7,
    )
    assert graph.friendship.nnz == before
    assert attacked1.friendship.nnz == before + 4
    assert (attacked1.friendship != attacked2.friendship).nnz == 0
    assert (act1.mentions != act2.mentions).nnz == 0
    assert summary1 == summary2
    assert summary1.incoming_activity == int(np.ceil(0.2 * summary1.v_hh))
