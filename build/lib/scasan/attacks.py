"""Attack injection and seed selection for the paper experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import sparse

from .graph import ActivityGraph, ScaSANGraph, sparse_from_edges


@dataclass(frozen=True)
class AttackSummary:
    friendship: int
    incoming_activity: int
    outgoing_activity: int
    v_hh: float
    v_ss: float


def without_natural_likes(graph: ScaSANGraph) -> ScaSANGraph:
    """Return the paper setting, which excludes mapped wall-post like edges."""

    result = graph.copy()
    shape = (result.n_users, result.n_users)
    result.like_created = sparse.csr_matrix(shape)
    result.like_forwarded = sparse.csr_matrix(shape)
    return result


def choose_honest_seeds(
    graph: ScaSANGraph,
    count: int,
    *,
    target_activity_count: int = 9,
    rng: np.random.Generator,
) -> np.ndarray:
    """Choose honest users with the paper's target activity count.

    If a transformed dataset has too few exact matches, the nearest non-zero
    activity-count groups are added in increasing distance from the target.
    """

    activities = graph.activity_count[: graph.n_honest]
    candidates: list[int] = []
    for distance in range(int(max(target_activity_count, activities.max(initial=0))) + 2):
        targets = {target_activity_count - distance, target_activity_count + distance}
        for target in sorted(t for t in targets if t > 0):
            candidates.extend(np.flatnonzero(activities == target).tolist())
        candidates = list(dict.fromkeys(candidates))
        if len(candidates) >= count:
            break
    if len(candidates) < count:
        candidates = np.flatnonzero(activities > 0).tolist()
    if len(candidates) < count:
        raise ValueError("not enough honest users with seed activities")
    return np.sort(rng.choice(np.asarray(candidates), size=count, replace=False))


def _unique_friend_pairs(
    graph: ScaSANGraph,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n_h, n = graph.n_honest, graph.n_users
    if n_h == n:
        raise ValueError("graph has no Sybil region")
    activity = graph.activity_count[:n_h]
    low_activity = np.flatnonzero(activity <= np.median(activity))
    sybils = np.arange(n_h, n)
    chosen: set[tuple[int, int]] = set()
    batch = max(1024, count * 2)
    while len(chosen) < count:
        honest_draw = rng.choice(low_activity, batch)
        sybil_draw = rng.choice(sybils, batch)
        for h, s in zip(honest_draw, sybil_draw):
            if graph.friendship[h, s] == 0:
                chosen.add((int(h), int(s)))
                if len(chosen) == count:
                    break
    pairs = np.asarray(sorted(chosen), dtype=np.int64)
    return pairs[:, 0], pairs[:, 1]


def _region_activities(
    graph: ActivityGraph,
    source_honest: bool,
    *,
    kind: int | None = None,
) -> np.ndarray:
    creator_honest = graph.creators < graph.n_honest
    mask = creator_honest if source_honest else ~creator_honest
    if kind is not None:
        mask &= graph.kinds == kind
    result = np.flatnonzero(mask)
    if result.size == 0:
        raise ValueError("an attack region has no eligible activities")
    return result


def _activity_attack_edges(
    graph: ScaSANGraph,
    activity_graph: ActivityGraph,
    count: int,
    *,
    incoming: bool,
    rng: np.random.Generator,
) -> tuple[dict[str, list[tuple[int, int]]], list[tuple[int, int]], list[tuple[int, int]]]:
    """Generate aggregate edges plus baseline mention/follow edges."""

    additions = {
        "mention_created": [],
        "mention_forwarded": [],
        "forward_to_created": [],
        "forward_to_forwarded": [],
        "like_created": [],
        "like_forwarded": [],
    }
    baseline_mentions: list[tuple[int, int]] = []
    baseline_follows: list[tuple[int, int]] = []

    source_honest = incoming
    target_honest = not incoming
    source_all = _region_activities(activity_graph, source_honest)
    source_forward = _region_activities(activity_graph, source_honest, kind=1)
    target_activities = _region_activities(activity_graph, target_honest)
    target_users = (
        np.arange(0, graph.n_honest)
        if target_honest
        else np.arange(graph.n_honest, graph.n_users)
    )
    source_users = (
        np.arange(0, graph.n_honest)
        if source_honest
        else np.arange(graph.n_honest, graph.n_users)
    )

    # Mention, forwarding and like are selected with equal probability exactly
    # as stated in the parameter-setting paragraph.
    for relation in rng.integers(0, 3, size=count):
        if relation == 0:
            activity = int(rng.choice(source_all))
            user = int(rng.choice(target_users))
            creator = int(activity_graph.creators[activity])
            key = "mention_created" if activity_graph.kinds[activity] == 0 else "mention_forwarded"
            additions[key].append((creator, user))
            baseline_mentions.append((activity, user))
        elif relation == 1:
            source_activity = int(rng.choice(source_forward))
            target_activity = int(rng.choice(target_activities))
            source_creator = int(activity_graph.creators[source_activity])
            target_creator = int(activity_graph.creators[target_activity])
            key = (
                "forward_to_created"
                if activity_graph.kinds[target_activity] == 0
                else "forward_to_forwarded"
            )
            additions[key].append((source_creator, target_creator))
            baseline_follows.append((source_activity, target_activity))
        else:
            user = int(rng.choice(source_users))
            target_activity = int(rng.choice(target_activities))
            target_creator = int(activity_graph.creators[target_activity])
            key = "like_created" if activity_graph.kinds[target_activity] == 0 else "like_forwarded"
            additions[key].append((user, target_creator))
    return additions, baseline_mentions, baseline_follows


def _add_pairs(
    matrix: sparse.csr_matrix,
    pairs: list[tuple[int, int]],
) -> sparse.csr_matrix:
    if not pairs:
        return matrix.copy()
    array = np.asarray(pairs, dtype=np.int64)
    return (matrix + sparse_from_edges(matrix.shape, array[:, 0], array[:, 1])).tocsr()


def inject_attacks(
    graph: ScaSANGraph,
    activity_graph: ActivityGraph,
    *,
    friendship_attacks: int = 3200,
    xi: float = 0.00002,
    zeta: float = 0.27,
    rng_seed: int = 2025,
) -> tuple[ScaSANGraph, ActivityGraph, AttackSummary]:
    """Inject all three attack families from Section 5.1.

    Every call starts from the supplied base graph; experiment sweeps therefore
    do not accidentally accumulate attacks across points.
    """

    rng = np.random.default_rng(rng_seed)
    result = graph.copy()
    activity_result = ActivityGraph(
        user_ids=activity_graph.user_ids.copy(),
        friendship=activity_graph.friendship.copy(),
        creators=activity_graph.creators.copy(),
        kinds=activity_graph.kinds.copy(),
        mentions=activity_graph.mentions.copy(),
        follows=activity_graph.follows.copy(),
        n_honest=activity_graph.n_honest,
    )
    v_hh, v_ss, _, _ = graph.interaction_volumes()
    n_incoming = int(math.ceil(xi * v_hh))
    n_outgoing = int(math.ceil(zeta * v_ss))

    if friendship_attacks:
        h, s = _unique_friend_pairs(result, friendship_attacks, rng)
        friend_add = sparse_from_edges(
            result.friendship.shape,
            np.r_[h, s],
            np.r_[s, h],
        )
        result.friendship = (result.friendship + friend_add).tocsr()
        result.friendship.data[:] = 1.0
        activity_result.friendship = result.friendship.copy()

    all_mentions: list[tuple[int, int]] = []
    all_follows: list[tuple[int, int]] = []
    for count, incoming in ((n_incoming, True), (n_outgoing, False)):
        additions, mentions, follows = _activity_attack_edges(
            result,
            activity_result,
            count,
            incoming=incoming,
            rng=rng,
        )
        for name, pairs in additions.items():
            setattr(result, name, _add_pairs(getattr(result, name), pairs))
        all_mentions.extend(mentions)
        all_follows.extend(follows)

    activity_result.mentions = _add_pairs(activity_result.mentions, all_mentions)
    activity_result.follows = _add_pairs(activity_result.follows, all_follows)
    summary = AttackSummary(friendship_attacks, n_incoming, n_outgoing, v_hh, v_ss)
    return result, activity_result, summary
