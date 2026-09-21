"""Sparse implementations of the three algorithms compared in the paper."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import math

import numpy as np
from scipy import sparse

from .graph import ActivityGraph, ScaSANGraph


def _row_normalize(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    matrix = matrix.tocsr().astype(np.float64)
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    inv = np.zeros_like(totals)
    np.divide(1.0, totals, out=inv, where=totals > 0)
    return (sparse.diags(inv) @ matrix).tocsr()


def _l2_delta(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right))


@dataclass(frozen=True)
class RunResult:
    scores: np.ndarray
    iterations: int
    converged: bool
    build_seconds: float
    propagation_seconds: float
    residual: float


class ScaSANOperator:
    """Matrix-free form of Algorithm 2's transition matrix.

    Dummy activity nodes are omitted.  This is equivalent to using the
    paper's "at most 3N" graph, but avoids zero rows and cuts memory use.
    """

    def __init__(
        self,
        graph: ScaSANGraph,
        seeds: np.ndarray,
        *,
        alpha: float,
        beta: float,
        gamma: float,
        lambda1: float,
        lambda2: float,
    ) -> None:
        if not (0 < alpha < 1 and 0 < alpha + beta < 1 and 0 < alpha + gamma < 1):
            raise ValueError("require 0 < alpha < 1, alpha + beta < 1, and alpha + gamma < 1")
        if lambda2 <= lambda1 or lambda1 < 1:
            raise ValueError("require lambda2 > lambda1 >= 1 (Figure 13 includes lambda1=1)")
        seeds = np.unique(np.asarray(seeds, dtype=np.int64))
        if seeds.size == 0 or seeds.min() < 0 or seeds.max() >= graph.n_users:
            raise ValueError("seeds must contain valid user indices")

        self.graph = graph
        self.seeds = seeds
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.lambda1, self.lambda2 = lambda1, lambda2
        self.created_users = graph.active_created
        self.forwarded_users = graph.active_forwarded
        self.n = graph.n_users
        self.nc = self.created_users.size
        self.nf = self.forwarded_users.size

        seed_user = np.zeros(self.n, dtype=np.float64)
        seed_user[seeds] = 1.0 / seeds.size
        self.seed_user = seed_user

        created_lookup = np.full(self.n, -1, dtype=np.int64)
        forwarded_lookup = np.full(self.n, -1, dtype=np.int64)
        created_lookup[self.created_users] = np.arange(self.nc)
        forwarded_lookup[self.forwarded_users] = np.arange(self.nf)
        self.created_lookup = created_lookup
        self.forwarded_lookup = forwarded_lookup

        seed_created = created_lookup[seeds]
        seed_forwarded = forwarded_lookup[seeds]
        seed_created = seed_created[seed_created >= 0]
        seed_forwarded = seed_forwarded[seed_forwarded >= 0]
        n_seed_activities = seed_created.size + seed_forwarded.size
        if n_seed_activities == 0:
            raise ValueError("at least one seed must have a created or forwarded activity")
        self.seed_created = np.zeros(self.nc, dtype=np.float64)
        self.seed_forwarded = np.zeros(self.nf, dtype=np.float64)
        self.seed_created[seed_created] = 1.0 / n_seed_activities
        self.seed_forwarded[seed_forwarded] = 1.0 / n_seed_activities

        self.friend = _row_normalize(graph.friendship)
        self.u_to_c, self.u_to_f = self._user_activity_transitions()
        self.c_to_u = self._created_user_transitions()
        self.f_to_u = self._forwarded_user_transitions()
        self.f_to_c, self.f_to_f = self._following_transitions()

    @property
    def size(self) -> int:
        return self.n + self.nc + self.nf

    def initial(self) -> np.ndarray:
        state = np.zeros(self.size, dtype=np.float64)
        state[: self.n] = self.seed_user
        return state

    def _fallback_rows(self, rows: np.ndarray) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        if rows.size == 0:
            return sparse.csr_matrix((self.n, self.nc)), sparse.csr_matrix((self.n, self.nf))
        c_seed = np.flatnonzero(self.seed_created)
        f_seed = np.flatnonzero(self.seed_forwarded)
        c_rows = np.repeat(rows, c_seed.size)
        f_rows = np.repeat(rows, f_seed.size)
        c_cols = np.tile(c_seed, rows.size)
        f_cols = np.tile(f_seed, rows.size)
        c_data = np.tile(self.seed_created[c_seed], rows.size)
        f_data = np.tile(self.seed_forwarded[f_seed], rows.size)
        return (
            sparse.csr_matrix((c_data, (c_rows, c_cols)), shape=(self.n, self.nc)),
            sparse.csr_matrix((f_data, (f_rows, f_cols)), shape=(self.n, self.nf)),
        )

    def _user_activity_transitions(self) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        g = self.graph
        c = sparse.diags(self.lambda2 * g.create_count, format="csr")[:, self.created_users]
        c = (c + g.like_created[:, self.created_users]).tocsr()
        f = sparse.diags(self.lambda1 * g.forward_count, format="csr")[:, self.forwarded_users]
        f = (f + g.like_forwarded[:, self.forwarded_users]).tocsr()
        totals = np.asarray(c.sum(axis=1)).ravel() + np.asarray(f.sum(axis=1)).ravel()
        inv = np.zeros_like(totals)
        np.divide(1.0, totals, out=inv, where=totals > 0)
        scale = sparse.diags(inv)
        c, f = (scale @ c).tocsr(), (scale @ f).tocsr()
        missing = np.flatnonzero(totals == 0)
        fallback_c, fallback_f = self._fallback_rows(missing)
        return (c + fallback_c).tocsr(), (f + fallback_f).tocsr()

    def _created_user_transitions(self) -> sparse.csr_matrix:
        owner_edges = sparse.csr_matrix(
            (
                self.graph.create_count[self.created_users],
                (np.arange(self.nc), self.created_users),
            ),
            shape=(self.nc, self.n),
        )
        mentions = self.graph.mention_created[self.created_users, :]
        return _row_normalize(owner_edges + mentions)

    def _forwarded_user_transitions(self) -> sparse.csr_matrix:
        owner_edges = sparse.csr_matrix(
            (
                self.graph.forward_count[self.forwarded_users],
                (np.arange(self.nf), self.forwarded_users),
            ),
            shape=(self.nf, self.n),
        )
        mentions = self.graph.mention_forwarded[self.forwarded_users, :]
        return _row_normalize(owner_edges + mentions)

    def _following_transitions(self) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        sc = self.graph.forward_to_created[self.forwarded_users, :][:, self.created_users]
        sf = self.graph.forward_to_forwarded[self.forwarded_users, :][:, self.forwarded_users]
        totals = np.asarray(sc.sum(axis=1)).ravel() + np.asarray(sf.sum(axis=1)).ravel()
        inv = np.zeros_like(totals)
        np.divide(1.0, totals, out=inv, where=totals > 0)
        scale = sparse.diags(inv)
        sc, sf = (scale @ sc).tocsr(), (scale @ sf).tocsr()

        # Algorithm 2 leaves this rare zero-denominator case implicit.  Sending
        # its gamma mass to seed activities preserves a stochastic transition.
        missing = np.flatnonzero(totals == 0)
        if missing.size:
            c_seed = np.flatnonzero(self.seed_created)
            f_seed = np.flatnonzero(self.seed_forwarded)
            sc += sparse.csr_matrix(
                (
                    np.tile(self.seed_created[c_seed], missing.size),
                    (np.repeat(missing, c_seed.size), np.tile(c_seed, missing.size)),
                ),
                shape=(self.nf, self.nc),
            )
            sf += sparse.csr_matrix(
                (
                    np.tile(self.seed_forwarded[f_seed], missing.size),
                    (np.repeat(missing, f_seed.size), np.tile(f_seed, missing.size)),
                ),
                shape=(self.nf, self.nf),
            )
        return sc.tocsr(), sf.tocsr()

    def step(self, state: np.ndarray) -> np.ndarray:
        u = state[: self.n]
        c = state[self.n : self.n + self.nc]
        f = state[self.n + self.nc :]
        out_u = self.beta * (u @ self.friend)
        out_u += self.alpha * u.sum() * self.seed_user
        out_u += (1.0 - self.alpha) * (c @ self.c_to_u)
        out_u += (1.0 - self.alpha - self.gamma) * (f @ self.f_to_u)

        activity_mass = self.alpha * (c.sum() + f.sum())
        out_c = (1.0 - self.alpha - self.beta) * (u @ self.u_to_c)
        out_c += activity_mass * self.seed_created
        out_c += self.gamma * (f @ self.f_to_c)
        out_f = (1.0 - self.alpha - self.beta) * (u @ self.u_to_f)
        out_f += activity_mass * self.seed_forwarded
        out_f += self.gamma * (f @ self.f_to_f)
        return np.concatenate((np.asarray(out_u).ravel(), np.asarray(out_c).ravel(), np.asarray(out_f).ravel()))


def theoretical_iterations(epsilon: float, alpha: float, beta: float, gamma: float, n_seeds: int) -> int:
    """Equation (2), including the leading factor of two."""

    contraction = 1.0 - min(1.0 - alpha - gamma, alpha + beta) * alpha / n_seeds
    return int(2 * math.ceil(math.log(0.5 * epsilon) / math.log(contraction)))


def sybil_scarepact(
    graph: ScaSANGraph,
    seeds: np.ndarray,
    *,
    alpha: float = 0.1,
    beta: float = 0.04,
    gamma: float = 0.425,
    lambda1: float = 2.0,
    lambda2: float = 3.0,
    epsilon: float = 1e-3,
    max_iterations: int = 100_000,
    convergence: str = "residual",
) -> RunResult:
    build_start = perf_counter()
    operator = ScaSANOperator(
        graph,
        seeds,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        lambda1=lambda1,
        lambda2=lambda2,
    )
    state = operator.initial()
    build_seconds = perf_counter() - build_start
    if convergence == "theoretical":
        target_iterations = min(theoretical_iterations(epsilon, alpha, beta, gamma, len(seeds)), max_iterations)
    elif convergence == "residual":
        target_iterations = max_iterations
    else:
        raise ValueError("convergence must be 'residual' or 'theoretical'")

    start = perf_counter()
    residual, converged = math.inf, False
    for iteration in range(1, target_iterations + 1):
        updated = operator.step(state)
        residual = _l2_delta(updated, state)
        state = updated
        if convergence == "residual" and residual <= epsilon:
            converged = True
            break
    else:
        iteration = target_iterations
    if convergence == "theoretical":
        converged = target_iterations < max_iterations or target_iterations == theoretical_iterations(
            epsilon, alpha, beta, gamma, len(seeds)
        )
    propagation_seconds = perf_counter() - start

    degree = graph.friendship_degree
    scores = np.divide(state[: graph.n_users], degree, out=np.zeros(graph.n_users), where=degree > 0)
    return RunResult(scores, iteration, converged, build_seconds, propagation_seconds, residual)


def sybil_friendship(
    graph: ScaSANGraph,
    seeds: np.ndarray,
    *,
    restart: float = 0.1,
    epsilon: float = 1e-3,
    max_iterations: int = 100_000,
) -> RunResult:
    """Friendship-only personalized random-walk baseline.

    The paper does not specify this variant's transition rule.  We use the
    natural friendship-only restriction: fixed restart to honest seeds and
    degree-normalized scores.  Its fixed ``restart`` is intentionally not
    changed in the alpha/beta sensitivity experiments.
    """

    build_start = perf_counter()
    transition = _row_normalize(graph.friendship)
    seed_dist = np.zeros(graph.n_users)
    seeds = np.unique(np.asarray(seeds, dtype=np.int64))
    seed_dist[seeds] = 1.0 / seeds.size
    state = seed_dist.copy()
    build_seconds = perf_counter() - build_start
    start = perf_counter()
    residual, converged = math.inf, False
    for iteration in range(1, max_iterations + 1):
        updated = (1.0 - restart) * (state @ transition) + restart * seed_dist
        updated = np.asarray(updated).ravel()
        residual = _l2_delta(updated, state)
        state = updated
        if residual <= epsilon:
            converged = True
            break
    propagation_seconds = perf_counter() - start
    degree = graph.friendship_degree
    scores = np.divide(state, degree, out=np.zeros_like(state), where=degree > 0)
    return RunResult(scores, iteration, converged, build_seconds, propagation_seconds, residual)


class SybilSocActOperator:
    """Matrix-free implementation of Algorithms 1--3 in Zhang et al. (2023)."""

    def __init__(
        self,
        graph: ActivityGraph,
        seeds: np.ndarray,
        *,
        jump: float = 0.15,
        k: int = 0,
        activity_steps: int = 1,
        user_layer_scale: float = 0.05,
        degree_decay: float = 0.9,
        activity_layer_scale: float = 0.5,
    ) -> None:
        if not 0 <= jump < 1 or k < 0 or activity_steps < 1:
            raise ValueError("invalid SybilSocActNet parameters")
        self.graph = graph
        self.n, self.a = graph.n_users, graph.n_activities
        self.k, self.activity_steps = k, activity_steps
        seeds = np.unique(np.asarray(seeds, dtype=np.int64))
        self.seed_user = np.zeros(self.n)
        self.seed_user[seeds] = 1.0 / seeds.size
        seed_acts = np.flatnonzero(np.isin(graph.creators, seeds))
        if seed_acts.size == 0:
            raise ValueError("SybilSocActNet requires at least one seed activity")
        self.seed_activity = np.zeros(self.a)
        self.seed_activity[seed_acts] = 1.0 / seed_acts.size

        self.friend = _row_normalize(graph.friendship)
        self.jump = jump
        self.follow = self._activity_follow_transition()
        self.ua = self._user_activity_transition()

        degree = np.asarray(graph.friendship.sum(axis=1)).ravel()
        activity_degree = np.bincount(graph.creators, minlength=self.n)
        safe_degree = np.maximum(degree, 1.0)
        user_lambda = user_layer_scale * np.power(degree_decay, np.log2(safe_degree))
        user_lambda[activity_degree == 0] = 1.0
        self.layer_probability = np.concatenate((user_lambda, np.full(self.a, activity_layer_scale)))

    def _activity_follow_transition(self) -> sparse.csr_matrix:
        base = _row_normalize(self.graph.follows)
        rowsum = np.asarray(self.graph.follows.sum(axis=1)).ravel()
        active = rowsum > 0
        transition = (1.0 - self.jump) * base
        seed_cols = np.flatnonzero(self.seed_activity)
        all_rows = np.arange(self.a)
        weights = np.where(active, self.jump, 1.0)
        teleport = sparse.csr_matrix(
            (
                np.repeat(weights, seed_cols.size) * np.tile(self.seed_activity[seed_cols], self.a),
                (np.repeat(all_rows, seed_cols.size), np.tile(seed_cols, self.a)),
            ),
            shape=(self.a, self.a),
        )
        return (transition + teleport).tocsr()

    def _user_activity_transition(self) -> sparse.csr_matrix:
        counts = np.bincount(self.graph.creators, minlength=self.n)
        act_ids = np.arange(self.a)
        user_to_act = sparse.csr_matrix(
            (
                np.divide(1.0, counts[self.graph.creators]),
                (self.graph.creators, act_ids),
            ),
            shape=(self.n, self.a),
        )
        no_activity = np.flatnonzero(counts == 0)
        user_self = sparse.csr_matrix(
            (np.ones(no_activity.size), (no_activity, no_activity)), shape=(self.n, self.n)
        )
        creator_edges = sparse.csr_matrix(
            (np.ones(self.a), (act_ids, self.graph.creators)), shape=(self.a, self.n)
        )
        act_to_user = _row_normalize(creator_edges + (self.graph.mentions > 0).astype(np.float64))
        return sparse.bmat(
            [[user_self, user_to_act], [act_to_user, sparse.csr_matrix((self.a, self.a))]],
            format="csr",
        )

    def _same_layer_step(self, state: np.ndarray) -> np.ndarray:
        users = (1.0 - self.jump) * (state[: self.n] @ self.friend)
        users += self.jump * state[: self.n].sum() * self.seed_user
        activities = state[self.n :]
        for _ in range(self.activity_steps):
            activities = activities @ self.follow
        return np.concatenate((np.asarray(users).ravel(), np.asarray(activities).ravel()))

    def step(self, state: np.ndarray) -> np.ndarray:
        same = self._same_layer_step(state * self.layer_probability)
        cross = state * (1.0 - self.layer_probability)
        for _ in range(2 * self.k + 1):
            cross = cross @ self.ua
        return same + np.asarray(cross).ravel()

    def initial(self) -> np.ndarray:
        return np.concatenate((self.seed_user.copy(), np.zeros(self.a)))


def sybil_socactnet(
    graph: ActivityGraph,
    seeds: np.ndarray,
    *,
    jump: float = 0.15,
    k: int = 0,
    activity_steps: int = 1,
    epsilon: float = 1e-3,
    max_iterations: int = 100_000,
) -> RunResult:
    build_start = perf_counter()
    op = SybilSocActOperator(graph, seeds, jump=jump, k=k, activity_steps=activity_steps)
    state = op.initial()
    build_seconds = perf_counter() - build_start
    start = perf_counter()
    residual, converged = math.inf, False
    for iteration in range(1, max_iterations + 1):
        updated = op.step(state)
        residual = _l2_delta(updated, state)
        state = updated
        if residual <= epsilon:
            converged = True
            break
    propagation_seconds = perf_counter() - start

    friend_sources = np.asarray((graph.friendship > 0).sum(axis=0)).ravel()
    mention_sources = np.asarray((graph.mentions > 0).sum(axis=0)).ravel()
    follow_sources_per_activity = np.asarray((graph.follows > 0).sum(axis=0)).ravel()
    follow_sources = np.bincount(graph.creators, weights=follow_sources_per_activity, minlength=graph.n_users)
    sources = friend_sources + mention_sources + follow_sources
    scores = np.divide(state[: graph.n_users], sources, out=np.zeros(graph.n_users), where=sources > 0)
    return RunResult(scores, iteration, converged, build_seconds, propagation_seconds, residual)
