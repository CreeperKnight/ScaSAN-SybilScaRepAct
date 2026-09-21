"""Sparse graph containers used by the ScaSAN experiments.

The paper writes all activity matrices as ``N x N`` matrices.  Keeping that
layout makes the implementation easy to audit while CSR storage keeps the
real Facebook experiment practical.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import sparse


CSR = sparse.csr_matrix


def _csr(matrix: sparse.spmatrix | np.ndarray, shape: tuple[int, int]) -> CSR:
    result = sparse.csr_matrix(matrix, dtype=np.float64, shape=shape)
    result.sum_duplicates()
    result.eliminate_zeros()
    return result


def sparse_from_edges(
    shape: tuple[int, int],
    rows: Iterable[int],
    cols: Iterable[int],
    values: Iterable[float] | None = None,
) -> CSR:
    """Build a CSR matrix and sum repeated edges."""

    row = np.asarray(list(rows), dtype=np.int64)
    col = np.asarray(list(cols), dtype=np.int64)
    data = np.ones(row.size, dtype=np.float64) if values is None else np.asarray(list(values), dtype=np.float64)
    return _csr(sparse.coo_matrix((data, (row, col)), shape=shape), shape)


@dataclass
class ScaSANGraph:
    """The weighted mixed graph from Section 3.1 of the paper.

    ``create_count`` and ``forward_count`` are the diagonals of C and F.
    The remaining activity matrices are split by destination/source activity
    type so each stored sparse matrix stays ``N x N``.
    """

    user_ids: np.ndarray
    friendship: CSR
    create_count: np.ndarray
    forward_count: np.ndarray
    mention_created: CSR
    mention_forwarded: CSR
    forward_to_created: CSR
    forward_to_forwarded: CSR
    like_created: CSR
    like_forwarded: CSR
    n_honest: int

    def __post_init__(self) -> None:
        self.user_ids = np.asarray(self.user_ids)
        n = self.user_ids.size
        shape = (n, n)
        self.friendship = _csr(self.friendship, shape)
        self.create_count = np.asarray(self.create_count, dtype=np.float64)
        self.forward_count = np.asarray(self.forward_count, dtype=np.float64)
        for name in (
            "mention_created",
            "mention_forwarded",
            "forward_to_created",
            "forward_to_forwarded",
            "like_created",
            "like_forwarded",
        ):
            setattr(self, name, _csr(getattr(self, name), shape))
        self.validate()

    @property
    def n_users(self) -> int:
        return int(self.user_ids.size)

    @property
    def honest_mask(self) -> np.ndarray:
        mask = np.zeros(self.n_users, dtype=bool)
        mask[: self.n_honest] = True
        return mask

    @property
    def sybil_mask(self) -> np.ndarray:
        return ~self.honest_mask

    @property
    def activity_count(self) -> np.ndarray:
        return self.create_count + self.forward_count

    @property
    def friendship_degree(self) -> np.ndarray:
        return np.asarray(self.friendship.sum(axis=1)).ravel()

    @property
    def active_created(self) -> np.ndarray:
        return np.flatnonzero(self.create_count > 0)

    @property
    def active_forwarded(self) -> np.ndarray:
        return np.flatnonzero(self.forward_count > 0)

    def validate(self) -> None:
        n = self.n_users
        if not 0 < self.n_honest <= n:
            raise ValueError("n_honest must be in [1, n_users]")
        if self.create_count.shape != (n,) or self.forward_count.shape != (n,):
            raise ValueError("activity-count vectors must have length n_users")
        if np.any(self.create_count < 0) or np.any(self.forward_count < 0):
            raise ValueError("activity counts must be non-negative")
        for name in (
            "friendship",
            "mention_created",
            "mention_forwarded",
            "forward_to_created",
            "forward_to_forwarded",
            "like_created",
            "like_forwarded",
        ):
            matrix = getattr(self, name)
            if matrix.shape != (n, n):
                raise ValueError(f"{name} must have shape {(n, n)}")
            if matrix.nnz and np.min(matrix.data) < 0:
                raise ValueError(f"{name} contains a negative weight")

    def copy(self) -> "ScaSANGraph":
        return ScaSANGraph(
            user_ids=self.user_ids.copy(),
            friendship=self.friendship.copy(),
            create_count=self.create_count.copy(),
            forward_count=self.forward_count.copy(),
            mention_created=self.mention_created.copy(),
            mention_forwarded=self.mention_forwarded.copy(),
            forward_to_created=self.forward_to_created.copy(),
            forward_to_forwarded=self.forward_to_forwarded.copy(),
            like_created=self.like_created.copy(),
            like_forwarded=self.like_forwarded.copy(),
            n_honest=self.n_honest,
        )

    def interaction_volumes(self) -> tuple[float, float, float, float]:
        """Return ``(V_HH, V_SS, V_IAA, V_OAA)`` from Definitions 2--3."""

        h = slice(0, self.n_honest)
        s = slice(self.n_honest, self.n_users)
        mats = (
            self.mention_created,
            self.mention_forwarded,
            self.forward_to_created,
            self.forward_to_forwarded,
            self.like_created,
            self.like_forwarded,
        )
        v_hh = sum(float(m[h, h].sum()) for m in mats)
        v_ss = sum(float(m[s, s].sum()) for m in mats)
        v_iaa = sum(float(m[h, s].sum()) for m in mats)
        v_oaa = sum(float(m[s, h].sum()) for m in mats)
        return v_hh, v_ss, v_iaa, v_oaa

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "metadata.npz",
            user_ids=self.user_ids,
            create_count=self.create_count,
            forward_count=self.forward_count,
            n_honest=np.asarray(self.n_honest, dtype=np.int64),
        )
        for name in (
            "friendship",
            "mention_created",
            "mention_forwarded",
            "forward_to_created",
            "forward_to_forwarded",
            "like_created",
            "like_forwarded",
        ):
            sparse.save_npz(directory / f"{name}.npz", getattr(self, name), compressed=True)

    @classmethod
    def load(cls, directory: str | Path) -> "ScaSANGraph":
        directory = Path(directory)
        with np.load(directory / "metadata.npz", allow_pickle=False) as meta:
            kwargs = {
                "user_ids": meta["user_ids"],
                "create_count": meta["create_count"],
                "forward_count": meta["forward_count"],
                "n_honest": int(meta["n_honest"]),
            }
        for name in (
            "friendship",
            "mention_created",
            "mention_forwarded",
            "forward_to_created",
            "forward_to_forwarded",
            "like_created",
            "like_forwarded",
        ):
            kwargs[name] = sparse.load_npz(directory / f"{name}.npz").tocsr()
        return cls(**kwargs)


@dataclass
class ActivityGraph:
    """Unaggregated SAN used by the SybilSocActNet/Sybil_SAN baseline."""

    user_ids: np.ndarray
    friendship: CSR
    creators: np.ndarray
    kinds: np.ndarray
    mentions: CSR
    follows: CSR
    n_honest: int

    def __post_init__(self) -> None:
        self.user_ids = np.asarray(self.user_ids)
        self.creators = np.asarray(self.creators, dtype=np.int64)
        self.kinds = np.asarray(self.kinds, dtype=np.int8)
        n, a = self.user_ids.size, self.creators.size
        self.friendship = _csr(self.friendship, (n, n))
        self.mentions = _csr(self.mentions, (a, n))
        self.follows = _csr(self.follows, (a, a))
        if self.creators.size and (self.creators.min() < 0 or self.creators.max() >= n):
            raise ValueError("activity creator index out of range")
        if self.kinds.shape != (a,) or np.any((self.kinds < 0) | (self.kinds > 1)):
            raise ValueError("kinds must contain one 0/1 value per activity")

    @property
    def n_users(self) -> int:
        return int(self.user_ids.size)

    @property
    def n_activities(self) -> int:
        return int(self.creators.size)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "activity_metadata.npz",
            user_ids=self.user_ids,
            creators=self.creators,
            kinds=self.kinds,
            n_honest=np.asarray(self.n_honest, dtype=np.int64),
        )
        sparse.save_npz(directory / "activity_friendship.npz", self.friendship, compressed=True)
        sparse.save_npz(directory / "activity_mentions.npz", self.mentions, compressed=True)
        sparse.save_npz(directory / "activity_follows.npz", self.follows, compressed=True)

    @classmethod
    def load(cls, directory: str | Path) -> "ActivityGraph":
        directory = Path(directory)
        with np.load(directory / "activity_metadata.npz", allow_pickle=False) as meta:
            return cls(
                user_ids=meta["user_ids"],
                creators=meta["creators"],
                kinds=meta["kinds"],
                n_honest=int(meta["n_honest"]),
                friendship=sparse.load_npz(directory / "activity_friendship.npz"),
                mentions=sparse.load_npz(directory / "activity_mentions.npz"),
                follows=sparse.load_npz(directory / "activity_follows.npz"),
            )
