"""Facebook WOSN 2009 preprocessing described in Section 5.1."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
from pathlib import Path
from urllib.request import urlopen
import shutil

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .graph import ActivityGraph, ScaSANGraph, sparse_from_edges


FACEBOOK_LINKS_URL = "https://socialnetworks.mpi-sws.org/data/facebook-links.txt.gz"
FACEBOOK_WALL_URL = "https://socialnetworks.mpi-sws.org/data/facebook-wall.txt.gz"


def download_facebook(destination: str | Path, *, overwrite: bool = False) -> tuple[Path, Path]:
    """Download the two files published by the dataset authors atomically."""

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    for url, name in (
        (FACEBOOK_LINKS_URL, "facebook-links.txt.gz"),
        (FACEBOOK_WALL_URL, "facebook-wall.txt.gz"),
    ):
        output = destination / name
        valid = False
        if output.exists() and not overwrite:
            try:
                with gzip.open(output, "rb") as handle:
                    while handle.read(1024 * 1024):
                        pass
                valid = True
            except (EOFError, OSError):
                valid = False
        if not valid:
            temporary = output.with_name(output.name + ".part")
            try:
                with urlopen(url) as response, temporary.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
                with gzip.open(temporary, "rb") as handle:
                    while handle.read(1024 * 1024):
                        pass
                temporary.replace(output)
            finally:
                if temporary.exists():
                    temporary.unlink()
        outputs.append(output)
    return outputs[0], outputs[1]


def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("rt", encoding="utf-8")


def read_edge_rows(path: str | Path) -> np.ndarray:
    """Read the first two integer columns from a Facebook data file."""

    rows: list[tuple[int, int]] = []
    with _open_text(Path(path)) as handle:
        for line_number, line in enumerate(handle, 1):
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                rows.append((int(parts[0]), int(parts[1])))
            except ValueError as exc:
                raise ValueError(f"invalid row {line_number} in {path}") from exc
    return np.asarray(rows, dtype=np.int64)


def read_wall_rows(path: str | Path) -> np.ndarray:
    """Read ``wall_owner, poster, timestamp`` rows."""

    rows: list[tuple[int, int, int]] = []
    with _open_text(Path(path)) as handle:
        for line_number, line in enumerate(handle, 1):
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                rows.append((int(parts[0]), int(parts[1]), int(parts[2])))
            except ValueError as exc:
                raise ValueError(f"invalid row {line_number} in {path}") from exc
    return np.asarray(rows, dtype=np.int64)


def _map_ids(values: np.ndarray, sorted_ids: np.ndarray) -> np.ndarray:
    mapped = np.searchsorted(sorted_ids, values)
    if np.any(mapped >= sorted_ids.size) or np.any(sorted_ids[mapped] != values):
        raise ValueError("encountered a user ID outside the ID map")
    return mapped


def _friendship_matrix(edges: np.ndarray, user_ids: np.ndarray) -> sparse.csr_matrix:
    u = _map_ids(edges[:, 0], user_ids)
    v = _map_ids(edges[:, 1], user_ids)
    keep = u != v
    u, v = u[keep], v[keep]
    matrix = sparse_from_edges((user_ids.size, user_ids.size), np.r_[u, v], np.r_[v, u])
    matrix.data[:] = 1.0
    matrix.eliminate_zeros()
    return matrix


def connect_to_largest_component(matrix: sparse.csr_matrix, rng: np.random.Generator) -> sparse.csr_matrix:
    """Apply the component repair described by the paper."""

    n_components, labels = connected_components(matrix, directed=False)
    if n_components <= 1:
        return matrix
    counts = np.bincount(labels)
    largest_label = int(np.argmax(counts))
    largest = np.flatnonzero(labels == largest_label)
    outside = np.flatnonzero(labels != largest_label)
    targets = rng.choice(largest, size=outside.size, replace=True)
    additions = sparse_from_edges(
        matrix.shape,
        np.r_[outside, targets],
        np.r_[targets, outside],
    )
    result = (matrix + additions).tocsr()
    result.data[:] = 1.0
    return result


@dataclass
class _ActivityBuild:
    creators: list[int]
    kinds: list[int]
    mentions_a: list[int]
    mentions_u: list[int]
    follows_src: list[int]
    follows_dst: list[int]
    create_count: np.ndarray
    forward_count: np.ndarray
    mc_r: list[int]
    mc_c: list[int]
    mf_r: list[int]
    mf_c: list[int]
    sc_r: list[int]
    sc_c: list[int]
    sf_r: list[int]
    sf_c: list[int]
    lc_r: list[int]
    lc_c: list[int]
    lf_r: list[int]
    lf_c: list[int]


def _empty_build(n: int) -> _ActivityBuild:
    return _ActivityBuild(
        [], [], [], [], [], [], np.zeros(n), np.zeros(n),
        [], [], [], [], [], [], [], [], [], [], [], [],
    )


def _add_aggregated_edge(build: _ActivityBuild, relation: str, src: int, dst: int, dst_kind: int | None = None) -> None:
    if relation == "mention":
        rows, cols = (build.mc_r, build.mc_c) if build.kinds[-1] == 0 else (build.mf_r, build.mf_c)
    elif relation == "follow":
        rows, cols = (build.sc_r, build.sc_c) if dst_kind == 0 else (build.sf_r, build.sf_c)
    elif relation == "like":
        rows, cols = (build.lc_r, build.lc_c) if dst_kind == 0 else (build.lf_r, build.lf_c)
    else:
        raise ValueError(relation)
    rows.append(src)
    cols.append(dst)


def _activities_from_walls(walls: np.ndarray, user_ids: np.ndarray) -> _ActivityBuild:
    n = user_ids.size
    build = _empty_build(n)
    owner = _map_ids(walls[:, 0], user_ids)
    poster = _map_ids(walls[:, 1], user_ids)
    timestamp = walls[:, 2]
    order = np.lexsort((timestamp, owner))
    owner, poster = owner[order], poster[order]

    start = 0
    while start < owner.size:
        end = start + 1
        while end < owner.size and owner[end] == owner[start]:
            end += 1
        wall_posters = poster[start:end]

        block_starts = np.r_[0, np.flatnonzero(wall_posters[1:] != wall_posters[:-1]) + 1]
        block_ends = np.r_[block_starts[1:], wall_posters.size]
        wall_activity_ids: list[int] = []
        for block_index, (left, right) in enumerate(zip(block_starts, block_ends), start=1):
            creator = int(wall_posters[left])
            kind = 0 if block_index % 2 == 1 else 1  # 0: create, 1: forward
            activity_id = len(build.creators)
            build.creators.append(creator)
            build.kinds.append(kind)
            wall_activity_ids.append(activity_id)
            if kind == 0:
                build.create_count[creator] += 1
            else:
                build.forward_count[creator] += 1

            if block_index > 1:
                previous_activity = wall_activity_ids[-2]
                previous_creator = build.creators[previous_activity]
                previous_kind = build.kinds[previous_activity]
                if kind == 1:
                    build.follows_src.append(activity_id)
                    build.follows_dst.append(previous_activity)
                    _add_aggregated_edge(build, "follow", creator, previous_creator, previous_kind)
                if right - left >= 2:
                    build.mentions_a.append(activity_id)
                    build.mentions_u.append(previous_creator)
                    _add_aggregated_edge(build, "mention", creator, previous_creator)

                # Each remaining post likes one preceding block, starting with
                # the most recent and moving backwards as stated in Section 5.1.
                n_likes = right - left - 2
                for offset in range(min(n_likes, block_index - 1)):
                    target_activity = wall_activity_ids[-2 - offset]
                    target_creator = build.creators[target_activity]
                    target_kind = build.kinds[target_activity]
                    _add_aggregated_edge(build, "like", creator, target_creator, target_kind)
        start = end
    return build


def preprocess_facebook(
    links_path: str | Path,
    wall_path: str | Path,
    *,
    rng_seed: int = 2025,
) -> tuple[ScaSANGraph, ActivityGraph]:
    """Convert raw WOSN files to the two graphs needed by the experiments."""

    links = read_edge_rows(links_path)
    walls = read_wall_rows(wall_path)
    user_ids = np.unique(np.r_[links.ravel(), walls[:, :2].ravel()])
    rng = np.random.default_rng(rng_seed)
    friendship = connect_to_largest_component(_friendship_matrix(links, user_ids), rng)
    build = _activities_from_walls(walls, user_ids)
    n, a = user_ids.size, len(build.creators)

    graph = ScaSANGraph(
        user_ids=user_ids,
        friendship=friendship,
        create_count=build.create_count,
        forward_count=build.forward_count,
        mention_created=sparse_from_edges((n, n), build.mc_r, build.mc_c),
        mention_forwarded=sparse_from_edges((n, n), build.mf_r, build.mf_c),
        forward_to_created=sparse_from_edges((n, n), build.sc_r, build.sc_c),
        forward_to_forwarded=sparse_from_edges((n, n), build.sf_r, build.sf_c),
        like_created=sparse_from_edges((n, n), build.lc_r, build.lc_c),
        like_forwarded=sparse_from_edges((n, n), build.lf_r, build.lf_c),
        n_honest=n,
    )
    activity_graph = ActivityGraph(
        user_ids=user_ids,
        friendship=friendship,
        creators=np.asarray(build.creators),
        kinds=np.asarray(build.kinds),
        mentions=sparse_from_edges((a, n), build.mentions_a, build.mentions_u),
        follows=sparse_from_edges((a, a), build.follows_src, build.follows_dst),
        n_honest=n,
    )
    return graph, activity_graph


def select_sybil_window(graph: ActivityGraph, window_size: int = 1000) -> np.ndarray:
    """Select the consecutive-ID window whose activity mean is closest to the global mean."""

    if window_size >= graph.n_users:
        raise ValueError("sybil window must be smaller than the honest graph")
    counts = np.bincount(graph.creators, minlength=graph.n_users).astype(np.float64)
    rolling = np.convolve(counts, np.ones(window_size), mode="valid") / window_size
    start = int(np.argmin(np.abs(rolling - counts.mean())))
    return np.arange(start, start + window_size, dtype=np.int64)


def _block_diag_copy(matrix: sparse.csr_matrix, selected: np.ndarray) -> sparse.csr_matrix:
    return sparse.block_diag((matrix, matrix[selected, :][:, selected]), format="csr")


def duplicate_sybil_region(
    graph: ScaSANGraph,
    activity_graph: ActivityGraph,
    selected: np.ndarray,
    *,
    rng_seed: int = 2025,
) -> tuple[ScaSANGraph, ActivityGraph]:
    """Duplicate the induced selected-user subgraph as the synthetic Sybil region."""

    selected = np.asarray(selected, dtype=np.int64)
    n_h, n_s = graph.n_users, selected.size
    rng = np.random.default_rng(rng_seed)
    sybil_friendship = graph.friendship[selected, :][:, selected]
    sybil_friendship = connect_to_largest_component(sybil_friendship, rng)
    friendship = sparse.block_diag((graph.friendship, sybil_friendship), format="csr")
    user_ids = np.r_[graph.user_ids, -1 - np.arange(n_s)]

    combined = ScaSANGraph(
        user_ids=user_ids,
        friendship=friendship,
        create_count=np.r_[graph.create_count, graph.create_count[selected]],
        forward_count=np.r_[graph.forward_count, graph.forward_count[selected]],
        mention_created=_block_diag_copy(graph.mention_created, selected),
        mention_forwarded=_block_diag_copy(graph.mention_forwarded, selected),
        forward_to_created=_block_diag_copy(graph.forward_to_created, selected),
        forward_to_forwarded=_block_diag_copy(graph.forward_to_forwarded, selected),
        like_created=_block_diag_copy(graph.like_created, selected),
        like_forwarded=_block_diag_copy(graph.like_forwarded, selected),
        n_honest=n_h,
    )

    user_local = np.full(n_h, -1, dtype=np.int64)
    user_local[selected] = np.arange(n_s)
    chosen_activities = np.flatnonzero(user_local[activity_graph.creators] >= 0)
    a_h, a_s = activity_graph.n_activities, chosen_activities.size
    copied_creators = n_h + user_local[activity_graph.creators[chosen_activities]]
    creators = np.r_[activity_graph.creators, copied_creators]
    kinds = np.r_[activity_graph.kinds, activity_graph.kinds[chosen_activities]]
    sybil_mentions = activity_graph.mentions[chosen_activities, :][:, selected]
    mentions = sparse.bmat(
        [
            [activity_graph.mentions, sparse.csr_matrix((a_h, n_s))],
            [sparse.csr_matrix((a_s, n_h)), sybil_mentions],
        ],
        format="csr",
    )
    sybil_follows = activity_graph.follows[chosen_activities, :][:, chosen_activities]
    follows = sparse.block_diag((activity_graph.follows, sybil_follows), format="csr")
    combined_activity = ActivityGraph(
        user_ids=user_ids,
        friendship=friendship,
        creators=creators,
        kinds=kinds,
        mentions=mentions,
        follows=follows,
        n_honest=n_h,
    )
    return combined, combined_activity


def prepare_paper_dataset(
    links_path: str | Path,
    wall_path: str | Path,
    output: str | Path,
    *,
    sybil_users: int = 1000,
    rng_seed: int = 2025,
) -> tuple[ScaSANGraph, ActivityGraph, np.ndarray]:
    honest, honest_activity = preprocess_facebook(links_path, wall_path, rng_seed=rng_seed)
    selected = select_sybil_window(honest_activity, sybil_users)
    graph, activity = duplicate_sybil_region(honest, honest_activity, selected, rng_seed=rng_seed)
    output = Path(output)
    graph.save(output)
    activity.save(output)
    np.save(output / "sybil_window.npy", selected)
    return graph, activity, selected
