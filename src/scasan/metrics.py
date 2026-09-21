"""Evaluation metrics and JSON-friendly result helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata


def auc_from_scores(scores: np.ndarray, n_honest: int) -> float:
    """AUC from the paper, with 0.5 credit for tied honest/Sybil pairs."""

    scores = np.asarray(scores, dtype=np.float64)
    if not 0 < n_honest < scores.size:
        raise ValueError("AUC requires non-empty honest and Sybil regions")
    ranks = rankdata(scores, method="average")
    n_h = n_honest
    n_s = scores.size - n_honest
    honest_rank_sum = float(ranks[:n_h].sum())
    # Mann-Whitney U equals the number of honest>sybil pairs plus half ties.
    u = honest_rank_sum - n_h * (n_h + 1) / 2.0
    return u / (n_h * n_s)


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(type(item).__name__)

    path.write_text(json.dumps(value, indent=2, default=default) + "\n", encoding="utf-8")
