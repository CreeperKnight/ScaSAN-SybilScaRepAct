"""End-to-end reproduction of every numerical experiment in the paper."""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from .algorithms import RunResult, sybil_friendship, sybil_scarepact, sybil_socactnet
from .attacks import choose_honest_seeds, inject_attacks, without_natural_likes
from .graph import ActivityGraph, ScaSANGraph
from .metrics import auc_from_scores, write_json
from .synthetic import aggregated_toy_activity_graph, toy_graphs


METHOD_LABELS = {
    "scasan": "SybilScaRepAct",
    "socact": "SybilSocActNet",
    "friendship": "SybilFriendship",
}
METHOD_STYLES = {
    "scasan": dict(marker="o", color="#1f77b4"),
    "socact": dict(marker="s", color="#d62728"),
    "friendship": dict(marker="^", color="#2ca02c"),
}


def _mean(records: list[dict[str, Any]], experiment: str, method: str, metric: str = "auc"):
    rows = [r for r in records if r["experiment"] == experiment and r["method"] == method]
    values = sorted({float(r["x"]) for r in rows})
    return np.asarray(values), np.asarray(
        [np.mean([float(r[metric]) for r in rows if float(r["x"]) == x]) for x in values]
    )


def _save_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in records for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _plot_methods(
    axis,
    records: list[dict[str, Any]],
    experiment: str,
    *,
    xlabel: str,
    ylabel: str = "AUC",
    metric: str = "auc",
    methods: tuple[str, ...] = ("scasan", "socact", "friendship"),
    xscale: str | None = None,
) -> None:
    for method in methods:
        x, y = _mean(records, experiment, method, metric)
        if x.size:
            axis.plot(x, y, label=METHOD_LABELS[method], **METHOD_STYLES[method])
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    if xscale:
        axis.set_xscale(xscale)


class ExperimentSuite:
    def __init__(self, data_dir: str | Path, config: dict[str, Any]) -> None:
        self.config = config
        self.base_graph = without_natural_likes(ScaSANGraph.load(data_dir))
        self.base_activity = ActivityGraph.load(data_dir)
        self.records: list[dict[str, Any]] = []
        self._graph_cache: dict[tuple[Any, ...], tuple[ScaSANGraph, ActivityGraph]] = {}
        self._run_cache: dict[tuple[Any, ...], RunResult] = {}

    @property
    def defaults(self) -> dict[str, Any]:
        return self.config["defaults"]

    def _seeds(self, count: int, repeat: int) -> np.ndarray:
        rng = np.random.default_rng(int(self.config["random_seed"]) + 100_003 * repeat + count)
        return choose_honest_seeds(
            self.base_graph,
            count,
            target_activity_count=int(self.config["seed_activity_count"]),
            rng=rng,
        )

    def _attacked(
        self,
        friend_attacks: int,
        xi: float,
        zeta: float,
        repeat: int,
    ) -> tuple[ScaSANGraph, ActivityGraph]:
        key = (int(friend_attacks), float(xi), float(zeta), repeat)
        if key not in self._graph_cache:
            graph, activity, _ = inject_attacks(
                self.base_graph,
                self.base_activity,
                friendship_attacks=int(friend_attacks),
                xi=float(xi),
                zeta=float(zeta),
                rng_seed=int(self.config["random_seed"]) + 10_007 * repeat,
            )
            self._graph_cache[key] = (graph, activity)
        return self._graph_cache[key]

    def _evaluate(
        self,
        experiment: str,
        x: float,
        *,
        repeat: int,
        attack: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        seed_count: int | None = None,
        methods: tuple[str, ...] = ("scasan", "socact", "friendship"),
        extra: dict[str, Any] | None = None,
    ) -> None:
        attack_values = {
            "friend_attacks": self.defaults["friendship_attacks"],
            "xi": self.defaults["xi"],
            "zeta": self.defaults["zeta"],
        }
        if attack:
            attack_values.update(attack)
        algorithm = {
            "alpha": self.defaults["alpha"],
            "beta": self.defaults["beta"],
            "gamma": self.defaults["gamma"],
            "lambda1": self.defaults["lambda1"],
            "lambda2": self.defaults["lambda2"],
            "epsilon": self.defaults["epsilon"],
            "max_iterations": self.defaults["max_iterations"],
        }
        if params:
            algorithm.update(params)
        seed_count = int(seed_count or self.defaults["seed_count"])
        seeds = self._seeds(seed_count, repeat)
        graph, activity = self._attacked(repeat=repeat, **attack_values)

        for method in methods:
            cache_key = (
                method,
                tuple(attack_values.values()),
                tuple(seeds.tolist()),
                tuple(sorted(algorithm.items())),
                repeat,
            )
            if method != "scasan":
                # Baseline parameters are fixed across ScaSAN-only sensitivity
                # sweeps, as described in Sections 5.3 and Appendix B.
                cache_key = (
                    method,
                    tuple(attack_values.values()),
                    tuple(seeds.tolist()),
                    algorithm["epsilon"],
                    algorithm["max_iterations"],
                    repeat,
                )
            if cache_key not in self._run_cache:
                if method == "scasan":
                    result = sybil_scarepact(
                        graph,
                        seeds,
                        alpha=float(algorithm["alpha"]),
                        beta=float(algorithm["beta"]),
                        gamma=float(algorithm["gamma"]),
                        lambda1=float(algorithm["lambda1"]),
                        lambda2=float(algorithm["lambda2"]),
                        epsilon=float(algorithm["epsilon"]),
                        max_iterations=int(algorithm["max_iterations"]),
                        convergence=self.defaults.get("convergence", "residual"),
                    )
                elif method == "socact":
                    result = sybil_socactnet(
                        activity,
                        seeds,
                        jump=float(self.config["sybil_socactnet"]["jump"]),
                        k=int(self.config["sybil_socactnet"]["k"]),
                        activity_steps=int(self.config["sybil_socactnet"]["activity_steps"]),
                        epsilon=float(algorithm["epsilon"]),
                        max_iterations=int(algorithm["max_iterations"]),
                    )
                elif method == "friendship":
                    result = sybil_friendship(
                        graph,
                        seeds,
                        restart=float(self.config["sybil_friendship"]["restart"]),
                        epsilon=float(algorithm["epsilon"]),
                        max_iterations=int(algorithm["max_iterations"]),
                    )
                else:
                    raise ValueError(method)
                self._run_cache[cache_key] = result
            result = self._run_cache[cache_key]
            row = {
                "experiment": experiment,
                "x": float(x),
                "repeat": repeat,
                "method": method,
                "auc": auc_from_scores(result.scores, graph.n_honest),
                "iterations": result.iterations,
                "converged": result.converged,
                "build_seconds": result.build_seconds,
                "propagation_seconds": result.propagation_seconds,
                "residual": result.residual,
                "seed_count": seed_count,
                **attack_values,
                **algorithm,
            }
            if extra:
                row.update(extra)
            self.records.append(row)

    def run(self, selected: set[str] | None = None) -> list[dict[str, Any]]:
        experiments = self.config["experiments"]
        repeats = int(self.config["repeats"])

        def enabled(name: str) -> bool:
            return selected is None or name in selected

        for repeat in range(repeats):
            if enabled("friendship_attack"):
                for value in experiments["friendship_attack"]:
                    self._evaluate(
                        "friendship_attack", value, repeat=repeat,
                        attack={"friend_attacks": int(value)},
                    )
            if enabled("incoming_activity_attack"):
                for value in experiments["incoming_activity_attack"]:
                    self._evaluate(
                        "incoming_activity_attack", value, repeat=repeat,
                        attack={"xi": float(value)},
                    )
            if enabled("alpha"):
                for value in experiments["alpha"]:
                    self._evaluate("alpha", value, repeat=repeat, params={"alpha": float(value)})
            if enabled("beta"):
                for value in experiments["beta"]:
                    self._evaluate("beta", value, repeat=repeat, params={"beta": float(value)})
            if enabled("outgoing_activity_attack"):
                for value in experiments["outgoing_activity_attack"]:
                    self._evaluate(
                        "outgoing_activity_attack", value, repeat=repeat,
                        attack={"zeta": float(value)},
                    )
            if enabled("seed_count"):
                for value in experiments["seed_count"]:
                    self._evaluate("seed_count", value, repeat=repeat, seed_count=int(value))
            if enabled("lambda1"):
                for value in experiments["lambda1"]:
                    self._evaluate("lambda1", value, repeat=repeat, params={"lambda1": float(value)})
            if enabled("lambda2"):
                for value in experiments["lambda2"]:
                    self._evaluate("lambda2", value, repeat=repeat, params={"lambda2": float(value)})
            if enabled("runtime"):
                for beta in experiments["runtime"]["beta"]:
                    for epsilon in experiments["runtime"]["epsilon"]:
                        self._evaluate(
                            f"runtime_beta_{beta}",
                            epsilon,
                            repeat=repeat,
                            params={"beta": float(beta), "epsilon": float(epsilon)},
                            methods=("scasan", "socact"),
                            extra={"runtime_beta": float(beta)},
                        )
        return self.records


def run_toy_experiments(config: dict[str, Any]) -> list[dict[str, Any]]:
    graph, activity = toy_graphs()
    aggregate_activity = aggregated_toy_activity_graph(graph)
    epsilon = float(config["defaults"]["epsilon"])
    maximum = int(config["defaults"]["max_iterations"])
    records: list[dict[str, Any]] = []
    for k in config["experiments"]["toy_k"]:
        result = sybil_socactnet(activity, np.asarray([2]), k=int(k), activity_steps=1, epsilon=epsilon, max_iterations=maximum)
        records.append({"experiment": "toy_k", "x": float(k), "method": "socact", "auc": auc_from_scores(result.scores, 3)})
        direct = sybil_socactnet(aggregate_activity, np.asarray([2]), k=int(k), activity_steps=1, epsilon=epsilon, max_iterations=maximum)
        records.append({"experiment": "toy_aggregate_k", "x": float(k), "method": "socact", "auc": auc_from_scores(direct.scores, 3)})
    for steps in config["experiments"]["toy_n"]:
        result = sybil_socactnet(activity, np.asarray([2]), k=5, activity_steps=int(steps), epsilon=epsilon, max_iterations=maximum)
        records.append({"experiment": "toy_n", "x": float(steps), "method": "socact", "auc": auc_from_scores(result.scores, 3)})
    return records


def plot_all(records: list[dict[str, Any]], output: str | Path) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    _plot_methods(axes[0], records, "friendship_attack", xlabel=r"$V_{FA}$")
    _plot_methods(axes[1], records, "incoming_activity_attack", xlabel=r"$\xi$")
    if axes[0].lines:
        axes[0].legend()
    fig.tight_layout()
    fig.savefig(output / "fig6_attack_strength.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    _plot_methods(axes[0], records, "alpha", xlabel=r"$\alpha$")
    _plot_methods(axes[1], records, "beta", xlabel=r"$\beta$")
    if axes[0].lines:
        axes[0].legend()
    fig.tight_layout()
    fig.savefig(output / "fig7_hyperparameters.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for axis, beta in zip(axes, (0.3, 0.4)):
        _plot_methods(
            axis,
            records,
            f"runtime_beta_{beta}",
            xlabel=r"$\epsilon$",
            ylabel="Propagation time (s)",
            metric="propagation_seconds",
            methods=("scasan", "socact"),
        )
        axis.invert_xaxis()
        axis.set_title(rf"$\beta={beta}$")
    if axes[0].lines:
        axes[0].legend()
    fig.tight_layout()
    fig.savefig(output / "fig8_runtime.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    _plot_methods(axes[0], records, "outgoing_activity_attack", xlabel=r"$\zeta$")
    _plot_methods(axes[1], records, "seed_count", xlabel="# seeds")
    if axes[0].lines:
        axes[0].legend()
    fig.tight_layout()
    fig.savefig(output / "fig12_outgoing_and_seeds.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    _plot_methods(axes[0], records, "lambda1", xlabel=r"$\lambda_1$")
    _plot_methods(axes[1], records, "lambda2", xlabel=r"$\lambda_2$")
    if axes[0].lines:
        axes[0].legend()
    fig.tight_layout()
    fig.savefig(output / "fig13_activity_weights.pdf")
    plt.close(fig)

    toy = [row for row in records if row["experiment"].startswith("toy")]
    if toy:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
        _plot_methods(axes[0], toy, "toy_k", xlabel="k", methods=("socact",))
        _plot_methods(axes[1], toy, "toy_n", xlabel="n", methods=("socact",))
        fig.tight_layout()
        fig.savefig(output / "fig3_toy_sensitivity.pdf")
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(4.6, 3.6))
        _plot_methods(axis, toy, "toy_aggregate_k", xlabel="k", methods=("socact",))
        fig.tight_layout()
        fig.savefig(output / "fig11_direct_application.pdf")
        plt.close(fig)


def run_from_config(
    config_path: str | Path,
    data_dir: str | Path,
    output: str | Path,
    *,
    selected: set[str] | None = None,
) -> list[dict[str, Any]]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    output = Path(output)
    suite = ExperimentSuite(data_dir, config)
    records = run_toy_experiments(config)
    records.extend(suite.run(selected))
    _save_csv(output / "metrics.csv", records)
    write_json(output / "metrics.json", records)
    plot_all(records, output)
    return records
