"""Command-line entry points for downloading, preparing and reproducing ScaSAN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .experiments import plot_all, run_from_config, run_toy_experiments
from .graph import ActivityGraph, ScaSANGraph
from .metrics import write_json
from .preprocess import download_facebook, prepare_paper_dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scasan", description="Reproduce the ScaSAN paper experiments")
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download", help="download the public Facebook WOSN 2009 files")
    download.add_argument("--output", default="data/raw")
    download.add_argument("--overwrite", action="store_true")

    prepare = sub.add_parser("prepare", help="build ScaSAN and unaggregated SAN graphs")
    prepare.add_argument("--links", default="data/raw/facebook-links.txt.gz")
    prepare.add_argument("--walls", default="data/raw/facebook-wall.txt.gz")
    prepare.add_argument("--output", default="data/processed/facebook")
    prepare.add_argument("--sybil-users", type=int, default=1000)
    prepare.add_argument("--seed", type=int, default=2025)

    inspect = sub.add_parser("inspect", help="print processed dataset statistics")
    inspect.add_argument("--data", default="data/processed/facebook")

    toy = sub.add_parser("toy", help="run Figures 3 and 11 without external data")
    toy.add_argument("--config", default="configs/paper.yaml")
    toy.add_argument("--output", default="results/toy")

    run = sub.add_parser("run", help="run the Facebook experiment suite")
    run.add_argument("--config", default="configs/paper.yaml")
    run.add_argument("--data", default="data/processed/facebook")
    run.add_argument("--output", default="results/paper")
    run.add_argument(
        "--experiments",
        nargs="*",
        help="optional subset: friendship_attack incoming_activity_attack alpha beta runtime "
        "outgoing_activity_attack seed_count lambda1 lambda2",
    )
    return parser


def _inspect(path: str | Path) -> dict:
    graph = ScaSANGraph.load(path)
    activity = ActivityGraph.load(path)
    v_hh, v_ss, v_iaa, v_oaa = graph.interaction_volumes()
    return {
        "users": graph.n_users,
        "honest_users": graph.n_honest,
        "sybil_users": graph.n_users - graph.n_honest,
        "friendship_undirected_edges": graph.friendship.nnz // 2,
        "active_created_aggregates": int(graph.active_created.size),
        "active_forwarded_aggregates": int(graph.active_forwarded.size),
        "unaggregated_activities": activity.n_activities,
        "V_HH": v_hh,
        "V_SS": v_ss,
        "V_IAA": v_iaa,
        "V_OAA": v_oaa,
    }


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "download":
        links, walls = download_facebook(args.output, overwrite=args.overwrite)
        print(links)
        print(walls)
    elif args.command == "prepare":
        graph, activity, selected = prepare_paper_dataset(
            args.links,
            args.walls,
            args.output,
            sybil_users=args.sybil_users,
            rng_seed=args.seed,
        )
        print(json.dumps({
            "users": graph.n_users,
            "activities": activity.n_activities,
            "sybil_window_start": int(selected[0]),
            "sybil_window_stop": int(selected[-1] + 1),
        }, indent=2))
    elif args.command == "inspect":
        print(json.dumps(_inspect(args.data), indent=2))
    elif args.command == "toy":
        with Path(args.config).open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        records = run_toy_experiments(config)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "metrics.json", records)
        plot_all(records, output)
        print(f"Wrote {len(records)} rows to {output}")
    elif args.command == "run":
        selected = set(args.experiments) if args.experiments else None
        records = run_from_config(args.config, args.data, args.output, selected=selected)
        print(f"Wrote {len(records)} rows to {args.output}")


if __name__ == "__main__":
    main()
