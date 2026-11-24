#!/usr/bin/env python3
"""Matched-trace benchmark for an SRv6/SFCPlanner-inspired KSP adaptation.

This is explicitly not a reproduction of Qiu et al.'s DRL-based SFCPlanner.
The simulator action is only a placement/routing endpoint pair, so this study
uses K-shortest paths as a path-cost and per-profile MSD candidate filter.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics.benchmark.benchmark_algorithm.algorithms import (
    run_jo_vppm,
    run_saf_h,
    run_sfcplanner_ksp_adaptation,
)
from src.analytics.benchmark.benchmark_algorithm.env_factory import make_env
from src.analytics.benchmark.revision.provenance import sha256_file

SEEDS = (42, 100, 2024, 8888, 9999, 7, 17, 73, 314, 2718,
         11, 23, 47, 89, 101, 211, 512, 1024, 4096, 8192)
CONDITIONS = {
    "elephant_stress": "heavy_tail",
    "burst_surge": "bursty",
    "chaos": "chaos",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topologies", nargs="+", default=["vietnam", "nsfnet", "geant2"])
    parser.add_argument("--scenarios", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--k-paths", type=int, default=5)
    parser.add_argument("--data", type=Path, default=ROOT / "data/real_telecom_combined.csv")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.steps <= 0 or args.k_paths <= 0 or not args.seeds:
        parser.error("steps, k-paths and seeds must be positive")

    torch.set_num_threads(1)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.output or ROOT / "results/rev1/sfcplanner_ksp_adaptation" / stamp
    out.mkdir(parents=True, exist_ok=False)
    model_root = ROOT / "results/models/retrain_v11_20261119_corrected"
    rows: list[dict] = []
    trace_checks = 0

    for topology in args.topologies:
        model = model_root / topology / f"dgrl_v11_final_{topology}.zip"
        normalizer = model_root / topology / f"vec_normalize_v11_{topology}.pkl"
        for artifact in (model, normalizer):
            if not artifact.is_file():
                raise FileNotFoundError(f"required current HARP artifact missing: {artifact}")
        for label in args.scenarios:
            env_scenario = CONDITIONS[label]

            def factory():
                env = make_env(topology, args.data, args.steps, 1.0, (100, 500), env_scenario)
                env.traffic_scenario = env_scenario
                return env

            for index, seed in enumerate(args.seeds):
                methods = ["harp", "saf_h", "ksp"]
                if index % 2:
                    methods.reverse()
                result = {}
                for method in methods:
                    if method == "harp":
                        run = run_jo_vppm(factory, model, normalizer, env_scenario,
                                          args.steps, seed, protected=True)
                    elif method == "saf_h":
                        run = run_saf_h(factory, env_scenario, args.steps, seed, protected=True)
                    else:
                        run = run_sfcplanner_ksp_adaptation(
                            factory, env_scenario, args.steps, seed,
                            protected=True, k_paths=args.k_paths,
                        )
                    summary = run.as_summary()
                    result[method] = summary
                    rows.append({
                        "topology": topology, "scenario": label, "env_scenario": env_scenario,
                        "seed": seed, "steps": args.steps, "k_paths": args.k_paths,
                        **summary,
                    })
                trace_hashes = {x["trace_sha256"] for x in result.values()}
                if len(trace_hashes) != 1:
                    raise RuntimeError(
                        f"offered trace mismatch for {topology}/{label}/seed={seed}: {trace_hashes}"
                    )
                trace_checks += 1
                print(
                    f"{topology} {label} seed={seed}: "
                    f"safe acceptance HARP={result['harp']['safe_acceptance_rate']:.3f}% "
                    f"SAF-H={result['saf_h']['safe_acceptance_rate']:.3f}% "
                    f"KSP-adapt={result['ksp']['safe_acceptance_rate']:.3f}% "
                    f"trace=matched", flush=True,
                )

    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (out / "per_seed.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    grouped = {}
    for row in rows:
        key = (row["topology"], row["scenario"])
        grouped.setdefault(key, {}).setdefault(row["algorithm"], []).append(row)
    summary_rows = []
    for (topology, scenario), algorithms in sorted(grouped.items()):
        for algorithm, records in sorted(algorithms.items()):
            vals = [r["safe_acceptance_rate"] for r in records]
            summary_rows.append({
                "topology": topology, "scenario": scenario, "algorithm": algorithm,
                "n_seeds": len(vals), "safe_acceptance_mean_pct": statistics.mean(vals),
                "safe_acceptance_sample_sd_pp": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "admitted_hard_violation_mean_pct": statistics.mean(
                    r["admitted_hard_violation_rate"] for r in records
                ),
                "decision_p95_mean_ms": statistics.mean(
                    r["decision_latency_p95_ms"] for r in records
                ),
            })
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    files = [Path(__file__), Path(__file__).parents[2] /
             "src/analytics/benchmark/benchmark_algorithm/algorithms.py"]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "matched offered-trace; protected common simulator admission",
        "baseline_name": "SFCPlanner-inspired K-shortest-path adaptation",
        "baseline_scope": (
            "Not a reproduction of Qiu et al. (2024). Simulator action is an endpoint pair; "
            "K shortest paths rank candidate pairs and filter requested depth against path-node MSD. "
            "The simulator executes/admit-checks the selected endpoint pair, not the candidate path."
        ),
        "topologies": args.topologies, "scenarios": args.scenarios,
        "environment_scenarios": CONDITIONS, "seeds": args.seeds, "steps": args.steps,
        "k_paths": args.k_paths, "data_path": str(args.data),
        "data_sha256": sha256_file(args.data), "trace_checks_equal": trace_checks,
        "protected_hard_constraints": ["CPU", "RAM", "pair bandwidth", "MSD"],
        "policy_order": "alternated by seed; each policy uses a fresh environment",
        "harp_model": str(model_root),
        "harp_artifacts": {
            topology: {
                "model_sha256": sha256_file(model_root / topology / f"dgrl_v11_final_{topology}.zip"),
                "normalizer_sha256": sha256_file(model_root / topology / f"vec_normalize_v11_{topology}.pkl"),
            } for topology in args.topologies
        },
        "source_sha256": {str(f.relative_to(ROOT)): sha256_file(f) for f in files},
        "git_revision": __import__("subprocess").run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip(),
        "git_worktree_dirty": bool(__import__("subprocess").run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"completed {len(rows)} policy-seed rows; matched-trace checks={trace_checks}; output={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
