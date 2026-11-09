from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import rankdata, wilcoxon

from .baseline_simulator import BASELINE_ALGORITHMS, BaselineSimulator
from .models import ScenarioSummary
from .scenarios import ALL_SCENARIOS


DEFAULT_SEEDS = [42, 100, 2024, 8888, 9999, 7, 17, 123, 2025, 31415]


def paired_statistics(
    proposed: Iterable[float],
    baseline: Iterable[float],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 2026,
) -> dict[str, float]:
    proposed_values = np.asarray(list(proposed), dtype=float)
    baseline_values = np.asarray(list(baseline), dtype=float)
    if proposed_values.shape != baseline_values.shape or proposed_values.size < 2:
        raise ValueError("Paired comparisons require equal arrays with at least two seeds.")

    differences = proposed_values - baseline_values
    if np.all(differences == 0):
        p_value = 1.0
        rank_biserial = 0.0
    else:
        p_value = float(wilcoxon(differences, zero_method="wilcox", alternative="two-sided").pvalue)
        nonzero = differences != 0
        ranks = rankdata(np.abs(differences[nonzero]))
        signed_ranks = np.sign(differences[nonzero]) * ranks
        rank_biserial = float(np.sum(signed_ranks) / np.sum(ranks))

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(bootstrap_samples, differences.size))
    bootstrap_means = differences[indices].mean(axis=1)
    ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
    return {
        "n_seeds": int(differences.size),
        "mean_paired_difference": float(differences.mean()),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "p_value": p_value,
        "rank_biserial": rank_biserial,
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running_max = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running_max = max(running_max, (count - rank) * float(p_values[index]))
        adjusted[index] = min(1.0, running_max)
    return adjusted.tolist()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(output_dir: Path, seeds: list[int], steps: int = 0) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = ALL_SCENARIOS
    summaries: list[tuple[int, ScenarioSummary]] = []
    for scenario in scenarios:
        print(f"Scenario: {scenario.name}")
        for seed in seeds:
            seed_dir = output_dir / scenario.name / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            simulator = BaselineSimulator(seed=seed, steps=steps)
            for algorithm in BASELINE_ALGORITHMS:
                algorithm_dir = seed_dir / algorithm
                algorithm_dir.mkdir(parents=True, exist_ok=True)
                _, summary = simulator.run(scenario, algorithm, algorithm_dir)
                summaries.append((seed, summary))
        print(f"  completed {len(seeds)} matched seeds")

    summary_rows = [{**asdict(summary), "seed": seed} for seed, summary in summaries]
    _write_csv(output_dir / "seed_summary.csv", summary_rows)

    stat_rows = []
    for scenario in scenarios:
        by_algorithm = {
            algorithm: {
                row["seed"]: row
                for row in summary_rows
                if row["scenario"] == scenario.name and row["algorithm"] == algorithm
            }
            for algorithm in BASELINE_ALGORITHMS
        }
        matched_seeds = [seed for seed in seeds if seed in by_algorithm["jo_vppm"] and seed in by_algorithm["saf_h"]]
        paired = paired_statistics(
            [float(by_algorithm["jo_vppm"][seed]["acceptance_rate"]) for seed in matched_seeds],
            [float(by_algorithm["saf_h"][seed]["acceptance_rate"]) for seed in matched_seeds],
        )
        stat_rows.append({"scenario": scenario.name, **paired})
    adjusted = holm_adjust([float(row["p_value"]) for row in stat_rows])
    for row, p_adjusted in zip(stat_rows, adjusted):
        row["p_holm"] = p_adjusted
    _write_csv(output_dir / "harp_vs_saf_h_paired.csv", stat_rows)

    model_path = Path(__import__("src.ai.dgrl_agent", fromlist=["DEFAULT_MODEL_PATH"]).DEFAULT_MODEL_PATH)
    metadata = {
        "topology": "Vietnam",
        "steps_per_scenario": steps or "scenario_default",
        "seeds": seeds,
        "scenarios": [scenario.name for scenario in scenarios],
        "algorithms": BASELINE_ALGORITHMS,
        "primary_comparison": "jo_vppm vs saf_h; same generated request trace and shared NetworkStateManager admission",
        "checkpoint_path": str(model_path),
        "checkpoint_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest() if model_path.exists() else None,
        "workload_scenario_source": "src/analytics/benchmark/benchmark_testbed/scenarios",
        "note": "Offline software baseline simulation; not live HIL or production evidence.",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run matched protected/unprotected baselines for the AIO-Mesh revision.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=0, help="Override default scenario lengths.")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        parser.error("Provide at least two unique seeds.")
    run_experiment(args.output_dir, seeds, args.steps)


if __name__ == "__main__":
    main()
