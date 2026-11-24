#!/usr/bin/env python3
"""Summarize paired seed-level results for the KSP adaptation benchmark."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon


def holm(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running = 0.0
    m = len(p_values)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def compare(rows: list[dict], treatment: str, control: str, rng: np.random.Generator) -> list[dict]:
    cells = sorted({(r["topology"], r["scenario"]) for r in rows})
    result = []
    for topology, scenario in cells:
        selected = [r for r in rows if r["topology"] == topology and r["scenario"] == scenario]
        by_method = {
            name: {int(r["seed"]): float(r["safe_acceptance_rate"])
                  for r in selected if r["algorithm"] == name}
            for name in (treatment, control)
        }
        if by_method[treatment].keys() != by_method[control].keys():
            raise ValueError(f"seed coverage mismatch for {topology}/{scenario}")
        seeds = sorted(by_method[treatment])
        delta = np.asarray([by_method[treatment][seed] - by_method[control][seed]
                            for seed in seeds], dtype=float)
        nonzero = delta[delta != 0]
        if len(nonzero):
            ranks = rankdata(np.abs(nonzero), method="average")
            rank_biserial = float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum())
            p_value = float(wilcoxon(delta, zero_method="wilcox", alternative="two-sided", method="auto").pvalue)
        else:
            rank_biserial, p_value = 0.0, 1.0
        draws = rng.choice(delta, size=(20000, len(delta)), replace=True).mean(axis=1)
        result.append({
            "topology": topology, "scenario": scenario, "treatment": treatment,
            "control": control, "n_seeds": len(seeds),
            "mean_delta_pp": float(delta.mean()),
            "sample_sd_delta_pp": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
            "bootstrap_95ci_low_pp": float(np.quantile(draws, 0.025)),
            "bootstrap_95ci_high_pp": float(np.quantile(draws, 0.975)),
            "zero_differences": int(np.count_nonzero(delta == 0)),
            "wilcoxon_p": p_value, "rank_biserial": rank_biserial,
        })
    adjusted = holm([r["wilcoxon_p"] for r in result])
    for row, p_adj in zip(result, adjusted):
        row["wilcoxon_p_holm_9"] = p_adj
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--bootstrap-seed", type=int, default=20261123)
    args = parser.parse_args()
    with (args.run_dir / "per_seed.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    all_hashes_match = True
    cells = {(r["topology"], r["scenario"], r["seed"]) for r in rows}
    for topology, scenario, seed in cells:
        hashes = {r["trace_sha256"] for r in rows
                  if r["topology"] == topology and r["scenario"] == scenario and r["seed"] == seed}
        all_hashes_match &= len(hashes) == 1
    max_violation = max(float(r["admitted_hard_violation_rate"]) for r in rows)
    ksp = "sfcplanner_ksp_adaptation_k5_protected"
    saf = "saf_h_protected"
    harp = "harp_protected"
    rng1 = np.random.default_rng(args.bootstrap_seed)
    rng2 = np.random.default_rng(args.bootstrap_seed + 1)
    paired = compare(rows, ksp, saf, rng1) + compare(rows, ksp, harp, rng2)
    with (args.run_dir / "paired_statistics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    report = {
        "policy_rows": len(rows), "matched_trace_seed_cells": len(cells),
        "all_trace_hashes_match": all_hashes_match,
        "max_admitted_hard_violation_rate_pct": max_violation,
        "comparisons": paired,
        "bootstrap": {"method": "paired seed-level percentile bootstrap", "resamples": 20000,
                      "seed_ksp_vs_saf_h": args.bootstrap_seed,
                      "seed_ksp_vs_harp": args.bootstrap_seed + 1},
        "multiple_testing": "Holm correction separately across nine topology-workload cells per contrast family",
    }
    (args.run_dir / "paired_statistics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "policy_rows": len(rows), "matched_trace_seed_cells": len(cells),
        "all_trace_hashes_match": all_hashes_match,
        "max_admitted_hard_violation_rate_pct": max_violation,
        "statistics": str(args.run_dir / "paired_statistics.csv"),
    }, indent=2))
    return 0 if all_hashes_match and max_violation == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
