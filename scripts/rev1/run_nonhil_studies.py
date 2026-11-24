#!/usr/bin/env python3
"""Reproducible offline studies for missed-feasible decisions and policy cost."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics.benchmark.benchmark_algorithm.algorithms import (
    _restore_env,
    _snapshot_env,
    run_jo_vppm,
    run_saf_h,
)
from src.analytics.benchmark.benchmark_algorithm.config import (
    AlgorithmBenchmarkConfig,
    SCALABILITY_TOPOLOGIES,
)
from src.analytics.benchmark.benchmark_algorithm.env_factory import make_env
from src.analytics.benchmark.benchmark_algorithm.metrics import RunMetrics
from src.analytics.benchmark.benchmark_algorithm.model_loader import (
    build_model_from_policy_weights,
    load_normalizer,
    normalized_obs,
)
from src.analytics.benchmark.revision.provenance import sha256_file


DEFAULT_DATA = ROOT / "data/real_telecom_combined.csv"
DEFAULT_OUT = ROOT / "results/rev1/nonhil_studies"
SEEDS = (42, 100, 2024, 8888, 9999, 7, 17, 73, 314, 2718)


def env_factory(topology: str, data: Path, steps: int):
    return lambda: make_env(topology, data, steps, 1.0, (100, 500), "bursty")


def run_oracle_policy(topology: str, policy: str, seeds: tuple[int, ...], steps: int,
                      stride: int, data: Path, model_path: Path, norm_path: Path) -> list[dict]:
    """Enumerate exact simulator feasibility at sampled pre-decision states.

    Each candidate is evaluated by the environment's own step() from the same
    snapshot. That intentionally includes its TTL-expiration phase before
    admission, which a direct pre-step action mask does not model.
    """
    rows: list[dict] = []
    factory = env_factory(topology, data, steps)
    for seed in seeds:
        env = factory()
        env.enforce_resource_constraints = True
        env.reset(seed=seed)
        np.random.seed(seed)
        random.seed(seed)
        model = normalizer = None
        if policy == "harp":
            model = build_model_from_policy_weights(env, model_path)
            normalizer = load_normalizer(norm_path, model.get_env())
            obs, _ = env.reset(seed=seed)

        sampled = rejected = avoidable = alt_total = empty = 0
        elapsed_oracle = 0.0
        for step_idx in range(steps):
            should_sample = step_idx % stride == 0
            snap = _snapshot_env(env) if should_sample else None
            mask_count = None
            if policy == "harp":
                mask = env.action_masks()
                mask_count = int(mask.sum())
                pred_mask = mask if mask_count else np.ones(env.action_space.n, dtype=bool)
                action, _ = model.predict(
                    normalized_obs(normalizer, obs),
                    action_masks=np.asarray([pred_mask]), deterministic=True,
                )
                chosen = action[0]
            else:
                place, route = _saf_h_choice(env)
                chosen = [place, route]

            alternatives = 0
            if should_sample:
                sampled += 1
                start = time.perf_counter()
                for place in range(env.num_nodes):
                    for route in range(env.num_nodes):
                        _restore_env(env, snap)
                        _, _, _, _, candidate_info = env.step([place, route])
                        alternatives += int(bool(candidate_info.get("accepted", False)))
                elapsed_oracle += time.perf_counter() - start
                _restore_env(env, snap)
                empty += int(mask_count == 0) if mask_count is not None else 0

            obs, _, done, _, info = env.step(chosen)
            if should_sample:
                rejected += int(not info.get("accepted", False))
                alt_total += alternatives
                avoidable += int(not info.get("accepted", False) and alternatives > 0)
            if done:
                obs, _ = env.reset(seed=seed)
        env.close()
        rows.append({
            "topology": topology, "policy": policy, "seed": seed, "steps": steps,
            "oracle_stride": stride, "sampled_decisions": sampled,
            "selected_rejections": rejected,
            "selected_rejections_with_feasible_alternative": avoidable,
            "avoidable_rejection_rate_pct": 100.0 * avoidable / sampled if sampled else 0.0,
            "mean_feasible_pairs_per_sample": alt_total / sampled if sampled else 0.0,
            "all_mask_empty_samples": empty,
            "oracle_wall_time_s": elapsed_oracle,
        })
        print(f"oracle {policy} seed={seed}: {avoidable}/{sampled} avoidable selected rejections", flush=True)
    return rows


def _saf_h_choice(env) -> tuple[int, int]:
    cpu_req = env._current_req.get("cpu", 0.0)
    ram_req = env._current_req.get("ram", 0.0)
    msd_req = env._current_req.get("msd", 1)
    bw_req = cpu_req * 10.0
    best_action, best_cost = None, np.inf
    for place in range(env.num_nodes):
        if (env._state[place * 3] + cpu_req > env.max_cpu or
                env._state[place * 3 + 1] + ram_req > env.max_ram):
            continue
        for route in range(env.num_nodes):
            if place != route and env._link_bw[place][route] < bw_req:
                continue
            if msd_req > env.node_msd_limits[route]:
                continue
            cost = (env.latency_matrix[place][route] * 0.5 +
                    ((env._state[place * 3] + cpu_req) / env.max_cpu) * 100.0 +
                    env.topo.get_hop_distance(place, route) * 5.0)
            if cost < best_cost:
                best_cost, best_action = cost, (place, route)
    return best_action if best_action is not None else (0, 0)


def run_cost(topology: str, seeds: tuple[int, ...], steps: int, data: Path,
             model_path: Path, norm_path: Path) -> dict:
    """Warm up both methods, then run serialized matched-seed measurements."""
    factory = env_factory(topology, data, steps)
    run_saf_h(factory, "bursty", steps, 2718, protected=True)
    run_jo_vppm(factory, model_path, norm_path, "bursty", steps, 2718, protected=True)
    rows = []
    for index, seed in enumerate(seeds):
        order = ("harp", "saf_h") if index % 2 == 0 else ("saf_h", "harp")
        measured = {}
        for method in order:
            if method == "harp":
                run = run_jo_vppm(factory, model_path, norm_path, "bursty", steps, seed, protected=True)
            else:
                run = run_saf_h(factory, "bursty", steps, seed, protected=True)
            measured[method] = run.as_summary()
        rows.append({"seed": seed, "execution_order": list(order), **measured})
        print(f"cost seed={seed} order={order} HARP p95={measured['harp']['decision_latency_p95_ms']:.4f}ms "
              f"SAF-H p95={measured['saf_h']['decision_latency_p95_ms']:.4f}ms", flush=True)
    return {
        "topology": topology, "scenario": "bursty", "steps_per_seed": steps,
        "seeds": list(seeds), "warmup_seed": 2718,
        "warmup_runs_excluded_per_policy": 1,
        "warmup_seed_also_in_measurement_set": 2718 in seeds,
        "measurement_order": "serialized in one process, alternating order by seed",
        "paired": {
            "harp_p95_minus_saf_h_p95_ms_mean": statistics.mean(
                r["harp"]["decision_latency_p95_ms"] - r["saf_h"]["decision_latency_p95_ms"] for r in rows),
            "harp_inference_p95_ms_mean": statistics.mean(r["harp"].get("policy_inference_ms_p95_ms", 0.0) for r in rows),
            "harp_decision_p95_ms_mean": statistics.mean(r["harp"]["decision_latency_p95_ms"] for r in rows),
            "saf_h_feasible_set_p95_ms_mean": statistics.mean(r["saf_h"].get("feasible_set_ms_p95_ms", 0.0) for r in rows),
            "saf_h_selection_p95_ms_mean": statistics.mean(r["saf_h"].get("heuristic_selection_ms_p95_ms", 0.0) for r in rows),
            "saf_h_decision_p95_ms_mean": statistics.mean(r["saf_h"]["decision_latency_p95_ms"] for r in rows),
        },
        "per_seed": rows,
    }


def run_scale(topologies: tuple[str, ...], seeds: tuple[int, ...], steps: int, data: Path) -> dict:
    """Matched, size-specific HARP/SAF-H policy timing on identical traces."""
    rows = []
    edge_counts = {}
    for topology in topologies:
        factory = env_factory(topology, data, steps)
        probe = factory()
        edge_counts[topology] = int(np.triu(probe.topo.adj_matrix, k=1).sum())
        probe.close()
        nodes = int(topology.split("_")[1])
        model_path = ROOT / f"results/models/v11_synthetic_{nodes}/dgrl_v11_final_synthetic_{nodes}.zip"
        norm_path = ROOT / f"results/models/v11_synthetic_{nodes}/vec_normalize_v11_synthetic_{nodes}.pkl"
        for path in (model_path, norm_path):
            if not path.is_file():
                raise FileNotFoundError(f"size-matched HARP artifact not found: {path}")
        # Exclude one warm-up for each policy and size; alternate execution order
        # to reduce systematic thermal/cache ordering effects.
        run_saf_h(factory, "bursty", steps, 2718, protected=True)
        run_jo_vppm(factory, model_path, norm_path, "bursty", steps, 2718, protected=True)
        for index, seed in enumerate(seeds):
            order = ("harp", "saf_h") if index % 2 == 0 else ("saf_h", "harp")
            measured = {}
            for method in order:
                if method == "harp":
                    run = run_jo_vppm(factory, model_path, norm_path, "bursty", steps, seed, protected=True)
                else:
                    run = run_saf_h(factory, "bursty", steps, seed, protected=True)
                measured[method] = run.as_summary()
            if measured["harp"]["trace_sha256"] != measured["saf_h"]["trace_sha256"]:
                raise RuntimeError(f"HARP/SAF-H offered traces differ: {topology} seed={seed}")
            rows.append({"topology": topology, "seed": seed, "execution_order": list(order), **measured})
            print(f"scale {topology} nodes={nodes} seed={seed} "
                  f"HARP mask/inference/decision p95="
                  f"{measured['harp'].get('mask_construction_ms_p95_ms', 0.0):.4f}/"
                  f"{measured['harp'].get('policy_inference_ms_p95_ms', 0.0):.4f}/"
                  f"{measured['harp'].get('decision_latency_p95_ms', 0.0):.4f}ms; "
                  f"SAF-H enumeration/selection/decision p95="
                  f"{measured['saf_h'].get('feasible_set_ms_p95_ms', 0.0):.4f}/"
                  f"{measured['saf_h'].get('heuristic_selection_ms_p95_ms', 0.0):.4f}/"
                  f"{measured['saf_h'].get('decision_latency_p95_ms', 0.0):.4f}ms", flush=True)
    aggregate = []
    for topology in topologies:
        group = [r for r in rows if r["topology"] == topology]
        nodes = int(topology.split("_")[1])
        entry = {
            "topology": topology, "nodes": nodes, "actions": nodes * nodes,
            "undirected_links": edge_counts[topology], "seeds": len(group),
        }
        for policy in ("harp", "saf_h"):
            pr = [row[policy] for row in group]
            entry[policy] = {
                "requests_mean": statistics.mean(r["requests"] for r in pr),
                "decision_p50_ms_mean": statistics.mean(r.get("decision_latency_p50_ms", 0.0) for r in pr),
                "decision_p95_ms_mean": statistics.mean(r.get("decision_latency_p95_ms", 0.0) for r in pr),
                "decision_p99_ms_mean": statistics.mean(r.get("decision_latency_p99_ms", 0.0) for r in pr),
                "wall_time_s_mean": statistics.mean(r.get("run_wall_time_s", 0.0) for r in pr),
            }
        entry["harp"].update({
            "mask_p50_ms_mean": statistics.mean(r.get("mask_construction_ms_p50_ms", 0.0) for r in [x["harp"] for x in group]),
            "mask_p95_ms_mean": statistics.mean(r.get("mask_construction_ms_p95_ms", 0.0) for r in [x["harp"] for x in group]),
            "inference_p50_ms_mean": statistics.mean(r.get("policy_inference_ms_p50_ms", 0.0) for r in [x["harp"] for x in group]),
            "inference_p95_ms_mean": statistics.mean(r.get("policy_inference_ms_p95_ms", 0.0) for r in [x["harp"] for x in group]),
            "normalization_p95_ms_mean": statistics.mean(r.get("normalization_ms_p95_ms", 0.0) for r in [x["harp"] for x in group]),
        })
        entry["saf_h"].update({
            "feasible_set_p50_ms_mean": statistics.mean(r.get("feasible_set_ms_p50_ms", 0.0) for r in [x["saf_h"] for x in group]),
            "feasible_set_p95_ms_mean": statistics.mean(r.get("feasible_set_ms_p95_ms", 0.0) for r in [x["saf_h"] for x in group]),
            "selection_p95_ms_mean": statistics.mean(r.get("heuristic_selection_ms_p95_ms", 0.0) for r in [x["saf_h"] for x in group]),
        })
        aggregate.append(entry)
    return {"policies": ["HARP", "SAF-H"], "scenario": "bursty", "steps_per_seed": steps,
            "seeds": list(seeds), "measurement_order": "serialized in one process, alternating policy order by seed",
            "warmup_runs_excluded_per_policy_and_topology": 1,
            "measurements": aggregate, "per_seed": rows,
            "scope": "synthetic topology simulator decisions; HARP component timing and SAF-H candidate enumeration/selection timing"}


def write_result(outdir: Path, name: str, payload: dict, manifest: dict) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", choices=("oracle", "cost", "scale", "all"), default="all")
    parser.add_argument("--topology", default="vietnam")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--oracle-stride", type=int, default=25)
    parser.add_argument("--scale-steps", type=int, default=300,
                        help="Decisions per seed for the quadratic topology-size timing sweep.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", type=Path, default=ROOT / "results/models/v11/dgrl_v11_final_vietnam.zip")
    parser.add_argument("--normalizer", type=Path, default=ROOT / "results/models/v11/vec_normalize_v11_vietnam.pkl")
    args = parser.parse_args()
    # The MLP/GAT inference workload is small; multiple BLAS worker threads can
    # oversubscribe the process and produce avoidable latency variance.
    torch.set_num_threads(1)
    seeds = tuple(args.seeds)
    if not args.data.is_file():
        raise FileNotFoundError(f"dataset not found: {args.data}")
    if args.study in ("oracle", "cost", "all"):
        for path in (args.model, args.normalizer):
            if not path.is_file():
                raise FileNotFoundError(f"HARP artifact not found: {path}")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_worktree_dirty": bool(__import__("subprocess").check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "study_source_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                Path(__file__).resolve(),
                ROOT / "src/analytics/benchmark/benchmark_algorithm/algorithms.py",
                ROOT / "src/analytics/benchmark/benchmark_algorithm/metrics.py",
                ROOT / "src/analytics/benchmark/benchmark_algorithm/env_factory.py",
                ROOT / "src/orchestration/jo_vdpr/env.py",
                ROOT / "src/orchestration/jo_vdpr/topology.py",
            )
        },
        "python": platform.python_version(), "platform": platform.platform(),
        "data_path": str(args.data.relative_to(ROOT) if args.data.is_relative_to(ROOT) else args.data),
        "data_sha256": sha256_file(args.data), "seeds": list(seeds), "steps": args.steps,
        "model_path": str(args.model.relative_to(ROOT) if args.model.is_relative_to(ROOT) else args.model),
        "model_sha256": sha256_file(args.model) if args.model.is_file() else None,
        "normalizer_path": str(args.normalizer.relative_to(ROOT) if args.normalizer.is_relative_to(ROOT) else args.normalizer),
        "normalizer_sha256": sha256_file(args.normalizer) if args.normalizer.is_file() else None,
        "cpu_count": os.cpu_count(),
        "cpu_model": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                            if line.lower().startswith("model name")), platform.processor())
                     if Path("/proc/cpuinfo").is_file() else platform.processor(),
        "memory_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
                        if hasattr(os, "sysconf") else None,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_thread_policy": "single intra-op thread for serialized simulator timing",
        "scale_model_artifacts": {
            topology: {
                "model": str((ROOT / f"results/models/v11_synthetic_{int(topology.split('_')[1])}/dgrl_v11_final_synthetic_{int(topology.split('_')[1])}.zip").relative_to(ROOT)),
                "model_sha256": sha256_file(ROOT / f"results/models/v11_synthetic_{int(topology.split('_')[1])}/dgrl_v11_final_synthetic_{int(topology.split('_')[1])}.zip")
                    if (ROOT / f"results/models/v11_synthetic_{int(topology.split('_')[1])}/dgrl_v11_final_synthetic_{int(topology.split('_')[1])}.zip").is_file() else None,
                "normalizer": str((ROOT / f"results/models/v11_synthetic_{int(topology.split('_')[1])}/vec_normalize_v11_synthetic_{int(topology.split('_')[1])}.pkl").relative_to(ROOT)),
                "normalizer_sha256": sha256_file(ROOT / f"results/models/v11_synthetic_{int(topology.split('_')[1])}/vec_normalize_v11_synthetic_{int(topology.split('_')[1])}.pkl")
                    if (ROOT / f"results/models/v11_synthetic_{int(topology.split('_')[1])}/vec_normalize_v11_synthetic_{int(topology.split('_')[1])}.pkl").is_file() else None,
            } for topology in SCALABILITY_TOPOLOGIES
        },
        "cuda_available": bool(torch.cuda.is_available()),
        "benchmark_workers": 1,
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.output / stamp
    if args.study in ("oracle", "all"):
        oracle = []
        for policy in ("harp", "saf_h"):
            oracle.extend(run_oracle_policy(args.topology, policy, seeds, args.steps,
                                            args.oracle_stride, args.data, args.model, args.normalizer))
        write_result(outdir, "missed_feasible_oracle", {"samples": oracle}, manifest)
    if args.study in ("cost", "all"):
        cost = run_cost(args.topology, seeds, args.steps, args.data, args.model, args.normalizer)
        write_result(outdir, "controlled_policy_cost", cost, manifest)
    if args.study in ("scale", "all"):
        manifest["scale_steps_per_seed"] = args.scale_steps
        scale = run_scale(tuple(SCALABILITY_TOPOLOGIES), seeds, args.scale_steps, args.data)
        write_result(outdir, "topology_scalability", scale, manifest)
    print(f"OUTPUT_DIR={outdir}", flush=True)


if __name__ == "__main__":
    main()
