# Kaggle Benchmark and Ablation Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one Kaggle notebook that smoke-tests and then runs the P0 benchmark and ablation matrices for NSFNET, GEANT2, and Vietnam.

**Architecture:** The notebook packages the current source and data, trains short-lived smoke checkpoints in an isolated output root, then evaluates both existing benchmark and strict ablation CLIs for each topology. A separate opt-in full stage trains production-length ablation checkpoints and runs the planned scenario/seed matrix, writing raw outputs and a reproducibility manifest.

**Tech Stack:** Python, Jupyter notebook JSON, Kaggle `/kaggle/input` and `/kaggle/working`, Stable-Baselines3 MaskablePPO, existing project CLI modules.

**Spec:** `docs/superpowers/specs/2026-11-06-kaggle-benchmark-ablation-notebook-design.md`

## Global Constraints

- Keep smoke artifacts and full-run artifacts in distinct directories.
- Smoke uses 512 training steps per variant/topology with one environment, then 64 evaluation requests; these outputs are pipeline diagnostics, not paper evidence.
- Full execution remains opt-in, default `RUN_FULL=False`, and requires all smoke checks to pass.
- Do not silently use old v11 checkpoints when training the current-source smoke/full `harp_full` policy.
- Full evaluation uses 3000 requests per seed, matched seed list, and records exact source/data/checkpoint hashes and runtime versions.
- Do not run training or benchmark experiments in the local workspace.

## Review Focus

- Old or mismatched checkpoints: training from current source must populate a model root that both benchmark and strict ablation loaders resolve; smoke must fail with the exact checkpoint path if not.
- Topology-specific output spaces: every trained checkpoint must be loaded only with its matching topology and normalizer.
- Missing/incorrect data path: preflight must validate the CSV before starting training.
- Partial matrix after a subprocess failure: stop and mark the topology/scenario failed; do not create a success archive.
- Full run accidentally starts during smoke: require explicit `RUN_FULL=True` and `SMOKE_PASSED=True`.

---

### Task 1: Add explicit benchmark seed selection

**Files:**
- Modify: `src/analytics/benchmark/benchmark_algorithm/config.py`
- Modify: `src/analytics/benchmark/benchmark_algorithm/run_all.py`
- Test: `tests/test_benchmark_seed_config.py`

**Interfaces:**
- `AlgorithmBenchmarkConfig` accepts an optional explicit evaluation seed list; `seeds` returns that list when set, `[42]` for legacy `single_seed`, otherwise the existing default list.
- CLI accepts `--seeds` as one or more integer values so Kaggle can run matched 10-seed comparisons in one output root.

- [x] Add tests for explicit seeds, legacy single seed, and defaults.
- [x] Run the focused test and confirm it fails because the CLI/config does not yet expose the override.
- [x] Add the optional seed field/property and `--seeds` CLI handling without changing existing defaults.
- [x] Re-run the focused test and verify explicit seed ordering and uniqueness behavior.

### Task 2: Create the staged Kaggle notebook

**Files:**
- Create: `notebook/rev1-benchmark-ablation-kaggle.ipynb`

**Interfaces:**
- Notebook configuration exposes topology list, smoke steps, full training steps by topology, scenarios, seeds, `RUN_FULL`, data path, and output roots.
- Smoke training calls `src.analytics.training.train_harp_ablation --variant all` for each topology with 512 steps, one env, and a separate smoke model root.
- Benchmark smoke calls `src.analytics.benchmark.benchmark_algorithm.run_all` with 64 steps, one seed, one scenario, and smoke model root.
- Strict ablation smoke calls `src.analytics.ablation.ablation_study` with 64 steps and the same smoke model root.
- Full benchmark runs `scenario=thesis` over the configured matched seed list; full strict ablations run each configured ablation workload over those seeds for each topology.
- Full aggregation computes paired HARP-vs-SAF-H acceptance statistics by topology/scenario: Wilcoxon signed-rank, bootstrap 95% CI for the paired mean difference, paired rank-biserial effect, and Holm-adjusted p-values across the predefined comparison family.

- [x] Write notebook cells for dependency/runtime capture, package/data discovery, source/checkpoint hash preflight, and topology validation.
- [x] Write smoke training and evaluator cells with fail-fast subprocess handling and output-row checks.
- [x] Gate full training/evaluation on `RUN_FULL=True` and complete smoke success.
- [x] Write full-matrix aggregation, manifest generation, and downloadable archive cells.

### Task 3: Verify the notebook artifact statically

**Files:**
- Verify: `notebook/rev1-benchmark-ablation-kaggle.ipynb`
- Verify: `src/analytics/benchmark/benchmark_algorithm/config.py`
- Verify: `src/analytics/benchmark/benchmark_algorithm/run_all.py`

- [x] Parse notebook JSON and compile every code cell without running any benchmark or training command.
- [x] Run focused seed-config tests only (`PYTHONPATH=. pytest -q tests/test_benchmark_seed_config.py`: 5 passed).
- [x] Inspect notebook configuration and confirm 3 topologies, smoke/full separation, 10 seeds, output roots, and manifest fields.
- [x] Inventory local `kaggle/results_kaggle` separately; notebook can inventory it only when attached to Kaggle as an input dataset.
- [x] Summarize P0s that remain dependent on Kaggle outputs vs additional code/paper work.
