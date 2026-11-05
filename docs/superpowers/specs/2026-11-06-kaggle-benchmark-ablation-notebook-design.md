# Kaggle Benchmark and Ablation Notebook Design

## Intent

Create one Kaggle notebook that runs benchmark and HARP ablation workflows for NSFNET, GEANT2, and Vietnam. It must first run a lightweight end-to-end smoke stage, then keep full experiments behind a separate explicit switch so a smoke run cannot accidentally start the long jobs.

## Goals

- Exercise source packaging, data lookup, model training/loading, benchmark evaluation, ablation evaluation, and artifact export for all three topologies.
- Use temporary smoke checkpoints trained from the current source, since the existing v11 checkpoints record `MultiDiscrete([N,N])` while current `JOVDPREnv` uses `Discrete(N*N)`.
- Train smoke checkpoints for 512 total steps per variant/topology with one environment, then run a short benchmark and ablation evaluation (64 requests each) on each topology. Smoke numbers are pipeline diagnostics only.
- If smoke succeeds, let the user explicitly enable the full stage. Full training uses the topology-specific configured training steps and full evaluation uses matched seeds, 3000 requests per run, and scenarios aligned with P0 in `docs/Rev1/todo.md`.
- Export per-seed CSV/JSON, resolved configuration, source/data/checkpoint hashes, environment information, logs, and the notebook output as one downloadable archive.
- If a prior `results_kaggle` directory is attached as Kaggle input, inventory and hash its artifacts separately; do not merge them into the new smoke or full outputs.

## Non-goals

- The notebook will not claim convergence or use smoke results as paper evidence.
- It will not modify the shared `JOVDPREnv` action contract to fit the old checkpoint metadata.
- It will not run HIL or hardware tests; Kaggle runs are offline simulation/training.
- It will not automatically continue into full experiments after smoke passes.

## Approaches

1. Smoke the existing v11 checkpoints only. This is quickest, but does not exercise the pipeline if their saved action spaces do not match the current environment.
2. Train temporary, short-run checkpoints from the current source, then run both benchmark and strict ablation evaluators. This verifies a coherent current-source pipeline without treating smoke outputs as results. **Selected by the user.**
3. Change the shared environment to match the old checkpoint. This risks changing the action/mask semantics used by training and other evaluators, so it is not part of this notebook task.

## Proposed notebook flow

1. Install pinned/recorded runtime dependencies and locate the uploaded project package and data.
2. Print source revision, topology node counts, dataset hash, and existing checkpoint metadata; fail early on missing files.
3. Train short-lived smoke checkpoints into an isolated `/kaggle/working` directory for each topology and ablation variant.
4. Run benchmark smoke and strict ablation smoke for each topology using those same smoke checkpoints. Verify expected files, seed/topology labels, row counts, and finite metrics.
5. Set `SMOKE_PASSED` only after all topology/evaluator checks succeed. Keep `RUN_FULL=False` by default; the full-run cell must raise unless smoke passed and the user changes the switch.
6. Full stage trains ablation variants using configured production steps, then runs the benchmark and ablation evaluation matrices over matched seeds and configured scenarios. Results are written to a different output directory from smoke artifacts.
7. Write a manifest and zip all outputs for download.

## Constraints and implementation calls

- Reuse `src.analytics.training.train_harp_ablation`, `src.analytics.benchmark.benchmark_algorithm.run_all`, and `src.analytics.ablation.ablation_study` as the authoritative entry points where possible.
- Save training output under a model root that both benchmark and strict ablation loaders already recognize.
- Benchmark HARP uses the `harp_full` checkpoint trained from the current source; don't silently fall back to incompatible v11 artifacts.
- Smoke checkpoints use 512 total steps, one environment, seed 42, and a fixed scenario. Full steps, scenarios, and seeds are notebook configuration values recorded in the manifest.
- A failed topology/evaluator blocks the full-run gate and identifies its command/log. Do not skip a failed run and report a partial matrix as complete.
- The notebook is created under `notebook/` alongside the existing Kaggle ablation notebook.

## Acceptance checks

- Notebook JSON parses and every code cell compiles.
- Smoke configuration includes all 3 topologies and invokes both benchmark and ablation paths.
- Smoke and full outputs are isolated; full execution is opt-in and gated on complete smoke success.
- Full artifacts include raw per-seed results and metadata sufficient to reproduce each run.
