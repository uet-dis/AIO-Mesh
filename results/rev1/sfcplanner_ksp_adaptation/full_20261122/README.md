# K-shortest-path adaptation benchmark

This benchmark evaluates a K-shortest-path (KSP) adaptation against the paper's primary learned policy, protected HARP. It is **not a reproduction of SFCPlanner**. The cited SFCPlanner is an online deep-reinforcement-learning planner; this simulator exposes only placement/routing endpoint pairs, without ingress, egress, or an ordered VNF chain. KSP therefore ranks endpoint-pair candidates, which are then evaluated by the simulator's common admission predicate.

## Protocol

- Policies: protected HARP, protected KSP adaptation (`K=5`), and protected SAF-H (secondary heuristic reference).
- Topologies: Vietnam, NSFNET, and GEANT2.
- Workloads: Elephant stress (`heavy_tail`), Burst surge (`bursty`), and Chaos.
- 20 matched seeds × 3,000 steps per topology/workload/policy cell.
- Fresh environment per policy; execution order alternated by seed.
- All policies used common simulator admission checks; offered-trace hashes match across all 180 seed cells.
- All protected runs recorded zero admitted modeled hard violations.

## Primary result: KSP adaptation versus HARP

The table reports paired seed-level safe-acceptance differences (KSP minus HARP, percentage points). Confidence intervals are paired percentile-bootstrap 95% CIs; `p_Holm` is the Wilcoxon signed-rank p-value adjusted across the nine topology/workload comparisons. Positive values favor KSP.

| Topology | Workload | Δ (pp) | 95% CI (pp) | p_Holm |
|---|---|---:|---:|---:|
| GEANT2 | bursty | -0.803 | [-0.972, -0.642] | 0.0007938 |
| GEANT2 | chaos | +0.230 | [-0.003, +0.457] | 0.05948 |
| GEANT2 | heavy_tail | -0.098 | [-0.285, +0.095] | 0.1589 |
| NSFNET | bursty | +0.868 | [+0.682, +1.050] | 0.0007938 |
| NSFNET | chaos | +0.842 | [+0.623, +1.055] | 0.0007938 |
| NSFNET | heavy_tail | +0.540 | [+0.370, +0.715] | 0.0007938 |
| VIETNAM | bursty | +1.165 | [+1.010, +1.320] | 0.0007938 |
| VIETNAM | chaos | +1.173 | [+1.013, +1.337] | 0.0007938 |
| VIETNAM | heavy_tail | +1.372 | [+1.108, +1.648] | 0.0007938 |

KSP adaptation had higher safe acceptance than HARP in all six Vietnam and NSFNET cells, with all six differences remaining significant after Holm correction. HARP led in GEANT2 Burst surge (−0.803 pp for KSP; adjusted `p=0.00079`). The GEANT2 Chaos and Elephant-stress contrasts were not significant after correction. Thus the result is topology-dependent; it does not support a blanket claim that KSP is better than HARP. The KSP adaptation also incurs greater simulator decision cost than SAF-H in these measurements; see `summary.csv` and `per_seed.csv` for policy timing.

SAF-H is retained as a secondary heuristic reference. Its near-equality with KSP on several complete, latency-derived graphs helps characterize the pair-ranking adaptation, but it is not the primary comparison for this experiment.

## Files and reproduction

- `harp_comparison.csv`: focused paired KSP-versus-HARP statistics.
- `paired_statistics.csv` / `.json`: paired results against HARP and SAF-H, with bootstrap intervals, Wilcoxon tests, and Holm correction.
- `per_seed.csv`: all 540 policy/seed records, including trace hashes and timing metrics.
- `summary.csv`: seed-level means and sample SDs by cell.
- `manifest.json`: model, normalizer, data, source, and run metadata hashes.

Reproduce from the repository root:

```bash
python scripts/rev1/run_sfcplanner_ksp_adaptation.py \
  --topologies vietnam nsfnet geant2 \
  --scenarios elephant_stress burst_surge chaos \
  --seeds 42 100 2024 8888 9999 7 17 73 314 2718 11 23 47 89 101 211 512 1024 4096 8192 \
  --steps 3000 --k-paths 5 --output results/rev1/sfcplanner_ksp_adaptation/<run-id>

python scripts/rev1/summarize_sfcplanner_ksp_adaptation.py \
  results/rev1/sfcplanner_ksp_adaptation/<run-id>
```
