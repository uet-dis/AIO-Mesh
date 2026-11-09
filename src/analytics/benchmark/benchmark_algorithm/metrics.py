from __future__ import annotations

from dataclasses import dataclass, field, asdict
from statistics import mean, pstdev
from typing import Dict, List

import numpy as np


@dataclass
class RunMetrics:
    algorithm: str
    scenario: str
    topology: str
    seed: int
    acceptance: List[bool] = field(default_factory=list)
    msd_violations: List[bool] = field(default_factory=list)
    latencies_ms: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    sla_violations: List[bool] = field(default_factory=list)
    evacuation_hits: List[bool] = field(default_factory=list)
    queue_latencies_ms: List[float] = field(default_factory=list)
    prop_latencies_ms: List[float] = field(default_factory=list)
    srv6_latencies_ms: List[float] = field(default_factory=list)
    candidate_action_counts: List[int] = field(default_factory=list)
    environment_feasible_action_counts: List[int] = field(default_factory=list)
    policy_allowed_action_counts: List[int] = field(default_factory=list)
    policy_pruning_rates: List[float] = field(default_factory=list)
    empty_mask_fallbacks: List[bool] = field(default_factory=list)
    selected_action_infeasible: List[bool] = field(default_factory=list)
    candidate_cpu_infeasible: List[bool] = field(default_factory=list)
    candidate_ram_infeasible: List[bool] = field(default_factory=list)
    candidate_msd_infeasible: List[bool] = field(default_factory=list)

    @property
    def requests(self) -> int:
        return len(self.acceptance)

    @property
    def acc_rate(self) -> float:
        return float(np.mean(self.acceptance) * 100.0) if self.acceptance else 0.0

    @property
    def msd_rate(self) -> float:
        return float(np.mean(self.msd_violations) * 100.0) if self.msd_violations else 0.0

    @property
    def avg_lat(self) -> float:
        return float(np.mean(self.latencies_ms)) if self.latencies_ms else 0.0

    @property
    def p95_lat(self) -> float:
        return float(np.percentile(self.latencies_ms, 95)) if self.latencies_ms else 0.0

    @property
    def sla_rate(self) -> float:
        return float(np.mean(self.sla_violations) * 100.0) if self.sla_violations else 0.0

    @property
    def evac_rate(self) -> float:
        return float(np.mean(self.evacuation_hits) * 100.0) if self.evacuation_hits else 0.0

    @property
    def avg_reward(self) -> float:
        return float(np.mean(self.rewards)) if self.rewards else 0.0

    @property
    def avg_queue_lat(self) -> float:
        return float(np.mean(self.queue_latencies_ms)) if self.queue_latencies_ms else 0.0

    @property
    def mean_candidate_actions(self) -> float:
        return float(np.mean(self.candidate_action_counts)) if self.candidate_action_counts else 0.0

    @property
    def mean_policy_allowed_actions(self) -> float:
        return float(np.mean(self.policy_allowed_action_counts)) if self.policy_allowed_action_counts else 0.0

    @property
    def mean_environment_feasible_actions(self) -> float:
        return float(np.mean(self.environment_feasible_action_counts)) if self.environment_feasible_action_counts else 0.0

    @property
    def mean_policy_pruning_rate(self) -> float:
        return float(np.mean(self.policy_pruning_rates)) if self.policy_pruning_rates else 0.0

    @property
    def empty_mask_fallback_rate(self) -> float:
        return float(np.mean(self.empty_mask_fallbacks) * 100.0) if self.empty_mask_fallbacks else 0.0

    @property
    def selected_action_infeasible_rate(self) -> float:
        return float(np.mean(self.selected_action_infeasible) * 100.0) if self.selected_action_infeasible else 0.0

    @property
    def candidate_cpu_infeasible_rate(self) -> float:
        return float(np.mean(self.candidate_cpu_infeasible) * 100.0) if self.candidate_cpu_infeasible else 0.0

    @property
    def candidate_ram_infeasible_rate(self) -> float:
        return float(np.mean(self.candidate_ram_infeasible) * 100.0) if self.candidate_ram_infeasible else 0.0

    @property
    def candidate_msd_infeasible_rate(self) -> float:
        return float(np.mean(self.candidate_msd_infeasible) * 100.0) if self.candidate_msd_infeasible else 0.0

    def rolling_acceptance(self, window: int = 200) -> np.ndarray:
        if not self.acceptance:
            return np.array([])
        arr = np.array(self.acceptance, dtype=float)
        if len(arr) < window:
            return np.array([self.acc_rate] * len(arr))
        ret = np.cumsum(arr)
        ret[window:] = ret[window:] - ret[:-window]
        return (ret[window - 1:] / window) * 100.0

    def collect(self, info: dict, reward: float) -> None:
        if info.get("skipped", False):
            return
        if "candidate_action_count" in info:
            self.candidate_action_counts.append(int(info["candidate_action_count"]))
            self.environment_feasible_action_counts.append(int(info["environment_feasible_action_count"]))
            self.policy_allowed_action_counts.append(int(info["policy_allowed_action_count"]))
            self.policy_pruning_rates.append(float(info["policy_pruning_rate"]))
            self.empty_mask_fallbacks.append(bool(info["mask_empty_fallback"]))
            self.selected_action_infeasible.append(not bool(info["selected_action_environment_feasible"]))
            self.candidate_cpu_infeasible.append(bool(info.get("candidate_cpu_infeasible", False)))
            self.candidate_ram_infeasible.append(bool(info.get("candidate_ram_infeasible", False)))
            self.candidate_msd_infeasible.append(bool(info.get("candidate_msd_infeasible", False)))
        accepted = bool(info.get("accepted", False))
        self.acceptance.append(accepted)
        self.msd_violations.append(bool(info.get("admitted_msd_violation", False)))
        self.rewards.append(float(reward))
        self.evacuation_hits.append(bool(info.get("evacuation_hit", False)))
        if accepted:
            lat = float(info.get("total_latency_ms", 0.0))
            self.latencies_ms.append(lat)
            self.queue_latencies_ms.append(float(info.get("queue_latency_ms", 0.0)))
            self.prop_latencies_ms.append(float(info.get("prop_latency_ms", 0.0)))
            self.srv6_latencies_ms.append(float(info.get("srv6_latency_ms", 0.0)))
            threshold = float(info.get("latency_threshold_ms", 80.0))
            self.sla_violations.append(lat > threshold)

    def as_summary(self) -> Dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "topology": self.topology,
            "scenario": self.scenario,
            "seed": self.seed,
            "requests": self.requests,
            "acceptance_rate": self.acc_rate,
            "msd_violation_rate": self.msd_rate,
            "sla_violation_rate": self.sla_rate,
            "avg_latency_ms": self.avg_lat,
            "p95_latency_ms": self.p95_lat,
            "avg_queue_latency_ms": self.avg_queue_lat,
            "evacuation_hit_rate": self.evac_rate,
            "avg_reward": self.avg_reward,
            "mean_candidate_actions": self.mean_candidate_actions,
            "mean_environment_feasible_actions": self.mean_environment_feasible_actions,
            "mean_policy_allowed_actions": self.mean_policy_allowed_actions,
            "mean_policy_pruning_rate": self.mean_policy_pruning_rate,
            "empty_mask_fallback_rate": self.empty_mask_fallback_rate,
            "selected_action_infeasible_rate": self.selected_action_infeasible_rate,
            "candidate_cpu_infeasible_rate": self.candidate_cpu_infeasible_rate,
            "candidate_ram_infeasible_rate": self.candidate_ram_infeasible_rate,
            "candidate_msd_infeasible_rate": self.candidate_msd_infeasible_rate,
        }


def aggregate_runs(runs: List[RunMetrics]) -> Dict[str, object]:
    if not runs:
        return {}
    keys = [
        ("acceptance_rate", "acc_rate"),
        ("msd_violation_rate", "msd_rate"),
        ("sla_violation_rate", "sla_rate"),
        ("avg_latency_ms", "avg_lat"),
        ("p95_latency_ms", "p95_lat"),
        ("avg_queue_latency_ms", "avg_queue_lat"),
        ("evacuation_hit_rate", "evac_rate"),
        ("avg_reward", "avg_reward"),
        ("mean_candidate_actions", "mean_candidate_actions"),
        ("mean_environment_feasible_actions", "mean_environment_feasible_actions"),
        ("mean_policy_allowed_actions", "mean_policy_allowed_actions"),
        ("mean_policy_pruning_rate", "mean_policy_pruning_rate"),
        ("empty_mask_fallback_rate", "empty_mask_fallback_rate"),
        ("selected_action_infeasible_rate", "selected_action_infeasible_rate"),
        ("candidate_cpu_infeasible_rate", "candidate_cpu_infeasible_rate"),
        ("candidate_ram_infeasible_rate", "candidate_ram_infeasible_rate"),
        ("candidate_msd_infeasible_rate", "candidate_msd_infeasible_rate"),
    ]
    out: Dict[str, object] = {
        "algorithm": runs[0].algorithm,
        "topology": runs[0].topology,
        "scenario": runs[0].scenario,
        "seeds": len(runs),
        "requests_mean": mean([r.requests for r in runs]),
    }
    for label, attr in keys:
        vals = [float(getattr(r, attr)) for r in runs]
        out[f"{label}_mean"] = mean(vals)
        out[f"{label}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
    return out
