from pathlib import Path

import pytest

from src.analytics.benchmark.benchmark_testbed.baseline_simulator import BaselineSimulator
from src.analytics.benchmark.benchmark_testbed.models import RequestSpec, ScenarioDefinition


@pytest.mark.parametrize("algorithm", ["greedy_protected", "decoupled_protected"])
def test_protected_baselines_reject_msd_infeasible_actions_through_admission(tmp_path: Path, algorithm: str):
    scenario = ScenarioDefinition(
        name="protected_admission_test",
        title="Protected admission test",
        description="A single request whose SID depth exceeds every node profile.",
        default_steps=1,
        request_factory=lambda step, total, rng: RequestSpec(
            service_type="Data",
            cpu_req=1.0,
            ram_req=1.0,
            msd_req=100,
            source_node=0,
            destination_node=1,
        ),
    )

    records, summary = BaselineSimulator(seed=7, steps=1).run(scenario, algorithm, tmp_path)

    assert len(records) == 1
    assert records[0].accepted is False
    assert records[0].status_code == 409
    assert records[0].reject_reason in {"placement_node_full", "routing_node_full"}
    assert summary.msd_violation_rate == 0.0
