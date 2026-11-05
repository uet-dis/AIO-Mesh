from types import SimpleNamespace

import numpy as np

from src.analytics.benchmark.benchmark_algorithm.config import scenario_jobs
from src.orchestration.jo_vdpr.env import JOVDPREnv


class FixedRng:
    def random(self):
        return 0.0


def test_chaos_uses_a_distinct_environment_scenario():
    assert scenario_jobs("chaos", "stress") == [("chaos", "chaos", "stress")]


def test_chaos_combines_burst_and_heavy_tail_request_pressure():
    env = JOVDPREnv.__new__(JOVDPREnv)
    env.repo = SimpleNamespace(
        get_next_entry=lambda step: {
            "cpu": 10.0,
            "ram": 4.0,
            "msd": 1,
            "service_type": "Data",
        }
    )
    env.max_cpu = 100.0
    env.node_msd_limits = np.array([10, 8])
    env.traffic_scenario = "chaos"
    env.np_random = FixedRng()

    env._load_request_for_step(4000)  # Burst peak plus a heavy-tail draw.

    assert env._current_req["cpu"] == 70.0
    assert env._current_req["msd"] == 4
