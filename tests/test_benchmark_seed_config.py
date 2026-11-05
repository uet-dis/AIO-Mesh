import pytest

from src.analytics.benchmark.benchmark_algorithm.config import AlgorithmBenchmarkConfig, EVAL_SEEDS
from src.analytics.benchmark.benchmark_algorithm.run_all import parse_args


def test_explicit_evaluation_seeds_override_legacy_defaults():
    config = AlgorithmBenchmarkConfig(evaluation_seeds=[42, 7, 17])

    assert config.seeds == [42, 7, 17]


def test_legacy_single_seed_keeps_seed_42():
    config = AlgorithmBenchmarkConfig(single_seed=True)

    assert config.seeds == [42]


def test_default_evaluation_seeds_remain_unchanged():
    config = AlgorithmBenchmarkConfig()

    assert config.seeds == EVAL_SEEDS


def test_benchmark_cli_accepts_explicit_seed_list():
    args = parse_args(["--seeds", "42", "7", "17"])

    assert args.seeds == [42, 7, 17]


def test_benchmark_cli_rejects_duplicate_seed_values():
    with pytest.raises(SystemExit):
        parse_args(["--seeds", "42", "42"])
