from src.analytics.benchmark.benchmark_testbed.rev1_experiment import paired_statistics


def test_paired_statistics_reports_positive_effect_and_confidence_interval():
    result = paired_statistics([8, 9, 10, 11, 12], [5, 6, 7, 8, 9], bootstrap_samples=500)

    assert result["mean_paired_difference"] == 3.0
    assert result["ci95_low"] == 3.0
    assert result["ci95_high"] == 3.0
    assert result["rank_biserial"] == 1.0
    assert 0.0 <= result["p_value"] <= 1.0
