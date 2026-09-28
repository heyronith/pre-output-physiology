"""Tests for Phase-22 sampling forecast (no model calls)."""

from __future__ import annotations

from pre_output_physiology.phase22_forecast import (
    PromptCounts,
    decide_support,
    forecast_additional_sampling,
    stability_bucket,
)


def _pc(pid: str, split: str, h: int, d: int, other: int = 0) -> PromptCounts:
    # Pack other into ambiguous for simplicity
    amb = other
    assert h + d + amb == 10
    return PromptCounts(
        prompt_id=pid,
        split=split,
        n_honest=h,
        n_deceptive_explicit=d,
        n_ambiguous=amb,
        n_exclude=0,
        n_deceptive_no_explicit=0,
        n_total=10,
    )


def test_stability_buckets() -> None:
    assert stability_bucket(10, 0) == "10/0_or_0/10"
    assert stability_bucket(0, 10) == "10/0_or_0/10"
    assert stability_bucket(9, 1) == "9/1_or_1/9"
    assert stability_bucket(2, 8) == "8/2_or_2/8"
    assert stability_bucket(5, 5) == "more_balanced_or_other"
    assert stability_bucket(3, 2) == "more_balanced_or_other"


def test_already_qualifying_stays_at_extra_zero() -> None:
    counts = [_pc(f"p{i:03d}", "train", 2, 2, 6) for i in range(30)]
    counts += [_pc(f"t{i:03d}", "test", 2, 2, 6) for i in range(10)]
    # Pad to not matter — forecast with extra=0 via total=10
    # Need exactly using total_rollouts=10
    res = forecast_additional_sampling(counts, total_rollouts=10, n_mc=200, seed=1)
    assert res["expected_train_qualifying"] == 30.0
    assert res["expected_test_qualifying"] == 10.0
    assert res["p_both_gates"] == 1.0


def test_decide_support_thresholds() -> None:
    low = [{"total_rollouts_per_prompt": 20, "p_both_gates": 0.1}]
    mid = [{"total_rollouts_per_prompt": 40, "p_both_gates": 0.6}]
    high = [{"total_rollouts_per_prompt": 80, "p_both_gates": 0.9}]
    assert decide_support(low)["status"].endswith("unsupported")
    assert decide_support(mid)["verdict"] == "supported"
    assert decide_support(high)["verdict"] == "supported_strong"
