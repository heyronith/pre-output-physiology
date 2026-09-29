"""Unit tests for Phase-22C updated K=20 sampling forecast."""

from __future__ import annotations

import json
from pathlib import Path

from pre_output_physiology.phase22c_forecast import (
    EXPECTED_TEST_AT_20,
    EXPECTED_TRAIN_AT_20,
    N_OBSERVED,
    N_PROMPTS,
    PHASE21_ONSET_RUN,
    PHASE22B_ONSET_RUN,
    clopper_pearson_zero_ucb,
    decide_support,
    forecast_model,
    load_prompt_counts,
    reproduce_k20_gates,
)

REPO = Path(__file__).resolve().parents[1]


def _load(run_id: str) -> list[dict]:
    path = REPO / "artifacts/runs" / run_id / "annotated.jsonl"
    return [
        json.loads(x)
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def test_clopper_pearson_zero_ucb() -> None:
    u = clopper_pearson_zero_ucb(20, 0.05)
    assert 0.13 < u < 0.15
    # rule-of-three ≈ 3/20 = 0.15; CP is slightly lower


def test_k20_reproduction() -> None:
    rows = _load(PHASE21_ONSET_RUN) + _load(PHASE22B_ONSET_RUN)
    counts = load_prompt_counts(rows)
    assert len(counts) == N_PROMPTS
    assert all(c.n_total == N_OBSERVED for c in counts)
    k20 = reproduce_k20_gates(counts)
    assert k20["reproduced"] is True
    assert k20["n_train_qualifying"] == EXPECTED_TRAIN_AT_20
    assert k20["n_test_qualifying"] == EXPECTED_TEST_AT_20


def test_structural_zeros_blocks_never_seen_class() -> None:
    """A prompt with 0 honest at K=20 cannot qualify under structural zeros."""
    from pre_output_physiology.phase22c_forecast import PromptCounts

    counts = [
        PromptCounts(
            prompt_id="roleplay_000",
            split="test",
            n_honest=0,
            n_deceptive_explicit=15,
            n_ambiguous=5,
            n_exclude=0,
            n_deceptive_no_explicit=0,
            n_total=20,
        ),
        # Already-qualifying filler so train gate isn't the issue
        *[
            PromptCounts(
                prompt_id=f"roleplay_{i:03d}",
                split="train",
                n_honest=5,
                n_deceptive_explicit=5,
                n_ambiguous=10,
                n_exclude=0,
                n_deceptive_no_explicit=0,
                n_total=20,
            )
            for i in range(1, 30)
        ],
    ]
    f = forecast_model(
        counts, total_rollouts=80, model="structural_zeros", n_mc=200, seed=1
    )
    # The sole test prompt can never qualify → test_q always 0
    assert f["expected_test_qualifying"] == 0.0
    assert f["p_test_ge_8"] == 0.0
    assert f["p_both_gates"] == 0.0


def test_jeffreys_can_rescue_zero_class() -> None:
    from pre_output_physiology.phase22c_forecast import PromptCounts

    counts = [
        PromptCounts(
            prompt_id="roleplay_000",
            split="test",
            n_honest=0,
            n_deceptive_explicit=15,
            n_ambiguous=5,
            n_exclude=0,
            n_deceptive_no_explicit=0,
            n_total=20,
        ),
    ]
    f = forecast_model(counts, total_rollouts=80, model="jeffreys", n_mc=500, seed=2)
    # Jeffreys places positive mass on honest → sometimes reaches ≥2 honest
    assert f["expected_test_qualifying"] > 0.0


def test_decide_support_unsupported_when_all_low() -> None:
    def _fake(p: float, k: int) -> dict:
        return {
            "total_rollouts_per_prompt": k,
            "p_both_gates": p,
            "p_train_ge_25": 1.0,
            "p_test_ge_8": p,
        }

    results = {
        "jeffreys": [_fake(0.1, k) for k in (30, 40, 60, 80)],
        "empirical_weak": [_fake(0.1, k) for k in (30, 40, 60, 80)],
        "structural_zeros": [_fake(0.0, k) for k in (30, 40, 60, 80)],
        "conservative_rare_event": [_fake(0.1, k) for k in (30, 40, 60, 80)],
    }
    d = decide_support(results)
    assert d["status"] == "phase22c_sampling_expansion_unsupported"


def test_decide_support_supported_when_rule_met() -> None:
    def _fake(p: float, k: int) -> dict:
        return {
            "total_rollouts_per_prompt": k,
            "p_both_gates": p,
            "p_train_ge_25": 1.0,
            "p_test_ge_8": p,
        }

    results = {
        "jeffreys": [_fake(0.9 if k >= 40 else 0.4, k) for k in (30, 40, 60, 80)],
        "empirical_weak": [
            _fake(0.85 if k >= 40 else 0.3, k) for k in (30, 40, 60, 80)
        ],
        "structural_zeros": [_fake(0.0, k) for k in (30, 40, 60, 80)],
        "conservative_rare_event": [
            _fake(0.6 if k >= 40 else 0.2, k) for k in (30, 40, 60, 80)
        ],
    }
    d = decide_support(results)
    assert d["status"] == "phase22c_sampling_expansion_supported"
    assert d["smallest_supported_k"] == 40
