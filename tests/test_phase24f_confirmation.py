"""Phase-24F locked-TEST confirmation unit tests (no GPU; no TEST open required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pre_output_physiology.phase24c_design import prompt_equal_balanced_weights
from pre_output_physiology.phase24f_confirmation import (
    ALLOWED_COORDINATE,
    CANDIDATE_LAYER,
    CANDIDATE_TIME,
    FROZEN_C,
    TEST_BOOTSTRAP_REPS,
    CoordinateViolationError,
    assert_frozen_coordinate,
    confirmatory_pass,
    delta_auroc_cluster_bootstrap,
    verify_frozen_provenance,
)

REPO = Path(__file__).resolve().parents[1]


def test_candidate_fixed_exactly_at_t1_l20() -> None:
    assert CANDIDATE_TIME == 1
    assert CANDIDATE_LAYER == 20
    assert ALLOWED_COORDINATE == (1, 20)
    assert_frozen_coordinate(1, 20)
    with pytest.raises(CoordinateViolationError):
        assert_frozen_coordinate(2, 20)
    with pytest.raises(CoordinateViolationError):
        assert_frozen_coordinate(1, 19)


def test_frozen_C_values() -> None:
    assert FROZEN_C == {
        "text": 0.01,
        "logits": 0.01,
        "surface": 0.1,
        "activation": 0.1,
        "surface_plus_activation": 10.0,
    }


def test_pass_criterion_requires_strict_ci_lower_bound() -> None:
    assert confirmatory_pass(delta_auroc=0.1, ci95_low=0.01) is True
    assert confirmatory_pass(delta_auroc=0.1, ci95_low=0.0) is False
    assert confirmatory_pass(delta_auroc=0.1, ci95_low=-0.01) is False
    assert confirmatory_pass(delta_auroc=0.0, ci95_low=0.01) is False
    assert confirmatory_pass(delta_auroc=-0.01, ci95_low=-0.1) is False


def test_cluster_bootstrap_10000_and_samples_prompts() -> None:
    y = np.array([0, 1, 0, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8, 0.15, 0.85])
    prompts = ["a", "a", "b", "b", "c", "c"]
    out = delta_auroc_cluster_bootstrap(
        y=y,
        surface_scores=s,
        combined_scores=s + 0.05,
        prompt_ids=prompts,
        n_reps=TEST_BOOTSTRAP_REPS,
        seed=1,
    )
    assert out["n_reps"] == 10_000
    assert "ci95" in out


def test_prompt_equal_weighting() -> None:
    labels = ["honest", "deceptive", "honest", "deceptive"]
    prompts = ["p1", "p1", "p2", "p2"]
    w = prompt_equal_balanced_weights(labels, prompts)
    assert abs(sum(w[:2]) - sum(w[2:])) < 1e-9


def test_frozen_provenance_verifies() -> None:
    ver = verify_frozen_provenance(REPO)
    assert ver["verified"] is True


def test_confirmatory_script_has_no_heatmap_or_search() -> None:
    src = (REPO / "scripts/run_phase24f_locked_test_confirmation.py").read_text(
        encoding="utf-8"
    )
    forbidden = (
        "svg_heatmap",
        "TEMPORAL_LANDMARKS",
        "select_candidate_region",
        "for L in range",
        "heatmap",
        "select_C_prompt_group_cv",
        "LOGISTIC_C_GRID",
    )
    for token in forbidden:
        assert token not in src, f"forbidden token in confirmatory script: {token}"
    assert "CANDIDATE_TIME" in src
    assert "CANDIDATE_LAYER" in src
    assert "assert_frozen_coordinate" in src


def test_no_alternate_coordinate_evaluable() -> None:
    with pytest.raises(CoordinateViolationError):
        assert_frozen_coordinate(32, 17)
