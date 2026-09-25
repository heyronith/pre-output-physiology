"""Temporal alignment / prompt-boundary consistency tests."""

from __future__ import annotations

import numpy as np
import pytest

from pre_output_physiology.trajectory import (
    COARSE_TRANSFORMER_BLOCKS,
    PREFIX_LENGTHS_K,
    PRIMARY_ANCHOR_LAYER,
    TemporalPoint,
    activation_generated_index,
    assert_future_tokens_excluded,
    compare_grouping_keys,
    prompt_boundary_activations_must_match,
)


def test_frozen_coarse_grid() -> None:
    assert COARSE_TRANSFORMER_BLOCKS == [0, 4, 8, 12, 16, 20, 24, 28, 31]
    assert PREFIX_LENGTHS_K == [0, 1, 2, 4, 8, 16]
    assert PRIMARY_ANCHOR_LAYER == 12
    assert 12 in COARSE_TRANSFORMER_BLOCKS


def test_k0_is_prompt_boundary() -> None:
    tp = TemporalPoint(0)
    assert tp.regime == "prompt_boundary_propensity"
    assert tp.generated_token_index is None
    assert activation_generated_index(0) is None


@pytest.mark.parametrize(
    ("k", "expected_idx"),
    [(1, 0), (2, 1), (4, 3), (8, 7), (16, 15)],
)
def test_activation_index_is_k_minus_one(k: int, expected_idx: int) -> None:
    """Off-by-one guard: state at generated index k-1 before predicting next."""
    assert activation_generated_index(k) == expected_idx
    assert TemporalPoint(k).generated_token_index == expected_idx
    assert TemporalPoint(k).regime == "early_trajectory_prediction"


def test_future_tokens_excluded_contract() -> None:
    assert_future_tokens_excluded(4, total_generated_tokens=10)
    with pytest.raises(ValueError):
        assert_future_tokens_excluded(11, total_generated_tokens=10)


def test_prompt_boundary_identical_prompts_must_match() -> None:
    acts = np.array([[1.0, 2.0], [1.0, 2.0], [0.0, 1.0]], dtype=float)
    groups = ["a", "a", "b"]
    out = prompt_boundary_activations_must_match(acts, groups)
    assert out["max_abs_within_group"] == 0.0


def test_prompt_boundary_mismatch_raises() -> None:
    acts = np.array([[1.0, 2.0], [1.0, 2.1]], dtype=float)
    with pytest.raises(AssertionError):
        prompt_boundary_activations_must_match(acts, ["g", "g"], atol=1e-5)


def test_grouping_key_agreement_detection() -> None:
    report = compare_grouping_keys(["p1_0", "p1_1", "p2_0"], ["h1", "h1", "h2"])
    assert report["agree"] is True
    assert report["scientific_key"] == "prompt_sha256"

    report2 = compare_grouping_keys(["p1_0", "p1_1"], ["h1", "h2"])
    assert report2["agree"] is False
    assert report2["n_prefixes_with_multiple_hashes"] == 1
