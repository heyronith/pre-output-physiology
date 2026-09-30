"""Phase-24E discovery unit tests (no GPU; no TEST access)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pre_output_physiology.phase24c_design import (
    TEMPORAL_LANDMARKS,
    prompt_equal_balanced_weights,
    prompt_group_folds,
    select_candidate_region,
)
from pre_output_physiology.phase24e_discovery import (
    SealViolationError,
    assert_not_test_path,
    assert_split_allowed,
    bf16_u16_to_float32,
    delta_h_vector,
    list_qualifying_bands,
    oof_log_odds,
    prefix_token_ids,
    prompt_cluster_bootstrap_ci,
    select_C_prompt_group_cv,
    survives_landmark,
    verify_frozen_input_hashes,
)

REPO = Path(__file__).resolve().parents[1]


def test_test_artifact_cannot_be_loaded_by_discovery_guards() -> None:
    with pytest.raises(SealViolationError):
        assert_not_test_path(
            "artifacts/phase24d_collection/by_split/LOCKED_TEST/labels_SEALED.json"
        )
    with pytest.raises(SealViolationError):
        assert_split_allowed("test")
    assert_split_allowed("train")
    assert_split_allowed("validation")


def test_no_future_token_leakage_and_landmark_alignment() -> None:
    gen = [10, 11, 12, 13, 14]
    assert prefix_token_ids(gen, 0) == []
    assert prefix_token_ids(gen, 2) == [10, 11]
    assert 13 not in prefix_token_ids(gen, 3)
    with pytest.raises(ValueError):
        prefix_token_ids(gen, 6)
    assert survives_landmark(5, 4) is True
    assert survives_landmark(4, 4) is False
    assert list(TEMPORAL_LANDMARKS) == [1, 2, 4, 8, 16, 32, 64]


def test_delta_h_calculation() -> None:
    h0 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ht = np.array([1.5, 0.0, 4.0], dtype=np.float32)
    d = delta_h_vector(ht, h0)
    np.testing.assert_allclose(d, [0.5, -2.0, 1.0])


def test_bf16_roundtrip_bits() -> None:
    # 1.0 in bf16 has same top bits as float32 1.0
    f = np.array([1.0, -2.0], dtype=np.float32)
    u16 = f.view(np.uint32) >> 16
    u16 = u16.astype(np.uint16)
    back = bf16_u16_to_float32(u16)
    np.testing.assert_allclose(back, f, rtol=1e-2)


def test_prompt_group_cv_isolation() -> None:
    folds = prompt_group_folds(["a", "b", "c", "d", "e"], n_folds=3)
    seen = set()
    for f in folds:
        for p in f:
            assert p not in seen
            seen.add(p)


def test_prompt_equal_weighting() -> None:
    labels = ["honest", "deceptive", "honest", "deceptive"]
    prompts = ["p1", "p1", "p2", "p2"]
    w = prompt_equal_balanced_weights(labels, prompts)
    assert abs(sum(w[:2]) - sum(w[2:])) < 1e-9


def test_oof_stacking_and_train_only_C() -> None:
    rng = np.random.default_rng(0)
    n = 40
    prompts = [f"p{i // 4}" for i in range(n)]
    y = np.array([i % 2 for i in range(n)])
    labels = ["honest" if yi == 0 else "deceptive" for yi in y]
    X = rng.normal(size=(n, 8))
    # make signal
    X[:, 0] += y * 2
    sel = select_C_prompt_group_cv(
        X=X, y=y, prompt_ids=prompts, labels_str=labels, kind="matrix"
    )
    assert sel["selected_C"] in (0.01, 0.1, 1.0, 10.0)
    oof = oof_log_odds(
        X=X,
        y=y,
        prompt_ids=prompts,
        labels_str=labels,
        kind="matrix",
        C=sel["selected_C"],
    )
    assert oof.shape == (n,)
    assert np.isfinite(oof).sum() > n // 2


def test_cluster_bootstrap_samples_prompts() -> None:
    y = np.array([0, 1, 0, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8, 0.15, 0.85])
    prompts = ["a", "a", "b", "b", "c", "c"]
    out = prompt_cluster_bootstrap_ci(
        y=y, scores_a=s, scores_b=s + 0.05, prompt_ids=prompts, n_reps=50, seed=1
    )
    assert "ci95" in out
    assert out["n_reps"] == 50


def test_exact_grid_and_candidate_rule() -> None:
    assert len(TEMPORAL_LANDMARKS) * 32 == 224
    grid = {t: {L: -0.1 for L in range(32)} for t in TEMPORAL_LANDMARKS}
    # earliest qualifying time = 2
    for L in range(3):
        grid[2][L] = 0.1
    grid[2][1] = 0.2
    for L in range(5, 10):
        grid[4][L] = 0.5
    cand = select_candidate_region(grid)
    assert cand is not None
    assert cand["candidate_time"] == 2
    assert cand["candidate_layer"] == 1
    bands = list_qualifying_bands(grid)
    assert any(b["time"] == 2 for b in bands)


def test_tie_break_lower_layer() -> None:
    grid = {t: {L: 0.0 for L in range(32)} for t in TEMPORAL_LANDMARKS}
    for L in range(3):
        grid[1][L] = 0.1
    cand = select_candidate_region(grid)
    assert cand["candidate_layer"] == 0


def test_contiguous_layer_median_threshold() -> None:
    grid = {t: {L: -0.1 for L in range(32)} for t in TEMPORAL_LANDMARKS}
    # three consecutive with Δ>0 but median < 0.05 → no candidate
    for L in (0, 1, 2):
        grid[1][L] = 0.04
    assert select_candidate_region(grid) is None
    # raise median above threshold
    for L in (0, 1, 2):
        grid[1][L] = 0.06
    cand = select_candidate_region(grid)
    assert cand is not None
    assert cand["candidate_time"] == 1
    assert cand["band"] == [0, 2]


def test_earliest_time_beats_later_stronger_band() -> None:
    grid = {t: {L: -0.1 for L in range(32)} for t in TEMPORAL_LANDMARKS}
    for L in range(3):
        grid[1][L] = 0.06
    for L in range(10, 20):
        grid[64][L] = 0.4
    cand = select_candidate_region(grid)
    assert cand["candidate_time"] == 1


def test_frozen_hashes_verify() -> None:
    ver = verify_frozen_input_hashes(REPO)
    assert ver["verified"] is True


def test_deterministic_analysis_seed_constant() -> None:
    from pre_output_physiology.phase24e_discovery import ANALYSIS_SEED

    assert ANALYSIS_SEED == 2405
