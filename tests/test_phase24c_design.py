"""Phase-24C design freeze unit tests (no GPU)."""

from __future__ import annotations

import hashlib

import numpy as np

from pre_output_physiology.phase24c_design import (
    CANDIDATE_MIN_MEDIAN_DELTA,
    DATASET_CONTENT_SHA256,
    FORECAST_SEED,
    K_GRID,
    N_PILOT_24B,
    STATUS_FORECAST_DISAGREE,
    STATUS_FROZEN,
    STATUS_K_NOT_SUPPORTED,
    TEMPORAL_LANDMARKS,
    assert_oof_stacking_no_leakage,
    build_train_val_test_split,
    cluster_bootstrap_indices,
    confirmatory_support,
    delta_h,
    dirichlet_multinomial_draw,
    locked_test_requires_frozen_candidate,
    prediction_step_surface_prefix_tokens,
    prompt_equal_balanced_weights,
    prompt_group_folds,
    select_candidate_region,
    select_k_from_forecasts,
    sha256_json,
    simulate_prompt_total_k,
    split_key,
)


def _fake_split():
    pilot = [f"pilot_{i:02d}" for i in range(16)]
    # 28 DEV = 16 pilot + 12 others
    others = [f"dev_{i:02d}" for i in range(12)]
    locked = [f"lock_{i:02d}" for i in range(11)]
    return build_train_val_test_split(
        pilot_24b_ids=pilot,
        development_eligible=pilot + others,
        locked_eligible=locked,
    )


def test_deterministic_split_and_pilot_train_only() -> None:
    a = _fake_split()
    b = _fake_split()
    assert a["train_prompt_ids"] == b["train_prompt_ids"]
    assert a["validation_prompt_ids"] == b["validation_prompt_ids"]
    assert a["test_prompt_ids"] == b["test_prompt_ids"]
    pilot = set(a["pilot_24b_prompt_ids"])
    assert pilot.isdisjoint(a["validation_prompt_ids"])
    assert pilot.isdisjoint(a["test_prompt_ids"])
    assert pilot.issubset(a["train_prompt_ids"])
    assert len(a["train_prompt_ids"]) == 20
    assert len(a["validation_prompt_ids"]) == 8
    assert len(a["test_prompt_ids"]) == 11


def test_locked_cannot_enter_train_or_val() -> None:
    s = _fake_split()
    locked = set(s["test_prompt_ids"])
    assert locked.isdisjoint(s["train_prompt_ids"])
    assert locked.isdisjoint(s["validation_prompt_ids"])


def test_split_key_deterministic() -> None:
    assert split_key("roleplay_003") == hashlib.sha256(
        b"phase24c|split|roleplay_003"
    ).hexdigest()


def test_dirichlet_and_simulate_reproducible() -> None:
    rng1 = np.random.default_rng(FORECAST_SEED)
    rng2 = np.random.default_rng(FORECAST_SEED)
    a = dirichlet_multinomial_draw(rng1, (1, 2, 3), n_draw=10)
    b = dirichlet_multinomial_draw(rng2, (1, 2, 3), n_draw=10)
    assert a == b
    assert sum(a) == 10
    tot = simulate_prompt_total_k(
        np.random.default_rng(1), (1, 2, 3), k=8, retain_observed=6
    )
    assert sum(tot) == 8
    assert tot[0] >= 1 and tot[1] >= 2 and tot[2] >= 3


def test_k_selection_and_disagreement_rules() -> None:
    def row(pass_all: bool, p_train: float = 0.95) -> dict:
        return {
            "all_gates_pass": pass_all,
            "p_train_ge10_of_20_ge2": p_train,
            "p_val_ge4_of_8_ge2": 0.9 if pass_all else 0.1,
            "p_test_ge5_of_11_ge2": 0.9 if pass_all else 0.1,
            "median_test_honest": 25 if pass_all else 5,
            "median_test_deceptive": 25 if pass_all else 5,
            "gates": {
                "train": pass_all,
                "validation": pass_all,
                "test_prompts": pass_all,
                "test_median_h": pass_all,
                "test_median_d": pass_all,
            },
        }

    primary = {"by_k": {str(k): row(False) for k in K_GRID}}
    hist = {"by_k": {str(k): row(False) for k in K_GRID}}
    assert select_k_from_forecasts(primary, hist)["status"] == STATUS_K_NOT_SUPPORTED

    primary["by_k"]["10"] = row(True)
    hist["by_k"]["10"] = row(True, p_train=0.95)
    sel = select_k_from_forecasts(primary, hist)
    assert sel["status"] == STATUS_FROZEN
    assert sel["selected_k"] == 10

    hist["by_k"]["10"] = row(False, p_train=0.2)
    hist["by_k"]["10"]["p_train_ge10_of_20_ge2"] = 0.2
    hist["by_k"]["10"]["p_val_ge4_of_8_ge2"] = 0.9
    hist["by_k"]["10"]["p_test_ge5_of_11_ge2"] = 0.9
    hist["by_k"]["10"]["median_test_honest"] = 25
    hist["by_k"]["10"]["median_test_deceptive"] = 25
    sel2 = select_k_from_forecasts(primary, hist)
    assert sel2["status"] == STATUS_FORECAST_DISAGREE


def test_temporal_no_future_token_alignment() -> None:
    gen = [10, 11, 12, 13]
    assert prediction_step_surface_prefix_tokens(gen, 0) == []
    assert prediction_step_surface_prefix_tokens(gen, 1) == [10]
    assert prediction_step_surface_prefix_tokens(gen, 3) == [10, 11, 12]
    # token t and future must not appear
    assert 13 not in prediction_step_surface_prefix_tokens(gen, 3)


def test_delta_h() -> None:
    assert delta_h([1.0, 2.0], [0.5, 0.5]) == [0.5, 1.5]


def test_grouped_folds_and_prompt_weights() -> None:
    folds = prompt_group_folds(["a", "b", "c", "d", "e"], n_folds=2)
    assert sorted(x for f in folds for x in f) == ["a", "b", "c", "d", "e"]
    # no duplicate prompts across folds
    seen = set()
    for f in folds:
        for p in f:
            assert p not in seen
            seen.add(p)
    labels = ["honest", "deceptive", "honest", "deceptive"]
    prompts = ["p1", "p1", "p2", "p2"]
    w = prompt_equal_balanced_weights(labels, prompts)
    assert len(w) == 4
    assert abs(sum(w[i] for i in range(2)) - sum(w[i] for i in range(2, 4))) < 1e-9


def test_candidate_region_and_tie_break() -> None:
    # time 1 fails; time 2 has band 0-2 with high median
    grid = {
        1: {L: -0.1 for L in range(32)},
        2: {L: 0.0 for L in range(32)},
        4: {L: 0.0 for L in range(32)},
    }
    for L in range(3):
        grid[2][L] = 0.1
    grid[2][1] = 0.2  # best in band
    # also a later time with better band — must choose earliest time 2
    for L in range(5, 10):
        grid[4][L] = 0.5
    cand = select_candidate_region(grid)
    assert cand is not None
    assert cand["candidate_time"] == 2
    assert cand["candidate_layer"] == 1
    assert cand["band"] == [0, 2]
    assert cand["band_median_delta_auroc"] >= CANDIDATE_MIN_MEDIAN_DELTA

    # tie: two layers same Δ → lower layer
    grid2 = {t: {L: 0.0 for L in range(32)} for t in TEMPORAL_LANDMARKS}
    for L in range(3):
        grid2[1][L] = 0.1
    cand2 = select_candidate_region(grid2)
    assert cand2["candidate_layer"] == 0


def test_locked_test_requires_candidate() -> None:
    assert locked_test_requires_frozen_candidate(None) is False
    assert (
        locked_test_requires_frozen_candidate(
            {"candidate_time": 4, "candidate_layer": 12}
        )
        is True
    )


def test_cluster_bootstrap_resamples_prompts() -> None:
    idx = cluster_bootstrap_indices(n_prompts=11, n_reps=100, seed=0)
    assert idx.shape == (100, 11)
    assert idx.min() >= 0 and idx.max() < 11


def test_confirmatory_support() -> None:
    assert confirmatory_support(0.1, 0.01, 0.2) is True
    assert confirmatory_support(0.1, -0.01, 0.2) is False
    assert confirmatory_support(-0.01, -0.02, 0.0) is False


def test_dataset_sha_constant() -> None:
    assert (
        DATASET_CONTENT_SHA256
        == "7d3e36dc9a2fa72ff823b651c2ade2e0fc34c5bc6267cdf9b7be8b046ed2e564"
    )


def test_sha_json_stable() -> None:
    assert sha256_json({"a": 1}) == sha256_json({"a": 1})


def test_k_grid_frozen() -> None:
    assert K_GRID == (6, 8, 10, 12, 16, 20)
    assert N_PILOT_24B == 16


def test_cross_fitted_stacking_leakage_prevention() -> None:
    fold_of = [0, 0, 1, 1, 2, 2]
    oof = [0, 0, 1, 1, 2, 2]
    assert assert_oof_stacking_no_leakage(
        fold_of=fold_of, base_pred_source_fold=oof
    )
    leaked = [0, 0, 1, 1, 2, 0]  # last sample used in-fold base pred
    assert not assert_oof_stacking_no_leakage(
        fold_of=fold_of, base_pred_source_fold=leaked
    )
