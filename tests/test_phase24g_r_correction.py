"""Phase 24G-R correction invariant tests (no GPU / no full grid)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pre_output_physiology.phase24c_design import (
    TEMPORAL_LANDMARKS,
    select_candidate_region,
)
from pre_output_physiology.phase24f_confirmation import confirmatory_pass
from pre_output_physiology.phase24g_diagnostics import PHASE24F_PRIMARY_HASH
from pre_output_physiology.phase24g_r_correction import (
    ANALYSIS_SEED,
    GUARANTEE,
    LOPO_N,
    NESTED_REPS,
    PRIOR_24G_ARTIFACTS,
    PROXY_COMPARE_NESTED,
    STARTING_SHA,
    STATUS,
    assert_nested_reps_exact,
    assert_no_proxy_in_primary,
    assert_phase24f_immutable,
    category_change_statement,
    compare_proxy_vs_full,
    snapshot_prior_24g_hashes,
    summarize_lopo,
    summarize_nested,
    verify_prior_24g_unchanged,
)

REPO = Path(__file__).resolve().parents[1]


def test_nested_reps_exactly_1000() -> None:
    assert NESTED_REPS == 1000
    assert_nested_reps_exact()


def test_lopo_n_is_28() -> None:
    assert LOPO_N == 28


def test_starting_sha_matches_authorization() -> None:
    assert STARTING_SHA == "8f2cbb67d0ad6a8ae4d0c8afb304729166f11a17"


def test_no_proxy_in_primary_diagnostics() -> None:
    assert_no_proxy_in_primary(False)
    with pytest.raises(RuntimeError, match="forbid proxy_delta"):
        assert_no_proxy_in_primary(True)


def test_full_surface_vs_surface_plus_activation_contrast_in_runner() -> None:
    """Corrected runner must call discovery with proxy_delta=False for primary."""
    src = (REPO / "scripts/run_phase24g_r_correction.py").read_text(encoding="utf-8")
    assert "proxy=False" in src
    assert "proxy_delta=proxy" in src
    assert "NESTED_REPS" in src
    assert "assert_no_proxy_in_primary" in src
    assert '"proxy": False' in src or "'proxy': False" in src
    # Primary LOPO payloads built with proxy=False
    assert "proxy=False, idx=i" in src or "proxy=False, idx=" in src
    lib = (REPO / "src/pre_output_physiology/phase24g_r_correction.py").read_text(
        encoding="utf-8"
    )
    assert "SURFACE_plus_ACTIVATION_minus_SURFACE" in lib
    assert "ACTIVATION − logits" not in src
    # Comparison-only proxy path exists but is labeled non-primary
    assert "comparison only; not primary" in src


def test_prompt_level_split_isolation() -> None:
    """Nested splits are prompt-level and disjoint."""
    prompts = [f"p{i:02d}" for i in range(28)]

    def make_splits(rep: int) -> tuple[set[str], set[str], set[str]]:
        rng = np.random.default_rng(ANALYSIS_SEED + rep)
        order = list(prompts)
        rng.shuffle(order)
        return set(order[:14]), set(order[14:21]), set(order[21:28])

    for rep in (0, 1, 7, 999):
        tr, va, ho = make_splits(rep)
        assert len(tr) == 14 and len(va) == 7 and len(ho) == 7
        assert tr.isdisjoint(va) and tr.isdisjoint(ho) and va.isdisjoint(ho)
        assert tr | va | ho == set(prompts)


def test_deterministic_nested_splits() -> None:
    prompts = [f"p{i:02d}" for i in range(28)]

    def make_splits(rep: int, seed: int = ANALYSIS_SEED):
        rng = np.random.default_rng(seed + rep)
        order = list(prompts)
        rng.shuffle(order)
        return tuple(order[:14]), tuple(order[14:21]), tuple(order[21:28])

    a = make_splits(42)
    b = make_splits(42)
    assert a == b
    assert make_splits(0) != make_splits(1)


def test_oof_stacking_leakage_prevention_contract() -> None:
    """oof_log_odds / stacked meta must be used in full (non-proxy) path."""
    diag = (REPO / "scripts/run_phase24g_diagnostics.py").read_text(encoding="utf-8")
    # Full path stacks OOF surface + OOF activation; proxy skips stacking
    assert "oof_log_odds" in diag
    assert "proxy_delta" in diag
    assert "Zc = np.stack([shared[\"oof_surf\"], oof_act], axis=1)" in diag
    # Proxy branch must not be the corrected primary path
    corr = (REPO / "scripts/run_phase24g_r_correction.py").read_text(encoding="utf-8")
    assert "proxy=False" in corr
    assert "assert_no_proxy_in_primary" in corr


def test_frozen_candidate_selection_rule() -> None:
    grid = {t: {L: float("-inf") for L in range(32)} for t in TEMPORAL_LANDMARKS}
    # Contiguous positive band at t=1, layers 18-22
    for L in range(18, 23):
        grid[1][L] = 0.05 + 0.01 * (L == 20)
    cand = select_candidate_region(grid)
    assert cand is not None
    assert cand["candidate_time"] == 1
    assert cand["candidate_layer"] == 20
    assert cand["band"] == [18, 22]


def test_phase24f_immutability() -> None:
    out = assert_phase24f_immutable(REPO)
    assert out["immutable"] is True
    assert out["primary_pass"] is False
    path = REPO / "artifacts/phase24f_confirmation/primary_metrics.json"
    blob = json.loads(path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == PHASE24F_PRIMARY_HASH
    assert confirmatory_pass(
        delta_auroc=float(blob["delta_auroc"]),
        ci95_low=float(blob["delta_auroc_bootstrap"]["ci95"][0]),
    ) is False


def test_prior_diagnostic_artifacts_not_overwritten_snapshot() -> None:
    snap = snapshot_prior_24g_hashes(REPO)
    assert snap["all_present"] is True
    for name in PRIOR_24G_ARTIFACTS:
        assert snap["artifacts"][name]["exists"] is True
        assert len(snap["artifacts"][name]["sha256"]) == 64
    check = verify_prior_24g_unchanged(REPO, snap)
    assert check["unchanged"] is True
    # Correction outputs must live in a separate directory
    out_dir = REPO / "artifacts/phase24g_r_correction"
    prior_dir = REPO / "artifacts/phase24g_diagnostics"
    assert out_dir.resolve() != prior_dir.resolve()


def test_summarize_and_compare_helpers() -> None:
    lopo_rows = [
        {
            "candidate": {
                "candidate_time": 1,
                "candidate_layer": 20,
                "band": [18, 22],
            },
            "delta_auroc": 0.1,
            "in_neighborhood": True,
        },
        {
            "candidate": {
                "candidate_time": 2,
                "candidate_layer": 10,
                "band": [8, 12],
            },
            "delta_auroc": 0.05,
            "in_neighborhood": False,
        },
    ]
    s = summarize_lopo(lopo_rows)
    assert s["n"] == 2
    assert s["proxy_used"] is False
    assert abs(s["frac_exact_t1_l20"] - 0.5) < 1e-9

    nested = [
        {
            "candidate": {"time": 1, "layer": 20, "band": [18, 22]},
            "discovery_delta": 0.2,
            "heldout_delta": 0.05,
            "optimism": 0.15,
        },
        {
            "candidate": None,
            "discovery_delta": float("nan"),
            "heldout_delta": float("nan"),
            "optimism": float("nan"),
        },
    ]
    ns = summarize_nested(nested)
    assert ns["n_reps_required"] == 1000
    assert ns["n_reps_completed"] == 2
    assert abs(ns["frac_candidate_exists"] - 0.5) < 1e-9

    cmp_ = compare_proxy_vs_full(lopo_rows, lopo_rows)
    assert cmp_["frac_time_agree"] == 1.0
    assert cmp_["frac_layer_agree"] == 1.0


def test_category_change_statement() -> None:
    assert category_change_statement("C", "C") == "preserves"
    assert category_change_statement("C", "A") == "changes"
    assert category_change_statement("C", "B") == "changes"
    assert category_change_statement("C", "D") == "weakens"


def test_status_and_guarantee_strings() -> None:
    assert STATUS == (
        "phase24g_r_corrected_diagnostics_complete_awaiting_independent_code_audit"
    )
    assert "1,000 NESTED" in GUARANTEE
    assert "PHASE-24F PRIMARY CONFIRMATORY FAILURE REMAINS FINAL" in GUARANTEE


def test_proxy_compare_nested_is_50_not_primary() -> None:
    assert PROXY_COMPARE_NESTED == 50
    assert PROXY_COMPARE_NESTED != NESTED_REPS


def test_runtime_stop_status_constant() -> None:
    from pre_output_physiology.phase24g_r_correction import STATUS_RUNTIME_STOP

    assert STATUS_RUNTIME_STOP == (
        "phase24g_r_stopped_full_pipeline_1000_nested_runtime_prohibitive"
    )
