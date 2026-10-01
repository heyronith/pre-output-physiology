"""Phase 24G-R2 Modal-500 unit tests (no Modal / no GPU)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pre_output_physiology.phase24c_design import (
    TEMPORAL_LANDMARKS,
    select_candidate_region,
)
from pre_output_physiology.phase24g_r2_modal500 import (
    AMENDMENT,
    ANALYSIS_SEED,
    GUARANTEE,
    LOPO_N,
    N_WORKERS,
    NESTED_REPS,
    STARTING_SHA,
    STATUS,
    assert_nested_reps_exact,
    assert_no_proxy_in_primary,
    assert_phase24f_immutable,
    atomic_write_json,
    config_hash,
    config_payload,
    load_all_nested_rows,
    nested_path,
    refuse_overwrite_completed,
    rep_id,
    scan_completed_nested,
    seed_for_rep,
    validate_nested_artifact,
    verify_prior_checkpoint_full_pipeline,
)

REPO = Path(__file__).resolve().parents[1]


def test_nested_reps_exactly_500() -> None:
    assert NESTED_REPS == 500
    assert_nested_reps_exact()


def test_amendment_recorded_prospectively() -> None:
    assert "1,000" in AMENDMENT or "1000" in AMENDMENT
    assert "500" in AMENDMENT
    assert "before" in AMENDMENT.lower() or "yet been observed" in AMENDMENT


def test_deterministic_rep_ids_and_seeds() -> None:
    assert rep_id(0) == "000"
    assert rep_id(499) == "499"
    with pytest.raises(ValueError):
        rep_id(500)
    assert seed_for_rep(0) == ANALYSIS_SEED
    assert seed_for_rep(42) == ANALYSIS_SEED + 42
    assert seed_for_rep(7) == seed_for_rep(7)


def test_atomic_write_and_non_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "rep_000.json"
    digest = atomic_write_json(path, {"ok": True, "n": 1})
    assert path.exists()
    assert len(digest) == 64
    assert json.loads(path.read_text())["ok"] is True
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        refuse_overwrite_completed(path)


def test_resumability_scan(tmp_path: Path) -> None:
    root = tmp_path / "results"
    # write three valid nested artifacts
    for rep in (0, 2, 5):
        blob = {
            "rep": rep,
            "rep_id": rep_id(rep),
            "seed": seed_for_rep(rep),
            "proxy": False,
            "candidate": None,
            "discovery_delta": float("nan"),
            "heldout_delta": float("nan"),
            "optimism": float("nan"),
            "split": {"tr": [], "va": [], "ho": []},
        }
        atomic_write_json(nested_path(root, rep), blob)
    scan = scan_completed_nested(root)
    assert scan["n_completed"] == 3
    assert 1 in scan["missing"]
    assert 0 in scan["completed"]


def test_aggregation_requires_all_500(tmp_path: Path) -> None:
    root = tmp_path / "results"
    atomic_write_json(
        nested_path(root, 0),
        {
            "rep": 0,
            "rep_id": "000",
            "seed": seed_for_rep(0),
            "proxy": False,
            "candidate": None,
            "discovery_delta": float("nan"),
            "heldout_delta": float("nan"),
            "optimism": float("nan"),
            "split": {"tr": [], "va": [], "ho": []},
        },
    )
    with pytest.raises(RuntimeError, match="cannot aggregate"):
        load_all_nested_rows(root)


def test_no_proxy_primary() -> None:
    assert_no_proxy_in_primary(False)
    with pytest.raises(RuntimeError):
        assert_no_proxy_in_primary(True)


def test_validate_rejects_proxy_nested() -> None:
    blob = {
        "rep": 0,
        "rep_id": "000",
        "seed": seed_for_rep(0),
        "proxy": True,
        "candidate": None,
        "discovery_delta": 0.1,
        "heldout_delta": 0.0,
        "optimism": 0.1,
        "split": {"tr": [], "va": [], "ho": []},
    }
    with pytest.raises(ValueError, match="proxy"):
        validate_nested_artifact(blob)


def test_frozen_candidate_rule() -> None:
    grid = {t: {L: float("-inf") for L in range(32)} for t in TEMPORAL_LANDMARKS}
    for L in range(18, 23):
        grid[1][L] = 0.05 + 0.01 * (L == 20)
    cand = select_candidate_region(grid)
    assert cand is not None
    assert cand["candidate_time"] == 1
    assert cand["candidate_layer"] == 20


def test_phase24f_immutable() -> None:
    out = assert_phase24f_immutable(REPO)
    assert out["immutable"] is True


def test_prior_4_of_28_checkpoint_is_full_pipeline() -> None:
    path = REPO / "artifacts/phase24g_r_correction/lopo_full_checkpoint.jsonl"
    rows = verify_prior_checkpoint_full_pipeline(path)
    assert len(rows) == 4
    assert all(r["proxy"] is False for r in rows)


def test_workers_and_status_constants() -> None:
    assert N_WORKERS == 8
    assert LOPO_N == 28
    assert STARTING_SHA == "be8bbdd8012ed0869d2b4b32c0000a417e45e0b7"
    assert STATUS.startswith("phase24g_r2_modal500")
    assert "500-REPETITION" in GUARANTEE
    assert "PHASE-24F REMAINS THE FINAL" in GUARANTEE


def test_config_hash_stable() -> None:
    a = config_hash(config_payload())
    b = config_hash(config_payload())
    assert a == b
    assert len(a) == 64


def test_optimism_interval_is_empirical_percentile_not_mean_ci() -> None:
    from pre_output_physiology.phase24g_r_correction import summarize_nested

    rows = []
    for opt in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        rows.append(
            {
                "discovery_delta": 0.5,
                "heldout_delta": 0.5 - opt,
                "optimism": opt,
                "candidate": {"time": 1, "layer": 20},
            }
        )
    s = summarize_nested(rows)
    assert "optimism_gap_ci95" not in s
    lo, hi = s["optimism_gap_empirical_percentile_025_975"]
    assert lo < hi
    # Matches sample quantiles, not a mean-CI construction.
    import numpy as np

    opts = [r["optimism"] for r in rows]
    assert lo == pytest.approx(float(np.quantile(opts, 0.025)))
    assert hi == pytest.approx(float(np.quantile(opts, 0.975)))


def test_provenance_sha_map_distinguishes_commits() -> None:
    path = REPO / "artifacts/phase24g_r2_modal500/provenance_sha_map.json"
    blob = json.loads(path.read_text(encoding="utf-8"))
    assert blob["analysis_logic_upload_commit"].startswith("650b81f")
    assert blob["successful_execution_wrapper_commit"].startswith("94add8a")
    assert blob["freeze_commit"].startswith("76fc2c4")
    assert blob["head_at_audit_repair_start"].startswith("a9c5f30")
    assert blob["volume_code_commit_txt"] == blob["analysis_logic_upload_commit"]
    assert blob["scientific_results_recomputed"] is False
    assert "not" in blob["note"].lower() and "wrapper" in blob["note"].lower()


def test_report_sha_fields_not_blank_or_swapped() -> None:
    text = (
        REPO / "reports/phase24g_r2_modal500_diagnostic_correction.md"
    ).read_text(encoding="utf-8")
    assert "**Starting SHA:** `be8bbdd8012ed0869d2b4b32c0000a417e45e0b7`" in text
    assert "**Freeze commit SHA:** `76fc2c4cedb0a31b33b7021b6e9a3f6db0a0348d`" in text
    assert "empirical 2.5" in text.lower()
    assert "confidence interval" in text.lower()
    assert "not" in text.lower().split("confidence interval")[0][-40:]
    # Starting must not be blank; freeze must not equal starting.
    assert "**Starting SHA:** \n" not in text
    assert (
        "**Freeze commit SHA:** `be8bbdd8012ed0869d2b4b32c0000a417e45e0b7`"
        not in text
    )
