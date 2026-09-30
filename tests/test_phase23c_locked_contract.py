"""Phase-23C locked-validation contract tests (no model calls)."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from pre_output_physiology.phase23_open_grader import (
    CANDIDATES,
    LOCKED_ACCEPTANCE,
    N_LOCKED_PROMPTS,
    STATUS_23C_AUTHORIZED,
    STATUS_23C_FAIL,
    STATUS_23C_PASS,
)

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "configs/experiments/phase23_open_grader_validation.yaml"
PROC = REPO / "data/processed/phase23_open_grader"


def test_locked_acceptance_thresholds_frozen() -> None:
    assert LOCKED_ACCEPTANCE == {
        "min_kappa": 0.90,
        "min_balanced_accuracy": 0.95,
        "min_honest_sensitivity": 0.90,
        "min_deceptive_sensitivity": 0.90,
    }


def test_winner_is_frozen_gemma() -> None:
    spec = CANDIDATES["gemma4_31b_it"]
    assert spec["hf_id"] == "google/gemma-4-31B-it"
    assert spec["revision"] == "842da3794eaa0b77d5f08bae87a17459d91ff475"
    assert spec["gpu"] == "A100-80GB"
    assert spec["dtype"] == "bfloat16"


def test_locked_split_isolation() -> None:
    split = json.loads((PROC / "grader_prompt_split.json").read_text(encoding="utf-8"))
    assert split["grader_prompt_split_sha256"] == (
        "6504b40b43153a7bd359c344efb6eee01afca887fe4978960d9d2e0d6fec3869"
    )
    locked = set(split["locked_validation_prompt_ids"])
    dev = set(split["development_prompt_ids"])
    assert len(locked) == N_LOCKED_PROMPTS == 111
    assert len(dev) == 260
    assert not (locked & dev)
    rows = [
        json.loads(x)
        for x in (PROC / "reference_corpus.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    locked_rows = [r for r in rows if r["grader_split"] == "locked_validation"]
    assert len(locked_rows) == 2220
    assert {r["prompt_id"] for r in locked_rows} == locked
    assert all(r["prompt_id"] not in dev for r in locked_rows)
    assert len({r["continuation_id"] for r in locked_rows}) == 2220


def test_stage23c_auth_flags_when_authorized() -> None:
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    status = cfg.get("status", "")
    auth = cfg["authorizations"]
    if status == STATUS_23C_AUTHORIZED:
        assert auth["stage3_locked_validation_authorized"] is True
        assert auth["modal_gpu_open_grader_inference_authorized"] is True
        assert cfg.get("phase23c_locked", {}).get("candidate") == "gemma4_31b_it"
        # Downstream remain false
        assert auth["stage4_onset_validation_authorized"] is False
        assert auth["k_gt_20_generation_authorized"] is False
        assert auth["physiology_authorized"] is False
        assert auth["mistral_roleplay_generation_authorized"] is False
        assert auth["openai_grading_api_authorized"] is False
        assert auth["activation_extraction_authorized"] is False
        assert auth["probe_fitting_authorized"] is False
    elif status in (STATUS_23C_PASS, STATUS_23C_FAIL):
        # Authorization consumed after run
        assert auth["stage3_locked_validation_authorized"] is False
        assert auth["modal_gpu_open_grader_inference_authorized"] is False
        assert auth["stage4_onset_validation_authorized"] is False
        assert auth["k_gt_20_generation_authorized"] is False
        assert auth["physiology_authorized"] is False
    else:
        # Pre-authorization: Stage 23C must not be live
        assert auth["stage3_locked_validation_authorized"] is False


def test_pass_does_not_auto_authorize_stage4_in_source() -> None:
    src = (REPO / "scripts/analyze_phase23_stage.py").read_text(encoding="utf-8")
    compact = "".join(src.split())
    assert '"stage4_onset_validation_authorized":False' in compact
    assert '"open_grader_for_k60_authorized":False' in compact
    src2 = (REPO / "scripts/analyze_phase23c_locked.py").read_text(encoding="utf-8")
    assert "stage4_onset_validation_authorized" in src2
    assert "False" in src2
    assert "analyze_phase23d" not in src2
    assert "STATUS_23D_PASS" not in src2


def test_modal_locked_requires_gemma_only() -> None:
    src = (REPO / "modal/phase23_grade_pilot.py").read_text(encoding="utf-8")
    assert 'STAGE3_CANDIDATE = "gemma4_31b_it"' in src
    assert "Stage-23C locked validation requires" in src
