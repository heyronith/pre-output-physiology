"""Validate Phase-23A freeze artifact contracts."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FREEZE = REPO / "artifacts/phase23a_pilot/freeze.json"
CFG = REPO / "configs/experiments/phase23_open_grader_validation.yaml"


def test_phase23a_freeze_present() -> None:
    assert FREEZE.is_file()
    data = json.loads(FREEZE.read_text(encoding="utf-8"))
    assert data["status"] == "phase23a_pilot_complete_awaiting_audit"
    assert data["stage2_winner_selected"] is False
    assert set(data["pilot_survivors_only"]) == {"gemma4_31b_it", "qwen35_27b"}
    assert data["candidates"]["gpt_oss_20b"]["decision"] == "ELIMINATED"
    assert data["candidates"]["gpt_oss_20b"]["metrics_scope"] == "158_of_200_valid_rows_only"
    assert data["candidates"]["qwen35_27b"]["decision"] == "SURVIVE"
    assert data["candidates"]["gemma4_31b_it"]["decision"] == "SURVIVE"
    for k, v in data["authorizations"].items():
        assert v is False, k
    for cand in ("gemma4_31b_it", "qwen35_27b", "gpt_oss_20b"):
        ev = data["evidence"][cand]
        jpath = REPO / ev["judgments_path"]
        assert jpath.is_file(), jpath
        assert len(ev["judgments_sha256"]) == 64


def test_phase23a_config_authorizations_false() -> None:
    text = CFG.read_text(encoding="utf-8")
    assert "status: phase23a_pilot_complete_awaiting_audit" in text
    assert "modal_gpu_open_grader_inference_authorized: false" in text
    assert "stage2_development_authorized: false" in text
    assert "stage3_locked_validation_authorized: false" in text
    assert "stage4_onset_validation_authorized: false" in text
    assert "k_gt_20_generation_authorized: false" in text
    assert "physiology_authorized: false" in text
    assert "mistral_roleplay_generation_authorized: false" in text
    assert "openai_grading_api_authorized: false" in text
