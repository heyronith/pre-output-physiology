"""Phase 3A methodology / config contract tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from pre_output_physiology.baselines import (
    SurfaceBaselineContract,
    build_prompt_plus_prefix_text,
    contract_as_dict,
    make_text_baseline_pipeline,
    visible_prompt_plus_prefix_text,
)
from pre_output_physiology.config import EXPERIMENTS_DIR, load_experiment_config
from pre_output_physiology.metrics import (
    bootstrap_auroc_ci_by_group,
    paired_delta_auroc_ci_by_group,
)
from pre_output_physiology.trajectory import (
    COARSE_TRANSFORMER_BLOCKS,
    PREFIX_LENGTHS_K,
    PRIMARY_ANCHOR_LAYER,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PINNED_MODEL_REV = "63a8b081895390a26e140280378bc85ec8bce07a"
PINNED_LASR_HF = "bf93584916fbd23121eca6f2017017df0ef3184f"


def test_phase3_experiment_status_awaits_gpu() -> None:
    cfg = load_experiment_config(EXPERIMENTS_DIR / "phase3_preoutput_scan.yaml")
    assert cfg.status in {
        "prepared_awaiting_gpu_authorization",
        "phase3b_dev_authorized",
        "phase3b_dev_reauthorized_truncated_prefix",
        "phase3b_dev_complete_awaiting_audit",
        "phase3b2_locked_authorized",
        "phase3b2_locked_complete_awaiting_audit",
    }


def test_phase3_yaml_frozen_scan_and_pins() -> None:
    raw = yaml.safe_load(
        (REPO_ROOT / "configs/experiments/phase3_preoutput_scan.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert raw["model_revision"] == PINNED_MODEL_REV
    assert raw["dataset_revision"] == PINNED_LASR_HF
    assert raw["coarse_scan"]["transformer_block_indices"] == COARSE_TRANSFORMER_BLOCKS
    assert raw["coarse_scan"]["prefix_lengths_k"] == PREFIX_LENGTHS_K
    assert raw["coarse_scan"]["primary_anchor_layer"] == PRIMARY_ANCHOR_LAYER
    assert float(raw["activation_probe"]["C"]) == 0.01
    assert raw["activation_probe"]["tune_C_per_cell"] is False
    assert raw["statistics"]["bootstrap_unit"] == "prompt_group"
    assert int(raw["statistics"]["bootstrap_resamples"]) >= 2000
    assert raw["onset_annotation"]["activation_dependent"] is False
    assert raw["onset_annotation"]["llm_api_allowed"] is False
    assert raw["phase3b_gpu"]["regime_c_authorized"] is False
    # Locked-test GPU may be authorized only after Phase 3B1 audit (Phase 3B2).
    assert isinstance(raw["phase3b_gpu"]["locked_test_gpu_authorized"], bool)
    if raw["status"] in {
        "phase3b2_locked_authorized",
        "phase3b2_locked_complete_awaiting_audit",
    }:
        assert raw["phase3b_gpu"]["locked_test_gpu_authorized"] is True
    else:
        assert raw["phase3b_gpu"]["locked_test_gpu_authorized"] is False
    assert raw["phase3b_gpu"]["gpu_type"] == "L40S"
    assert float(raw["phase3b_gpu"]["hard_budget_usd"]) == 40.0
    assert raw["primary_endpoints"]["regime_a"]["layer"] == 12
    assert raw["primary_endpoints"]["regime_a"]["k"] == 0
    assert raw["primary_endpoints"]["regime_b"]["k"] == 1
    assert raw["eligibility"]["k_gt_0"] == (
        "canonical_response_token_length_strictly_greater_than_k"
    )


def test_surface_baseline_contract() -> None:
    c = SurfaceBaselineContract()
    d = contract_as_dict(c)
    assert d["primary_surface_baseline"] == "prompt_plus_prefix"
    assert d["select_C_on"] == "phase3_validation_only"
    assert d["C_grid"] == [0.1, 1.0, 10.0]
    assert d["prompt_plus_prefix_alignment"] == (
        "canonical_full_sequence_response_token_offsets"
    )
    pipe = make_text_baseline_pipeline(C=1.0)
    assert "features" in pipe.named_steps
    assert build_prompt_plus_prefix_text("P", ["a", "b", "c"], 2) == "Pab"
    assert build_prompt_plus_prefix_text("P", ["a", "b"], 0) == "P"
    offsets = [(0, 1), (1, 2)]
    assert visible_prompt_plus_prefix_text("P", "xyz", 0, offsets) == "P"
    assert visible_prompt_plus_prefix_text("P", "xyz", 1, offsets) == "Px"
    assert visible_prompt_plus_prefix_text("P", "xyz", 2, offsets) == "Pxy"


def test_group_bootstrap_helpers_run() -> None:
    y = np.array([0, 0, 1, 1, 0, 1])
    s_a = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.7])
    s_b = np.array([0.4, 0.5, 0.6, 0.55, 0.45, 0.5])
    g = np.array(["g0", "g0", "g1", "g1", "g2", "g2"])
    out = bootstrap_auroc_ci_by_group(y, s_a, g, n_bootstrap=50, seed=0)
    assert out["bootstrap_unit"] == "prompt_group"
    delta = paired_delta_auroc_ci_by_group(y, s_a, s_b, g, n_bootstrap=50, seed=0)
    assert "delta_auroc" in delta
    assert delta["bootstrap_unit"] == "prompt_group"


def test_phase3_protocol_docs_exist() -> None:
    assert (REPO_ROOT / "docs/phase3_protocol.md").is_file()
    assert (REPO_ROOT / "docs/phase4_control_plan.md").is_file()
    protocol = (REPO_ROOT / "docs/phase3_protocol.md").read_text(encoding="utf-8")
    assert "prompt-boundary" in protocol.lower()
    assert "early-trajectory" in protocol.lower() or "early_trajectory" in protocol
    assert "pre-deceptive" in protocol.lower()
    assert "prompt_sha256" in protocol
    plan = (REPO_ROOT / "docs/phase4_control_plan.md").read_text(encoding="utf-8")
    assert "KNOWN-TRUTH" in plan
    assert "FALSE-BELIEF" in plan


def test_decision_log_contains_phase3a_decisions() -> None:
    text = (REPO_ROOT / "docs/decision_log.md").read_text(encoding="utf-8")
    for did in (
        "D019",
        "D020",
        "D021",
        "D022",
        "D023",
        "D024",
        "D025",
        "D026",
        "D027",
        "D028",
        "D029",
        "D030",
        "D031",
        "D032",
        "D033",
        "D034",
        "D035",
        "D036",
        "D037",
        "D038",
        "D039",
        "D040",
        "D041",
        "D042",
        "D043",
        "D044",
    ):
        assert did in text, did


def test_protocol_states_canonical_token_boundary() -> None:
    protocol = (REPO_ROOT / "docs/phase3_protocol.md").read_text(encoding="utf-8")
    assert "canonical tokenization" in protocol.lower() or "canonical" in protocol
    assert "input_formatted + model_outputs" in protocol
    assert "character boundary" in protocol.lower() or "prompt/response" in protocol
