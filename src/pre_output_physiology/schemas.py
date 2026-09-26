"""Pydantic schemas for Phase 1 configuration contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

PRIMARY_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
UNPINNED_REVISION_SENTINEL = "TO_BE_PINNED_BEFORE_PHASE2"


class ModelConfig(BaseModel):
    """Frozen primary-model inference contract."""

    model_id: str
    revision: str
    dtype: Literal["bfloat16", "float16", "float32"]
    quantization: Any = None
    trust_remote_code: bool = False
    device_policy: str
    seed: int
    notes: str | None = None

    @field_validator("model_id")
    @classmethod
    def model_id_must_match_frozen_checkpoint(cls, value: str) -> str:
        if value != PRIMARY_MODEL_ID:
            raise ValueError(
                f"Primary model_id must be {PRIMARY_MODEL_ID!r}; got {value!r}. "
                "Do not substitute another checkpoint without explicit approval."
            )
        return value

    @field_validator("quantization")
    @classmethod
    def quantization_must_be_null(cls, value: Any) -> None:
        if value is not None:
            raise ValueError(
                "Primary scientific configuration must set quantization: null. "
                "Quantized models are debugging-only and must be labelled separately."
            )
        return None

    @field_validator("revision")
    @classmethod
    def revision_must_not_silently_be_main(cls, value: str) -> str:
        if value.strip().lower() == "main":
            raise ValueError(
                "Do not silently pin revision to 'main'. "
                f"Use an exact commit hash or {UNPINNED_REVISION_SENTINEL}."
            )
        return value


class ExperimentConfig(BaseModel):
    """Lightweight experiment YAML contract for future phases (not executed in Phase 1)."""

    experiment_id: str
    phase: str
    status: Literal[
        "scaffolded",
        "authorized",
        "prepared_awaiting_gpu_authorization",
        "phase3b_dev_authorized",
        "phase3b_dev_reauthorized_truncated_prefix",
        "phase3b_dev_complete_awaiting_audit",
        "phase3b2_locked_authorized",
        "phase3b2_locked_complete_awaiting_audit",
        "phase4a_design_frozen_awaiting_pilot",
        "phase4b_pilot_complete_awaiting_audit",
        "phase4b_repilot_pass_awaiting_audit",
        "phase4b_repilot_fail_hold",
        "phase4c_controlled_prefix_pilot_authorized",
        "phase4c_controlled_prefix_pilot_pass_awaiting_audit",
        "phase4c_controlled_prefix_pilot_fail_hold",
        "phase4d_final_behavior_generation_authorized",
        "phase4d_final_behavior_complete_awaiting_audit",
        "phase4e_specificity_extraction_authorized",
        "phase4e_specificity_complete_awaiting_audit",
        "phase4f_natural_token_diagnostic_authorized",
        "phase4f_natural_token_diagnostic_complete_awaiting_audit",
        "phase5a_design_frozen_pilot_authorized",
        "phase5a_behavior_pilot_pass_awaiting_audit",
        "phase5a_behavior_pilot_hold",
        "phase5b_discovery_behavior_authorized",
        "phase5b_discovery_behavior_complete_awaiting_audit",
        "phase5c_discovery_physiology_authorized",
        "phase5c_candidate_selection_complete_awaiting_audit",
        "phase5c_candidate_gate_fail_hold",
        "phase5d_locked_test_authorized",
        "phase5d_locked_test_complete_awaiting_audit",
        "phase6a_specificity_design_frozen_pilot_authorized",
        "phase6a_specificity_pilot_pass_awaiting_audit",
        "phase6a_specificity_pilot_hold",
        "phase6b_factorial_frozen_probe_authorized",
        "phase6b_factorial_primary_complete_awaiting_audit",
        "phase7a_policy_choice_design_frozen_pilot_authorized",
        "phase7a_policy_choice_pilot_pass_awaiting_audit",
        "phase7a_policy_choice_pilot_hold",
        "phase8a_policy_frontier_calibration_authorized",
        "phase8a_policy_frontier_frozen_awaiting_audit",
        "phase8a_policy_frontier_hold",
        "phase9a_policy_flip_diagnostic_authorized",
        "phase9a_policy_flip_diagnostic_complete_awaiting_audit",
        "phase10a_risk_frontier_calibration_authorized",
        "phase10a_risk_frontier_frozen_awaiting_audit",
        "phase10a_risk_frontier_sanity_hold",
        "phase10a_risk_frontier_hold",
        "running",
        "complete",
        "blocked",
    ]
    description: str
    model_config_path: str
    positive_control_dataset: str | None = None
    measurement_regimes: list[str] = Field(default_factory=list)
    notes: str | None = None
