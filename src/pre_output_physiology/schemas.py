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
        "running",
        "complete",
        "blocked",
    ]
    description: str
    model_config_path: str
    positive_control_dataset: str | None = None
    measurement_regimes: list[str] = Field(default_factory=list)
    notes: str | None = None
