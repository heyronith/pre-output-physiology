"""Configuration contract tests — no GPU, downloads, or paid APIs."""

from __future__ import annotations

from pathlib import Path

import yaml

from pre_output_physiology import PRIMARY_MODEL_ID
from pre_output_physiology.config import (
    EXPERIMENTS_DIR,
    PRIMARY_MODEL_CONFIG,
    load_experiment_config,
    load_model_config,
)
from pre_output_physiology.schemas import ModelConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
PINNED_MODEL_REV = "63a8b081895390a26e140280378bc85ec8bce07a"


def test_primary_model_yaml_parses_and_matches_frozen_id() -> None:
    raw = yaml.safe_load(PRIMARY_MODEL_CONFIG.read_text(encoding="utf-8"))
    assert raw["model_id"] == PRIMARY_MODEL_ID
    assert raw["quantization"] is None
    assert raw["revision"] == PINNED_MODEL_REV
    assert raw["revision"].lower() != "main"


def test_load_model_config_via_package() -> None:
    cfg = load_model_config()
    assert isinstance(cfg, ModelConfig)
    assert cfg.model_id == PRIMARY_MODEL_ID
    assert cfg.quantization is None
    assert cfg.dtype in {"bfloat16", "float16", "float32"}
    assert cfg.seed == 0


def test_experiment_configs_status_and_model_path() -> None:
    phase2 = load_experiment_config(EXPERIMENTS_DIR / "phase2_positive_control.yaml")
    assert phase2.status == "authorized"
    assert phase2.model_config_path.endswith("mistral_7b_instruct_v02.yaml")
    phase3 = load_experiment_config(EXPERIMENTS_DIR / "phase3_preoutput_scan.yaml")
    assert phase3.status == "scaffolded"
    assert phase3.model_config_path.endswith("mistral_7b_instruct_v02.yaml")


def test_model_schema_rejects_quantization_and_wrong_model(tmp_path: Path) -> None:
    good = {
        "model_id": PRIMARY_MODEL_ID,
        "revision": PINNED_MODEL_REV,
        "dtype": "bfloat16",
        "quantization": None,
        "trust_remote_code": False,
        "device_policy": "modal_full_precision_gpu_primary",
        "seed": 0,
    }
    assert ModelConfig.model_validate(good).model_id == PRIMARY_MODEL_ID

    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ModelConfig.model_validate({**good, "model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1"})
    with pytest.raises(ValidationError):
        ModelConfig.model_validate({**good, "quantization": "bitsandbytes-4bit"})
    with pytest.raises(ValidationError):
        ModelConfig.model_validate({**good, "revision": "main"})
