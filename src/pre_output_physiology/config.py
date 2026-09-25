"""Typed loading helpers for frozen YAML configuration contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from pre_output_physiology.schemas import ExperimentConfig, ModelConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
MODELS_DIR = CONFIGS_DIR / "models"
EXPERIMENTS_DIR = CONFIGS_DIR / "experiments"

PRIMARY_MODEL_CONFIG = MODELS_DIR / "mistral_7b_instruct_v02.yaml"


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Parse a YAML file into a plain dict. Does not download models or run inference."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at root of {path}, got {type(data).__name__}")
    return data


def load_model_config(path: Path | str | None = None) -> ModelConfig:
    """Load and validate a model configuration YAML."""
    config_path = Path(path) if path is not None else PRIMARY_MODEL_CONFIG
    try:
        return ModelConfig.model_validate(load_yaml(config_path))
    except ValidationError as exc:
        raise ValueError(f"Invalid model config at {config_path}") from exc


def load_experiment_config(path: Path | str) -> ExperimentConfig:
    """Load and validate an experiment configuration YAML."""
    config_path = Path(path)
    try:
        return ExperimentConfig.model_validate(load_yaml(config_path))
    except ValidationError as exc:
        raise ValueError(f"Invalid experiment config at {config_path}") from exc
