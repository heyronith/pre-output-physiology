"""Phase 2 configuration / provenance contract tests."""

from __future__ import annotations

from pathlib import Path

import yaml

from pre_output_physiology import PRIMARY_MODEL_ID
from pre_output_physiology.config import load_model_config
from pre_output_physiology.provenance import assert_not_main_revision, build_run_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
PINNED_MODEL_REV = "63a8b081895390a26e140280378bc85ec8bce07a"
PINNED_LASR_HF = "bf93584916fbd23121eca6f2017017df0ef3184f"
PINNED_APOLLO = "f8ec4010e74927394709dffa22b97bdf8cd5a62f"
PINNED_LASR_CODE = "f4c6ad69b10a5436a2e819c69009431802a0f5f7"


def test_model_config_phase2_pin() -> None:
    cfg = load_model_config()
    assert cfg.model_id == PRIMARY_MODEL_ID
    assert cfg.revision == PINNED_MODEL_REV
    assert cfg.quantization is None
    assert cfg.dtype == "bfloat16"
    assert_not_main_revision("model", cfg.revision)


def test_dataset_config_pins() -> None:
    raw = yaml.safe_load(
        (REPO_ROOT / "configs/datasets/roleplay_deception.yaml").read_text(encoding="utf-8")
    )
    assert raw["lasr_hf_revision"] == PINNED_LASR_HF
    assert raw["apollo_revision"] == PINNED_APOLLO
    assert raw["lasr_code_revision"] == PINNED_LASR_CODE
    assert raw["train_jsonl"] == "roleplaying/mistral_7b_incentivised_train.jsonl"
    assert raw["test_jsonl"] == "roleplaying/mistral_7b_incentivised_test.jsonl"
    for key in ("lasr_hf_revision", "apollo_revision", "lasr_code_revision"):
        assert_not_main_revision(key, raw[key])


def test_experiment_config_phase2_probe_and_layer() -> None:
    raw = yaml.safe_load(
        (REPO_ROOT / "configs/experiments/phase2_positive_control.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert raw["status"] == "authorized"
    assert raw["instrumentation"]["transformer_block_index"] == 12
    assert raw["instrumentation"]["hook_module_path"] == "model.model.layers[12]"
    assert raw["instrumentation"]["aggregation"] == "attention_mask_mean_non_padding"
    assert float(raw["probe"]["C"]) == 0.01
    assert int(raw["probe"]["random_state"]) == 42
    assert raw["gpu"]["gpu_type"] == "L40S"
    assert int(raw["gpu"]["num_gpus"]) == 1


def test_run_manifest_builder_requires_core_fields() -> None:
    base = {
        "run_id": "x",
        "timestamp": "t",
        "git_commit": "c",
        "model_id": PRIMARY_MODEL_ID,
        "model_revision": PINNED_MODEL_REV,
        "dataset_revision": PINNED_LASR_HF,
        "config_file": "configs/experiments/phase2_positive_control.yaml",
        "random_seed": 42,
        "hardware": "modal",
        "gpu_type": "L40S",
        "software_versions": {},
        "precision": "bfloat16",
        "generation_parameters": {},
        "activation_locations_collected": {},
        "output_artifact_hashes": {"a": "b"},
    }
    man = build_run_manifest(**base)
    assert man["model_revision"] == PINNED_MODEL_REV
