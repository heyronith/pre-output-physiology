#!/usr/bin/env python3
"""Phase 2 integrity checks (inexpensive; no model download)."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.activations import (  # noqa: E402
    HOOK_MODULE_PATH_PHASE2,
    TRANSFORMER_BLOCK_INDEX_PHASE2,
    assert_layer_contract,
)

PRIMARY_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
PINNED_MODEL_REV = "63a8b081895390a26e140280378bc85ec8bce07a"
PINNED_LASR_CODE = "f4c6ad69b10a5436a2e819c69009431802a0f5f7"
PINNED_LASR_HF = "bf93584916fbd23121eca6f2017017df0ef3184f"
PINNED_APOLLO = "f8ec4010e74927394709dffa22b97bdf8cd5a62f"


class Result:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.passes: list[str] = []

    def ok(self, msg: str) -> None:
        self.passes.append(msg)
        print(f"PASS  {msg}")

    def fail(self, msg: str) -> None:
        self.failures.append(msg)
        print(f"FAIL  {msg}")


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(path)
    return data


def tracked_files() -> list[str]:
    out = subprocess.check_output(["git", "-C", str(REPO_ROOT), "ls-files"], text=True)
    return [line for line in out.splitlines() if line]


def main() -> int:
    # Phase 1 must still pass
    phase1 = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/validate_phase1.py")],
        cwd=REPO_ROOT,
    )
    result = Result()
    if phase1.returncode == 0:
        result.ok("phase1 validation passed")
    else:
        result.fail("phase1 validation failed")

    model = load_yaml(REPO_ROOT / "configs/models/mistral_7b_instruct_v02.yaml")
    ds = load_yaml(REPO_ROOT / "configs/datasets/roleplay_deception.yaml")
    exp = load_yaml(REPO_ROOT / "configs/experiments/phase2_positive_control.yaml")

    if model.get("model_id") == PRIMARY_MODEL_ID:
        result.ok("model_id pinned")
    else:
        result.fail(f"bad model_id {model.get('model_id')}")

    if model.get("revision") == PINNED_MODEL_REV:
        result.ok("exact Mistral revision pinned")
    else:
        result.fail(f"model revision {model.get('revision')} != {PINNED_MODEL_REV}")

    if str(model.get("revision", "")).lower() == "main":
        result.fail("model revision is main")
    if model.get("quantization") is None:
        result.ok("quantization null")
    else:
        result.fail("quantization enabled")

    if ds.get("lasr_hf_revision") == PINNED_LASR_HF and ds.get("lasr_hf_revision") != "main":
        result.ok("LASR HF dataset revision pinned")
    else:
        result.fail("LASR HF revision mismatch")

    if ds.get("apollo_revision") == PINNED_APOLLO:
        result.ok("Apollo revision pinned")
    else:
        result.fail("Apollo revision mismatch")

    if ds.get("lasr_code_revision") == PINNED_LASR_CODE:
        result.ok("LASR code revision pinned")
    else:
        result.fail("LASR code revision mismatch")

    for key in ("lasr_hf_revision", "apollo_revision", "lasr_code_revision"):
        if str(ds.get(key, "")).lower() == "main":
            result.fail(f"{key} is main")

    try:
        assert_layer_contract(
            int(exp["instrumentation"]["transformer_block_index"]),
            exp["instrumentation"]["hook_module_path"],
        )
        result.ok(
            f"transformer block index {TRANSFORMER_BLOCK_INDEX_PHASE2} / {HOOK_MODULE_PATH_PHASE2}"
        )
    except ValueError as exc:
        result.fail(str(exc))

    probe = exp.get("probe", {})
    if float(probe.get("C", -1)) == 0.01:
        result.ok("probe C=0.01")
    else:
        result.fail(f"probe C={probe.get('C')}")
    if int(probe.get("random_state", -1)) == 42:
        result.ok("probe seed 42")
    else:
        result.fail("probe seed mismatch")
    if exp.get("instrumentation", {}).get("aggregation") == "attention_mask_mean_non_padding":
        result.ok("primary aggregation is token mean with padding excluded")
    else:
        result.fail("aggregation method mismatch")

    # Prefer canonical full run if present; else any run with metrics.
    summary_dir = REPO_ROOT / "artifacts" / "phase2_summaries"
    run_dirs = sorted((REPO_ROOT / "artifacts" / "runs").glob("phase2_*")) if (
        REPO_ROOT / "artifacts" / "runs"
    ).exists() else []
    metrics_ok = False
    manifest_ok = False
    hash_ok = False
    for run_dir in reversed(run_dirs):
        metrics = run_dir / "metrics.json"
        manifest = run_dir / "run_manifest.json"
        if metrics.is_file() and manifest.is_file():
            metrics_ok = True
            manifest_ok = True
            man = json.loads(manifest.read_text(encoding="utf-8"))
            hashes = man.get("output_artifact_hashes") or {}
            if hashes:
                hash_ok = True
            break
    if summary_dir.exists() and any(summary_dir.glob("*_metrics_summary.json")):
        result.ok("metrics summary artifact present")
    elif metrics_ok:
        result.ok("metrics file exists in a run dir")
    else:
        result.fail("metrics file missing")

    if manifest_ok:
        result.ok("run manifest exists")
    else:
        result.fail("run manifest missing")
    if hash_ok:
        result.ok("scientific artifact hashes recorded")
    else:
        result.fail("artifact hashes missing")

    tracked = tracked_files()
    forbidden = []
    for rel in tracked:
        if rel.startswith("artifacts/runs/"):
            forbidden.append(rel)
        if rel.endswith((".safetensors", ".pkl", ".bin", ".pt", ".pth")) and not rel.endswith(
            ".example"
        ):
            forbidden.append(rel)
        if re.search(r"(^|/)\.env$", rel) or rel.endswith((".pem", ".key")):
            forbidden.append(rel)
    if forbidden:
        result.fail(f"tracked forbidden artifacts/secrets: {forbidden}")
    else:
        result.ok("no primary artifacts or secrets tracked by git")

    print()
    print(f"Passed: {len(result.passes)}")
    print(f"Failed: {len(result.failures)}")
    if result.failures:
        print("Phase 2 validation FAILED")
        return 1
    print("Phase 2 validation PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
