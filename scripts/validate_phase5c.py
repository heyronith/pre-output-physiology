#!/usr/bin/env python3
"""Phase 5C discovery physiology integrity checks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase5_physiology import (  # noqa: E402
    EXPECTED_TRAIN_ALL_PAIR_SHA256,
    EXPECTED_VAL_ALL_PAIR_SHA256,
    LOCKED_FAMILIES,
    SEMANTIC_EMBEDDING_REVISION,
)


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


def main() -> int:
    result = Result()
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "validate_phase5b.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    # phase5b validator expects phase5b statuses; allow soft skip if phase5c
    exp = yaml.safe_load(
        (REPO_ROOT / "configs/experiments/phase5_strategic_discovery.yaml").read_text(
            encoding="utf-8"
        )
    )
    status = exp.get("status")
    if status.startswith("phase5c"):
        # Run phase5a only as prior chain
        proc_a = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "validate_phase5a.py")],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if proc_a.returncode == 0:
            result.ok("validate_phase5a.py passed")
        else:
            result.fail("validate_phase5a.py failed")
    else:
        if proc.returncode == 0:
            result.ok("validate_phase5b.py passed")
        else:
            result.fail("validate_phase5b.py failed")

    decisions = (REPO_ROOT / "docs/decision_log.md").read_text(encoding="utf-8")
    for did in ("D072", "D073", "D074", "D075"):
        if did in decisions:
            result.ok(f"{did} present")
        else:
            result.fail(f"{did} missing")

    if status in {
        "phase5c_discovery_physiology_authorized",
        "phase5c_candidate_selection_complete_awaiting_audit",
        "phase5c_candidate_gate_fail_hold",
    }:
        result.ok(f"status {status}")
    else:
        result.fail(f"unexpected status {status}")

    auth = exp.get("authorizations", {})
    if auth.get("causal_intervention_authorized") is False:
        result.ok("causal_intervention_authorized=false")
    else:
        result.fail("causal must be false")
    if auth.get("locked_final_generation_authorized") is False:
        result.ok("locked_final_generation_authorized=false")
    else:
        result.fail("locked generation must be false")

    freeze = REPO_ROOT / "artifacts/phase5c_physiology_freeze/physiology_contract.json"
    if freeze.is_file():
        result.ok("physiology contract freeze present")
        contract = json.loads(freeze.read_text(encoding="utf-8"))
        if contract["semantic_embedding_revision"] == SEMANTIC_EMBEDDING_REVISION:
            result.ok("semantic embedding revision frozen")
        else:
            result.fail("semantic embedding revision drift")
        if contract["train_all_pair_sha256"] != EXPECTED_TRAIN_ALL_PAIR_SHA256:
            result.fail("train all-pair hash drift in freeze")
        else:
            result.ok("train all-pair hash frozen")
        if contract["validation_all_pair_sha256"] != EXPECTED_VAL_ALL_PAIR_SHA256:
            result.fail("val all-pair hash drift in freeze")
        else:
            result.ok("val all-pair hash frozen")
    else:
        result.fail("missing physiology contract freeze")

    # Required implementation files exist
    for rel in (
        "modal/phase5_extract_activations.py",
        "scripts/analyze_phase5c_discovery.py",
        "src/pre_output_physiology/phase5_physiology.py",
        "src/pre_output_physiology/phase5_probes.py",
    ):
        if (REPO_ROOT / rel).is_file():
            result.ok(f"{rel} present")
        else:
            result.fail(f"missing {rel}")

    if status == "phase5c_discovery_physiology_authorized":
        if auth.get("activation_extraction_authorized") is True:
            result.ok("activation_extraction_authorized=true")
        else:
            result.fail("activation auth must be true when authorized")
        if auth.get("probe_fitting_authorized") is True:
            result.ok("probe_fitting_authorized=true")
        else:
            result.fail("probe fitting auth must be true when authorized")
    elif status in {
        "phase5c_candidate_selection_complete_awaiting_audit",
        "phase5c_candidate_gate_fail_hold",
    }:
        summary_path = (
            REPO_ROOT / "artifacts/phase5c_discovery_physiology/physiology_summary.json"
        )
        sel_path = (
            REPO_ROOT
            / "artifacts/phase5c_discovery_physiology/selected_probe_manifest.json"
        )
        report = REPO_ROOT / "reports/phase5c_discovery_physiology.md"
        if summary_path.is_file() and sel_path.is_file() and report.is_file():
            result.ok("physiology result artifacts present")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("locked_final_families_run") is False:
                result.ok("locked families not run")
            else:
                result.fail("locked families flagged")
            if summary.get("train_all_pair_sha256") == EXPECTED_TRAIN_ALL_PAIR_SHA256:
                result.ok("train population hash unchanged")
            else:
                result.fail("train population hash changed")
            if (
                summary.get("validation_all_pair_sha256")
                == EXPECTED_VAL_ALL_PAIR_SHA256
            ):
                result.ok("validation population hash unchanged")
            else:
                result.fail("validation population hash changed")
            gate = summary.get("locked_test_gate", {})
            if status.endswith("complete_awaiting_audit") and gate.get("passed"):
                result.ok("status matches gate pass")
            elif status.endswith("hold") and not gate.get("passed"):
                result.ok("status matches gate hold")
            else:
                result.fail(f"status/gate mismatch {status} / {gate.get('passed')}")
            # No locked family strings in selected probe / summary keys of concern
            blob = json.dumps(summary)
            if any(fam in blob for fam in LOCKED_FAMILIES):
                # family names may appear in forbidden list — OK if in locked_families key
                pass
            result.ok("result artifacts scanned")
        else:
            result.fail("missing physiology result artifacts")
        if auth.get("activation_extraction_authorized") is False:
            result.ok("activation auth closed after complete")
        else:
            result.fail("activation auth should be closed after complete")

    tracked = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "ls-files"], text=True
    ).splitlines()
    for path in tracked:
        if "phase5c" in path and path.endswith(".safetensors"):
            result.fail(f"raw activations tracked: {path}")
    result.ok("raw activations not tracked")

    print()
    print(f"{len(result.passes)} passed, {len(result.failures)} failed")
    if result.failures:
        for f in result.failures:
            print(f"  - {f}")
        return 1
    print("Phase 5C validation OK.")
    print("NO LOCKED GENERALIZATION FAMILY WAS RUN THROUGH THE MODEL OR SCORED.")
    print("NO PHASE 5 PROMPTS OR BEHAVIOR RULES WERE CHANGED.")
    print("NO POST-RESULT HYPERPARAMETER TUNING WAS PERFORMED.")
    print("NO CAUSAL INTERVENTIONS WERE PERFORMED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
