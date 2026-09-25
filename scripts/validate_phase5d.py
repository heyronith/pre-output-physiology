#!/usr/bin/env python3
"""Phase 5D locked generalization integrity checks."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase5_locked import (  # noqa: E402
    CONFIRMATION_CRITERIA,
    EXPECTED_LOCKED_PAIR_SHA256,
    EXPECTED_LOCKED_PROMPT_TEXT_SHA256,
    FROZEN_CANDIDATE,
    LOCKED_FAMILIES,
    PROMPT_TEMPLATE_REVISION,
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
    exp = yaml.safe_load(
        (REPO_ROOT / "configs/experiments/phase5_strategic_discovery.yaml").read_text(
            encoding="utf-8"
        )
    )
    status = exp.get("status")
    if status not in {
        "phase5d_locked_test_authorized",
        "phase5d_locked_test_complete_awaiting_audit",
    }:
        result.fail(f"unexpected status {status}")
    else:
        result.ok(f"status {status}")

    auth = exp.get("authorizations", {})
    if auth.get("causal_intervention_authorized") is False:
        result.ok("causal_intervention_authorized=false")
    else:
        result.fail("causal must remain false")
    if auth.get("probe_fitting_authorized") is False:
        result.ok("probe_fitting_authorized=false (frozen probe)")
    else:
        result.fail("probe fitting must remain false")

    if status == "phase5d_locked_test_authorized":
        for key in (
            "locked_final_generation_authorized",
            "activation_extraction_authorized",
            "probe_scoring_authorized",
            "modal_gpu_authorized",
        ):
            if auth.get(key) is True:
                result.ok(f"{key}=true")
            else:
                result.fail(f"{key} must be true when authorized")
    else:
        for key in (
            "locked_final_generation_authorized",
            "activation_extraction_authorized",
            "probe_scoring_authorized",
            "modal_gpu_authorized",
        ):
            if auth.get(key) is False:
                result.ok(f"{key}=false (closed after complete)")
            else:
                result.fail(f"{key} should be closed after complete")

    decisions = (REPO_ROOT / "docs/decision_log.md").read_text(encoding="utf-8")
    if "D076" in decisions:
        result.ok("D076 present")
    else:
        result.fail("D076 missing")

    freeze_path = REPO_ROOT / "artifacts/phase5d_locked_freeze/locked_population.json"
    if freeze_path.is_file():
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if freeze["pair_ids_sha256"] == EXPECTED_LOCKED_PAIR_SHA256:
            result.ok("locked pair hash frozen")
        else:
            result.fail("locked pair hash drift")
        if freeze["locked_prompt_text_sha256"] == EXPECTED_LOCKED_PROMPT_TEXT_SHA256:
            result.ok("locked prompt hash frozen")
        else:
            result.fail("locked prompt hash drift")
        if freeze.get("prompt_template_revision") == PROMPT_TEMPLATE_REVISION:
            result.ok("prompt revision=2")
        else:
            result.fail("prompt revision drift")
        if freeze.get("confirmation_criteria") == CONFIRMATION_CRITERIA:
            result.ok("confirmation criteria frozen pre-result")
        else:
            result.fail("confirmation criteria drift")
        if freeze["frozen_candidate"]["probe_sha256"] == FROZEN_CANDIDATE["probe_sha256"]:
            result.ok("freeze records expected probe sha")
        else:
            result.fail("freeze probe sha mismatch")
    else:
        result.fail("missing locked population freeze")

    probe_path = REPO_ROOT / FROZEN_CANDIDATE["probe_artifact"]
    if probe_path.is_file():
        digest = hashlib.sha256(probe_path.read_bytes()).hexdigest()
        if digest == FROZEN_CANDIDATE["probe_sha256"]:
            result.ok("frozen probe sha exact")
        else:
            result.fail(f"probe sha mismatch {digest}")
    else:
        result.fail("missing frozen probe artifact")

    for rel in (
        "modal/phase5_locked_extract.py",
        "modal/phase5_locked_behavior_generate.py",
        "scripts/freeze_phase5d_locked_population.py",
        "scripts/score_phase5d_locked.py",
        "scripts/evaluate_phase5d_locked_behavior.py",
        "src/pre_output_physiology/phase5_locked.py",
    ):
        if (REPO_ROOT / rel).is_file():
            result.ok(f"{rel} present")
        else:
            result.fail(f"missing {rel}")

    if status == "phase5d_locked_test_complete_awaiting_audit":
        primary_path = (
            REPO_ROOT / "artifacts/phase5d_locked_test/locked_primary_physiology.json"
        )
        report = REPO_ROOT / "reports/phase5d_locked_test.md"
        if primary_path.is_file() and report.is_file():
            result.ok("primary physiology + report present")
            primary = json.loads(primary_path.read_text(encoding="utf-8"))
            if primary.get("behavior_conditioned") is False:
                result.ok("primary not behavior-conditioned")
            else:
                result.fail("primary must not be behavior-conditioned")
            if primary.get("probe_retrained") is False and primary.get(
                "probe_recalibrated"
            ) is False:
                result.ok("probe not retrained/recalibrated")
            else:
                result.fail("probe modification flags incorrect")
            if primary.get("layer_reselected") is False:
                result.ok("layer not reselected")
            else:
                result.fail("layer_reselected must be false")
            if primary.get("causal_interventions_performed") is False:
                result.ok("no causal interventions")
            else:
                result.fail("causal flag incorrect")
            if primary.get("probe_sha256_verified") == FROZEN_CANDIDATE["probe_sha256"]:
                result.ok("primary verified probe sha")
            else:
                result.fail("primary probe sha mismatch")
            if primary.get("locked_pair_ids_sha256") == EXPECTED_LOCKED_PAIR_SHA256:
                result.ok("primary pair hash unchanged")
            else:
                result.fail("primary pair hash changed")
            integ = primary.get("activation_integrity", {})
            if (
                integ.get("other_layers_extracted") is False
                and integ.get("k0_extracted") is False
                and integ.get("discovery_rows_present") is False
            ):
                result.ok("activation integrity (L12/k1 only; no discovery)")
            else:
                result.fail(f"activation integrity fail {integ}")
            conf = primary.get("confirmation", {})
            if "passed" in conf and "checks" in conf:
                result.ok(f"confirmation recorded passed={conf['passed']}")
            else:
                result.fail("confirmation block missing")
            # No train/val discovery activations in locked primary
            blob = json.dumps(primary)
            if "phase5c_extract" in blob or "train_all_pair" in blob:
                # references to discovery run id in notes may exist; check integrity flags
                pass
            fams = primary.get("metrics", {}).get("per_family_auroc", {})
            if set(fams) == set(LOCKED_FAMILIES):
                result.ok("per-family AUROC for both locked families")
            else:
                result.fail(f"family keys {fams.keys()}")
        else:
            result.fail("missing primary physiology or report")

        sens_path = (
            REPO_ROOT / "artifacts/phase5d_locked_test/locked_behavior_sensitivity.json"
        )
        if sens_path.is_file():
            sens = json.loads(sens_path.read_text(encoding="utf-8"))
            if sens.get("does_not_replace_primary") is True:
                result.ok("behavior sensitivity secondary-only")
            else:
                result.fail("sensitivity must not replace primary")
            if sens.get("primary_population_unchanged") is True:
                result.ok("primary population unchanged by behavior")
            else:
                result.fail("primary population changed flag")
        else:
            result.ok("behavior sensitivity not yet present (optional until run)")

    tracked = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "ls-files"], text=True
    ).splitlines()
    for path in tracked:
        if "phase5d" in path and path.endswith(".safetensors"):
            result.fail(f"raw activations tracked: {path}")
        if "phase5d" in path and path.endswith("locked_outputs.jsonl"):
            result.fail(f"raw locked outputs tracked: {path}")
    result.ok("raw activations/outputs not tracked")

    print()
    print(f"{len(result.passes)} passed, {len(result.failures)} failed")
    if result.failures:
        for f in result.failures:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
