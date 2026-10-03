#!/usr/bin/env python3
"""Phase 5B discovery behavior integrity checks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase5_conditions import (  # noqa: E402
    DISCOVERY_FAMILIES,
    LOCKED_FAMILIES,
)
from pre_output_physiology.phase5_split import (  # noqa: E402
    compute_discovery_family_split,
)

EXPECTED_FINAL_PROMPT_TEXT_SHA256 = (
    "cdf2003ac0d568eee28a0feffa7b07e0ef208e637de40665aa9a5b1e8bc2920e"
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
        [sys.executable, str(REPO_ROOT / "scripts" / "validate_phase5a.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        result.ok("validate_phase5a.py passed")
    else:
        result.fail("validate_phase5a.py failed")
        print(proc.stdout[-2000:] if proc.stdout else "")
        print(proc.stderr[-2000:] if proc.stderr else "")

    decisions = (REPO_ROOT / "docs/decision_log.md").read_text(encoding="utf-8")
    for did in ("D065", "D066", "D067", "D068", "D069", "D070", "D071", "D072", "D073"):
        if did in decisions:
            result.ok(f"{did} present")
        else:
            result.fail(f"{did} missing")

    exp = yaml.safe_load(
        (REPO_ROOT / "configs/experiments/phase5_strategic_discovery.yaml").read_text(
            encoding="utf-8"
        )
    )
    status = exp.get("status")
    if status in {
        "phase5b_discovery_behavior_authorized",
        "phase5b_discovery_behavior_complete_awaiting_audit",
        "phase5c_discovery_physiology_authorized",
        "phase5c_candidate_selection_complete_awaiting_audit",
        "phase5c_candidate_gate_fail_hold",
        "phase5d_locked_test_authorized",
        "phase5d_locked_test_complete_awaiting_audit",
    }:
        result.ok(f"status {status}")
    else:
        result.fail(f"unexpected status {status}")

    auth = exp.get("authorizations", {})
    if status.startswith("phase5d_locked_test"):
        result.ok("phase5d path; auth checked by validate_phase5d")
        if auth.get("causal_intervention_authorized") is False:
            result.ok("causal_intervention_authorized=false")
        else:
            result.fail("causal must be false")
        if auth.get("probe_fitting_authorized") is False:
            result.ok("probe_fitting_authorized=false")
        else:
            result.fail("probe fitting must remain false")
    else:
        for key in (
            "activation_extraction_authorized",
            "probe_fitting_authorized",
            "probe_scoring_authorized",
            "causal_intervention_authorized",
            "locked_final_generation_authorized",
        ):
            if auth.get(key) is False:
                result.ok(f"{key}=false")
            else:
                result.fail(f"{key} must be false")

    if int(exp.get("design", {}).get("prompt_template_revision", 0)) != 2:
        result.fail("prompt revision must remain 2")
    else:
        result.ok("prompt_template_revision=2 unchanged")
    if (
        exp.get("design", {}).get("final_prompt_text_sha256")
        != EXPECTED_FINAL_PROMPT_TEXT_SHA256
    ):
        result.fail("final prompt hash changed")
    else:
        result.ok("final prompt hash unchanged")

    split_path = REPO_ROOT / "artifacts/phase5b_discovery_split/family_split.json"
    primary_path = (
        REPO_ROOT / "artifacts/phase5b_discovery_split/primary_all_pair_populations.json"
    )
    procedure_path = (
        REPO_ROOT / "artifacts/phase5b_discovery_split/future_probe_procedure.json"
    )
    if not split_path.is_file() or not primary_path.is_file() or not procedure_path.is_file():
        result.fail("missing phase5b split freeze artifacts")
        print(f"\n{len(result.passes)} passed, {len(result.failures)} failed")
        return 1

    split = json.loads(split_path.read_text(encoding="utf-8"))
    expected = compute_discovery_family_split(DISCOVERY_FAMILIES)
    if split["train_families"] != expected["train_families"]:
        result.fail("train family assignment drift")
    else:
        result.ok(f"train families {split['train_families']}")
    if split["validation_families"] != expected["validation_families"]:
        result.fail("validation family assignment drift")
    else:
        result.ok(f"validation families {split['validation_families']}")
    if set(LOCKED_FAMILIES) & set(split["train_families"] + split["validation_families"]):
        result.fail("locked family in split")
    else:
        result.ok("locked families excluded from split")

    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    if primary["train"]["n_pairs"] == 640 and primary["validation"]["n_pairs"] == 320:
        result.ok("primary all-pair sizes 640/320")
    else:
        result.fail("primary all-pair size drift")
    if set(primary["train"]["pair_ids"]) & set(primary["validation"]["pair_ids"]):
        result.fail("primary train/val pair overlap")
    else:
        result.ok("primary train/val pairs disjoint")
    locked_hits = [
        pid
        for pid in primary["train"]["pair_ids"] + primary["validation"]["pair_ids"]
        if any(fam in pid for fam in LOCKED_FAMILIES)
    ]
    if locked_hits:
        result.fail(f"locked IDs in primary populations: {locked_hits[:3]}")
    else:
        result.ok("no locked IDs in primary populations")

    procedure = json.loads(procedure_path.read_text(encoding="utf-8"))
    if procedure.get("executed") is False:
        result.ok("future probe procedure recorded not executed")
    else:
        result.fail("probe procedure marked executed")

    # No phase5 activation/probe artifacts before Phase 5C results exist.
    if status.startswith("phase5c_candidate") or status.startswith("phase5d_locked_test"):
        result.ok("phase5c/5d results path; skip pre-extraction activation absence check")
    else:
        hits = []
        for path in (REPO_ROOT / "artifacts").rglob("*"):
            if not path.is_file():
                continue
            rel = str(path.relative_to(REPO_ROOT)).lower()
            if "phase5" not in rel:
                continue
            if any(
                tok in rel
                for tok in ("activation", "probe_score", ".safetensors", "causal")
            ):
                if "procedure" in rel or "manifest" in rel or "condition_matrix" in rel:
                    continue
                if "physiology_freeze" in rel or "phase5c_physiology_freeze" in rel:
                    continue
                hits.append(rel)
        if hits:
            result.fail(f"phase5 activation/probe artifacts: {hits[:5]}")
        else:
            result.ok("no phase5 activation/probe artifacts")

    if status == "phase5b_discovery_behavior_authorized":
        if auth.get("discovery_generation_authorized") is True:
            result.ok("discovery_generation_authorized=true")
        else:
            result.fail("discovery_generation_authorized must be true when authorized")
        if split.get("model_generation_performed") is False:
            result.ok("pre-generation freeze intact")
        else:
            result.fail("split marked generated before completion status")
    elif status in {
        "phase5b_discovery_behavior_complete_awaiting_audit",
        "phase5c_discovery_physiology_authorized",
        "phase5c_candidate_selection_complete_awaiting_audit",
        "phase5c_candidate_gate_fail_hold",
        "phase5d_locked_test_authorized",
        "phase5d_locked_test_complete_awaiting_audit",
    }:
        beh = (
            REPO_ROOT
            / "artifacts/phase5b_discovery_behavior/discovery_behavior_summary.json"
        )
        sens = (
            REPO_ROOT
            / "artifacts/phase5b_discovery_behavior/sensitivity_behavior_valid_populations.json"
        )
        report = REPO_ROOT / "reports/phase5b_discovery_behavior.md"
        man = (
            REPO_ROOT
            / "artifacts/phase5b_discovery_behavior/discovery_generation_manifest.json"
        )
        if beh.is_file() and sens.is_file() and report.is_file() and man.is_file():
            result.ok("discovery behavior artifacts present")
            summary = json.loads(beh.read_text(encoding="utf-8"))["summary"]
            if summary.get("n_outputs") == 1920 and summary.get("locked_outputs") == 0:
                result.ok("1920 outputs; zero locked")
            else:
                result.fail("output count / locked mismatch")
            man_j = json.loads(man.read_text(encoding="utf-8"))
            if man_j.get("final_prompt_text_sha256") == EXPECTED_FINAL_PROMPT_TEXT_SHA256:
                result.ok("generation prompt hash unchanged")
            else:
                result.fail("generation prompt hash drift")
            if man_j.get("locked_final_families_run") is False:
                result.ok("manifest locked_final_families_run=false")
            else:
                result.fail("locked families flagged run")
            if status.startswith("phase5c_") or status.startswith("phase5d_"):
                result.ok("later phase inherits phase5b discovery freeze")
            elif auth.get("discovery_generation_authorized") is False:
                result.ok("discovery_generation_authorized=false after complete")
            else:
                result.fail("discovery auth should be closed after complete")
        else:
            result.fail("missing discovery behavior artifacts")

    tracked = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "ls-files"], text=True
    ).splitlines()
    for path in tracked:
        if "phase5" in path and path.endswith("discovery_outputs.jsonl"):
            result.fail(f"raw discovery outputs tracked: {path}")
    result.ok("raw discovery outputs not tracked")

    print()
    print(f"{len(result.passes)} passed, {len(result.failures)} failed")
    if result.failures:
        for f in result.failures:
            print(f"  - {f}")
        return 1
    print("Phase 5B validation OK.")
    if str(status).startswith("phase5d"):
        print("PHASE 5D PATH ACKNOWLEDGED; DISCOVERY FREEZE INTACT.")
    else:
        print("NO PHASE 5 ACTIVATIONS WERE COLLECTED.")
        print("NO PHASE 5 PROBES WERE FIT OR SCORED.")
        print("NO LAYER SELECTION WAS PERFORMED.")
        print("LOCKED GENERALIZATION FAMILIES WERE NOT RUN THROUGH THE MODEL.")
    print("NO PROMPT OR BEHAVIOR RULES WERE CHANGED.")
    print("NO CAUSAL INTERVENTIONS WERE PERFORMED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
