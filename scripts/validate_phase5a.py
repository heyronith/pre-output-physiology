#!/usr/bin/env python3
"""Phase 5A design + pilot integrity checks."""

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
    FORBIDDEN_PROMPT_SUBSTRINGS,
    LOCKED_FAMILIES,
    PHASE4_FAMILIES,
    SCENARIO_FAMILIES,
    assert_s2_s3_template_symmetry,
    build_condition_prompt,
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


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    result = Result()
    for script in (
        "validate_phase1.py",
        "validate_phase2.py",
        "validate_phase3a.py",
        "validate_phase3b_dev.py",
        "validate_phase3b_locked.py",
        "validate_phase4a.py",
    ):
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / script)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            result.ok(f"{script} passed")
        else:
            result.fail(f"{script} failed")

    decisions = (REPO_ROOT / "docs/decision_log.md").read_text(encoding="utf-8")
    for did in ("D065", "D066", "D067", "D068", "D069", "D070", "D071"):
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
        "phase5a_design_frozen_pilot_authorized",
        "phase5a_behavior_pilot_pass_awaiting_audit",
        "phase5a_behavior_pilot_hold",
        "phase5b_discovery_behavior_authorized",
        "phase5b_discovery_behavior_complete_awaiting_audit",
        "phase5c_discovery_physiology_authorized",
        "phase5c_candidate_selection_complete_awaiting_audit",
        "phase5c_candidate_gate_fail_hold",
    }:
        result.ok(f"status {status}")
    else:
        result.fail(f"unexpected status {status}")

    auth = exp.get("authorizations", {})
    for key in (
        "causal_intervention_authorized",
        "locked_final_generation_authorized",
    ):
        if auth.get(key) is False:
            result.ok(f"{key}=false")
        else:
            result.fail(f"{key} must be false")
    # Activation/probe auth may be true only during authorized Phase 5C extraction.
    if status == "phase5c_discovery_physiology_authorized":
        for key in (
            "activation_extraction_authorized",
            "probe_fitting_authorized",
            "probe_scoring_authorized",
        ):
            if auth.get(key) is True:
                result.ok(f"{key}=true (phase5c authorized)")
            else:
                result.fail(f"{key} must be true when phase5c authorized")
    else:
        for key in (
            "activation_extraction_authorized",
            "probe_fitting_authorized",
            "probe_scoring_authorized",
        ):
            if auth.get(key) is False:
                result.ok(f"{key}=false")
            else:
                result.fail(f"{key} must be false")

    if set(SCENARIO_FAMILIES) & set(PHASE4_FAMILIES):
        result.fail("Phase 5 families overlap Phase 4")
    else:
        result.ok("no Phase-4 family name overlap")
    if len(SCENARIO_FAMILIES) == 8:
        result.ok("exactly 8 families")
    else:
        result.fail(f"n_families={len(SCENARIO_FAMILIES)}")
    if len(DISCOVERY_FAMILIES) == 6 and len(LOCKED_FAMILIES) == 2:
        result.ok("6 discovery + 2 locked")
    else:
        result.fail("discovery/locked split wrong")

    matrix_path = REPO_ROOT / "artifacts/phase5a_design/condition_matrix.json"
    if not matrix_path.is_file():
        result.fail("missing condition_matrix.json")
        print(f"\n{len(result.passes)} passed, {len(result.failures)} failed")
        return 1
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if matrix.get("n_final_per_family") == 160 and matrix.get("n_final_base_scenarios") == 1280:
        result.ok("160/family and 1280 base scenarios")
    else:
        result.fail("final scenario counts drift")
    if matrix.get("n_final_prompts") == 2560:
        result.ok("2560 prompts")
    else:
        result.fail("prompt count drift")
    if matrix.get("neutral_prefix_token_id") == 12107:
        result.ok("neutral prefix token id=12107 verified")
    else:
        result.fail(f"unexpected prefix id {matrix.get('neutral_prefix_token_id')}")
    if matrix.get("phase4_family_overlap") is False:
        result.ok("matrix records no Phase-4 overlap")
    else:
        result.fail("phase4 overlap flagged")
    if matrix.get("locked_model_outputs_exist") is False:
        result.ok("locked outputs absent in matrix")
    else:
        result.fail("locked outputs flagged")

    data = REPO_ROOT / "data/processed/phase5_design"
    final_sc = _load_jsonl(data / "final_base_scenarios.jsonl")
    pilot_sc = _load_jsonl(data / "pilot_base_scenarios.jsonl")
    final_pr = _load_jsonl(data / "final_candidate_prompts.jsonl")
    pilot_pr = _load_jsonl(data / "pilot_candidate_prompts.jsonl")
    if len(final_sc) == 1280 and len(pilot_sc) == 32:
        result.ok("scenario JSONL counts")
    else:
        result.fail("scenario JSONL count drift")
    if len(final_pr) == 2560 and len(pilot_pr) == 64:
        result.ok("prompt JSONL counts")
    else:
        result.fail("prompt JSONL count drift")

    final_ids = {s["base_scenario_id"] for s in final_sc}
    pilot_ids = {s["base_scenario_id"] for s in pilot_sc}
    locked_ids = {s["base_scenario_id"] for s in final_sc if s["pool"] == "locked"}
    discovery_ids = {s["base_scenario_id"] for s in final_sc if s["pool"] == "discovery"}
    if final_ids & pilot_ids:
        result.fail("pilot/final ID overlap")
    else:
        result.ok("pilot/final IDs disjoint")
    if locked_ids & discovery_ids:
        result.fail("locked/discovery overlap")
    else:
        result.ok("locked/discovery disjoint")
    if len(locked_ids) == 320 and len(discovery_ids) == 960:
        result.ok("locked=320 discovery=960")
    else:
        result.fail("pool size drift")

    # Phase-4 ID collision check if phase4 design present
    p4 = REPO_ROOT / "data/processed/phase4_design/final_base_scenarios.jsonl"
    if p4.is_file():
        p4_ids = {
            json.loads(line)["base_scenario_id"]
            for line in p4.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if p4_ids & final_ids or p4_ids & pilot_ids:
            result.fail("Phase-4 scenario ID overlap")
        else:
            result.ok("no Phase-4 scenario ID overlap")
    else:
        result.ok("Phase-4 design JSONL absent locally (skipped ID check)")

    # Symmetry + forbidden lexicon on samples
    try:
        for sc in final_sc[:5] + pilot_sc[:3]:
            assert_s2_s3_template_symmetry(sc)
            for cid in ("S2_strategic_honesty", "S3_strategic_deception"):
                prompt = build_condition_prompt(condition_id=cid, scenario=sc)
                low = prompt.lower()
                for term in FORBIDDEN_PROMPT_SUBSTRINGS:
                    import re

                    if re.search(rf"\b{re.escape(term)}\b", low):
                        raise AssertionError(term)
        result.ok("S2/S3 symmetry + forbidden lexicon samples OK")
    except Exception as exc:  # noqa: BLE001
        result.fail(f"symmetry/lexicon: {exc}")

    # Counterbalance: within each family, record == option_a about half
    bal_ok = True
    for fam in SCENARIO_FAMILIES:
        fam_rows = [s for s in final_sc if s["family"] == fam]
        a_rec = sum(1 for s in fam_rows if s["record_state"] == s["state_option_a"])
        if abs(a_rec - 80) > 1:
            bal_ok = False
            result.fail(f"{fam} record balance {a_rec}/160")
    if bal_ok:
        result.ok("per-family record-state counterbalance")

    # Locked final prompts must not have model outputs in runs
    locked_output_hits = []
    runs = REPO_ROOT / "artifacts/runs"
    if runs.is_dir():
        for path in runs.rglob("*.jsonl"):
            if "phase5" not in path.name and "phase5" not in str(path):
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("split") == "final" and row.get("pool") == "locked":
                    locked_output_hits.append(str(path))
                    break
    if locked_output_hits:
        result.fail(f"locked final outputs present: {locked_output_hits[:3]}")
    else:
        result.ok("no locked-final model outputs")

    # No activation / probe artifacts for phase5
    hits = []
    for path in (REPO_ROOT / "artifacts").rglob("*"):
        if not path.is_file():
            continue
        rel = str(path.relative_to(REPO_ROOT)).lower()
        if "phase5" not in rel:
            continue
        if any(tok in rel for tok in ("activation", "probe_score", ".safetensors", "causal")):
            if "condition_matrix" in rel or "manifest" in rel:
                continue
            hits.append(rel)
    if hits:
        result.fail(f"phase5 activation/probe artifacts: {hits[:5]}")
    else:
        result.ok("no phase5 activation/probe artifacts")

    # Pilot artifacts if status is pass/hold or later Phase 5B
    if status in {
        "phase5a_behavior_pilot_pass_awaiting_audit",
        "phase5a_behavior_pilot_hold",
        "phase5b_discovery_behavior_authorized",
        "phase5b_discovery_behavior_complete_awaiting_audit",
        "phase5c_discovery_physiology_authorized",
        "phase5c_candidate_selection_complete_awaiting_audit",
        "phase5c_candidate_gate_fail_hold",
    }:
        beh = REPO_ROOT / "artifacts/phase5a_pilot/pilot_behavior_summary.json"
        report = REPO_ROOT / "reports/phase5a_pilot.md"
        if beh.is_file() and report.is_file():
            result.ok("pilot summary/report present")
            summary = json.loads(beh.read_text(encoding="utf-8"))["summary"]
            gates = summary["gates"]["all_operational_gates_pass"]
            if status == "phase5a_behavior_pilot_hold":
                if not gates:
                    result.ok("status matches gate hold")
                else:
                    result.fail(f"status/gates mismatch {status} / {gates}")
            elif gates:
                result.ok("status matches gate pass")
            else:
                result.fail(f"status/gates mismatch {status} / {gates}")
        else:
            result.fail("missing pilot summary/report")

    tracked = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "ls-files"], text=True
    ).splitlines()
    for path in tracked:
        if path.endswith("pilot_outputs.jsonl") and "phase5" in path:
            result.fail(f"raw pilot tracked: {path}")
    result.ok("raw pilot outputs not tracked")

    print()
    print(f"{len(result.passes)} passed, {len(result.failures)} failed")
    if result.failures:
        for f in result.failures:
            print(f"  - {f}")
        return 1
    print("Phase 5A validation OK.")
    print("NO PHASE 5 ACTIVATIONS WERE COLLECTED.")
    print("NO PHASE 5 PROBES WERE FIT OR SCORED.")
    print("LOCKED GENERALIZATION FAMILIES WERE NOT RUN THROUGH THE MODEL.")
    print("NO CAUSAL INTERVENTIONS WERE PERFORMED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
