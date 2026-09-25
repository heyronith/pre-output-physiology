#!/usr/bin/env python3
"""Phase 4C controlled-prefix pilot integrity checks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase4_behavior import (  # noqa: E402
    PILOT_BEHAVIOR_MIN_PER_CONDITION,
    PILOT_C2_C3_PAIRED_MIN,
)
from pre_output_physiology.phase4_conditions import (  # noqa: E402
    CONDITION_ORDER,
    N_PILOT_BASE_SCENARIOS,
)

MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
FROZEN_PROBE_L12_K1 = "3bbf0c78e97d8ff14b028592e6e6ab3215651e973e0be1f004ce66e2b9b7aa65"
FROZEN_PROBE_L12_K0 = "dc14569a4e229cadc55fd3f6e6fef4e33f4eac4f58d00e83916238d5274f4f29"
PILOT_TEXT = "74e03eee1ad8b7ecbbb81846b8119ddfc074021d6ebb3713d41ff0639d80d57c"
FINAL_TEXT = "84942a71e61a1aed67c872d4ffd69dc6e533f6295fcb0b3b93fa78d93f97ae89"
REV0 = "phase4b_pilot_20260925T165130Z_ee4d615c"
REV1 = "phase4b_pilot_rev1_20260925T172314Z_33fb5209"


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
    for script in (
        "validate_phase1.py",
        "validate_phase2.py",
        "validate_phase3a.py",
        "validate_phase3b_dev.py",
        "validate_phase3b_locked.py",
        "validate_phase4a.py",
        "validate_phase4b_pilot.py",
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
            print((proc.stdout or "")[-1200:])

    decisions = (REPO_ROOT / "docs/decision_log.md").read_text(encoding="utf-8")
    for did in ("D057", "D058", "D059", "D060"):
        if did in decisions:
            result.ok(f"{did} present")
        else:
            result.fail(f"{did} missing")

    exp = yaml.safe_load(
        (REPO_ROOT / "configs/experiments/phase4_specificity.yaml").read_text(
            encoding="utf-8"
        )
    )
    status = exp.get("status")
    if status in {
        "phase4c_controlled_prefix_pilot_pass_awaiting_audit",
        "phase4c_controlled_prefix_pilot_fail_hold",
    }:
        result.ok(f"status {status}")
    else:
        result.fail(f"unexpected status {status}")
    auth = exp.get("authorizations", {})
    for key in (
        "final_generation_authorized",
        "activation_extraction_authorized",
        "causal_intervention_authorized",
    ):
        if auth.get(key) is False:
            result.ok(f"{key}=false")
        else:
            result.fail(f"{key} must be false")
    if auth.get("probe_scoring_authorized", False) is False:
        result.ok("probe_scoring_authorized=false")
    else:
        result.fail("probe scoring authorized")

    if exp.get("design", {}).get("pilot_template_revision") != 1:
        result.fail("prompt revision must remain 1")
    else:
        result.ok("prompt revision remains exactly 1")

    mx = json.loads(
        (REPO_ROOT / "artifacts/phase4a_summaries/condition_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    if mx.get("pilot_prompt_text_sha256") == PILOT_TEXT:
        result.ok("revision-1 pilot prompt-text hash unchanged")
    else:
        result.fail("pilot prompt-text hash drifted")
    if mx.get("final_prompt_text_sha256") == FINAL_TEXT:
        result.ok("revision-1 final prompt-text hash unchanged")
    else:
        result.fail("final prompt-text hash drifted")

    for run_id in (REV0, REV1):
        if (REPO_ROOT / "artifacts/runs" / run_id / "pilot_outputs.jsonl").is_file():
            result.ok(f"prior pilot preserved: {run_id}")
        else:
            result.fail(f"missing prior pilot {run_id}")

    man_path = (
        REPO_ROOT / "artifacts/phase4c_controlled_prefix_pilot/pilot_generation_manifest.json"
    )
    beh_path = (
        REPO_ROOT / "artifacts/phase4c_controlled_prefix_pilot/pilot_behavior_summary.json"
    )
    report = REPO_ROOT / "reports/phase4c_controlled_prefix_pilot.md"
    if not man_path.is_file() or not beh_path.is_file() or not report.is_file():
        result.fail("missing phase4c controlled-prefix summary/report")
        print(f"\n{len(result.passes)} passed, {len(result.failures)} failed")
        return 1
    result.ok("phase4c summary artifacts present")

    man = json.loads(man_path.read_text(encoding="utf-8"))
    beh = json.loads(beh_path.read_text(encoding="utf-8"))
    summary = beh["summary"]

    if man.get("prompt_template_revision") == 1 and man.get(
        "prompt_template_changed_after_rev1"
    ) is False:
        result.ok("no further prompt-template changes")
    else:
        result.fail("prompt template change flags wrong")
    if man.get("first_token_sampled") is False and man.get(
        "controlled_prefix_token_id"
    ) == 12107:
        result.ok("controlled prefix exactly 12107; not sampled")
    else:
        result.fail("controlled prefix provenance wrong")
    if man.get("n_pilot_outputs") == 144 and man.get("n_final_outputs") == 0:
        result.ok("144 pilot / zero final outputs")
    else:
        result.fail("output counts drift")
    if man.get("model_revision") == MODEL_REVISION and man.get("do_sample") is False:
        result.ok("model revision + greedy continuation")
    else:
        result.fail("generation settings drift")
    if man.get("prefix_integrity_count") == 144:
        result.ok("prefix integrity 144/144")
    else:
        result.fail("prefix integrity incomplete")
    if man.get("activations_collected") is False and man.get("probe_scores_computed") is False:
        result.ok("no activations/probes")
    else:
        result.fail("activation/probe flags set")

    raw_rel = man.get("raw_outputs_path_gitignored") or man.get("raw_outputs_path")
    raw_path = REPO_ROOT / raw_rel if raw_rel else None
    if raw_path and raw_path.is_file():
        rows = [
            json.loads(line)
            for line in raw_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) != 144:
            result.fail(f"raw rows != 144 ({len(rows)})")
        else:
            result.ok("raw outputs = 144")
        if any(r.get("split") != "pilot" for r in rows):
            result.fail("non-pilot rows present")
        else:
            result.ok("all rows pilot split")
        if any("final_" in r.get("base_scenario_id", "") for r in rows):
            result.fail("final scenario IDs present")
        else:
            result.ok("same pilot scenario IDs only")
        ok_prefix = all(
            r.get("controlled_prefix_supplied") is True
            and r.get("first_token_sampled") is False
            and r.get("controlled_prefix_token_id") == 12107
            and r.get("generated_token_ids", [None])[0] == 12107
            for r in rows
        )
        if ok_prefix:
            result.ok("every row has supplied prefix 12107 first")
        else:
            result.fail("prefix integrity broken in raw rows")
    else:
        result.fail("missing raw outputs")

    if summary.get("gates", {}).get("behavior_min_per_condition") != (
        PILOT_BEHAVIOR_MIN_PER_CONDITION
    ) or summary.get("gates", {}).get("c2_c3_paired_min") != PILOT_C2_C3_PAIRED_MIN:
        result.fail("behavioral thresholds changed")
    else:
        result.ok("behavioral gates unchanged (20/24 and 18/24)")

    for cid in CONDITION_ORDER:
        pc = summary["per_condition"].get(cid)
        if pc and pc.get("n_total") == N_PILOT_BASE_SCENARIOS:
            result.ok(f"{cid} n=24")
        else:
            result.fail(f"{cid} n drift")

    probe_man = json.loads(
        (REPO_ROOT / "artifacts/phase4a_summaries/frozen_probe_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    k1 = probe_man["probes"]["probe_l12_k1"]["artifact_npz_sha256"]
    k0 = probe_man["probes"]["probe_l12_k0"]["artifact_npz_sha256"]
    if k1 == FROZEN_PROBE_L12_K1 and k0 == FROZEN_PROBE_L12_K0:
        result.ok("frozen probe hashes unchanged")
    else:
        result.fail("frozen probe hashes drifted")

    score_hits = list((REPO_ROOT / "artifacts").rglob("*phase4*probe*score*"))
    score_hits += list((REPO_ROOT / "artifacts").rglob("*phase4*auroc*"))
    if score_hits:
        result.fail(f"probe-score artifacts present: {score_hits[:3]}")
    else:
        result.ok("no probe-score artifacts")

    tracked = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "ls-files"], text=True
    ).splitlines()
    for path in tracked:
        if path.endswith(("pilot_outputs.jsonl", ".safetensors", ".npz")):
            if "summaries" not in path and "manifest" not in path:
                if path.endswith("pilot_outputs.jsonl") or path.endswith(
                    (".safetensors", ".npz")
                ):
                    result.fail(f"raw/tensor tracked: {path}")
    result.ok("raw outputs not tracked")

    gates_pass = summary.get("gates", {}).get("all_operational_gates_pass")
    # For controlled-prefix, require prefix integrity + behavioral gates.
    # all_operational_gates_pass includes first-token which is integrity here.
    if status == "phase4c_controlled_prefix_pilot_pass_awaiting_audit" and gates_pass:
        result.ok("status matches gate pass")
    elif status == "phase4c_controlled_prefix_pilot_fail_hold" and not gates_pass:
        result.ok("status matches gate fail")
    else:
        result.fail(f"status/gates mismatch: {status} / {gates_pass}")

    print()
    print(f"{len(result.passes)} passed, {len(result.failures)} failed")
    if result.failures:
        for f in result.failures:
            print(f"  - {f}")
        return 1
    print("Phase 4C controlled-prefix validation OK.")
    print("NO ADDITIONAL PROMPT-TEMPLATE REVISION WAS PERFORMED.")
    print("NO PHASE 4 ACTIVATIONS WERE COLLECTED.")
    print("NO PHASE 4 PROBE SCORES WERE COMPUTED.")
    print("FINAL PHASE 4 SCENARIOS WERE NOT GENERATED.")
    print("NO CAUSAL ACTIVATION INTERVENTIONS WERE PERFORMED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
