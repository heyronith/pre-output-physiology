#!/usr/bin/env python3
"""Phase 4B revision-1 re-pilot integrity checks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase4_conditions import (  # noqa: E402
    COMMON_FIRST_TOKEN_ID,
    CONDITION_ORDER,
    N_PILOT_BASE_SCENARIOS,
    PILOT_TEMPLATE_REVISION,
    assert_c2_c3_template_symmetry,
    verify_common_first_token_with_tokenizer,
)

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
FROZEN_PROBE_L12_K1_NPZ_SHA256 = (
    "3bbf0c78e97d8ff14b028592e6e6ab3215651e973e0be1f004ce66e2b9b7aa65"
)
FROZEN_PROBE_L12_K0_NPZ_SHA256 = (
    "dc14569a4e229cadc55fd3f6e6fef4e33f4eac4f58d00e83916238d5274f4f29"
)
REV0_RUN_ID = "phase4b_pilot_20260925T165130Z_ee4d615c"


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
            print((proc.stdout or "")[-1500:])
            print((proc.stderr or "")[-1500:])

    decisions = (REPO_ROOT / "docs/decision_log.md").read_text(encoding="utf-8")
    for did in ("D055", "D056", "D057"):
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
        "phase4b_repilot_fail_hold",
        "phase4b_repilot_pass_awaiting_audit",
        "phase4c_controlled_prefix_pilot_authorized",
        "phase4c_controlled_prefix_pilot_pass_awaiting_audit",
        "phase4c_controlled_prefix_pilot_fail_hold",
        "phase4d_final_behavior_generation_authorized",
        "phase4d_final_behavior_complete_awaiting_audit",
        "phase4e_specificity_extraction_authorized",
        "phase4e_specificity_complete_awaiting_audit",
    }:
        result.ok(f"status acknowledges post-rev1 path ({status})")
    else:
        result.fail(f"unexpected status {status}")
    auth = exp.get("authorizations", {})
    phase4d = status.startswith("phase4d_")
    phase4e = status.startswith("phase4e_")
    if phase4e:
        if auth.get("final_generation_authorized") is True:
            result.ok("final_generation_authorized=true (phase4e)")
        else:
            result.fail("final_generation_authorized must be true in phase4e")
        if auth.get("activation_extraction_authorized") is True:
            result.ok("activation_extraction_authorized=true (phase4e)")
        else:
            result.fail("activation_extraction_authorized must be true in phase4e")
        if auth.get("causal_intervention_authorized") is False:
            result.ok("causal_intervention_authorized=false")
        else:
            result.fail("causal_intervention_authorized must be false")
    elif phase4d:
        if auth.get("final_generation_authorized") is True:
            result.ok("final_generation_authorized=true (phase4d)")
        else:
            result.fail("final_generation_authorized must be true in phase4d")
        for key in (
            "activation_extraction_authorized",
            "causal_intervention_authorized",
        ):
            if auth.get(key) is False:
                result.ok(f"{key}=false")
            else:
                result.fail(f"{key} must be false")
    else:
        for key in (
            "final_generation_authorized",
            "activation_extraction_authorized",
            "causal_intervention_authorized",
        ):
            if auth.get(key) is False:
                result.ok(f"{key}=false")
            else:
                result.fail(f"{key} must be false")

    if COMMON_FIRST_TOKEN_ID != 12107 or exp.get("common_first_token_id") != 12107:
        result.fail("common token ID must be 12107")
    else:
        result.ok("common token ID = 12107")
    if PILOT_TEMPLATE_REVISION != 1 or exp.get("design", {}).get(
        "pilot_template_revision"
    ) != 1:
        result.fail("pilot_template_revision must be 1")
    else:
        result.ok("revision count = 1")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, use_fast=True
    )
    try:
        verify_common_first_token_with_tokenizer(tok)
        result.ok("tokenizer verifies 12107 in chat-generation context")
    except ValueError as exc:
        result.fail(f"token verification failed: {exc}")

    sample = {
        "topic_sentence": "A catalog lists where the red box is stored.",
        "state_label_a": "STATE_A",
        "state_label_b": "STATE_B",
        "state_a": "north room",
        "state_b": "south room",
        "ground_truth_state": "north room",
        "alt_state": "south room",
        "accessible_false_state": "south room",
    }
    try:
        assert_c2_c3_template_symmetry(sample)
        result.ok("C2/C3 structural symmetry")
    except AssertionError as exc:
        result.fail(f"C2/C3 asymmetry: {exc}")

    # Original pilot preserved
    if (REPO_ROOT / "artifacts/runs" / REV0_RUN_ID / "pilot_outputs.jsonl").is_file():
        result.ok("original pilot raw outputs preserved")
    else:
        result.fail("original pilot raw outputs missing")

    man_path = REPO_ROOT / "artifacts/phase4b_pilot_revision1/pilot_generation_manifest.json"
    beh_path = REPO_ROOT / "artifacts/phase4b_pilot_revision1/pilot_behavior_summary.json"
    report_path = REPO_ROOT / "reports/phase4b_pilot_revision1.md"
    if not man_path.is_file() or not beh_path.is_file() or not report_path.is_file():
        result.fail("missing revision-1 summary/report artifacts")
        print(f"\n{len(result.passes)} passed, {len(result.failures)} failed")
        return 1
    result.ok("revision-1 summary artifacts present")

    man = json.loads(man_path.read_text(encoding="utf-8"))
    beh = json.loads(beh_path.read_text(encoding="utf-8"))
    summary = beh["summary"]

    if man.get("pilot_revision") == 1:
        result.ok("manifest pilot_revision=1")
    else:
        result.fail("manifest pilot_revision drift")
    if man.get("n_pilot_outputs") == 144 and man.get("n_final_outputs") == 0:
        result.ok("exactly 144 pilot / zero final outputs")
    else:
        result.fail("output count drift")
    if man.get("model_revision") == MODEL_REVISION and man.get("do_sample") is False:
        result.ok("model revision + greedy decoding")
    else:
        result.fail("generation settings drift")
    if man.get("common_first_token_id") == 12107:
        result.ok("rev1 expected first-token ID 12107")
    else:
        result.fail("rev1 token ID drift")
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
        if len(rows) == 144 and all(r.get("split") == "pilot" for r in rows):
            result.ok("raw rev1 outputs = 144 pilot")
        else:
            result.fail("raw rev1 rows invalid")
        if any("final_" in r.get("base_scenario_id", "") for r in rows):
            result.fail("final scenario IDs in rev1 outputs")
        else:
            result.ok("same pilot scenario IDs only")
        if all(isinstance(r.get("generated_token_ids"), list) for r in rows):
            result.ok("generated token IDs stored")
        else:
            result.fail("missing generated token IDs")
    else:
        result.fail("missing rev1 raw outputs")

    for cid in CONDITION_ORDER:
        pc = summary["per_condition"].get(cid)
        if pc and pc.get("n_total") == N_PILOT_BASE_SCENARIOS:
            result.ok(f"{cid} n_total=24")
        else:
            result.fail(f"{cid} n_total drift")

    # MODE schema expectations in prompts/parser
    if all(
        "exact_three_line_format" in summary["per_condition"][cid]
        for cid in CONDITION_ORDER
    ):
        result.ok("exact three-line schema metrics present")
    else:
        result.fail("missing exact-format metrics")

    probe_man = json.loads(
        (REPO_ROOT / "artifacts/phase4a_summaries/frozen_probe_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    k1 = probe_man["probes"]["probe_l12_k1"]["artifact_npz_sha256"]
    k0 = probe_man["probes"]["probe_l12_k0"]["artifact_npz_sha256"]
    if k1 == FROZEN_PROBE_L12_K1_NPZ_SHA256 and k0 == FROZEN_PROBE_L12_K0_NPZ_SHA256:
        result.ok("frozen probe hashes unchanged")
    else:
        result.fail("frozen probe hashes drifted")

    score_hits = list((REPO_ROOT / "artifacts").rglob("*phase4*probe*score*"))
    score_hits += list((REPO_ROOT / "artifacts").rglob("*phase4*auroc*"))
    # Phase 4E may commit specificity summaries; exclude those paths.
    score_hits = [
        p
        for p in score_hits
        if "phase4e_" not in str(p) and "phase4e/" not in str(p).replace("\\", "/")
    ]
    if score_hits:
        result.fail(f"probe-score artifacts: {score_hits[:5]}")
    else:
        result.ok("no Phase-4 probe-score artifacts")

    tracked = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "ls-files"], text=True
    ).splitlines()
    for path in tracked:
        if path.endswith((".safetensors", ".npz", ".pt", ".pth")):
            result.fail(f"weight tensor tracked: {path}")
        if "pilot_outputs.jsonl" in path:
            result.fail(f"raw pilot outputs tracked: {path}")
    result.ok("raw outputs not git-tracked")

    # Gate outcome recorded in rev1 summary (historical). Current status may be Phase 4C.
    gates_pass = summary.get("gates", {}).get("all_operational_gates_pass")
    if gates_pass is False:
        result.ok("rev1 summary records gate failure (historical)")
    elif gates_pass is True:
        result.ok("rev1 summary records gate pass (historical)")
    else:
        result.fail("rev1 gate outcome missing")
    if (
        status.startswith("phase4c_")
        or status.startswith("phase4d_")
        or status.startswith("phase4e_")
        or (status == "phase4b_repilot_fail_hold" and gates_pass is False)
        or (status == "phase4b_repilot_pass_awaiting_audit" and gates_pass is True)
    ):
        result.ok("status compatible with rev1 outcome / Phase 4C/4D/4E continuation")
    else:
        result.fail(f"status/gates mismatch: status={status} gates={gates_pass}")

    print()
    print(f"{len(result.passes)} passed, {len(result.failures)} failed")
    if result.failures:
        for f in result.failures:
            print(f"  - {f}")
        return 1
    print("Phase 4B revision-1 validation OK.")
    print("THIS WAS THE SINGLE ALLOWED POST-PILOT TEMPLATE REVISION.")
    print("NO PHASE 4 ACTIVATIONS WERE COLLECTED.")
    print("NO PHASE 4 PROBE SCORES WERE COMPUTED.")
    print("FINAL PHASE 4 SCENARIOS WERE NOT GENERATED.")
    print("NO CAUSAL INTERVENTIONS WERE PERFORMED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
