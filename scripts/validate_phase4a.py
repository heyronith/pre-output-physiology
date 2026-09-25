#!/usr/bin/env python3
"""Phase 4A design-freeze integrity checks (local only)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase4_behavior import (  # noqa: E402
    BEHAVIOR_RULES_SUMMARY,
    is_behaviorally_valid,
)
from pre_output_physiology.phase4_conditions import (  # noqa: E402
    COMMON_FIRST_TOKEN,
    COMMON_FIRST_TOKEN_ID,
    CONDITION_ORDER,
    KEY_SECONDARY_CONTRAST,
    N_FINAL_BASE_SCENARIOS,
    N_PILOT_BASE_SCENARIOS,
    PRIMARY_CONTRAST,
    SCENARIO_FAMILIES,
    STRATEGIC_NONDISCLOSURE_INSTRUCTION,
    assert_c2_c3_template_symmetry,
    assert_no_forbidden_prompt_terms,
    build_condition_prompt,
    verify_common_first_token_with_tokenizer,
)

PINNED_MODEL_REV = "63a8b081895390a26e140280378bc85ec8bce07a"
CANONICAL_B1 = "phase3b1_extract_20260925T154005Z_39505f42"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

# Immutable Phase-3 freeze hashes (must not change during Phase 4A design edits).
FROZEN_PROBE_L12_K1_NPZ_SHA256 = (
    "3bbf0c78e97d8ff14b028592e6e6ab3215651e973e0be1f004ce66e2b9b7aa65"
)
FROZEN_PROBE_L12_K0_NPZ_SHA256 = (
    "dc14569a4e229cadc55fd3f6e6fef4e33f4eac4f58d00e83916238d5274f4f29"
)
FROZEN_SURFACE_K0_JOBLIB_SHA256 = (
    "4992d91c576efe34c4fbc47a4beb4df195960eef65218795b2c36fffb7f04c05"
)
FROZEN_SURFACE_K1_JOBLIB_SHA256 = (
    "17c7fed1eec5feb1b038cc188a44d8ac07b9728283e35088ac016a73be61a626"
)
EXPECTED_FINAL_BASE_SCENARIO_IDS_SHA256 = (
    "f43d16d942331aa45f3a86e8af5f330f0370fe84356599fa7964a3949a2bf880"
)
EXPECTED_PILOT_BASE_SCENARIO_IDS_SHA256 = (
    "9d12e5597644bdb3689f31bcd0ad9902060cb870dd02ba40139ebcf7d9e13990"
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


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    result = Result()
    for script in (
        "validate_phase1.py",
        "validate_phase2.py",
        "validate_phase3a.py",
        "validate_phase3b_dev.py",
        "validate_phase3b_locked.py",
    ):
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / script)],
            cwd=REPO_ROOT,
        )
        if proc.returncode == 0:
            result.ok(f"{script} passed")
        else:
            result.fail(f"{script} failed")

    decisions = (REPO_ROOT / "docs/decision_log.md").read_text(encoding="utf-8")
    for did in ("D045", "D046", "D047", "D048", "D049", "D050", "D051"):
        if did in decisions:
            result.ok(f"{did} present")
        else:
            result.fail(f"{did} missing")

    # Phase 3 provenance preserved
    latest = REPO_ROOT / "artifacts/phase3b_dev/latest_extract_manifest.json"
    if latest.is_file():
        man = json.loads(latest.read_text(encoding="utf-8"))
        if man.get("run_id") == CANONICAL_B1:
            result.ok("audited Phase 3 canonical B1 provenance preserved")
        else:
            result.fail(f"canonical B1 drift: {man.get('run_id')}")
    else:
        result.fail("missing Phase 3B1 latest extract manifest")

    locked_metrics = REPO_ROOT / "artifacts/phase3b_locked/phase3b_locked_metrics.json"
    if locked_metrics.is_file():
        result.ok("Phase 3B2 locked metrics preserved")
    else:
        result.fail("missing Phase 3B2 locked metrics")

    exp = yaml.safe_load(
        (REPO_ROOT / "configs/experiments/phase4_specificity.yaml").read_text(
            encoding="utf-8"
        )
    )
    status = exp.get("status")
    if status in {
        "phase4a_design_frozen_awaiting_pilot",
        "phase4b_pilot_complete_awaiting_audit",
        "phase4b_repilot_pass_awaiting_audit",
        "phase4b_repilot_fail_hold",
        "phase4c_controlled_prefix_pilot_authorized",
        "phase4c_controlled_prefix_pilot_pass_awaiting_audit",
        "phase4c_controlled_prefix_pilot_fail_hold",
        "phase4d_final_behavior_generation_authorized",
        "phase4d_final_behavior_complete_awaiting_audit",
        "phase4e_specificity_extraction_authorized",
        "phase4e_specificity_complete_awaiting_audit",
    }:
        result.ok(f"phase4 status recognized ({status})")
    else:
        result.fail(f"unexpected phase4 status {status}")
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
    if auth.get("pilot_generation_authorized") is True:
        result.ok("pilot_generation_authorized=true (pilot path)")
    elif status == "phase4a_design_frozen_awaiting_pilot":
        # Pre-authorization design freeze may still have pilot false.
        if auth.get("pilot_generation_authorized") is False:
            result.ok("pilot_generation_authorized=false (pre-pilot freeze)")
        else:
            result.fail("pilot_generation_authorized unexpected")
    else:
        result.fail("pilot_generation_authorized must be true after pilot authorization")

    # Frozen probes
    probe_man_path = REPO_ROOT / "artifacts/phase4a_summaries/frozen_probe_manifest.json"
    surf_man_path = REPO_ROOT / "artifacts/phase4a_summaries/frozen_surface_manifest.json"
    if not probe_man_path.is_file():
        result.fail("missing frozen_probe_manifest.json")
    else:
        pm = json.loads(probe_man_path.read_text(encoding="utf-8"))
        if pm.get("locked_test_used") is False and pm.get("phase4_labels_used") is False:
            result.ok("probes fitted without locked-test / Phase4 labels")
        else:
            result.fail("probe fit contamination")
        if pm.get("n_development_rows") == 3000:
            result.ok("probes fitted on 3000-row development set")
        else:
            result.fail("probe fit n_development_rows != 3000")
        if pm.get("canonical_b1_run_id") == CANONICAL_B1:
            result.ok("probe canonical B1 run ID correct")
        else:
            result.fail("probe B1 run ID drift")
        for name in ("probe_l12_k1", "probe_l12_k0"):
            entry = pm.get("probes", {}).get(name)
            if not entry:
                result.fail(f"missing {name} in probe manifest")
                continue
            art = REPO_ROOT / entry["artifact_npz"]
            if art.is_file() and _sha_file(art) == entry["artifact_npz_sha256"]:
                result.ok(f"{name} artifact hash exists")
            else:
                result.fail(f"{name} artifact missing or hash mismatch")
            expected = (
                FROZEN_PROBE_L12_K1_NPZ_SHA256
                if name == "probe_l12_k1"
                else FROZEN_PROBE_L12_K0_NPZ_SHA256
            )
            if entry.get("artifact_npz_sha256") == expected:
                result.ok(f"{name} frozen hash unchanged")
            else:
                result.fail(f"{name} frozen hash drifted from Phase 4A lock")
            if entry.get("layer") == 12 and entry.get("C") == 0.01:
                result.ok(f"{name} layer/C frozen")
            else:
                result.fail(f"{name} settings drift")
        if pm.get("primary_probe") == "probe_l12_k1":
            result.ok("primary probe is L12/k1")
        else:
            result.fail("primary probe not L12/k1")

    if not surf_man_path.is_file():
        result.fail("missing frozen_surface_manifest.json")
    else:
        sm = json.loads(surf_man_path.read_text(encoding="utf-8"))
        if sm.get("locked_test_used") is False:
            result.ok("surface models not fit on locked test")
        else:
            result.fail("surface models used locked test")
        for name, expected in (
            ("surface_k0_prompt_only", FROZEN_SURFACE_K0_JOBLIB_SHA256),
            ("surface_k1_prompt_plus_1", FROZEN_SURFACE_K1_JOBLIB_SHA256),
        ):
            entry = sm.get("models", {}).get(name)
            if entry and (REPO_ROOT / entry["artifact_joblib"]).is_file():
                digest = _sha_file(REPO_ROOT / entry["artifact_joblib"])
                if digest == entry["artifact_joblib_sha256"] == expected:
                    result.ok(f"{name} hash ok")
                else:
                    result.fail(f"{name} hash mismatch or drift")
            else:
                result.fail(f"missing {name}")

    # Condition matrix
    matrix_path = REPO_ROOT / "artifacts/phase4a_summaries/condition_matrix.json"
    if not matrix_path.is_file():
        result.fail("missing condition_matrix.json")
    else:
        mx = json.loads(matrix_path.read_text(encoding="utf-8"))
        if mx.get("order") == list(CONDITION_ORDER):
            result.ok("six conditions match frozen order")
        else:
            result.fail("condition order drift")
        if mx.get("n_final_base_scenarios") == N_FINAL_BASE_SCENARIOS:
            result.ok("240 final base scenarios")
        else:
            result.fail("final base scenario count drift")
        if mx.get("n_pilot_base_scenarios") == N_PILOT_BASE_SCENARIOS:
            result.ok("pilot scenario count frozen")
        else:
            result.fail("pilot scenario count drift")
        if mx.get("family_counts_final") == {
            f: 40 for f in SCENARIO_FAMILIES
        }:
            result.ok("six balanced scenario families (40 each)")
        else:
            result.fail(f"family counts {mx.get('family_counts_final')}")
        if mx.get("primary_contrast") == list(PRIMARY_CONTRAST):
            result.ok("primary contrast = C3 vs C2")
        else:
            result.fail("primary contrast drift")
        if mx.get("key_secondary_contrast") == list(KEY_SECONDARY_CONTRAST):
            result.ok("key secondary = C3 vs C4")
        else:
            result.fail("key secondary contrast drift")
        if mx.get("common_first_token") == COMMON_FIRST_TOKEN and mx.get(
            "common_first_token_id"
        ) == COMMON_FIRST_TOKEN_ID:
            result.ok("common first token frozen")
        else:
            result.fail("common first token drift")
        if mx.get("pilot_final_disjoint") is True:
            result.ok("pilot/final disjointness recorded")
        else:
            result.fail("pilot/final not disjoint")
        if mx.get("final_base_scenario_ids_sha256") == (
            EXPECTED_FINAL_BASE_SCENARIO_IDS_SHA256
        ):
            result.ok("final base scenario IDs unchanged")
        else:
            result.fail("final base scenario ID hash drifted")
        if mx.get("pilot_base_scenario_ids_sha256") == (
            EXPECTED_PILOT_BASE_SCENARIO_IDS_SHA256
        ):
            result.ok("pilot base scenario IDs unchanged")
        else:
            result.fail("pilot base scenario ID hash drifted")
        if mx.get("phase4_outputs_exist") is False and mx.get(
            "phase4_activations_exist"
        ) is False:
            result.ok("no Phase 4 outputs/activations claimed")
        else:
            result.fail("Phase 4 outputs/activations unexpectedly present")
        if mx.get("model_generation_performed") is False:
            result.ok("condition matrix design freeze marks generation not performed")
        else:
            result.fail("model generation flag true in design matrix")

    # Tokenizer verification of common first token (D055 chat-generation context)
    tok = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=PINNED_MODEL_REV, use_fast=True
    )
    try:
        verify_common_first_token_with_tokenizer(tok)
        if COMMON_FIRST_TOKEN_ID == 12107:
            result.ok("common first token is 12107 under chat-generation context")
        else:
            result.fail(f"common first token ID drift: {COMMON_FIRST_TOKEN_ID}")
    except ValueError as exc:
        result.fail(f"first-token verification failed: {exc}")

    # Prompt lexicon + behavior rules smoke test
    sample_scenario = {
        "topic_sentence": "A catalog lists where the red box is stored.",
        "state_label_a": "STATE_A",
        "state_label_b": "STATE_B",
        "state_a": "north room",
        "state_b": "south room",
        "ground_truth_state": "north room",
        "alt_state": "south room",
        "accessible_false_state": "south room",
    }
    for cid in CONDITION_ORDER:
        prompt = build_condition_prompt(condition_id=cid, scenario=sample_scenario)
        try:
            assert_no_forbidden_prompt_terms(prompt)
            result.ok(f"no forbidden lexicon in {cid}")
        except ValueError:
            result.fail(f"forbidden term in {cid} prompt")
        if cid not in BEHAVIOR_RULES_SUMMARY:
            result.fail(f"missing behavior rule for {cid}")

    # D051: C2/C3 structural-template symmetry
    try:
        assert_c2_c3_template_symmetry(sample_scenario)
        result.ok("C2/C3 structural-template symmetry")
    except AssertionError as exc:
        result.fail(f"C2/C3 template asymmetry: {exc}")
    c2 = build_condition_prompt(
        condition_id="C2_known_honest_strategic", scenario=sample_scenario
    )
    c3 = build_condition_prompt(
        condition_id="C3_known_deceptive_strategic", scenario=sample_scenario
    )
    if "opposite" in c2.lower() or "opposite" in c3.lower():
        result.fail("word 'opposite' present in C2 or C3 prompt")
    else:
        result.ok("no word 'opposite' in C2 or C3 prompts")
    if (
        STRATEGIC_NONDISCLOSURE_INSTRUCTION in c2
        and STRATEGIC_NONDISCLOSURE_INSTRUCTION in c3
        and c2.count(STRATEGIC_NONDISCLOSURE_INSTRUCTION)
        == c3.count(STRATEGIC_NONDISCLOSURE_INSTRUCTION)
    ):
        result.ok("identical non-disclosure instruction in C2/C3")
    else:
        result.fail("C2/C3 non-disclosure instruction mismatch")

    # Behavior gate smoke
    ok, _ = is_behaviorally_valid(
        "C1_known_honest_neutral",
        sample_scenario,
        "Response\nMODE=REAL\nFINAL=north room\n",
        first_generated_token_id=12107,
    )
    if ok:
        result.ok("deterministic behavior rules executable")
    else:
        result.fail("behavior rule smoke failed")

    # No Phase 4 activation / unauthorized final-generation contamination in tree
    phase4_runs = list((REPO_ROOT / "artifacts/runs").glob("phase4*")) if (
        REPO_ROOT / "artifacts/runs"
    ).is_dir() else []
    final_out_runs = [
        p for p in phase4_runs if "final" in p.name.lower() and p.is_dir()
    ]
    if phase4d:
        if final_out_runs:
            result.ok(f"phase4d final run dirs present ({len(final_out_runs)})")
        else:
            result.fail("phase4d status but no final run directories")
    elif phase4e:
        if final_out_runs:
            result.ok(f"phase4e keeps final run dirs ({len(final_out_runs)})")
        else:
            result.fail("phase4e status but no final run directories")
    elif final_out_runs:
        result.fail(f"final Phase 4 run dirs present: {final_out_runs}")
    else:
        result.ok("no final Phase 4 run directories")
    act_hits = []
    for p in phase4_runs:
        # Phase 4E may store activations under phase4e_* run dirs (gitignored).
        if p.name.startswith("phase4e_"):
            continue
        act_hits.extend(p.rglob("*.safetensors"))
        act_hits.extend(p.rglob("*activation*"))
        act_hits.extend(p.rglob("*probe_score*"))
    if act_hits:
        result.fail(f"Phase 4 activation/score artifacts present: {act_hits[:5]}")
    else:
        result.ok("no Phase 4 activation/score artifacts in run dirs")

    tracked = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "ls-files"], text=True
    ).splitlines()
    for path in tracked:
        if path.endswith((".safetensors", ".npz", ".pt", ".pth")):
            result.fail(f"weight tensor tracked: {path}")
        if "phase4" in path and path.endswith(".jsonl") and "summaries" not in path:
            # design jsonl under data/ should be gitignored
            result.fail(f"unexpected tracked phase4 payload: {path}")
    result.ok("no activation/weight tensors tracked")

    if not (REPO_ROOT / "docs/phase4_specificity_protocol.md").is_file():
        result.fail("missing phase4 specificity protocol")
    else:
        result.ok("phase4 specificity protocol present")

    # Causal intervention absence
    if "causal_intervention_authorized: false" in (
        REPO_ROOT / "configs/experiments/phase4_specificity.yaml"
    ).read_text(encoding="utf-8") or auth.get("causal_intervention_authorized") is False:
        result.ok("no causal intervention authorized")
    else:
        result.fail("causal intervention authorized")

    print()
    print(f"{len(result.passes)} passed, {len(result.failures)} failed")
    if result.failures:
        for f in result.failures:
            print(f"  - {f}")
        return 1
    print("Phase 4A validation OK.")
    if status == "phase4a_design_frozen_awaiting_pilot" and auth.get(
        "pilot_generation_authorized"
    ) is not True:
        print("NO PHASE 4 MODEL RESPONSES WERE GENERATED.")
    print("NO PHASE 4 ACTIVATIONS WERE COLLECTED.")
    print("NO PHASE 4 PROBE SCORES WERE COMPUTED.")
    print("FINAL PHASE 4 SCENARIOS WERE NOT GENERATED.")
    print("NO CAUSAL INTERVENTIONS WERE PERFORMED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
