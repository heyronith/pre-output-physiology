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
    FIRST_TOKEN_CANDIDATES,
    KEY_SECONDARY_CONTRAST,
    N_FINAL_BASE_SCENARIOS,
    N_PILOT_BASE_SCENARIOS,
    PRIMARY_CONTRAST,
    SCENARIO_FAMILIES,
    assert_no_forbidden_prompt_terms,
    build_condition_prompt,
)

PINNED_MODEL_REV = "63a8b081895390a26e140280378bc85ec8bce07a"
CANONICAL_B1 = "phase3b1_extract_20260925T154005Z_39505f42"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"


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
    for did in ("D045", "D046", "D047", "D048", "D049", "D050"):
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
    if exp.get("status") == "phase4a_design_frozen_awaiting_pilot":
        result.ok("phase4 status awaiting pilot")
    else:
        result.fail(f"unexpected phase4 status {exp.get('status')}")
    auth = exp.get("authorizations", {})
    for key in (
        "pilot_generation_authorized",
        "final_generation_authorized",
        "activation_extraction_authorized",
        "causal_intervention_authorized",
        "modal_gpu_authorized",
    ):
        if auth.get(key) is False:
            result.ok(f"{key}=false")
        else:
            result.fail(f"{key} must be false")

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
        for name in ("surface_k0_prompt_only", "surface_k1_prompt_plus_1"):
            entry = sm.get("models", {}).get(name)
            if entry and (REPO_ROOT / entry["artifact_joblib"]).is_file():
                if _sha_file(REPO_ROOT / entry["artifact_joblib"]) == entry[
                    "artifact_joblib_sha256"
                ]:
                    result.ok(f"{name} hash ok")
                else:
                    result.fail(f"{name} hash mismatch")
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
        if mx.get("phase4_outputs_exist") is False and mx.get(
            "phase4_activations_exist"
        ) is False:
            result.ok("no Phase 4 outputs/activations claimed")
        else:
            result.fail("Phase 4 outputs/activations unexpectedly present")
        if mx.get("model_generation_performed") is False:
            result.ok("no Phase 4 model generation performed")
        else:
            result.fail("model generation flag true")

    # Tokenizer verification of common first token
    tok = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=PINNED_MODEL_REV, use_fast=True
    )
    prefix = "[INST] say something [/INST]"
    p = tok.encode(prefix, add_special_tokens=False)
    chosen = None
    for cand in FIRST_TOKEN_CANDIDATES:
        delta = tok.encode(prefix + cand, add_special_tokens=False)[len(p) :]
        if len(delta) == 1:
            chosen = (cand, int(delta[0]))
            break
    if chosen == (COMMON_FIRST_TOKEN, COMMON_FIRST_TOKEN_ID):
        result.ok("common first token is exactly one tokenizer token")
    else:
        result.fail(f"first-token verification failed: {chosen}")

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

    # Behavior gate smoke
    ok, _ = is_behaviorally_valid(
        "C1_known_honest_neutral",
        sample_scenario,
        "Response\nThe state is north room.\nFINAL=north room\n",
    )
    if ok:
        result.ok("deterministic behavior rules executable")
    else:
        result.fail("behavior rule smoke failed")

    # No Phase 4 outputs / activations / modal jobs in tree
    phase4_runs = list((REPO_ROOT / "artifacts/runs").glob("phase4*")) if (
        REPO_ROOT / "artifacts/runs"
    ).is_dir() else []
    if not phase4_runs:
        result.ok("no Phase 4 run directories")
    else:
        result.fail(f"Phase 4 run dirs present: {phase4_runs}")

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
    print("NO PHASE 4 MODEL RESPONSES WERE GENERATED.")
    print("NO PHASE 4 ACTIVATIONS WERE COLLECTED.")
    print("NO PHASE 4 MODAL GPU JOBS WERE RUN.")
    print("NO CAUSAL INTERVENTIONS WERE PERFORMED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
