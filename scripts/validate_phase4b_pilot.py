#!/usr/bin/env python3
"""Phase 4B pilot integrity checks (behavior-only; no activations/probes)."""

from __future__ import annotations

import hashlib
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
)

MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
FROZEN_PROBE_L12_K1_NPZ_SHA256 = (
    "3bbf0c78e97d8ff14b028592e6e6ab3215651e973e0be1f004ce66e2b9b7aa65"
)
FROZEN_PROBE_L12_K0_NPZ_SHA256 = (
    "dc14569a4e229cadc55fd3f6e6fef4e33f4eac4f58d00e83916238d5274f4f29"
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

    # Prior validators
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
            print(proc.stdout[-2000:] if proc.stdout else "")
            print(proc.stderr[-2000:] if proc.stderr else "")

    decisions = (REPO_ROOT / "docs/decision_log.md").read_text(encoding="utf-8")
    for did in ("D052", "D053", "D054"):
        if did in decisions:
            result.ok(f"{did} present")
        else:
            result.fail(f"{did} missing")

    exp = yaml.safe_load(
        (REPO_ROOT / "configs/experiments/phase4_specificity.yaml").read_text(
            encoding="utf-8"
        )
    )
    if exp.get("status") == "phase4b_pilot_complete_awaiting_audit":
        result.ok("status phase4b_pilot_complete_awaiting_audit")
    else:
        result.fail(f"unexpected status {exp.get('status')}")
    auth = exp.get("authorizations", {})
    if auth.get("final_generation_authorized") is False:
        result.ok("final_generation_authorized=false")
    else:
        result.fail("final generation authorized")
    if auth.get("activation_extraction_authorized") is False:
        result.ok("activation_extraction_authorized=false")
    else:
        result.fail("activation extraction authorized")
    if auth.get("causal_intervention_authorized") is False:
        result.ok("causal_intervention_authorized=false")
    else:
        result.fail("causal intervention authorized")

    man_path = REPO_ROOT / "artifacts/phase4b_pilot/pilot_generation_manifest.json"
    beh_path = REPO_ROOT / "artifacts/phase4b_pilot/pilot_behavior_summary.json"
    report_path = REPO_ROOT / "reports/phase4b_pilot.md"
    if not man_path.is_file():
        result.fail("missing pilot_generation_manifest.json")
        print(f"\n{len(result.passes)} passed, {len(result.failures)} failed")
        return 1
    if not beh_path.is_file():
        result.fail("missing pilot_behavior_summary.json")
        print(f"\n{len(result.passes)} passed, {len(result.failures)} failed")
        return 1
    if not report_path.is_file():
        result.fail("missing reports/phase4b_pilot.md")
    else:
        result.ok("pilot report present")

    man = json.loads(man_path.read_text(encoding="utf-8"))
    beh = json.loads(beh_path.read_text(encoding="utf-8"))
    summary = beh["summary"]

    if man.get("n_pilot_outputs") == 144:
        result.ok("exactly 144 pilot outputs")
    else:
        result.fail(f"n_pilot_outputs={man.get('n_pilot_outputs')}")
    if man.get("n_final_outputs") == 0:
        result.ok("zero final-scenario outputs")
    else:
        result.fail("final-scenario outputs recorded")
    if man.get("model_revision") == MODEL_REVISION:
        result.ok("exact model revision")
    else:
        result.fail("model revision drift")
    if man.get("do_sample") is False:
        result.ok("greedy decoding (do_sample=false)")
    else:
        result.fail("do_sample not false")
    if man.get("max_new_tokens") == 128 and man.get("batch_size") == 1:
        result.ok("max_new_tokens=128 batch_size=1")
    else:
        result.fail("generation settings drift")
    if man.get("activations_collected") is False and man.get("probe_scores_computed") is False:
        result.ok("no activations or probe scores in generation manifest")
    else:
        result.fail("activation/probe flags set in generation manifest")
    if man.get("common_first_token_id") == COMMON_FIRST_TOKEN_ID:
        result.ok("common first token ID frozen at 2963")
    else:
        result.fail("common first token ID drift")

    # Raw outputs must exist locally and store token IDs
    raw_rel = man.get("raw_outputs_path_gitignored") or man.get("raw_outputs_path")
    if not raw_rel:
        result.fail("missing raw outputs path in manifest")
    else:
        raw_path = REPO_ROOT / raw_rel
        if not raw_path.is_file():
            result.fail(f"raw pilot outputs missing: {raw_path}")
        else:
            rows = [
                json.loads(line)
                for line in raw_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(rows) != 144:
                result.fail(f"raw rows != 144 ({len(rows)})")
            else:
                result.ok("raw pilot outputs = 144")
            if any(r.get("split") != "pilot" for r in rows):
                result.fail("non-pilot rows in raw outputs")
            else:
                result.ok("all raw rows are pilot split")
            if any("final_" in r.get("base_scenario_id", "") for r in rows):
                result.fail("final scenario IDs present in pilot outputs")
            else:
                result.ok("no final-scenario IDs in pilot outputs")
            if all(
                isinstance(r.get("generated_token_ids"), list)
                and r.get("first_generated_token_id") is not None
                for r in rows
            ):
                result.ok("generated token IDs stored")
            else:
                result.fail("missing generated token IDs")
            if any(r.get("activation_extracted") or r.get("probe_scored") for r in rows):
                result.fail("activation/probe flags on raw rows")
            else:
                result.ok("raw rows have no activation/probe scoring")

    # Behavior summary integrity
    if summary.get("n_eval_rows") == 144:
        result.ok("behavior eval covers 144 rows")
    else:
        result.fail("behavior eval row count drift")
    if summary.get("activations_collected") is False and summary.get(
        "probe_scores_computed"
    ) is False:
        result.ok("behavior summary bans activations/probes")
    else:
        result.fail("behavior summary claims activations/probes")
    for cid in CONDITION_ORDER:
        pc = summary["per_condition"].get(cid)
        if not pc or pc.get("n_total") != N_PILOT_BASE_SCENARIOS:
            result.fail(f"{cid} n_total != 24")
        else:
            result.ok(f"{cid} n_total=24")

    # First-token compliance must be ID-based in eval rows
    eval_rows = beh.get("eval_rows", [])
    if eval_rows and all("first_token_id_compliant" in r for r in eval_rows):
        result.ok("first-token compliance recorded via token-ID field")
    else:
        result.fail("missing first_token_id_compliant on eval rows")

    # Frozen probe hashes unchanged
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

    # No Phase-4 probe score artifact
    score_hits = list((REPO_ROOT / "artifacts").rglob("*phase4*probe*score*"))
    score_hits += list((REPO_ROOT / "artifacts").rglob("*phase4*auroc*"))
    if score_hits:
        result.fail(f"Phase-4 probe score artifacts present: {score_hits[:5]}")
    else:
        result.ok("no Phase-4 probe score artifacts")

    # Raw weight tensors still not tracked
    tracked = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "ls-files"], text=True
    ).splitlines()
    for path in tracked:
        if path.endswith((".safetensors", ".npz", ".pt", ".pth")):
            result.fail(f"weight tensor tracked: {path}")
        if "phase4b" in path and path.endswith("pilot_outputs.jsonl"):
            result.fail(f"raw pilot outputs tracked: {path}")
    result.ok("raw pilot outputs not git-tracked")

    print()
    print(f"{len(result.passes)} passed, {len(result.failures)} failed")
    if result.failures:
        for f in result.failures:
            print(f"  - {f}")
        return 1
    print("Phase 4B pilot validation OK.")
    print("NO PHASE 4 ACTIVATIONS WERE COLLECTED.")
    print("NO PHASE 4 PROBE SCORES WERE COMPUTED.")
    print("FINAL PHASE 4 SCENARIOS WERE NOT GENERATED.")
    print("NO CAUSAL INTERVENTIONS WERE PERFORMED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
