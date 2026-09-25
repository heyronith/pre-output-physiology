#!/usr/bin/env python3
"""Phase 4D final behavior integrity checks (no activations / no C5)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
FROZEN_PROBE_L12_K1 = "3bbf0c78e97d8ff14b028592e6e6ab3215651e973e0be1f004ce66e2b9b7aa65"
FROZEN_PROBE_L12_K0 = "dc14569a4e229cadc55fd3f6e6fef4e33f4eac4f58d00e83916238d5274f4f29"
FINAL_TEXT = "84942a71e61a1aed67c872d4ffd69dc6e533f6295fcb0b3b93fa78d93f97ae89"
FINAL_SCEN = "f43d16d942331aa45f3a86e8af5f330f0370fe84356599fa7964a3949a2bf880"
SELECTED = (
    "C1_known_honest_neutral",
    "C2_known_honest_strategic",
    "C3_known_deceptive_strategic",
    "C4_false_belief_honest",
    "C6_counterfactual_nondeceptive",
)
HELD = "C5_uncertain_honest"
CONTRAST_NAMES = (
    "C3_vs_C2",
    "C3_vs_C4",
    "C3_vs_C1",
    "C3_vs_C6",
    "C2_vs_C1",
    "C4_vs_C1",
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
    for script in (
        "validate_phase1.py",
        "validate_phase2.py",
        "validate_phase3a.py",
        "validate_phase3b_dev.py",
        "validate_phase3b_locked.py",
        "validate_phase4a.py",
        "validate_phase4b_pilot.py",
        "validate_phase4b_repilot.py",
        "validate_phase4c_controlled_prefix.py",
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
            print((proc.stderr or "")[-800:])

    decisions = (REPO_ROOT / "docs/decision_log.md").read_text(encoding="utf-8")
    for did in ("D061", "D062"):
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
        "phase4d_final_behavior_complete_awaiting_audit",
        "phase4e_specificity_extraction_authorized",
        "phase4e_specificity_complete_awaiting_audit",
    }:
        result.ok(f"status {status}")
    elif status == "phase4d_final_behavior_generation_authorized":
        result.fail(
            "status still generation_authorized — complete evaluation first"
        )
    else:
        result.fail(f"unexpected status {status}")

    auth = exp.get("authorizations", {})
    if auth.get("final_generation_authorized") is True:
        result.ok("final_generation_authorized=true")
    else:
        result.fail("final_generation_authorized must be true")
    phase4e = isinstance(status, str) and status.startswith("phase4e_")
    if phase4e:
        if auth.get("activation_extraction_authorized") is True:
            result.ok("activation_extraction_authorized=true (phase4e)")
        else:
            result.fail("activation_extraction_authorized must be true in phase4e")
        if auth.get("probe_scoring_authorized") is True:
            result.ok("probe_scoring_authorized=true (phase4e)")
        else:
            result.fail("probe_scoring_authorized must be true in phase4e")
        if auth.get("causal_intervention_authorized") is False:
            result.ok("causal_intervention_authorized=false")
        else:
            result.fail("causal_intervention_authorized must be false")
    else:
        for key in (
            "activation_extraction_authorized",
            "causal_intervention_authorized",
            "probe_scoring_authorized",
        ):
            if auth.get(key) is False:
                result.ok(f"{key}=false")
            else:
                result.fail(f"{key} must be false")

    design = exp.get("design", {})
    if design.get("pilot_template_revision") != 1:
        result.fail("prompt revision must remain 1")
    else:
        result.ok("prompt revision remains exactly 1")
    if design.get("prompt_template_changed_after_rev1") is not False:
        result.fail("prompt_template_changed_after_rev1 must be false")
    else:
        result.ok("no prompt-template change after rev1")
    if design.get("held_condition") != HELD:
        result.fail("held_condition must be C5")
    else:
        result.ok("C5 held")
    if design.get("final_prompt_text_sha256") != FINAL_TEXT:
        result.fail("final prompt-text hash drifted in yaml")
    else:
        result.ok("yaml final prompt-text hash unchanged")
    if design.get("final_scenario_ids_sha256") != FINAL_SCEN:
        result.fail("final scenario hash drifted in yaml")
    else:
        result.ok("yaml final scenario hash unchanged")

    mx = json.loads(
        (REPO_ROOT / "artifacts/phase4a_summaries/condition_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    if mx.get("final_prompt_text_sha256") == FINAL_TEXT:
        result.ok("matrix final prompt-text hash unchanged")
    else:
        result.fail("matrix final prompt-text hash drifted")
    if mx.get("final_base_scenario_ids_sha256") == FINAL_SCEN:
        result.ok("matrix final scenario hash unchanged")
    else:
        result.fail("matrix final scenario hash drifted")

    summary_dir = REPO_ROOT / "artifacts/phase4d_final_behavior"
    man_path = summary_dir / "final_generation_manifest.json"
    beh_path = summary_dir / "final_behavior_summary.json"
    elig_path = summary_dir / "frozen_contrast_eligibility.json"
    report = REPO_ROOT / "reports/phase4d_final_behavior.md"
    for path in (man_path, beh_path, elig_path, report):
        if path.is_file():
            result.ok(f"present {path.relative_to(REPO_ROOT)}")
        else:
            result.fail(f"missing {path.relative_to(REPO_ROOT)}")

    if not all(p.is_file() for p in (man_path, beh_path, elig_path)):
        print()
        print(f"{len(result.passes)} passed, {len(result.failures)} failed")
        return 1

    man = json.loads(man_path.read_text(encoding="utf-8"))
    beh = json.loads(beh_path.read_text(encoding="utf-8"))
    elig = json.loads(elig_path.read_text(encoding="utf-8"))
    summary = beh.get("summary", {})

    if man.get("n_final_outputs") == 1200:
        result.ok("1200 final outputs recorded")
    else:
        result.fail(f"n_final_outputs={man.get('n_final_outputs')}")
    if man.get("n_c5_outputs") == 0 and summary.get("n_c5_outputs") == 0:
        result.ok("zero C5 outputs")
    else:
        result.fail("C5 outputs non-zero")
    if man.get("n_pilot_outputs") == 0 and summary.get("n_pilot_outputs") == 0:
        result.ok("zero pilot outputs")
    else:
        result.fail("pilot outputs non-zero")
    if man.get("prompt_template_revision") == 1:
        result.ok("manifest prompt revision=1")
    else:
        result.fail("manifest prompt revision drifted")
    if man.get("controlled_prefix_token_id") == 12107:
        result.ok("controlled prefix=12107")
    else:
        result.fail("controlled prefix drifted")
    if man.get("first_token_sampled") is False:
        result.ok("first_token_sampled=false")
    else:
        result.fail("first_token_sampled must be false")
    if man.get("model_revision") == MODEL_REVISION:
        result.ok("model revision pinned")
    else:
        result.fail("model revision drift")
    if summary.get("prefix_integrity_count") == 1200:
        result.ok("prefix integrity 1200/1200")
    else:
        result.fail(f"prefix integrity {summary.get('prefix_integrity_count')}")

    per = summary.get("per_condition", {})
    for cid in SELECTED:
        c = per.get(cid, {})
        if c.get("n_total") == 240:
            result.ok(f"{cid} n=240")
        else:
            result.fail(f"{cid} n={c.get('n_total')}")

    if HELD in per:
        result.fail("C5 present in per_condition summary")
    else:
        result.ok("C5 absent from behavior summary")

    contrasts = elig.get("contrasts", {})
    for name in CONTRAST_NAMES:
        c = contrasts.get(name)
        if not c:
            result.fail(f"missing contrast {name}")
            continue
        ids = c.get("paired_base_scenario_ids", [])
        digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        if c.get("paired_ids_sha256") == digest and c.get("n_paired_valid") == len(ids):
            result.ok(f"{name} eligibility hash consistent (N={len(ids)})")
        else:
            result.fail(f"{name} eligibility hash mismatch")

    if elig.get("held_condition") == HELD:
        result.ok("eligibility records C5 HOLD")
    else:
        result.fail("eligibility missing C5 HOLD")

    # Frozen probes unchanged
    for name, expected in (
        ("probe_l12_k1.npz", FROZEN_PROBE_L12_K1),
        ("probe_l12_k0.npz", FROZEN_PROBE_L12_K0),
    ):
        path = REPO_ROOT / "artifacts/phase4_models" / name
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest == expected:
                result.ok(f"frozen {name} hash unchanged")
            else:
                result.fail(f"frozen {name} hash drifted")
        else:
            result.fail(f"missing {name}")

    # No activation / probe-score / causal artifacts under phase4d paths
    hits = []
    for path in (REPO_ROOT / "artifacts").rglob("*"):
        if not path.is_file():
            continue
        rel = str(path.relative_to(REPO_ROOT))
        low = rel.lower()
        if "phase4e" in low:
            continue
        if any(
            tok in low
            for tok in (
                "phase4d",
                "phase4_final",
                "final_behavior",
            )
        ):
            if any(
                tok in low
                for tok in (
                    "activation",
                    "hidden_state",
                    "probe_score",
                    "logit_lens",
                    "causal",
                    "intervention",
                    ".safetensors",
                )
            ):
                hits.append(rel)
    if hits:
        result.fail(f"activation/probe/causal artifacts: {hits[:5]}")
    else:
        result.ok("no phase4d activation/probe/causal artifacts")

    # Raw outputs gitignored
    tracked = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "ls-files"], text=True
    ).splitlines()
    for path in tracked:
        if "final_outputs.jsonl" in path or path.endswith(
            ("phase4d_final_outputs.jsonl",)
        ):
            result.fail(f"raw final outputs tracked: {path}")
    result.ok("raw final outputs not tracked")

    # Eval rows consistency if present
    eval_rows = beh.get("eval_rows", [])
    if len(eval_rows) == 1200:
        result.ok("eval_rows length 1200")
        counts = Counter(r["condition_id"] for r in eval_rows)
        if HELD in counts:
            result.fail("C5 in eval_rows")
        else:
            result.ok("no C5 in eval_rows")
        for cid in SELECTED:
            if counts[cid] != 240:
                result.fail(f"eval {cid} count {counts[cid]}")
    else:
        result.fail(f"eval_rows length {len(eval_rows)}")

    print()
    print(f"{len(result.passes)} passed, {len(result.failures)} failed")
    if result.failures:
        for f in result.failures:
            print(f"  - {f}")
        return 1
    print("Phase 4D final behavior validation OK.")
    print("NO PHASE 4 ACTIVATIONS WERE COLLECTED.")
    print("NO PHASE 4 PROBE SCORES WERE COMPUTED.")
    print("NO PROMPT-TEMPLATE CHANGES WERE MADE.")
    print("NO CAUSAL INTERVENTIONS WERE PERFORMED.")
    print("C5 PRODUCED ZERO OUTPUTS AND REMAINS HOLD.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
