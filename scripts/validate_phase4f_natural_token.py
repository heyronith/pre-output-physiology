#!/usr/bin/env python3
"""Phase 4F natural-token diagnostic integrity checks."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_PROBE_L12_K1 = "3bbf0c78e97d8ff14b028592e6e6ab3215651e973e0be1f004ce66e2b9b7aa65"
ELIG_C3_C2 = "07dcb4567aaf98cea443327bca2cc336f082ab035c645ccd4c4b28dcc97503b2"
ELIG_C3_C4 = "c03c70d06cd5e487621781b8d0a5aa65354857c578a0842b7e216c894e045d22"


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
        "validate_phase4d_final_behavior.py",
        "validate_phase4e_specificity.py",
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
    if "D064" in decisions:
        result.ok("D064 present")
    else:
        result.fail("D064 missing")

    exp = yaml.safe_load(
        (REPO_ROOT / "configs/experiments/phase4_specificity.yaml").read_text(
            encoding="utf-8"
        )
    )
    status = exp.get("status")
    if status == "phase4f_natural_token_diagnostic_complete_awaiting_audit":
        result.ok(f"status {status}")
    else:
        result.fail(f"unexpected status {status}")
    auth = exp.get("authorizations", {})
    if auth.get("causal_intervention_authorized") is False:
        result.ok("causal_intervention_authorized=false")
    else:
        result.fail("causal must remain unauthorized")

    summary_dir = REPO_ROOT / "artifacts/phase4f_natural_token_diagnostic"
    for name in (
        "extraction_manifest.json",
        "diagnostic_summary.json",
        "b1_compatibility_preflight.json",
        "eligibility_confirmation.json",
        "natural_probe_scores.json",
    ):
        if (summary_dir / name).is_file():
            result.ok(f"present {name}")
        else:
            result.fail(f"missing {name}")
    report = REPO_ROOT / "reports/phase4f_natural_token_diagnostic.md"
    if report.is_file():
        result.ok("report present")
    else:
        result.fail("missing report")

    if not (summary_dir / "diagnostic_summary.json").is_file():
        print(f"\n{len(result.passes)} passed, {len(result.failures)} failed")
        return 1

    man = json.loads((summary_dir / "extraction_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads(
        (summary_dir / "diagnostic_summary.json").read_text(encoding="utf-8")
    )
    preflight = json.loads(
        (summary_dir / "b1_compatibility_preflight.json").read_text(encoding="utf-8")
    )
    elig_c = json.loads(
        (summary_dir / "eligibility_confirmation.json").read_text(encoding="utf-8")
    )

    if preflight.get("passed") and float(preflight.get("min_cosine_all", 0)) >= 0.9999:
        result.ok(f"preflight min cosine {preflight['min_cosine_all']}")
    else:
        result.fail("preflight failed")
    if man.get("n_rows") == 1200 and man.get("full_response_generated") is False:
        result.ok("1200 natural-k1 rows; no full responses")
    else:
        result.fail("integrity incomplete")
    if man.get("n_c5_rows") == 0 and summary.get("c5_status") == "HOLD":
        result.ok("C5 HOLD")
    else:
        result.fail("C5 not held")
    if (
        summary.get("probes_retrained") is False
        and summary.get("eligibility_changed") is False
        and summary.get("full_responses_regenerated") is False
        and elig_c.get("unchanged") is True
    ):
        result.ok("no retrain / eligibility change / regen")
    else:
        result.fail("mutation flags")
    if summary.get("n_bootstrap", 0) >= 5000:
        result.ok(f"n_bootstrap={summary.get('n_bootstrap')}")
    else:
        result.fail("bootstrap < 5000")

    for name, expected in (("C3_vs_C2", ELIG_C3_C2), ("C3_vs_C4", ELIG_C3_C4)):
        if elig_c["contrasts"][name]["paired_ids_sha256"] == expected:
            result.ok(f"{name} eligibility unchanged")
        else:
            result.fail(f"{name} eligibility drift")

    for name in ("C3_vs_C2", "C3_vs_C4"):
        c = summary.get("contrasts", {}).get(name, {})
        if "natural" in c and "auroc" in c["natural"]:
            result.ok(f"{name} natural metrics present")
        else:
            result.fail(f"missing {name} natural metrics")

    digest = hashlib.sha256(
        (REPO_ROOT / "artifacts/phase4_models/probe_l12_k1.npz").read_bytes()
    ).hexdigest()
    if digest == FROZEN_PROBE_L12_K1:
        result.ok("frozen L12/k1 probe unchanged")
    else:
        result.fail("probe hash drifted")

    tracked = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "ls-files"], text=True
    ).splitlines()
    for path in tracked:
        if "activations_l12_natural_k1.safetensors" in path:
            result.fail(f"raw activations tracked: {path}")
    result.ok("raw activations not tracked")

    print()
    print(f"{len(result.passes)} passed, {len(result.failures)} failed")
    if result.failures:
        for f in result.failures:
            print(f"  - {f}")
        return 1
    print("Phase 4F natural-token diagnostic validation OK.")
    print("NO PROBES WERE RETRAINED OR RECALIBRATED.")
    print("NO ELIGIBILITY SETS WERE CHANGED.")
    print("NO FULL PHASE 4 RESPONSES WERE REGENERATED.")
    print("NO CAUSAL INTERVENTIONS WERE PERFORMED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
