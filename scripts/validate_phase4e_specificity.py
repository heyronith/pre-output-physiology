#!/usr/bin/env python3
"""Phase 4E specificity integrity checks."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

FROZEN_PROBE_L12_K1 = "3bbf0c78e97d8ff14b028592e6e6ab3215651e973e0be1f004ce66e2b9b7aa65"
FROZEN_PROBE_L12_K0 = "dc14569a4e229cadc55fd3f6e6fef4e33f4eac4f58d00e83916238d5274f4f29"
ELIG_C3_C2 = "07dcb4567aaf98cea443327bca2cc336f082ab035c645ccd4c4b28dcc97503b2"
ELIG_C3_C4 = "c03c70d06cd5e487621781b8d0a5aa65354857c578a0842b7e216c894e045d22"
CONTRASTS = (
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
        "validate_phase4d_final_behavior.py",
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
    for did in ("D061", "D062", "D063"):
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
    if status == "phase4e_specificity_complete_awaiting_audit":
        result.ok(f"status {status}")
    else:
        result.fail(f"unexpected status {status}")
    auth = exp.get("authorizations", {})
    if auth.get("activation_extraction_authorized") is True:
        result.ok("activation_extraction_authorized=true")
    else:
        result.fail("activation_extraction_authorized must be true")
    if auth.get("probe_scoring_authorized") is True:
        result.ok("probe_scoring_authorized=true")
    else:
        result.fail("probe_scoring_authorized must be true")
    if auth.get("causal_intervention_authorized") is False:
        result.ok("causal_intervention_authorized=false")
    else:
        result.fail("causal must remain unauthorized")

    summary_dir = REPO_ROOT / "artifacts/phase4e_specificity"
    for name in (
        "extraction_manifest.json",
        "specificity_summary.json",
        "b1_compatibility_preflight.json",
        "eligibility_confirmation.json",
        "probe_scores.json",
    ):
        path = summary_dir / name
        if path.is_file():
            result.ok(f"present {name}")
        else:
            result.fail(f"missing {name}")
    report = REPO_ROOT / "reports/phase4e_specificity.md"
    if report.is_file():
        result.ok("report present")
    else:
        result.fail("missing report")

    if not (summary_dir / "specificity_summary.json").is_file():
        print(f"\n{len(result.passes)} passed, {len(result.failures)} failed")
        return 1

    man = json.loads((summary_dir / "extraction_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads(
        (summary_dir / "specificity_summary.json").read_text(encoding="utf-8")
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
        result.fail("preflight failed or below gate")

    if man.get("n_rows") == 1200 and man.get("prefix_identity_count") == 1200:
        result.ok("1200/1200 prefix identity")
    else:
        result.fail("activation integrity incomplete")
    if man.get("n_c5_rows") == 0 and summary.get("c5_status") == "HOLD":
        result.ok("C5 HOLD / zero C5 rows")
    else:
        result.fail("C5 not held")
    if man.get("future_response_tokens_present") is False:
        result.ok("no future response tokens")
    else:
        result.fail("future tokens flagged")
    if man.get("condition_labels_in_extractor") is False:
        result.ok("extractor label-free")
    else:
        result.fail("extractor had labels")
    if (
        man.get("probes_retrained") is False
        and summary.get("probes_retrained") is False
        and summary.get("probes_recalibrated") is False
    ):
        result.ok("no retrain/recalibrate")
    else:
        result.fail("probe mutation flagged")
    if (
        man.get("eligibility_changed") is False
        and summary.get("eligibility_changed") is False
        and elig_c.get("unchanged") is True
    ):
        result.ok("eligibility unchanged")
    else:
        result.fail("eligibility changed")
    if man.get("causal_interventions_performed") is False:
        result.ok("no causal interventions")
    else:
        result.fail("causal flagged")

    if summary.get("n_bootstrap", 0) >= 5000:
        result.ok(f"n_bootstrap={summary.get('n_bootstrap')}")
    else:
        result.fail("bootstrap < 5000")

    # Eligibility hashes
    src_elig = json.loads(
        (
            REPO_ROOT
            / "artifacts/phase4d_final_behavior/frozen_contrast_eligibility.json"
        ).read_text(encoding="utf-8")
    )
    for name, expected in (("C3_vs_C2", ELIG_C3_C2), ("C3_vs_C4", ELIG_C3_C4)):
        digest = src_elig["contrasts"][name]["paired_ids_sha256"]
        if digest == expected and elig_c["contrasts"][name]["paired_ids_sha256"] == expected:
            result.ok(f"{name} eligibility hash unchanged")
        else:
            result.fail(f"{name} eligibility hash drift")

    for endpoint in ("L12_k1", "L12_k0"):
        block = summary.get("contrasts", {}).get(endpoint, {})
        for name in CONTRASTS:
            if name in block and "auroc" in block[name]:
                result.ok(f"{endpoint} {name} metrics present")
            else:
                result.fail(f"missing {endpoint} {name}")

    # Frozen probes
    for name, expected in (
        ("probe_l12_k1.npz", FROZEN_PROBE_L12_K1),
        ("probe_l12_k0.npz", FROZEN_PROBE_L12_K0),
    ):
        path = REPO_ROOT / "artifacts/phase4_models" / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest == expected:
            result.ok(f"frozen {name} unchanged")
        else:
            result.fail(f"frozen {name} drifted")

    # Raw activations not tracked
    tracked = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "ls-files"], text=True
    ).splitlines()
    for path in tracked:
        if "activations_l12_k0_k1.safetensors" in path:
            result.fail(f"raw activations tracked: {path}")
    result.ok("raw activations not tracked")

    print()
    print(f"{len(result.passes)} passed, {len(result.failures)} failed")
    if result.failures:
        for f in result.failures:
            print(f"  - {f}")
        return 1
    print("Phase 4E specificity validation OK.")
    print("NO PROBES WERE RETRAINED OR RECALIBRATED.")
    print("NO ELIGIBILITY SETS WERE CHANGED.")
    print("C5 REMAINED ON HOLD.")
    print("NO CAUSAL INTERVENTIONS WERE PERFORMED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
