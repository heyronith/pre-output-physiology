#!/usr/bin/env python3
"""Validate preserved Phase 4B revision-0 pilot artifacts (historical)."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

REV0_RUN_ID = "phase4b_pilot_20260925T165130Z_ee4d615c"
REV0_EXPECTED_TOKEN_ID = 2963  # historical incorrect freeze used during rev0


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
    exp = yaml.safe_load(
        (REPO_ROOT / "configs/experiments/phase4_specificity.yaml").read_text(
            encoding="utf-8"
        )
    )
    auth = exp.get("authorizations", {})
    status = exp.get("status", "")
    if status.startswith("phase4e_") or status.startswith("phase4d_"):
        if auth.get("final_generation_authorized") is True:
            result.ok(f"final_generation_authorized=true ({status.split('_')[0]})")
        else:
            result.fail("final_generation_authorized must be true in phase4d/e")
    elif auth.get("final_generation_authorized") is False:
        result.ok("final_generation_authorized=false")
    else:
        result.fail("final generation authorized")

    man_path = REPO_ROOT / "artifacts/phase4b_pilot/pilot_generation_manifest.json"
    beh_path = REPO_ROOT / "artifacts/phase4b_pilot/pilot_behavior_summary.json"
    report_path = REPO_ROOT / "reports/phase4b_pilot.md"
    raw_path = REPO_ROOT / "artifacts/runs" / REV0_RUN_ID / "pilot_outputs.jsonl"

    for path, label in (
        (man_path, "rev0 generation manifest"),
        (beh_path, "rev0 behavior summary"),
        (report_path, "rev0 report"),
        (raw_path, "rev0 raw outputs"),
    ):
        if path.is_file():
            result.ok(f"{label} preserved")
        else:
            result.fail(f"missing {label}: {path}")

    if not man_path.is_file() or not beh_path.is_file() or not raw_path.is_file():
        print(f"\n{len(result.passes)} passed, {len(result.failures)} failed")
        return 1

    man = json.loads(man_path.read_text(encoding="utf-8"))
    beh = json.loads(beh_path.read_text(encoding="utf-8"))
    if man.get("run_id") == REV0_RUN_ID:
        result.ok("rev0 run_id preserved")
    else:
        result.fail(f"rev0 run_id drift: {man.get('run_id')}")
    if man.get("pilot_revision", 0) == 0:
        result.ok("pilot_revision=0 recorded")
    else:
        result.fail("rev0 pilot_revision not 0")
    if man.get("n_pilot_outputs") == 144 and man.get("n_final_outputs") == 0:
        result.ok("rev0 has 144 pilot / 0 final outputs")
    else:
        result.fail("rev0 output counts drift")
    if man.get("activations_collected") is False and man.get("probe_scores_computed") is False:
        result.ok("rev0 no activations/probes")
    else:
        result.fail("rev0 activation/probe contamination")

    rows = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) == 144 and all(r.get("split") == "pilot" for r in rows):
        result.ok("rev0 raw 144 pilot rows intact")
    else:
        result.fail("rev0 raw rows corrupted")
    if all(r.get("first_generated_token_id") == 12107 for r in rows):
        result.ok("rev0 observed first-token ID was 12107 (diagnosis preserved)")
    else:
        result.fail("rev0 first-token diagnosis drift")
    # Historical freeze used 2963 as expected criterion in that eval.
    if man.get("common_first_token_id") == REV0_EXPECTED_TOKEN_ID:
        result.ok("rev0 manifest records historical expected ID 2963")
    else:
        result.fail("rev0 historical expected token ID missing")

    if beh.get("summary", {}).get("n_eval_rows") == 144:
        result.ok("rev0 behavior summary intact")
    else:
        result.fail("rev0 behavior summary drift")

    print()
    print(f"{len(result.passes)} passed, {len(result.failures)} failed")
    if result.failures:
        for f in result.failures:
            print(f"  - {f}")
        return 1
    print("Phase 4B revision-0 preservation OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
