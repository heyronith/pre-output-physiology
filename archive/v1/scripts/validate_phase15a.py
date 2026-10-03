#!/usr/bin/env python3
"""Phase 15A proposition-onset reanalysis integrity checks."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase15_onset import (  # noqa: E402
    GUARANTEE,
    N_CONTINUATIONS,
    PHASE14_RAW_CONTINUATIONS_SHA256,
    PHASE14_RUN_ID,
    RULE_HASH,
    RULE_MANIFEST,
    RULE_VERSION,
    STATUS_AUTHORIZED,
    STATUS_HOLD,
    STATUS_PROMISING,
    TEMPERATURE_GRID,
    annotate_continuation,
    coverage_audit_vs_phase14_assignment,
    rates_by_temperature,
    replication_worthy,
    rule_manifest,
)

STATUSES = {STATUS_AUTHORIZED, STATUS_PROMISING, STATUS_HOLD}
CFG = REPO_ROOT / "configs/experiments/phase15_same_prompt_onset_reanalysis.yaml"
ART = REPO_ROOT / "artifacts/phase15a_onset_reanalysis"
RAW = (
    REPO_ROOT
    / "artifacts/runs"
    / PHASE14_RUN_ID
    / "calibration_continuations.jsonl"
)
P14_CFG = REPO_ROOT / "configs/experiments/phase14_same_prompt_trajectory.yaml"


def main() -> int:
    fails: list[str] = []
    n_pass = 0

    def check(cond: bool, msg: str) -> None:
        nonlocal n_pass
        if cond:
            n_pass += 1
            print(f"PASS  {msg}")
        else:
            fails.append(msg)
            print(f"FAIL  {msg}")

    cfg = yaml.safe_load(CFG.read_text("utf-8"))
    status = cfg["status"]
    check(status in STATUSES, f"status {status}")
    auth = cfg["authorizations"]
    for k in (
        "model_calls_authorized",
        "final_model_calls_authorized",
        "locked_family_model_calls_authorized",
        "activation_extraction_authorized",
        "probe_fitting_authorized",
        "phase5_probe_scoring_authorized",
        "resampling_authorized",
        "temperature_extension_authorized",
        "prompt_changes_authorized",
        "causal_intervention_authorized",
    ):
        check(auth.get(k) is False, f"{k}=false")
    pre = status == STATUS_AUTHORIZED
    if pre:
        check(auth.get("reanalysis_authorized") is True, "pre-run reanalysis authorized")
    else:
        check(auth.get("reanalysis_authorized") is False, "post-run reanalysis closed")

    check(
        cfg["phase14_merge_sha"] == "9ab4e9a5b805674bddbc5922a3efb330b9910cef",
        "phase14 merge",
    )
    p14 = yaml.safe_load(P14_CFG.read_text("utf-8"))
    check(
        p14["status"] == "phase14a_same_prompt_trajectory_calibration_hold",
        "Phase 14A remains HOLD",
    )
    dec = (REPO_ROOT / "docs/decision_log.md").read_text("utf-8")
    if pre:
        check("### D121" in dec, "D121")
    else:
        check(all(f"### {d}" in dec for d in ("D121", "D122")), "D121-D122")
    check(list(cfg["design"]["temperature_grid"]) == list(TEMPERATURE_GRID), "T grid")
    check(cfg["design"]["phase14_run_id"] == PHASE14_RUN_ID, "source run id")
    check(
        cfg["design"]["raw_continuations_sha256"] == PHASE14_RAW_CONTINUATIONS_SHA256,
        "raw sha in config",
    )

    check(RAW.is_file(), "raw continuations present")
    raw_sha = hashlib.sha256(RAW.read_bytes()).hexdigest()
    check(raw_sha == PHASE14_RAW_CONTINUATIONS_SHA256, "raw sha verifies")
    rows = [
        json.loads(x) for x in RAW.read_text("utf-8").splitlines() if x.strip()
    ]
    check(len(rows) == N_CONTINUATIONS, "1536 continuations")
    check(
        all(r["prompt_group_id"].startswith("calib_") for r in rows),
        "calibration-only rows",
    )

    fresh = rule_manifest()
    check(fresh["rule_hash"] == RULE_HASH == RULE_MANIFEST["rule_hash"], "rule hash stable")
    check(cfg["design"]["rule_version"] == RULE_VERSION, "rule version in config")

    if not pre:
        rules = json.loads((ART / "annotation_rules.json").read_text("utf-8"))
        check(rules["rule_hash"] == RULE_HASH, "frozen rules artifact")
        summary = json.loads((ART / "reanalysis_summary.json").read_text("utf-8"))
        check(len(summary["annotated_rows"]) == 1536, "1536 annotated")
        check(summary["raw_continuations_sha256"] == raw_sha, "summary raw sha")
        check(summary["rule_hash"] == RULE_HASH, "summary rule hash")
        check(
            summary["phase14_status_unchanged"]
            == "phase14a_same_prompt_trajectory_calibration_hold",
            "summary keeps Phase14 HOLD",
        )
        check(summary["model_calls"] == 0, "zero model calls")
        check(summary["selects_t_star"] is False, "no T* selection")
        # Recompute from raw
        ann = [annotate_continuation(r) for r in rows]
        rates = rates_by_temperature(ann)
        worth = replication_worthy(rates)
        check(
            worth["replication_worthy_temperatures"]
            == summary["replication"]["replication_worthy_temperatures"],
            "replication-worthy recomputes",
        )
        check(summary["status"] == status, "status matches summary")
        if status == STATUS_HOLD:
            check(
                not worth["replication_worthy_temperatures"],
                "hold ⇔ no replication-worthy T",
            )
        else:
            check(
                bool(worth["replication_worthy_temperatures"]),
                "promising ⇔ some replication-worthy T",
            )
        cov = coverage_audit_vs_phase14_assignment(ann)
        check(cov["phase15_covers_all_phase14_assignments"], "covers P14 assignments")
        check(
            cfg["reanalysis_result"]["status"] == status,
            "result status in config",
        )
        check((REPO_ROOT / "reports/phase15a_onset_reanalysis.md").is_file(), "report")
        report_txt = (REPO_ROOT / "reports/phase15a_onset_reanalysis.md").read_text("utf-8")
        check(GUARANTEE in report_txt, "guarantee")
        # No activation artifacts
        check(
            not [
                p
                for p in (REPO_ROOT / "artifacts").rglob("*")
                if p.is_file()
                and "phase15" in str(p)
                and p.suffix in {".safetensors", ".npz", ".pt", ".pth"}
            ],
            "no activations",
        )

    print()
    print(f"{n_pass} passed, {len(fails)} failed")
    for f in fails:
        print(f"  - {f}")
    if fails:
        return 1
    print("Phase 15A validation OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
