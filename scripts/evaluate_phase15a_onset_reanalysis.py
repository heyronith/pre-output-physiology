#!/usr/bin/env python3
"""Phase 15A: proposition-onset reanalysis of frozen Phase-14 calibration (no model calls)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

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
    STATUS_HOLD,
    STATUS_PROMISING,
    TEMPERATURE_GRID,
    annotate_continuation,
    coverage_audit_vs_phase14_assignment,
    cross_tab_validity,
    policy_by_onset_category,
    rates_by_temperature,
    recovery_vs_phase14,
    replication_worthy,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

RAW_PATH = (
    REPO_ROOT
    / "artifacts/runs"
    / PHASE14_RUN_ID
    / "calibration_continuations.jsonl"
)


def write_report(summary: dict, path: Path) -> None:
    rates = summary["rates_by_temperature"]
    worth = summary["replication"]
    lines = [
        "# Phase 15A — same-prompt proposition-onset reanalysis",
        "",
        f"**Source run:** `{summary['phase14_run_id']}`  ",
        f"**Raw continuations SHA256:** `{summary['raw_continuations_sha256']}`  ",
        f"**Annotation rule:** `{summary['rule_version']}` / `{summary['rule_hash']}`  ",
        f"**Status:** `{summary['status']}`  ",
        "**Phase 14A status (unchanged):** `phase14a_same_prompt_trajectory_calibration_hold`  ",
        "",
        "Calibration-data reanalysis only. Phase-15 validity admits exact candidate "
        "mentions without an assignment proposition (`candidate_mention_only`). "
        "No T* selection; no final prompts; no model calls.",
        "",
        "## Onset category counts",
        "",
        "| Category | n |",
        "| --- | ---: |",
    ]
    for cat, n in summary["onset_category_counts"].items():
        lines.append(f"| `{cat}` | {n} |")
    lines += [
        "",
        "## Old vs new validity",
        "",
        "| Cross-tab | n |",
        "| --- | ---: |",
    ]
    for k, n in summary["validity_cross_tab"].items():
        lines.append(f"| `{k}` | {n} |")
    lines += [
        "",
        "## Per-temperature Phase-15-valid feasibility",
        "",
        "| T | Valid frac | Assign leak | Record frac | Alternate frac | "
        "≥1/class | ≥3/class | Median valid/prompt | Replication-worthy |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    worthy = set(worth["replication_worthy_temperatures"])
    for t in TEMPERATURE_GRID:
        r = rates[str(t)]
        lines.append(
            f"| {t} | {r['valid_fraction']:.3f} | "
            f"{r['assignment_proposition_leak_rate']:.3f} | "
            f"{r['record_fraction']:.3f} | {r['alternate_fraction']:.3f} | "
            f"{r['n_prompt_groups_ge1_each_class']}/32 | "
            f"{r['n_prompt_groups_ge3_each_class']}/32 | "
            f"{r['median_valid_continuations_per_prompt']:.1f} | "
            f"{float(t) in worthy} |"
        )
    lines += [
        "",
        f"**Replication-worthy temperatures:** "
        f"{worth['replication_worthy_temperatures']}  ",
        "**Selects T\\*:** false  ",
        "",
        "## Newly recovered vs Phase-14 validity",
        "",
        "| T | Newly valid | New record | New alternate | Groups ↑ alternate | "
        "Groups newly with alternate |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for t in TEMPERATURE_GRID:
        g = summary["recovery_vs_phase14"][str(t)]
        lines.append(
            f"| {t} | {g['newly_valid_continuations']} | {g['newly_valid_record']} | "
            f"{g['newly_valid_alternate']} | "
            f"{g['prompt_groups_with_increased_alternate']} | "
            f"{g['prompt_groups_newly_gaining_any_alternate']} |"
        )
    lines += [
        "",
        "## Final policy rates by Stage-1 onset category (descriptive)",
        "",
    ]
    for t in TEMPERATURE_GRID:
        lines.append(f"### T={t}")
        lines.append("")
        lines.append("| Category | n | Record frac | Alternate frac |")
        lines.append("| --- | ---: | ---: | ---: |")
        for cat, pol in summary["policy_by_onset_category"][str(t)].items():
            rf = pol["record_fraction"]
            af = pol["alternate_fraction"]
            lines.append(
                f"| `{cat}` | {pol['n']} | "
                f"{'—' if rf is None else f'{rf:.3f}'} | "
                f"{'—' if af is None else f'{af:.3f}'} |"
            )
        lines.append("")
    lines += [GUARANTEE, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw",
        default=str(RAW_PATH),
        help="Frozen Phase-14 calibration_continuations.jsonl",
    )
    ap.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "artifacts/phase15a_onset_reanalysis"),
    )
    ap.add_argument(
        "--report",
        default=str(REPO_ROOT / "reports/phase15a_onset_reanalysis.md"),
    )
    args = ap.parse_args()
    raw_path = Path(args.raw)
    raw_bytes = raw_path.read_bytes()
    raw_sha = hashlib.sha256(raw_bytes).hexdigest()
    if raw_sha != PHASE14_RAW_CONTINUATIONS_SHA256:
        raise SystemExit(
            f"raw continuation SHA mismatch: {raw_sha} != {PHASE14_RAW_CONTINUATIONS_SHA256}"
        )
    rows = [
        json.loads(x)
        for x in raw_bytes.decode("utf-8").splitlines()
        if x.strip()
    ]
    if len(rows) != N_CONTINUATIONS:
        raise SystemExit(f"expected {N_CONTINUATIONS} rows, got {len(rows)}")
    if any(not r["prompt_group_id"].startswith("calib_") for r in rows):
        raise SystemExit("non-calibration prompt in source run")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Freeze annotation rules BEFORE recomputing behavior statistics.
    write_json(out / "annotation_rules.json", RULE_MANIFEST)
    if RULE_HASH != RULE_MANIFEST["rule_hash"]:
        raise SystemExit("rule hash inconsistency")

    annotated = [annotate_continuation(r) for r in rows]
    for r in annotated:
        if r["rule_hash"] != RULE_HASH:
            raise SystemExit("per-row rule hash drift")

    rates = rates_by_temperature(annotated)
    worth = replication_worthy(rates)
    status = (
        STATUS_PROMISING
        if worth["replication_worthy_temperatures"]
        else STATUS_HOLD
    )
    cov = coverage_audit_vs_phase14_assignment(annotated)
    if not cov["phase15_covers_all_phase14_assignments"]:
        raise SystemExit(f"Phase-15 missed Phase-14 assignments: {cov}")

    summary = {
        "created_at": utc_now_iso(),
        "phase": "phase15a",
        "phase14_run_id": PHASE14_RUN_ID,
        "phase14_status_unchanged": "phase14a_same_prompt_trajectory_calibration_hold",
        "raw_continuations_sha256": raw_sha,
        "n_continuations": len(annotated),
        "rule_version": RULE_VERSION,
        "rule_hash": RULE_HASH,
        "onset_category_counts": dict(
            Counter(r["onset_category"] for r in annotated)
        ),
        "validity_cross_tab": cross_tab_validity(annotated),
        "rates_by_temperature": {str(k): v for k, v in rates.items()},
        "replication": worth,
        "recovery_vs_phase14": recovery_vs_phase14(annotated),
        "policy_by_onset_category": policy_by_onset_category(annotated),
        "coverage_audit": cov,
        "status": status,
        "selects_t_star": False,
        "final_model_calls": 0,
        "locked_model_calls": 0,
        "model_calls": 0,
        "activations_collected": False,
        "probe_scored": False,
        "resampling": False,
        "temperature_grid_extended": False,
        "prompt_changed": False,
        "guarantee": GUARANTEE,
    }
    write_json(out / "reanalysis_summary.json", {**summary, "annotated_rows": annotated})
    write_json(
        out / "reanalysis_manifest.json",
        {
            "phase14_run_id": PHASE14_RUN_ID,
            "raw_continuations_sha256": raw_sha,
            "rule_version": RULE_VERSION,
            "rule_hash": RULE_HASH,
            "n_continuations": len(annotated),
            "status": status,
            "model_calls": 0,
            "activations_collected": False,
        },
    )
    write_report(summary, Path(args.report))
    print(
        json.dumps(
            {
                "status": status,
                "rule_hash": RULE_HASH,
                "onset_category_counts": summary["onset_category_counts"],
                "validity_cross_tab": summary["validity_cross_tab"],
                "replication_worthy_temperatures": worth[
                    "replication_worthy_temperatures"
                ],
                "rates": {
                    str(t): {
                        k: rates[t][k]
                        for k in (
                            "valid_fraction",
                            "assignment_proposition_leak_rate",
                            "record_fraction",
                            "alternate_fraction",
                            "n_prompt_groups_ge1_each_class",
                            "n_prompt_groups_ge3_each_class",
                        )
                    }
                    for t in TEMPERATURE_GRID
                },
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
