#!/usr/bin/env python3
"""Evaluate Phase 5A pilot outputs (deterministic; no probes/activations)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase5_behavior import (  # noqa: E402
    evaluate_row,
    summarize_pilot_behavior,
)
from pre_output_physiology.phase5_conditions import CONDITION_ORDER  # noqa: E402
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_report(summary: dict, manifest: dict, path: Path) -> None:
    gates = summary["gates"]
    lines = [
        "# Phase 5A behavior pilot report",
        "",
        f"**Run ID:** `{manifest.get('run_id')}`  ",
        f"**Git SHA:** `{manifest.get('git_commit')}`  ",
        f"**Neutral prefix token ID:** `{manifest.get('neutral_prefix_token_id')}`  ",
        "",
        "## Scope",
        "",
        "- 64 pilot prompts (32 base × S2/S3)",
        "- Greedy BF16 Mistral-7B-Instruct-v0.2",
        "- Deterministic one-line `Response <STATE>` parser",
        "- No activations, probe fitting/scoring, or locked-final generation",
        "",
        f"**All operational gates passed:** `{gates['all_operational_gates_pass']}`  ",
        f"**S2∩S3 paired valid:** `{summary['n_s2_s3_paired_valid']}` / 32  ",
        "",
        "## Per-condition",
        "",
        "| Condition | N | Schema | First-token | Valid |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for cid in CONDITION_ORDER:
        c = summary["per_condition"][cid]
        lines.append(
            f"| {cid} | {c['n_total']} | {c['schema_compliant']} | "
            f"{c['first_token_id_compliant']} | {c['behaviorally_valid']} |"
        )
    lines.extend(["", "## Failure reasons", ""])
    for cid in CONDITION_ORDER:
        lines.append(f"### {cid}")
        reasons = summary["per_condition"][cid]["failure_reason_counts"]
        if not reasons:
            lines.append("- (none)")
        else:
            for reason, count in sorted(reasons.items(), key=lambda x: (-x[1], x[0])):
                lines.append(f"- `{reason}`: {count}")
        lines.append("")
        lines.append("First-token ID counts:")
        for tid, cnt in sorted(
            summary["per_condition"][cid]["first_token_id_counts"].items(),
            key=lambda x: -x[1],
        ):
            lines.append(f"- `{tid}`: {cnt}")
        lines.append("")
    lines.extend(
        [
            "## Guarantees",
            "",
            "NO PHASE 5 ACTIVATIONS WERE COLLECTED.  ",
            "NO PHASE 5 PROBES WERE FIT OR SCORED.  ",
            "LOCKED GENERALIZATION FAMILIES WERE NOT RUN THROUGH THE MODEL.  ",
            "NO CAUSAL INTERVENTIONS WERE PERFORMED.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--summary-dir",
        default=str(REPO_ROOT / "artifacts/phase5a_pilot"),
    )
    parser.add_argument(
        "--report-path",
        default=str(REPO_ROOT / "reports/phase5a_pilot.md"),
    )
    parser.add_argument(
        "--expected-first-token-id",
        type=int,
        default=None,
        help="If set, require first generated token ID equality",
    )
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    raw = run_dir / "pilot_outputs.jsonl"
    man_path = run_dir / "pilot_generation_manifest.json"
    rows = _load_jsonl(raw)
    if len(rows) != 64:
        raise SystemExit(f"expected 64 outputs, got {len(rows)}")
    if any(r.get("split") != "pilot" for r in rows):
        raise SystemExit("non-pilot rows")
    if any(r.get("activation_extracted") for r in rows):
        raise SystemExit("activations flagged")
    matrix = json.loads(
        (REPO_ROOT / "artifacts/phase5a_design/condition_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    expected_ft = args.expected_first_token_id
    if expected_ft is None:
        expected_ft = int(matrix["neutral_prefix_token_id"])
    # Soft: evaluate both with and without first-token requirement for reporting.
    eval_strict = [
        evaluate_row(r, expected_first_token_id=expected_ft) for r in rows
    ]
    eval_schema = [evaluate_row(r, expected_first_token_id=None) for r in rows]
    summary = summarize_pilot_behavior(eval_strict)
    summary["schema_only_without_first_token_gate"] = summarize_pilot_behavior(
        eval_schema
    )
    summary["expected_first_token_id"] = expected_ft

    run_manifest = json.loads(man_path.read_text(encoding="utf-8"))
    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    commit_manifest = {
        **{k: v for k, v in run_manifest.items() if k != "raw_outputs_path"},
        "raw_outputs_gitignored": True,
        "raw_outputs_path_gitignored": run_manifest.get("raw_outputs_path"),
        "evaluated_at": utc_now_iso(),
        "expected_first_token_id": expected_ft,
        "activations_collected": False,
        "probes_fit_or_scored": False,
        "locked_final_families_run": False,
    }
    write_json(summary_dir / "pilot_generation_manifest.json", commit_manifest)
    write_json(
        summary_dir / "pilot_behavior_summary.json",
        {
            "created_at": utc_now_iso(),
            "run_id": run_manifest["run_id"],
            "summary": summary,
            "eval_rows": eval_strict,
        },
    )
    _write_report(summary, commit_manifest, Path(args.report_path))
    print(
        json.dumps(
            {
                "run_id": run_manifest["run_id"],
                "all_gates_pass": summary["gates"]["all_operational_gates_pass"],
                "n_paired": summary["n_s2_s3_paired_valid"],
                "per_condition_valid": {
                    cid: summary["per_condition"][cid]["behaviorally_valid"]
                    for cid in CONDITION_ORDER
                },
                "schema_only_gates": summary["schema_only_without_first_token_gate"][
                    "gates"
                ]["all_operational_gates_pass"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
