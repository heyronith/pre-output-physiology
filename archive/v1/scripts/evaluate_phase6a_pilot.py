#!/usr/bin/env python3
"""Evaluate Phase 6A factorial pilot outputs (deterministic; no probes/activations)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase6_behavior import (  # noqa: E402
    evaluate_row,
    summarize_pilot_behavior,
)
from pre_output_physiology.phase6_conditions import (  # noqa: E402
    CONDITION_ORDER,
    N_PILOT_BASE_SCENARIOS,
    N_PILOT_PROMPTS,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

EXPECTED_RESPONSE = {
    "F00": "record",
    "F10": "record",
    "F01": "alternate",
    "F11": "alternate",
}


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_report(summary: dict, manifest: dict, path: Path) -> None:
    gates = summary["gates"]
    revision = manifest.get("prompt_template_revision")
    lines = [
        f"# Phase 6A factorial behavior pilot (revision {revision})",
        "",
        f"**Run ID:** `{manifest.get('run_id')}`  ",
        f"**Git SHA:** `{manifest.get('git_commit')}`  ",
        f"**Pilot prompt SHA256:** `{manifest.get('pilot_prompt_text_sha256')}`  ",
        f"**Response prefix token ID:** `{manifest.get('neutral_prefix_token_id')}`  ",
        f"**GPU wall / cost:** `{manifest.get('wall_seconds', 0):.1f}s` / "
        f"`${manifest.get('estimated_cost_usd', 0):.4f}`  ",
        "",
        "## Scope",
        "",
        f"- {N_PILOT_PROMPTS} pilot prompts ({N_PILOT_BASE_SCENARIOS} base × F00/F10/F01/F11)",
        "- Greedy BF16 Mistral-7B-Instruct-v0.2 (pinned revision)",
        "- Deterministic one-line `Response <STATE>` parser; valid = communication target",
        "- No activations, no frozen-probe scoring, no final-dataset generation",
        "",
        f"**All operational gates passed:** `{gates['all_operational_gates_pass']}`  ",
        f"**All-four paired-valid:** `{summary['n_all_four_valid']}` / "
        f"{N_PILOT_BASE_SCENARIOS} (gate ≥ {gates['all_four_min']})  ",
        f"**Valid runs all first-token 12107:** `{gates['valid_runs_first_token_ok']}`  ",
        "",
        "## Per-condition",
        "",
        "| Condition | Expected | N | Schema | First-token | Valid | Gate |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for cid in CONDITION_ORDER:
        c = summary["per_condition"][cid]
        lines.append(
            f"| {cid} | {EXPECTED_RESPONSE[cid]} | {c['n_total']} | {c['schema_compliant']} | "
            f"{c['first_token_id_compliant']} | {c['behaviorally_valid']} | "
            f"{c['behavior_gate_pass']} |"
        )
    lines.extend(["", "## Failure reasons and first-token distributions", ""])
    for cid in CONDITION_ORDER:
        c = summary["per_condition"][cid]
        lines.append(f"### {cid}")
        if not c["failure_reason_counts"]:
            lines.append("- (no failures)")
        for reason, count in sorted(
            c["failure_reason_counts"].items(), key=lambda x: (-x[1], x[0])
        ):
            lines.append(f"- `{reason}`: {count}")
        lines.append("")
        lines.append("First-token ID counts:")
        for tid, cnt in sorted(c["first_token_id_counts"].items(), key=lambda x: -x[1]):
            lines.append(f"- `{tid}`: {cnt}")
        lines.append("")
    lines.extend(
        [
            "## Guarantees",
            "",
            "NO PHASE 6 ACTIVATIONS WERE COLLECTED.  ",
            "THE FROZEN PHASE-5 PROBE WAS NOT SCORED, RETRAINED, RECALIBRATED, OR MODIFIED.  ",
            "NO FINAL PHASE-6 SCENARIOS WERE RUN THROUGH THE MODEL.  ",
            "NO CAUSAL INTERVENTIONS WERE PERFORMED.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--summary-dir", default=str(REPO_ROOT / "artifacts/phase6a_pilot")
    )
    parser.add_argument(
        "--report-path", default=str(REPO_ROOT / "reports/phase6a_pilot.md")
    )
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    rows = _load_jsonl(run_dir / "pilot_outputs.jsonl")
    if len(rows) != N_PILOT_PROMPTS:
        raise SystemExit(f"expected {N_PILOT_PROMPTS} outputs, got {len(rows)}")
    if any(r.get("split") != "pilot" for r in rows):
        raise SystemExit("non-pilot rows")
    if any(r.get("activation_extracted") or r.get("probe_scored") for r in rows):
        raise SystemExit("activation/probe flags set")
    matrix = json.loads(
        (REPO_ROOT / "artifacts/phase6a_design/condition_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    expected_ft = int(matrix["neutral_prefix_token_id"])
    eval_rows = [evaluate_row(r, expected_first_token_id=expected_ft) for r in rows]
    summary = summarize_pilot_behavior(eval_rows, expected_first_token_id=expected_ft)
    summary["expected_first_token_id"] = expected_ft

    run_manifest = json.loads(
        (run_dir / "pilot_generation_manifest.json").read_text(encoding="utf-8")
    )
    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    commit_manifest = {
        **{k: v for k, v in run_manifest.items() if k != "raw_outputs_path"},
        "raw_outputs_gitignored": True,
        "raw_outputs_path_gitignored": run_manifest.get("raw_outputs_path"),
        "evaluated_at": utc_now_iso(),
        "expected_first_token_id": expected_ft,
    }
    write_json(summary_dir / "pilot_generation_manifest.json", commit_manifest)
    write_json(
        summary_dir / "pilot_behavior_summary.json",
        {
            "created_at": utc_now_iso(),
            "run_id": run_manifest["run_id"],
            "summary": summary,
            "eval_rows": eval_rows,
        },
    )
    _write_report(summary, commit_manifest, Path(args.report_path))
    print(
        json.dumps(
            {
                "run_id": run_manifest["run_id"],
                "all_gates_pass": summary["gates"]["all_operational_gates_pass"],
                "n_all_four_valid": summary["n_all_four_valid"],
                "per_condition_valid": {
                    cid: summary["per_condition"][cid]["behaviorally_valid"]
                    for cid in CONDITION_ORDER
                },
                "failure_reasons": {
                    cid: summary["per_condition"][cid]["failure_reason_counts"]
                    for cid in CONDITION_ORDER
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
