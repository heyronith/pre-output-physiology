#!/usr/bin/env python3
"""Local deterministic evaluation of Phase 4B pilot outputs (no probes/activations)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase4_behavior import (  # noqa: E402
    PILOT_BEHAVIOR_MIN_PER_CONDITION,
    PILOT_C2_C3_PAIRED_MIN,
    PILOT_FIRST_TOKEN_MIN_PER_CONDITION,
    evaluate_pilot_row,
    summarize_pilot_behavior,
)
from pre_output_physiology.phase4_conditions import (  # noqa: E402
    COMMON_FIRST_TOKEN,
    COMMON_FIRST_TOKEN_ID,
    CONDITION_ORDER,
)
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
    revision = manifest.get("pilot_revision", 0)
    kind = manifest.get("pilot_kind", "sampled")
    lines = [
        f"# Phase 4 pilot report (rev {revision}; kind={kind})",
        "",
        f"**Run ID:** `{manifest.get('run_id')}`  ",
        f"**Git SHA (generation):** `{manifest.get('git_commit')}`  ",
        f"**Pilot revision:** `{revision}`  ",
        f"**Pilot kind:** `{kind}`  ",
        "**Status:** operational compliance pilot only (D053/D058)  ",
        "",
        "## Scope",
        "",
        "- 144 pilot prompts only (24 base × 6 conditions)",
        "- Greedy BF16 Mistral-7B-Instruct-v0.2 generation",
        "- Deterministic behavior parser only",
        "- **No** activations, probe scores, or final-scenario generation",
        "",
        "## Generation",
        "",
        f"- Model revision: `{manifest.get('model_revision')}`",
        f"- Expected first-token ID: `{manifest.get('common_first_token_id')}`",
        f"- first_token_sampled: `{manifest.get('first_token_sampled')}`",
        f"- controlled_prefix_token_id: `{manifest.get('controlled_prefix_token_id')}`",
        f"- do_sample: `{manifest.get('do_sample')}`",
        "- max_continuation_tokens: "
        f"`{manifest.get('max_continuation_tokens', manifest.get('max_new_tokens'))}`",
        f"- GPU wall seconds: `{manifest.get('wall_seconds')}`",
        f"- Estimated cost USD: `{manifest.get('estimated_cost_usd')}`",
        "",
        "## Gates",
        "",
    ]
    if kind == "controlled_prefix":
        lines.append(
            "- Prefix integrity: 144/144 supplied token ID 12107 (engineering check)"
        )
    else:
        lines.append(
            f"- First-token ID {COMMON_FIRST_TOKEN_ID} ≥ "
            f"{PILOT_FIRST_TOKEN_MIN_PER_CONDITION}/24 per condition"
        )
    lines.extend(
        [
            f"- Behavioral validity ≥ {PILOT_BEHAVIOR_MIN_PER_CONDITION}/24 per condition",
            f"- C2∩C3 paired valid ≥ {PILOT_C2_C3_PAIRED_MIN}/24 base scenarios",
            "",
            f"**Prefix integrity count:** `{summary.get('prefix_integrity_count', 'n/a')}`  ",
            f"**All behavioral gates passed:** `{gates['all_operational_gates_pass']}`  ",
            "**Recommend freeze revision-1 prompts + controlled-prefix method:** "
            f"`{summary.get('recommend_freeze_revision1_templates')}`  ",
            "",
            "## Per-condition results",
            "",
            "| Condition | N | FT-ID | 3-line | MODE | FINAL | Valid |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for cid in CONDITION_ORDER:
        c = summary["per_condition"][cid]
        lines.append(
            f"| {cid} | {c['n_total']} | {c['first_token_id_compliant']} | "
            f"{c.get('exact_three_line_format', 'n/a')} | "
            f"{c.get('mode_compliant', 'n/a')} | "
            f"{c['valid_FINAL_marker']} | {c['behaviorally_valid']} |"
        )
    lines.extend(
        [
            "",
            f"**C2/C3 paired valid scenarios:** {summary['n_c2_c3_paired_valid']} / 24",
            "",
            "## Failure reasons",
            "",
        ]
    )
    for cid in CONDITION_ORDER:
        reasons = summary["per_condition"][cid]["failure_reason_counts"]
        lines.append(f"### {cid}")
        if not reasons:
            lines.append("- (none)")
        else:
            for reason, count in sorted(reasons.items(), key=lambda x: (-x[1], x[0])):
                lines.append(f"- `{reason}`: {count}")
        lines.append("")
    lines.extend(
        [
            "## Contamination / analysis guarantees",
            "",
            "THIS WAS THE SINGLE ALLOWED POST-PILOT TEMPLATE REVISION.  "
            if revision == 1
            else "",
            "NO PHASE 4 ACTIVATIONS WERE COLLECTED.  ",
            "NO PHASE 4 PROBE SCORES WERE COMPUTED.  ",
            "FINAL PHASE 4 SCENARIOS WERE NOT GENERATED.  ",
            "NO CAUSAL INTERVENTIONS WERE PERFORMED.",
            "",
        ]
    )
    # Drop empty string lines from conditional
    cleaned = [ln for ln in lines if ln is not None]
    path.write_text("\n".join(cleaned) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        required=True,
        help="artifacts/runs/<phase4b_pilot_...>/ containing pilot_outputs.jsonl",
    )
    parser.add_argument(
        "--pilot-revision",
        type=int,
        default=1,
        help="0 = historical; 1 = template revision re-pilot; use with --kind",
    )
    parser.add_argument(
        "--kind",
        choices=("sampled", "controlled_prefix"),
        default="sampled",
        help="sampled = free first-token pilots; controlled_prefix = Phase 4C",
    )
    parser.add_argument(
        "--summary-dir",
        default="",
    )
    parser.add_argument(
        "--report-path",
        default="",
    )
    args = parser.parse_args()
    revision = int(args.pilot_revision)
    kind = args.kind
    if args.summary_dir:
        summary_dir = Path(args.summary_dir)
    elif kind == "controlled_prefix":
        summary_dir = REPO_ROOT / "artifacts/phase4c_controlled_prefix_pilot"
    elif revision == 0:
        summary_dir = REPO_ROOT / "artifacts/phase4b_pilot"
    else:
        summary_dir = REPO_ROOT / "artifacts/phase4b_pilot_revision1"
    if args.report_path:
        report_path = Path(args.report_path)
    elif kind == "controlled_prefix":
        report_path = REPO_ROOT / "reports/phase4c_controlled_prefix_pilot.md"
    elif revision == 0:
        report_path = REPO_ROOT / "reports/phase4b_pilot.md"
    else:
        report_path = REPO_ROOT / "reports/phase4b_pilot_revision1.md"

    run_dir = Path(args.run_dir)
    raw = run_dir / "pilot_outputs.jsonl"
    run_manifest_path = run_dir / "pilot_generation_manifest.json"
    if not raw.is_file():
        raise SystemExit(f"missing {raw}")
    if not run_manifest_path.is_file():
        raise SystemExit(f"missing {run_manifest_path}")

    rows = _load_jsonl(raw)
    if len(rows) != 144:
        raise SystemExit(f"expected 144 outputs, got {len(rows)}")
    if any(r.get("split") != "pilot" for r in rows):
        raise SystemExit("non-pilot rows present")
    if any(r.get("activation_extracted") for r in rows):
        raise SystemExit("activation_extracted true in pilot outputs")
    if any(r.get("probe_scored") for r in rows):
        raise SystemExit("probe_scored true in pilot outputs")

    if kind == "controlled_prefix":
        bad = [
            r["example_id"]
            for r in rows
            if not (
                r.get("controlled_prefix_supplied") is True
                and r.get("first_token_sampled") is False
                and r.get("controlled_prefix_token_id") == 12107
                and r.get("first_generated_token_id") == 12107
                and isinstance(r.get("generated_token_ids"), list)
                and r["generated_token_ids"]
                and r["generated_token_ids"][0] == 12107
            )
        ]
        if bad:
            raise SystemExit(
                f"STOP: controlled-prefix integrity failed for {len(bad)} rows "
                f"(e.g. {bad[:3]})"
            )

    eval_rows = [evaluate_pilot_row(r) for r in rows]
    summary = summarize_pilot_behavior(eval_rows)
    if kind == "controlled_prefix":
        summary["prefix_integrity_count"] = 144
        summary["prefix_integrity_required"] = 144
        summary["first_token_sampled"] = False
        summary["controlled_prefix_token_id"] = 12107
        summary["k1_interpretation"] = "controlled-prefix k1"
        # First-token gate is integrity (all 12107 by construction), not sampling.
        summary["gates"]["first_token_gate_is_prefix_integrity"] = True
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))

    summary_dir.mkdir(parents=True, exist_ok=True)
    commit_manifest = {
        **{k: v for k, v in run_manifest.items() if k != "raw_outputs_path"},
        "raw_outputs_gitignored": True,
        "raw_outputs_path_gitignored": run_manifest.get("raw_outputs_path"),
        "evaluated_at": utc_now_iso(),
        "common_first_token": COMMON_FIRST_TOKEN,
        "common_first_token_id": COMMON_FIRST_TOKEN_ID,
        "n_pilot_outputs": 144,
        "n_final_outputs": 0,
        "pilot_revision": revision,
        "pilot_kind": kind,
        "behavior_parser": "pre_output_physiology.phase4_behavior",
        "activations_collected": False,
        "probe_scores_computed": False,
        "final_scenarios_generated": False,
    }
    if kind == "controlled_prefix":
        commit_manifest.update(
            {
                "first_token_sampled": False,
                "controlled_prefix_token_id": 12107,
                "controlled_prefix_text": "Response",
                "controlled_prefix_supplied": True,
                "prompt_template_revision": 1,
                "prompt_template_changed_after_rev1": False,
                "prefix_integrity_count": 144,
                "k1_interpretation": "controlled-prefix k1",
            }
        )
    write_json(summary_dir / "pilot_generation_manifest.json", commit_manifest)

    behavior_payload = {
        "created_at": utc_now_iso(),
        "run_id": run_manifest["run_id"],
        "git_commit_generation": run_manifest["git_commit"],
        "pilot_revision": revision,
        "summary": summary,
        "eval_rows": eval_rows,
    }
    write_json(summary_dir / "pilot_behavior_summary.json", behavior_payload)
    _write_report(summary, commit_manifest, report_path)

    print(
        json.dumps(
            {
                "run_id": run_manifest["run_id"],
                "pilot_revision": revision,
                "all_gates_pass": summary["gates"]["all_operational_gates_pass"],
                "recommend_freeze_revision1": summary.get(
                    "recommend_freeze_revision1_templates"
                ),
                "n_c2_c3_paired": summary["n_c2_c3_paired_valid"],
                "per_condition_behaviorally_valid": {
                    cid: summary["per_condition"][cid]["behaviorally_valid"]
                    for cid in CONDITION_ORDER
                },
                "report": str(report_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
