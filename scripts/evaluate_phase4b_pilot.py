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
    lines = [
        "# Phase 4B pilot report",
        "",
        f"**Run ID:** `{manifest.get('run_id')}`  ",
        f"**Git SHA (generation):** `{manifest.get('git_commit')}`  ",
        "**Status:** operational compliance pilot only (D053)  ",
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
        f"- do_sample: `{manifest.get('do_sample')}`",
        f"- max_new_tokens: `{manifest.get('max_new_tokens')}`",
        f"- GPU wall seconds: `{manifest.get('wall_seconds')}`",
        f"- Estimated cost USD: `{manifest.get('estimated_cost_usd')}`",
        "",
        "## Operational gates (D054)",
        "",
        f"- First-token ID 2963 ≥ {PILOT_FIRST_TOKEN_MIN_PER_CONDITION}/24 per condition",
        f"- Behavioral validity ≥ {PILOT_BEHAVIOR_MIN_PER_CONDITION}/24 per condition",
        f"- C2∩C3 paired valid ≥ {PILOT_C2_C3_PAIRED_MIN}/24 base scenarios",
        "",
        f"**All gates passed:** `{gates['all_operational_gates_pass']}`  ",
        f"**Recommend freeze templates unchanged:** "
        f"`{summary['recommend_freeze_templates_unchanged']}`  ",
        f"**Template revision recommended:** `{summary['template_revision_recommended']}`  ",
        "",
        "## Per-condition results",
        "",
        "| Condition | N | FT-ID ok | FINAL | Valid | FT gate | Beh gate |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for cid in CONDITION_ORDER:
        c = summary["per_condition"][cid]
        lines.append(
            f"| {cid} | {c['n_total']} | {c['first_token_id_compliant']} | "
            f"{c['valid_FINAL_marker']} | {c['behaviorally_valid']} | "
            f"{c['first_token_gate_pass']} | {c['behavior_gate_pass']} |"
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
    diag = summary.get("diagnostic_leading_space_response_token")
    if diag:
        lines.extend(
            [
                "## Diagnostic (not a gate): leading-space Response token",
                "",
                diag["note"],
                "",
                f"- Observed first-token ID on all "
                f"`{diag['n_matching_12107']}` rows: `{diag['observed_first_token_id']}`",
                f"- Frozen expected ID: `{diag['frozen_expected_first_token_id']}`",
                f"- If expected ID were 12107: first-token OK = "
                f"`{diag['n_first_token_ok_if_expected_12107']}/144`; "
                f"behaviorally valid = "
                f"`{diag['n_behaviorally_valid_if_expected_12107']}/144`",
                "",
                "### Proposed single template revision (D050)",
                "",
                "1. Re-verify `common_first_token` in true post-`[/INST]` generation "
                "context, where the model emits the leading-space BPE form "
                "(`12107` = `Response`), and freeze that ID if it is the one-token "
                "form actually produced.",
                "2. In the same revision, tighten machine-readable format instructions "
                "(require exactly one `FINAL=` line; forbid alternate MODE spellings; "
                "strengthen C3 reward-target compliance without adding forbidden "
                "deception lexicon).",
                "",
                "Do not generate final scenarios until that revision is audited.",
                "",
            ]
        )
    lines.extend(
        [
            "## Contamination / analysis guarantees",
            "",
            "NO PHASE 4 ACTIVATIONS WERE COLLECTED.  ",
            "NO PHASE 4 PROBE SCORES WERE COMPUTED.  ",
            "FINAL PHASE 4 SCENARIOS WERE NOT GENERATED.  ",
            "NO CAUSAL INTERVENTIONS WERE PERFORMED.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        required=True,
        help="artifacts/runs/<phase4b_pilot_...>/ containing pilot_outputs.jsonl",
    )
    parser.add_argument(
        "--summary-dir",
        default=str(REPO_ROOT / "artifacts/phase4b_pilot"),
    )
    parser.add_argument(
        "--report-path",
        default=str(REPO_ROOT / "reports/phase4b_pilot.md"),
    )
    args = parser.parse_args()
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

    eval_rows = [evaluate_pilot_row(r) for r in rows]
    summary = summarize_pilot_behavior(eval_rows)

    # Diagnostic only (not an operational gate): leading-space Response token 12107.
    alt_id = 12107
    alt_valid = 0
    alt_ft = 0
    alt_reasons: dict[str, dict[str, int]] = {cid: {} for cid in CONDITION_ORDER}
    for r in rows:
        from pre_output_physiology.phase4_behavior import is_behaviorally_valid as _ibv
        from pre_output_physiology.phase4_behavior import parse_response as _pr

        parsed = _pr(
            r["output_text"],
            first_generated_token_id=r.get("first_generated_token_id"),
            expected_first_token_id=alt_id,
        )
        ok, reason = _ibv(
            r["condition_id"],
            {
                "ground_truth_state": r["ground_truth_state"],
                "alt_state": r["alt_state"],
                "accessible_false_state": r["accessible_false_state"],
            },
            r["output_text"],
            first_generated_token_id=r.get("first_generated_token_id"),
            expected_first_token_id=alt_id,
        )
        alt_ft += int(parsed.first_token_ok)
        alt_valid += int(ok)
        if not ok:
            bucket = alt_reasons[r["condition_id"]]
            bucket[reason] = bucket.get(reason, 0) + 1
    summary["diagnostic_leading_space_response_token"] = {
        "note": (
            "All 144 first generated token IDs were 12107, which decodes to the "
            "leading-space BPE form of 'Response'. Frozen criterion remains 2963."
        ),
        "observed_first_token_id": 12107,
        "frozen_expected_first_token_id": COMMON_FIRST_TOKEN_ID,
        "n_matching_12107": sum(
            1 for r in rows if r.get("first_generated_token_id") == 12107
        ),
        "n_first_token_ok_if_expected_12107": alt_ft,
        "n_behaviorally_valid_if_expected_12107": alt_valid,
        "failure_reason_counts_if_expected_12107": alt_reasons,
    }

    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))

    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Commit-safe generation manifest (no raw output text)
    commit_manifest = {
        **{k: v for k, v in run_manifest.items() if k != "raw_outputs_path"},
        "raw_outputs_gitignored": True,
        "raw_outputs_path_gitignored": run_manifest.get("raw_outputs_path"),
        "evaluated_at": utc_now_iso(),
        "common_first_token": COMMON_FIRST_TOKEN,
        "common_first_token_id": COMMON_FIRST_TOKEN_ID,
        "n_pilot_outputs": 144,
        "n_final_outputs": 0,
        "behavior_parser": "pre_output_physiology.phase4_behavior",
        "activations_collected": False,
        "probe_scores_computed": False,
        "final_scenarios_generated": False,
    }
    write_json(summary_dir / "pilot_generation_manifest.json", commit_manifest)

    behavior_payload = {
        "created_at": utc_now_iso(),
        "run_id": run_manifest["run_id"],
        "git_commit_generation": run_manifest["git_commit"],
        "summary": summary,
        "eval_rows": eval_rows,
    }
    write_json(summary_dir / "pilot_behavior_summary.json", behavior_payload)
    _write_report(summary, run_manifest, Path(args.report_path))

    print(
        json.dumps(
            {
                "run_id": run_manifest["run_id"],
                "all_gates_pass": summary["gates"]["all_operational_gates_pass"],
                "recommend_freeze": summary["recommend_freeze_templates_unchanged"],
                "n_c2_c3_paired": summary["n_c2_c3_paired_valid"],
                "per_condition_behaviorally_valid": {
                    cid: summary["per_condition"][cid]["behaviorally_valid"]
                    for cid in CONDITION_ORDER
                },
                "report": args.report_path,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
