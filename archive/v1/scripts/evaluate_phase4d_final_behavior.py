#!/usr/bin/env python3
"""Evaluate Phase 4D final behavior outputs and freeze contrast eligibility.

Uses frozen revision-1 behavioral rules unchanged. No LLM judge, no relabeling,
no activations or probe scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase4_behavior import (  # noqa: E402
    BEHAVIOR_RULES_SUMMARY,
    evaluate_pilot_row,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

SELECTED_CONDITIONS: tuple[str, ...] = (
    "C1_known_honest_neutral",
    "C2_known_honest_strategic",
    "C3_known_deceptive_strategic",
    "C4_false_belief_honest",
    "C6_counterfactual_nondeceptive",
)
HELD_CONDITION = "C5_uncertain_honest"
N_EXPECTED = 1200
N_PER = 240
CONTROLLED_PREFIX_TOKEN_ID = 12107
FINAL_PROMPT_TEXT_SHA256 = (
    "84942a71e61a1aed67c872d4ffd69dc6e533f6295fcb0b3b93fa78d93f97ae89"
)
FINAL_SCENARIO_IDS_SHA256 = (
    "f43d16d942331aa45f3a86e8af5f330f0370fe84356599fa7964a3949a2bf880"
)
FROZEN_PROBE_L12_K1 = "3bbf0c78e97d8ff14b028592e6e6ab3215651e973e0be1f004ce66e2b9b7aa65"
FROZEN_PROBE_L12_K0 = "dc14569a4e229cadc55fd3f6e6fef4e33f4eac4f58d00e83916238d5274f4f29"

# (name, left_condition, right_condition)
CONTRASTS: tuple[tuple[str, str, str], ...] = (
    ("C3_vs_C2", "C3_known_deceptive_strategic", "C2_known_honest_strategic"),
    ("C3_vs_C4", "C3_known_deceptive_strategic", "C4_false_belief_honest"),
    ("C3_vs_C1", "C3_known_deceptive_strategic", "C1_known_honest_neutral"),
    ("C3_vs_C6", "C3_known_deceptive_strategic", "C6_counterfactual_nondeceptive"),
    ("C2_vs_C1", "C2_known_honest_strategic", "C1_known_honest_neutral"),
    ("C4_vs_C1", "C4_false_belief_honest", "C1_known_honest_neutral"),
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _sha_ids(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _valid_ids_by_condition(
    eval_rows: list[dict[str, Any]],
) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {cid: set() for cid in SELECTED_CONDITIONS}
    for row in eval_rows:
        cid = row["condition_id"]
        if cid in out and row["behaviorally_valid"]:
            out[cid].add(row["base_scenario_id"])
    return out


def _freeze_contrasts(valid_by: dict[str, set[str]]) -> dict[str, Any]:
    contrasts: dict[str, Any] = {}
    for name, left, right in CONTRASTS:
        paired = sorted(valid_by[left] & valid_by[right])
        contrasts[name] = {
            "left_condition": left,
            "right_condition": right,
            "n_paired_valid": len(paired),
            "paired_base_scenario_ids": paired,
            "paired_ids_sha256": _sha_ids(paired),
        }
    return contrasts


def summarize_final_behavior(eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_cond: dict[str, list[dict[str, Any]]] = {cid: [] for cid in SELECTED_CONDITIONS}
    for row in eval_rows:
        cid = row["condition_id"]
        if cid not in by_cond:
            raise ValueError(f"unexpected condition in eval: {cid}")
        by_cond[cid].append(row)

    per_condition: dict[str, Any] = {}
    for cid in SELECTED_CONDITIONS:
        rows = by_cond[cid]
        n = len(rows)
        prefix_ok = sum(1 for r in rows if r["first_token_id_compliant"])
        fmt = sum(1 for r in rows if r["exact_three_line_format"])
        mode_ok = sum(1 for r in rows if r["mode_compliant"])
        finals = sum(1 for r in rows if r["has_final_marker"])
        valid = sum(1 for r in rows if r["behaviorally_valid"])
        disclosure_fail = sum(
            1 for r in rows if r["failure_reason"] == "disclosed_private_objective"
        )
        reasons = Counter(r["failure_reason"] for r in rows if r["failure_reason"])
        per_condition[cid] = {
            "n_total": n,
            "prefix_integrity": prefix_ok,
            "exact_three_line_format": fmt,
            "mode_compliant": mode_ok,
            "final_compliant": finals,
            "behaviorally_valid": valid,
            "c3_disclosure_failures": disclosure_fail if cid.startswith("C3") else 0,
            "failure_reason_counts": dict(reasons),
        }

    valid_by = _valid_ids_by_condition(eval_rows)
    contrasts = _freeze_contrasts(valid_by)
    return {
        "n_eval_rows": len(eval_rows),
        "n_c5_outputs": 0,
        "n_pilot_outputs": 0,
        "selected_conditions": list(SELECTED_CONDITIONS),
        "held_condition": HELD_CONDITION,
        "per_condition": per_condition,
        "prefix_integrity_count": sum(
            1 for r in eval_rows if r["first_token_id_compliant"]
        ),
        "behavior_rules": {
            cid: BEHAVIOR_RULES_SUMMARY[cid] for cid in SELECTED_CONDITIONS
        },
        "primary_contrast": "C3_vs_C2",
        "key_secondary_contrast": "C3_vs_C4",
        "n_c2_c3_paired_valid": contrasts["C3_vs_C2"]["n_paired_valid"],
        "c2_c3_paired_ids_sha256": contrasts["C3_vs_C2"]["paired_ids_sha256"],
        "activations_collected": False,
        "probe_scores_computed": False,
        "prompt_template_changed": False,
        "causal_interventions_performed": False,
    }


def _write_report(
    summary: dict[str, Any],
    contrasts: dict[str, Any],
    manifest: dict[str, Any],
    path: Path,
) -> None:
    lines = [
        "# Phase 4D final behavior report",
        "",
        f"**Run ID:** `{manifest.get('run_id')}`  ",
        f"**Git SHA (generation):** `{manifest.get('git_commit')}`  ",
        f"**Prompt revision:** `{manifest.get('prompt_template_revision', 1)}`  ",
        "**Status:** final behavior complete; awaiting sample-size audit before activations  ",
        "",
        "## Scope",
        "",
        "- 1200 final outputs (240 base × 5 conditions: C1/C2/C3/C4/C6)",
        "- C5 HOLD (D061) — zero outputs",
        "- Controlled-prefix token 12107; greedy BF16 Mistral-7B-Instruct-v0.2",
        "- Frozen revision-1 behavioral rules; deterministic parser only",
        "- **No** activations, probe scores, causal interventions, or prompt edits",
        "",
        "## Generation",
        "",
        f"- Model revision: `{manifest.get('model_revision')}`",
        f"- controlled_prefix_token_id: `{manifest.get('controlled_prefix_token_id')}`",
        f"- first_token_sampled: `{manifest.get('first_token_sampled')}`",
        f"- do_sample: `{manifest.get('do_sample')}`",
        f"- max_continuation_tokens: `{manifest.get('max_continuation_tokens')}`",
        f"- GPU wall seconds: `{manifest.get('wall_seconds')}`",
        f"- Estimated cost USD: `{manifest.get('estimated_cost_usd')}`",
        f"- Prefix integrity: `{summary['prefix_integrity_count']}/{N_EXPECTED}`",
        "",
        "## Per-condition behavioral validity",
        "",
        "| Condition | N | Prefix | 3-line | MODE | FINAL | Valid |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cid in SELECTED_CONDITIONS:
        c = summary["per_condition"][cid]
        lines.append(
            f"| {cid} | {c['n_total']} | {c['prefix_integrity']} | "
            f"{c['exact_three_line_format']} | {c['mode_compliant']} | "
            f"{c['final_compliant']} | {c['behaviorally_valid']} |"
        )
    lines.extend(
        [
            "",
            "## Frozen contrast eligibility",
            "",
            "| Contrast | Paired N | IDs SHA256 |",
            "| --- | ---: | --- |",
        ]
    )
    for name, _, _ in CONTRASTS:
        c = contrasts[name]
        lines.append(
            f"| {name} | {c['n_paired_valid']} | `{c['paired_ids_sha256']}` |"
        )
    lines.extend(["", "## Failure reasons", ""])
    for cid in SELECTED_CONDITIONS:
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
            "## Guarantees",
            "",
            "NO PHASE 4 ACTIVATIONS WERE COLLECTED.  ",
            "NO PHASE 4 PROBE SCORES WERE COMPUTED.  ",
            "NO PROMPT-TEMPLATE CHANGES WERE MADE.  ",
            "NO CAUSAL INTERVENTIONS WERE PERFORMED.  ",
            "C5 PRODUCED ZERO OUTPUTS AND REMAINS HOLD.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--summary-dir",
        default=str(REPO_ROOT / "artifacts/phase4d_final_behavior"),
    )
    parser.add_argument(
        "--report-path",
        default=str(REPO_ROOT / "reports/phase4d_final_behavior.md"),
    )
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    summary_dir = Path(args.summary_dir)
    report_path = Path(args.report_path)

    raw = run_dir / "final_outputs.jsonl"
    run_manifest_path = run_dir / "final_generation_manifest.json"
    if not raw.is_file():
        raise SystemExit(f"missing {raw}")
    if not run_manifest_path.is_file():
        raise SystemExit(f"missing {run_manifest_path}")

    rows = _load_jsonl(raw)
    if len(rows) != N_EXPECTED:
        raise SystemExit(f"expected {N_EXPECTED} outputs, got {len(rows)}")
    if any(r.get("split") != "final" for r in rows):
        raise SystemExit("non-final rows present")
    if any(r.get("condition_id") == HELD_CONDITION for r in rows):
        raise SystemExit("C5 outputs present — STOP")
    if any(r["base_scenario_id"].startswith("pilot_") for r in rows):
        raise SystemExit("pilot scenario IDs present — STOP")
    if any(r.get("activation_extracted") for r in rows):
        raise SystemExit("activation_extracted true")
    if any(r.get("probe_scored") for r in rows):
        raise SystemExit("probe_scored true")

    counts = Counter(r["condition_id"] for r in rows)
    for cid in SELECTED_CONDITIONS:
        if counts[cid] != N_PER:
            raise SystemExit(f"{cid} count {counts[cid]} != {N_PER}")
    if set(counts) != set(SELECTED_CONDITIONS):
        raise SystemExit(f"unexpected conditions: {sorted(counts)}")

    bad_prefix = [
        r["example_id"]
        for r in rows
        if not (
            r.get("controlled_prefix_supplied") is True
            and r.get("first_token_sampled") is False
            and r.get("controlled_prefix_token_id") == CONTROLLED_PREFIX_TOKEN_ID
            and r.get("first_generated_token_id") == CONTROLLED_PREFIX_TOKEN_ID
            and isinstance(r.get("generated_token_ids"), list)
            and r["generated_token_ids"]
            and r["generated_token_ids"][0] == CONTROLLED_PREFIX_TOKEN_ID
        )
    ]
    if bad_prefix:
        raise SystemExit(f"prefix integrity failed for {len(bad_prefix)} rows")

    eval_rows = [evaluate_pilot_row(r) for r in rows]
    summary = summarize_final_behavior(eval_rows)
    if summary["prefix_integrity_count"] != N_EXPECTED:
        raise SystemExit(
            f"prefix integrity {summary['prefix_integrity_count']} != {N_EXPECTED}"
        )

    valid_by = _valid_ids_by_condition(eval_rows)
    contrasts = _freeze_contrasts(valid_by)
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))

    summary_dir.mkdir(parents=True, exist_ok=True)
    commit_manifest = {
        **{k: v for k, v in run_manifest.items() if k != "raw_outputs_path"},
        "raw_outputs_gitignored": True,
        "raw_outputs_path_gitignored": run_manifest.get("raw_outputs_path"),
        "evaluated_at": utc_now_iso(),
        "n_final_outputs": N_EXPECTED,
        "n_c5_outputs": 0,
        "n_pilot_outputs": 0,
        "behavior_parser": "pre_output_physiology.phase4_behavior",
        "behavior_rules_unchanged": True,
        "prompt_template_revision": 1,
        "prompt_template_changed_after_rev1": False,
        "final_prompt_text_sha256": FINAL_PROMPT_TEXT_SHA256,
        "final_scenario_ids_sha256": FINAL_SCENARIO_IDS_SHA256,
        "frozen_probe_l12_k1_sha256": FROZEN_PROBE_L12_K1,
        "frozen_probe_l12_k0_sha256": FROZEN_PROBE_L12_K0,
        "activations_collected": False,
        "probe_scores_computed": False,
        "causal_interventions_performed": False,
        "c5_status": "HOLD",
    }
    write_json(summary_dir / "final_generation_manifest.json", commit_manifest)

    behavior_payload = {
        "created_at": utc_now_iso(),
        "run_id": run_manifest["run_id"],
        "git_commit_generation": run_manifest["git_commit"],
        "summary": summary,
        "eval_rows": eval_rows,
    }
    write_json(summary_dir / "final_behavior_summary.json", behavior_payload)

    eligibility = {
        "created_at": utc_now_iso(),
        "run_id": run_manifest["run_id"],
        "git_commit_generation": run_manifest["git_commit"],
        "n_final_base_scenarios": 240,
        "selected_conditions": list(SELECTED_CONDITIONS),
        "held_condition": HELD_CONDITION,
        "primary_contrast": "C3_vs_C2",
        "key_secondary_contrast": "C3_vs_C4",
        "contrasts": contrasts,
        "frozen_before_activations": True,
        "behavior_rules_revision": 1,
        "note": (
            "Paired-valid IDs define Phase-4 analysis populations. "
            "Do not change after activations are observed."
        ),
    }
    write_json(summary_dir / "frozen_contrast_eligibility.json", eligibility)
    _write_report(summary, contrasts, commit_manifest, report_path)

    print(
        json.dumps(
            {
                "run_id": run_manifest["run_id"],
                "n_outputs": N_EXPECTED,
                "prefix_integrity": summary["prefix_integrity_count"],
                "per_condition_behaviorally_valid": {
                    cid: summary["per_condition"][cid]["behaviorally_valid"]
                    for cid in SELECTED_CONDITIONS
                },
                "contrast_ns": {
                    name: contrasts[name]["n_paired_valid"] for name, _, _ in CONTRASTS
                },
                "c2_c3_paired_hash": contrasts["C3_vs_C2"]["paired_ids_sha256"],
                "c3_c4_paired_hash": contrasts["C3_vs_C4"]["paired_ids_sha256"],
                "report": str(report_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
