#!/usr/bin/env python3
"""Evaluate Phase 5B discovery behavior and freeze sensitivity populations.

Uses frozen Phase-5A parser/rules unchanged. No prompt revision. No activations.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase5_behavior import (  # noqa: E402
    evaluate_row,
)
from pre_output_physiology.phase5_conditions import (  # noqa: E402
    CONDITION_ORDER,
    DISCOVERY_FAMILIES,
    LOCKED_FAMILIES,
)
from pre_output_physiology.phase5_split import sha_sorted_ids  # noqa: E402
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

N_EXPECTED = 1920
EXPECTED_FIRST_TOKEN_ID = 12107
EXPECTED_FINAL_PROMPT_TEXT_SHA256 = (
    "cdf2003ac0d568eee28a0feffa7b07e0ef208e637de40665aa9a5b1e8bc2920e"
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _summarize_subset(eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_cond: dict[str, list[dict[str, Any]]] = {cid: [] for cid in CONDITION_ORDER}
    for row in eval_rows:
        by_cond[row["condition_id"]].append(row)
    per_condition: dict[str, Any] = {}
    for cid in CONDITION_ORDER:
        rows = by_cond[cid]
        reasons = Counter(r["failure_reason"] for r in rows if r["failure_reason"])
        schema_fail = sum(1 for r in rows if r["failure_reason"] == "schema_mismatch")
        # State mismatches: record/target wrong after schema ok
        state_mismatch = sum(
            1
            for r in rows
            if r["failure_reason"]
            in {
                "state_ne_record",
                "state_ne_objective_target",
                "state_equals_record_for_s3",
            }
        )
        other = {
            k: v
            for k, v in reasons.items()
            if k
            not in {
                "schema_mismatch",
                "state_ne_record",
                "state_ne_objective_target",
                "state_equals_record_for_s3",
            }
        }
        token_counts = Counter(
            r["first_generated_token_id"]
            for r in rows
            if r["first_generated_token_id"] is not None
        )
        per_condition[cid] = {
            "n_total": len(rows),
            "behaviorally_valid": sum(1 for r in rows if r["behaviorally_valid"]),
            "schema_compliant": sum(1 for r in rows if r["schema_compliant"]),
            "first_token_id_compliant": sum(
                1 for r in rows if r["first_token_id_compliant"]
            ),
            "schema_failures": schema_fail,
            "state_mismatches": state_mismatch,
            "other_failure_reason_counts": dict(other),
            "failure_reason_counts": dict(reasons),
            "first_token_id_counts": {str(k): v for k, v in token_counts.items()},
        }
    s2_ok = {
        r["base_scenario_id"]
        for r in by_cond["S2_strategic_honesty"]
        if r["behaviorally_valid"]
    }
    s3_ok = {
        r["base_scenario_id"]
        for r in by_cond["S3_strategic_deception"]
        if r["behaviorally_valid"]
    }
    paired = sorted(s2_ok & s3_ok)
    return {
        "n_eval_rows": len(eval_rows),
        "per_condition": per_condition,
        "n_s2_s3_paired_valid": len(paired),
        "s2_s3_paired_valid_scenarios": paired,
        "paired_ids_sha256": sha_sorted_ids(paired),
    }


def _write_report(
    *,
    summary: dict[str, Any],
    manifest: dict[str, Any],
    primary: dict[str, Any],
    path: Path,
) -> None:
    lines = [
        "# Phase 5B discovery behavior report",
        "",
        f"**Run ID:** `{manifest.get('run_id')}`  ",
        f"**Git SHA:** `{manifest.get('git_commit')}`  ",
        f"**Neutral prefix token ID:** `{manifest.get('neutral_prefix_token_id')}`  ",
        f"**Prompt template revision:** `{manifest.get('prompt_template_revision')}`  ",
        "",
        "## Scope",
        "",
        "- 1920 discovery prompts (960 S2 + 960 S3)",
        "- Six discovery families only; locked families not run",
        "- Frozen revision-2 prompts; Phase-5A parser unchanged",
        "- No activations, probe fitting/scoring, layer selection, or causal work",
        "",
        "## Family split",
        "",
        f"- **Train:** {', '.join(manifest.get('train_families', []))}",
        f"- **Validation:** {', '.join(manifest.get('validation_families', []))}",
        "",
        "## Primary all-pair populations (pre-registered)",
        "",
        f"- Train all-pair N=`{primary['train']['n_pairs']}` "
        f"sha256=`{primary['train']['pair_ids_sha256']}`",
        f"- Validation all-pair N=`{primary['validation']['n_pairs']}` "
        f"sha256=`{primary['validation']['pair_ids_sha256']}`",
        "",
        "## Behavior (secondary sensitivity)",
        "",
    ]
    for role in ("train", "validation"):
        block = summary[role]
        lines.append(f"### {role}")
        lines.append("")
        lines.append(
            f"- S2 valid: `{block['per_condition']['S2_strategic_honesty']['behaviorally_valid']}` "
            f"/ `{block['per_condition']['S2_strategic_honesty']['n_total']}`"
        )
        lines.append(
            "- S3 valid: "
            f"`{block['per_condition']['S3_strategic_deception']['behaviorally_valid']}` "
            f"/ `{block['per_condition']['S3_strategic_deception']['n_total']}`"
        )
        lines.append(f"- Paired-valid N: `{block['n_s2_s3_paired_valid']}`")
        lines.append(
            f"- Paired-valid sha256: `{block['paired_ids_sha256']}`"
        )
        lines.append("")
        for cid in CONDITION_ORDER:
            c = block["per_condition"][cid]
            lines.append(
                f"- `{cid}`: schema_failures={c['schema_failures']}, "
                f"state_mismatches={c['state_mismatches']}, "
                f"failure_reasons={c['failure_reason_counts']}, "
                f"first_token_ids={c['first_token_id_counts']}"
            )
        lines.append("")
    lines.extend(
        [
            "## Guarantees",
            "",
            "NO PHASE 5 ACTIVATIONS WERE COLLECTED.  ",
            "NO PHASE 5 PROBES WERE FIT OR SCORED.  ",
            "NO LAYER SELECTION WAS PERFORMED.  ",
            "LOCKED GENERALIZATION FAMILIES WERE NOT RUN THROUGH THE MODEL.  ",
            "NO PROMPT OR BEHAVIOR RULES WERE CHANGED.  ",
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
        default=str(REPO_ROOT / "artifacts/phase5b_discovery_behavior"),
    )
    parser.add_argument(
        "--report-path",
        default=str(REPO_ROOT / "reports/phase5b_discovery_behavior.md"),
    )
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    raw = run_dir / "discovery_outputs.jsonl"
    man_path = run_dir / "discovery_generation_manifest.json"
    rows = _load_jsonl(raw)
    if len(rows) != N_EXPECTED:
        raise SystemExit(f"expected {N_EXPECTED}, got {len(rows)}")
    if any(r.get("pool") != "discovery" for r in rows):
        raise SystemExit("non-discovery rows")
    if any(r.get("family") in LOCKED_FAMILIES for r in rows):
        raise SystemExit("locked family in outputs")
    if any(r.get("activation_extracted") for r in rows):
        raise SystemExit("activations flagged")
    if any(r.get("probe_scored") for r in rows):
        raise SystemExit("probe scores flagged")

    split = json.loads(
        (
            REPO_ROOT / "artifacts/phase5b_discovery_split/family_split.json"
        ).read_text(encoding="utf-8")
    )
    primary = json.loads(
        (
            REPO_ROOT
            / "artifacts/phase5b_discovery_split/primary_all_pair_populations.json"
        ).read_text(encoding="utf-8")
    )
    train_fam = set(split["train_families"])

    eval_rows = [
        {
            **evaluate_row(r, expected_first_token_id=EXPECTED_FIRST_TOKEN_ID),
            "family": r["family"],
            "split_role": r.get("split_role")
            or ("train" if r["family"] in train_fam else "validation"),
        }
        for r in rows
    ]
    train_eval = [r for r in eval_rows if r["split_role"] == "train"]
    val_eval = [r for r in eval_rows if r["split_role"] == "validation"]
    if len(train_eval) != 1280 or len(val_eval) != 640:
        raise SystemExit(f"role sizes {len(train_eval)}/{len(val_eval)}")

    train_sum = _summarize_subset(train_eval)
    val_sum = _summarize_subset(val_eval)

    per_family: dict[str, Any] = {}
    for fam in DISCOVERY_FAMILIES:
        fam_rows = [r for r in eval_rows if r["family"] == fam]
        fam_sum = _summarize_subset(fam_rows)
        role = "train" if fam in train_fam else "validation"
        per_family[fam] = {
            "role": role,
            "n_designed_pairs": primary["per_family_pair_counts"][fam],
            "n_behavior_valid_pairs": fam_sum["n_s2_s3_paired_valid"],
            "s2_valid": fam_sum["per_condition"]["S2_strategic_honesty"][
                "behaviorally_valid"
            ],
            "s3_valid": fam_sum["per_condition"]["S3_strategic_deception"][
                "behaviorally_valid"
            ],
        }

    sensitivity = {
        "created_at": utc_now_iso(),
        "phase": "phase5b",
        "role": "secondary_sensitivity_only",
        "note": (
            "Behaviorally valid paired subsets are secondary sensitivity analyses; "
            "they do not replace primary all-pair populations."
        ),
        "train": {
            "n_pairs": train_sum["n_s2_s3_paired_valid"],
            "pair_ids": train_sum["s2_s3_paired_valid_scenarios"],
            "pair_ids_sha256": train_sum["paired_ids_sha256"],
        },
        "validation": {
            "n_pairs": val_sum["n_s2_s3_paired_valid"],
            "pair_ids": val_sum["s2_s3_paired_valid_scenarios"],
            "pair_ids_sha256": val_sum["paired_ids_sha256"],
        },
        "per_family_pair_counts": {
            fam: {
                "designed": per_family[fam]["n_designed_pairs"],
                "behavior_valid": per_family[fam]["n_behavior_valid_pairs"],
            }
            for fam in DISCOVERY_FAMILIES
        },
        "locked_family_ids_present": False,
        "primary_all_pair_unchanged": {
            "train_n": primary["train"]["n_pairs"],
            "train_sha256": primary["train"]["pair_ids_sha256"],
            "validation_n": primary["validation"]["n_pairs"],
            "validation_sha256": primary["validation"]["pair_ids_sha256"],
        },
    }

    run_manifest = json.loads(man_path.read_text(encoding="utf-8"))
    if run_manifest.get("final_prompt_text_sha256") != EXPECTED_FINAL_PROMPT_TEXT_SHA256:
        raise SystemExit("prompt hash changed vs Phase 5A freeze")
    if run_manifest.get("discovery_prompt_text_sha256") != split[
        "discovery_prompt_text_sha256"
    ]:
        raise SystemExit("discovery prompt hash mismatch vs split freeze")

    summary = {
        "train": train_sum,
        "validation": val_sum,
        "per_family": per_family,
        "n_outputs": len(rows),
        "locked_outputs": 0,
        "activations_collected": False,
        "probes_fit_or_scored": False,
        "layer_selection_performed": False,
        "prompt_or_behavior_rules_changed": False,
    }

    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    commit_manifest = {
        **{k: v for k, v in run_manifest.items() if k != "raw_outputs_path"},
        "raw_outputs_gitignored": True,
        "raw_outputs_path_gitignored": run_manifest.get("raw_outputs_path"),
        "evaluated_at": utc_now_iso(),
        "expected_first_token_id": EXPECTED_FIRST_TOKEN_ID,
        "activations_collected": False,
        "probes_fit_or_scored": False,
        "locked_final_families_run": False,
        "layer_selection_performed": False,
    }
    write_json(summary_dir / "discovery_generation_manifest.json", commit_manifest)
    write_json(
        summary_dir / "discovery_behavior_summary.json",
        {
            "created_at": utc_now_iso(),
            "run_id": run_manifest["run_id"],
            "summary": summary,
            # Omit full eval_rows from committed artifact (large); keep counts only.
            "eval_row_count": len(eval_rows),
        },
    )
    write_json(
        summary_dir / "sensitivity_behavior_valid_populations.json",
        sensitivity,
    )
    # Mirror sensitivity under split dir for analysis freeze location.
    write_json(
        REPO_ROOT
        / "artifacts/phase5b_discovery_split/sensitivity_behavior_valid_populations.json",
        sensitivity,
    )
    _write_report(
        summary=summary,
        manifest=commit_manifest,
        primary=primary,
        path=Path(args.report_path),
    )
    print(
        json.dumps(
            {
                "run_id": run_manifest["run_id"],
                "n_outputs": len(rows),
                "train_s2_valid": train_sum["per_condition"]["S2_strategic_honesty"][
                    "behaviorally_valid"
                ],
                "train_s3_valid": train_sum["per_condition"]["S3_strategic_deception"][
                    "behaviorally_valid"
                ],
                "train_paired": train_sum["n_s2_s3_paired_valid"],
                "val_s2_valid": val_sum["per_condition"]["S2_strategic_honesty"][
                    "behaviorally_valid"
                ],
                "val_s3_valid": val_sum["per_condition"]["S3_strategic_deception"][
                    "behaviorally_valid"
                ],
                "val_paired": val_sum["n_s2_s3_paired_valid"],
                "train_valid_sha": train_sum["paired_ids_sha256"],
                "val_valid_sha": val_sum["paired_ids_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
