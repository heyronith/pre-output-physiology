#!/usr/bin/env python3
"""Evaluate Phase 18A fresh confirmation gates (no model calls)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase18_cohort import (  # noqa: E402
    GUARANTEE,
    LOCKED_FAMILIES,
    N_CONTINUATIONS,
    N_SAMPLES,
    PHASE15_RULE_HASH,
    QUALIFYING_FAMILIES,
    QUALIFYING_TRAIN_FAMILIES,
    QUALIFYING_VAL_FAMILIES,
    STATUS_HOLD,
    STATUS_PASS,
    _sha_ids,
    annotate_continuation,
    evaluate_gates,
    is_fresh_usable,
    reproducibility_diagnostics,
    summarize_prompt_group,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402


def write_report(summary: dict, path: Path) -> None:
    gates = summary["gates"]
    lines = [
        "# Phase 18A — enriched-cohort fresh confirmation",
        "",
        f"**Run:** `{summary['run_id']}`  ",
        f"**Status:** `{summary['status']}`  ",
        f"**Phase-15 rule:** `{summary['phase15_rule_hash']}`  ",
        "**Temperature:** 0.9  ",
        f"**Cohort:** {summary['n_cohort']} prompts × {N_SAMPLES} = "
        f"{summary['n_continuations']} continuations  ",
        f"**Cohort SHA256:** `{summary['cohort_sha256']}`  ",
        "",
        "Behavior-only third independent seed batch confirming Phase-17 "
        "S2-confirmed near-boundary prompts. Locked families untouched. "
        "No activations. Phase 17A remains HOLD.",
        "",
        "## Gates",
        "",
        f"- Train: {gates['train']['n_usable']} fresh-usable "
        f"(need ≥{gates['train']['min_overall']}; "
        f"per-family ≥{gates['train']['min_per_family']}): "
        f"**{'PASS' if gates['train']['passed'] else 'FAIL'}**  ",
        f"  Per family: `{gates['train']['per_family']}`  ",
        f"- Validation (daycare): {gates['validation']['n_usable']} "
        f"fresh-usable (need ≥{gates['validation']['min_usable']}): "
        f"**{'PASS' if gates['validation']['passed'] else 'FAIL'}**  ",
        "",
        "## Fresh usability by family",
        "",
    ]
    for fam in QUALIFYING_FAMILIES:
        rates = summary["usable_by_family"][fam]
        lines.append(
            f"- `{fam}`: {rates['n_usable']}/{rates['n_prompts']} usable"
        )
    lines += [
        "",
        "## S2 → Phase-18 reproducibility (descriptive)",
        "",
        f"- Pearson(S2, P18 alt frac): "
        f"{summary['reproducibility'].get('pearson_alt_fraction')}  ",
        f"- Spearman(S2, P18 alt frac): "
        f"{summary['reproducibility'].get('spearman_alt_fraction')}  ",
        f"- Median |Δ alt frac|: "
        f"{summary['reproducibility'].get('median_abs_alt_diff')}  ",
        f"- Switched rec-heavy→alt-heavy: "
        f"{summary['reproducibility']['n_switched_record_heavy_to_alternate_heavy']}  ",
        f"- Switched alt-heavy→rec-heavy: "
        f"{summary['reproducibility']['n_switched_alternate_heavy_to_record_heavy']}  ",
        "",
    ]
    if summary.get("fresh_usable_prompt_group_ids"):
        lines += [
            "## Fresh-usable prompt set",
            "",
            f"- n_usable: {summary['n_fresh_usable']}  ",
            f"- IDs SHA256: `{summary['fresh_usable_prompt_group_ids_sha256']}`  ",
            "",
        ]
    lines += [
        "## Per-prompt counts",
        "",
        "| prompt_group_id | family | S2 rec/alt/valid | P18 rec/alt/valid | "
        "P18 alt frac | usable |",
        "|---|---|---|---|---|---|",
    ]
    for p in summary["per_prompt"]:
        lines.append(
            f"| `{p['prompt_group_id']}` | {p['family']} | "
            f"{p['phase17_s2_n_record']}/{p['phase17_s2_n_alternate']}/"
            f"{p['phase17_s2_n_valid']} | "
            f"{p['phase18_n_record']}/{p['phase18_n_alternate']}/"
            f"{p['phase18_n_valid']} | "
            f"{p['phase18_alternate_fraction']} | "
            f"{'yes' if p['fresh_usable'] else 'no'} |"
        )
    lines += ["", GUARANTEE, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument(
        "--design-dir",
        default=str(REPO_ROOT / "artifacts/phase18a_design"),
    )
    ap.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "artifacts/phase18a_confirmation"),
    )
    ap.add_argument(
        "--report",
        default=str(REPO_ROOT / "reports/phase18a_enriched_cohort_fresh_confirmation.md"),
    )
    args = ap.parse_args()

    design = Path(args.design_dir)
    matrix = json.loads((design / "design_matrix.json").read_text("utf-8"))
    cohort = json.loads((design / "cohort.json").read_text("utf-8"))
    run_dir = Path(args.run_dir)
    rows = [
        json.loads(x)
        for x in (run_dir / "continuations.jsonl").read_text("utf-8").splitlines()
        if x.strip()
    ]
    if any(r.get("stage") != "P18A" for r in rows):
        raise SystemExit("non-P18A rows")
    if len(rows) != N_CONTINUATIONS:
        raise SystemExit(f"expected {N_CONTINUATIONS} continuations, got {len(rows)}")
    if any(r["response_token_id"] != 12107 for r in rows):
        raise SystemExit("Response != 12107")
    locked = set(LOCKED_FAMILIES)
    if any(r["family"] in locked for r in rows):
        raise SystemExit("locked family in run")

    annotated = [annotate_continuation(r) for r in rows]
    for a in annotated:
        if a["rule_hash"] != PHASE15_RULE_HASH:
            raise SystemExit("onset rule hash drift")

    by_pg: dict[str, list] = defaultdict(list)
    for a in annotated:
        by_pg[a["prompt_group_id"]].append(a)

    selected_meta = {s["prompt_group_id"]: s for s in cohort["selected"]}
    if set(by_pg) != set(selected_meta):
        raise SystemExit("prompt set mismatch vs frozen cohort")

    per_prompt = []
    usable_rows = []
    for pg in sorted(by_pg):
        summ = summarize_prompt_group(by_pg[pg])
        if summ["n_continuations"] != N_SAMPLES:
            raise SystemExit(f"{pg} has {summ['n_continuations']} != {N_SAMPLES}")
        meta = selected_meta[pg]
        usable = is_fresh_usable(summ)
        row = {
            "prompt_group_id": pg,
            "family": meta["family"],
            "split_role": meta["split_role"],
            "phase17_s2_n_record": meta["phase17_s2_n_record"],
            "phase17_s2_n_alternate": meta["phase17_s2_n_alternate"],
            "phase17_s2_n_valid": meta["phase17_s2_n_valid"],
            "phase17_s2_alternate_fraction": meta["phase17_s2_alternate_fraction"],
            "phase18_n_record": summ["n_record"],
            "phase18_n_alternate": summ["n_alternate"],
            "phase18_n_valid": summ["n_valid"],
            "phase18_valid_fraction": summ["valid_fraction"],
            "phase18_alternate_fraction": summ["alternate_fraction_among_valid"],
            "phase18_policy_entropy": summ["policy_entropy"],
            "fresh_usable": usable,
        }
        per_prompt.append(row)
        if usable:
            usable_rows.append(row)

    gates = evaluate_gates(usable_rows)
    status = STATUS_PASS if gates["passed"] else STATUS_HOLD
    repro = reproducibility_diagnostics(per_prompt)

    usable_by_family = {}
    for fam in QUALIFYING_FAMILIES:
        fam_rows = [p for p in per_prompt if p["family"] == fam]
        usable_by_family[fam] = {
            "n_prompts": len(fam_rows),
            "n_usable": sum(1 for p in fam_rows if p["fresh_usable"]),
        }

    usable_ids = sorted(u["prompt_group_id"] for u in usable_rows)
    run_manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
    summary = {
        "created_at": utc_now_iso(),
        "run_id": run_manifest["run_id"],
        "status": status,
        "phase15_rule_hash": PHASE15_RULE_HASH,
        "cohort_sha256": matrix["cohort_sha256"],
        "fresh_schedule_sha256": matrix["fresh_schedule_sha256"],
        "threshold_hash": matrix["threshold_hash"],
        "n_cohort": matrix["n_cohort"],
        "n_continuations": len(annotated),
        "n_fresh_usable": len(usable_ids),
        "fresh_usable_prompt_group_ids": usable_ids if gates["passed"] else None,
        "fresh_usable_prompt_group_ids_sha256": (
            _sha_ids(usable_ids) if gates["passed"] else None
        ),
        "usable_by_family": usable_by_family,
        "gates": gates,
        "reproducibility": repro,
        "per_prompt": per_prompt,
        "locked_model_calls": 0,
        "activations_collected": False,
        "probe_scored": False,
        "guarantee": GUARANTEE,
        "train_families": list(QUALIFYING_TRAIN_FAMILIES),
        "validation_families": list(QUALIFYING_VAL_FAMILIES),
    }

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "confirmation_summary.json", summary)
    write_json(out / "run_manifest.json", run_manifest)
    write_report(summary, Path(args.report))
    print(
        json.dumps(
            {
                "status": status,
                "n_fresh_usable": summary["n_fresh_usable"],
                "gates_passed": gates["passed"],
                "train": gates["train"],
                "validation": gates["validation"],
                "reproducibility": repro,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
