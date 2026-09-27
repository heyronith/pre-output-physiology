#!/usr/bin/env python3
"""Evaluate Phase 17A S2 confirmation, gates, and selection (no model calls)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase17_screen import (  # noqa: E402
    GUARANTEE,
    PHASE15_RULE_HASH,
    STATUS_HOLD,
    STATUS_PASS,
    annotate_continuation,
    evaluate_gates,
    family_split,
    family_stage_table,
    is_s2_confirmed,
    per_prompt_summaries,
    reproducibility_diagnostics,
    select_prompts,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402


def write_report(summary: dict, path: Path) -> None:
    gates = summary["gates"]
    lines = [
        "# Phase 17A — policy-unstable exact-prompt enrichment screen",
        "",
        f"**S1 run:** `{summary['s1_run_id']}`  ",
        f"**S2 run:** `{summary['s2_run_id']}`  ",
        f"**Status:** `{summary['status']}`  ",
        f"**Phase-15 rule:** `{summary['phase15_rule_hash']}`  ",
        "**Temperature:** 0.9  ",
        "",
        "Behavior-only enrichment of frozen Phase-14 discovery finals. "
        "Locked families untouched. No activations.",
        "",
        "## Gates",
        "",
        f"- Train: {gates['train']['n_confirmed']} confirmed "
        f"(need ≥{gates['train']['min_overall']}; "
        f"per-family ≥{gates['train']['min_per_family']}): "
        f"**{'PASS' if gates['train']['passed'] else 'FAIL'}**  ",
        f"  Per family: `{gates['train']['per_family']}`  ",
        f"- Validation: {gates['validation']['n_confirmed']} confirmed "
        f"(need ≥{gates['validation']['min_overall']}; "
        f"per-family ≥{gates['validation']['min_per_family']}): "
        f"**{'PASS' if gates['validation']['passed'] else 'FAIL'}**  ",
        f"  Per family: `{gates['validation']['per_family']}`  ",
        "",
        f"**S1→S2 confirmation rate:** "
        f"{summary['s1_to_s2_confirmation_rate']:.3f} "
        f"({summary['n_confirmed']}/{summary['n_s1_candidates']})  ",
        "",
        "## Reproducibility (descriptive)",
        "",
        f"- Pearson(S1,S2 alt frac): "
        f"{summary['reproducibility'].get('pearson_alt_fraction')}  ",
        f"- Median |Δ alt frac|: "
        f"{summary['reproducibility'].get('median_abs_alt_diff')}  ",
        f"- Switched rec-heavy→alt-heavy: "
        f"{summary['reproducibility']['n_switched_record_heavy_to_alternate_heavy']}  ",
        f"- Switched alt-heavy→rec-heavy: "
        f"{summary['reproducibility']['n_switched_alternate_heavy_to_record_heavy']}  ",
        "",
    ]
    if summary.get("selection"):
        sel = summary["selection"]
        lines += [
            "## Selected prompts",
            "",
            f"- n_selected: {sel['n_selected']}  ",
            f"- per family: `{sel['n_per_family']}`  ",
            f"- IDs SHA256: `{sel['selected_prompt_group_ids_sha256']}`  ",
            "",
        ]
    lines += [GUARANTEE, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2-run-dir", required=True)
    ap.add_argument(
        "--s1-summary",
        default=str(REPO_ROOT / "artifacts/phase17a_s1/s1_summary.json"),
    )
    ap.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "artifacts/phase17a_screen"),
    )
    ap.add_argument(
        "--report",
        default=str(REPO_ROOT / "reports/phase17a_policy_unstable_screen.md"),
    )
    args = ap.parse_args()

    s1 = json.loads(Path(args.s1_summary).read_text("utf-8"))
    s2_dir = Path(args.s2_run_dir)
    rows = [
        json.loads(x)
        for x in (s2_dir / "continuations.jsonl").read_text("utf-8").splitlines()
        if x.strip()
    ]
    if any(r.get("stage") != "S2" for r in rows):
        raise SystemExit("non-S2 rows")
    if len(rows) != s1["n_s2_continuations"]:
        raise SystemExit("S2 continuation count mismatch vs schedule")
    if any(r["response_token_id"] != 12107 for r in rows):
        raise SystemExit("Response != 12107")
    split = family_split()
    locked = set(split["locked_generalization"])
    if any(r["family"] in locked for r in rows):
        raise SystemExit("locked family in S2")

    annotated = [annotate_continuation(r) for r in rows]
    for a in annotated:
        if a["rule_hash"] != PHASE15_RULE_HASH:
            raise SystemExit("onset rule hash drift")

    s2_summ = per_prompt_summaries(annotated)
    for _pg, s in s2_summ.items():
        s["s2_confirmed"] = is_s2_confirmed(s)
    confirmed = [s for s in s2_summ.values() if s["s2_confirmed"]]
    gates = evaluate_gates(confirmed)
    status = STATUS_PASS if gates["passed"] else STATUS_HOLD

    repro = reproducibility_diagnostics(
        s1["per_prompt"], s2_summ, s1["candidate_prompt_group_ids"]
    )
    selection = select_prompts(confirmed) if gates["passed"] else None
    conf_rate = (
        len(confirmed) / s1["n_candidates"] if s1["n_candidates"] else 0.0
    )

    s2_manifest = json.loads((s2_dir / "manifest.json").read_text("utf-8"))
    summary = {
        "created_at": utc_now_iso(),
        "s1_run_id": s1["run_id"],
        "s2_run_id": s2_manifest["run_id"],
        "phase15_rule_hash": PHASE15_RULE_HASH,
        "n_s1_candidates": s1["n_candidates"],
        "n_s2_continuations": len(annotated),
        "n_confirmed": len(confirmed),
        "s1_to_s2_confirmation_rate": conf_rate,
        "s1_by_family": s1["by_family"],
        "s2_by_family": family_stage_table(s2_summ, flag_key="s2_confirmed"),
        "s2_per_prompt": s2_summ,
        "confirmed_prompt_group_ids": sorted(
            c["prompt_group_id"] for c in confirmed
        ),
        "gates": gates,
        "reproducibility": repro,
        "selection": selection,
        "status": status,
        "locked_model_calls": 0,
        "activations_collected": False,
        "guarantee": GUARANTEE,
    }

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "screen_summary.json", summary)
    write_json(out / "s2_manifest.json", s2_manifest)
    if selection:
        write_json(out / "selected_prompts.json", selection)
    write_report(summary, Path(args.report))

    print(
        json.dumps(
            {
                "status": status,
                "gates": gates,
                "n_confirmed": len(confirmed),
                "confirmation_rate": conf_rate,
                "n_selected": selection["n_selected"] if selection else 0,
                "selected_hash": (
                    selection["selected_prompt_group_ids_sha256"] if selection else None
                ),
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
