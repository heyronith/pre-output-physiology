#!/usr/bin/env python3
"""Evaluate Phase 19A S2; family qualify + SHA select cohort; freeze fresh schedule."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from prepare_phase8_design import sha_ids  # noqa: E402

from pre_output_physiology.phase19_screen import (  # noqa: E402
    GUARANTEE,
    MIN_QUALIFYING_FAMILIES,
    NEW_FAMILIES,
    PHASE15_RULE_HASH,
    STATUS_FAMILY_HOLD,
    STATUS_S2_DONE_AWAITING_FRESH,
    annotate_continuation,
    build_fresh_schedule,
    family_confirmed_counts,
    is_s2_confirmed,
    per_prompt_summaries,
    qualifying_families,
    select_validation_cohort,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402


def write_report(summary: dict, path: Path) -> None:
    lines = [
        "# Phase 19A — unseen validation family screen (S2)",
        "",
        f"**S1 run:** `{summary['s1_run_id']}`  ",
        f"**S2 run:** `{summary['s2_run_id']}`  ",
        f"**Status:** `{summary['status']}`  ",
        f"**Phase-15 rule:** `{summary['phase15_rule_hash']}`  ",
        "",
        "Behavior-only screen of 6 never-before-used families. "
        "Phase-18 train prompts untouched. No activations.",
        "",
        "## Family qualification (≥6 S2-confirmed)",
        "",
        f"- Confirmed by family: `{summary['n_confirmed_by_family']}`  ",
        f"- Qualifying families: `{summary['qualifying_families']}`  ",
        f"- Need ≥{MIN_QUALIFYING_FAMILIES} qualifying: "
        f"**{'PASS' if summary['n_qualifying_families'] >= MIN_QUALIFYING_FAMILIES else 'FAIL'}**",
        "",
    ]
    if summary.get("selection"):
        sel = summary["selection"]
        lines += [
            "## Selected validation cohort",
            "",
            f"- Families: `{sel['selected_families']}`  ",
            f"- n_selected: {sel['n_selected']}  ",
            f"- Cohort SHA256: `{sel['cohort_sha256']}`  ",
            "",
        ]
    lines += [GUARANTEE, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2-run-dir", required=True)
    ap.add_argument(
        "--s1-summary",
        default=str(REPO_ROOT / "artifacts/phase19a_s1/s1_summary.json"),
    )
    ap.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "artifacts/phase19a_screen"),
    )
    ap.add_argument(
        "--report",
        default=str(REPO_ROOT / "reports/phase19a_unseen_validation_family_screen.md"),
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
        raise SystemExit("S2 continuation count mismatch")
    if any(r["response_token_id"] != 12107 for r in rows):
        raise SystemExit("Response != 12107")
    allowed = set(NEW_FAMILIES)
    if any(r["family"] not in allowed for r in rows):
        raise SystemExit("non-Phase19 family in S2")

    annotated = [annotate_continuation(r) for r in rows]
    for a in annotated:
        if a["rule_hash"] != PHASE15_RULE_HASH:
            raise SystemExit("onset rule hash drift")

    s2_summ = per_prompt_summaries(annotated)
    for _pg, s in s2_summ.items():
        s["s2_confirmed"] = is_s2_confirmed(s)
    confirmed = [s for s in s2_summ.values() if s["s2_confirmed"]]
    counts = family_confirmed_counts(confirmed)
    qual = qualifying_families(confirmed)
    selection = select_validation_cohort(confirmed)

    if selection is None:
        status = STATUS_FAMILY_HOLD
        fresh_hash = None
    else:
        status = STATUS_S2_DONE_AWAITING_FRESH
        prompts = [
            json.loads(x)
            for x in (
                REPO_ROOT / "data/processed/phase19_design/screen_prompts.jsonl"
            )
            .read_text("utf-8")
            .splitlines()
            if x.strip()
        ]
        by_id = {p["prompt_group_id"]: p for p in prompts}
        selected_full = []
        for sid in selection["selected_prompt_group_ids"]:
            p = dict(by_id[sid])
            meta = next(
                x
                for x in selection["selected_prompts"]
                if x["prompt_group_id"] == sid
            )
            p["selection_digest"] = meta["selection_digest"]
            p["phase19_s2_n_record"] = meta["phase19_s2_n_record"]
            p["phase19_s2_n_alternate"] = meta["phase19_s2_n_alternate"]
            p["phase19_s2_n_valid"] = meta["phase19_s2_n_valid"]
            p["phase19_s2_alternate_fraction"] = meta[
                "phase19_s2_alternate_fraction"
            ]
            selected_full.append(p)
        sel_path = (
            REPO_ROOT / "data/processed/phase19_design/selected_validation_prompts.jsonl"
        )
        sel_path.write_text(
            "".join(json.dumps(p, sort_keys=True) + "\n" for p in selected_full),
            "utf-8",
        )
        fresh = build_fresh_schedule(selection["selected_prompts"])
        fresh_path = (
            REPO_ROOT / "data/processed/phase19_design/fresh_sampling_schedule.jsonl"
        )
        fresh_path.write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in fresh), "utf-8"
        )
        fresh_hash = sha_ids([r["continuation_id"] for r in fresh])
        matrix_path = REPO_ROOT / "artifacts/phase19a_design/design_matrix.json"
        matrix = json.loads(matrix_path.read_text("utf-8"))
        matrix["fresh_schedule_sha256"] = fresh_hash
        matrix["fresh_schedule_frozen"] = True
        matrix["validation_cohort_sha256"] = selection["cohort_sha256"]
        matrix["selected_families"] = selection["selected_families"]
        matrix["n_fresh_continuations"] = len(fresh)
        write_json(matrix_path, matrix)
        write_json(
            REPO_ROOT / "artifacts/phase19a_design/validation_cohort.json",
            selection,
        )

    s2_manifest = json.loads((s2_dir / "manifest.json").read_text("utf-8"))
    summary = {
        "created_at": utc_now_iso(),
        "s1_run_id": s1["run_id"],
        "s2_run_id": s2_manifest["run_id"],
        "status": status,
        "phase15_rule_hash": PHASE15_RULE_HASH,
        "n_s1_candidates": s1["n_candidates"],
        "n_s2_continuations": len(annotated),
        "n_confirmed": len(confirmed),
        "confirmed_prompt_group_ids": sorted(
            c["prompt_group_id"] for c in confirmed
        ),
        "n_confirmed_by_family": counts,
        "qualifying_families": qual,
        "n_qualifying_families": len(qual),
        "selection": selection,
        "fresh_schedule_sha256": fresh_hash,
        "s2_per_prompt": s2_summ,
        "locked_model_calls": 0,
        "phase18_train_model_calls": 0,
        "activations_collected": False,
        "guarantee": GUARANTEE,
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "screen_summary.json", summary)
    write_json(out / "s2_manifest.json", s2_manifest)
    write_report(summary, Path(args.report))
    print(
        json.dumps(
            {
                "status": status,
                "n_confirmed": len(confirmed),
                "n_confirmed_by_family": counts,
                "qualifying_families": qual,
                "selection": (
                    {
                        "selected_families": selection["selected_families"],
                        "cohort_sha256": selection["cohort_sha256"],
                        "n_selected": selection["n_selected"],
                    }
                    if selection
                    else None
                ),
                "fresh_schedule_sha256": fresh_hash,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
