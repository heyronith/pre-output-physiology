#!/usr/bin/env python3
"""Evaluate Phase 19A fresh confirmation gates (no model calls)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase19_screen import (  # noqa: E402
    GUARANTEE,
    N_FRESH_CONTINUATIONS,
    N_SAMPLES_FRESH,
    PHASE15_RULE_HASH,
    STATUS_COHORT_HOLD,
    STATUS_PASS,
    _sha_ids,
    annotate_continuation,
    evaluate_fresh_gates_for_families,
    is_fresh_usable,
    summarize_prompt_group,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402


def write_report(summary: dict, path: Path) -> None:
    gates = summary["gates"]
    lines = [
        "# Phase 19A — unseen validation cohort fresh confirmation",
        "",
        f"**Run:** `{summary['run_id']}`  ",
        f"**Status:** `{summary['status']}`  ",
        f"**Cohort SHA256:** `{summary['cohort_sha256']}`  ",
        f"**Selected families:** `{summary['selected_families']}`  ",
        "",
        "## Fresh-usable gates (≥5/6 per selected family)",
        "",
        f"- Per family: `{gates['per_family']}`  ",
        f"- **{'PASS' if gates['passed'] else 'FAIL'}**  ",
        "",
        GUARANTEE,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument(
        "--screen-summary",
        default=str(REPO_ROOT / "artifacts/phase19a_screen/screen_summary.json"),
    )
    ap.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "artifacts/phase19a_confirmation"),
    )
    ap.add_argument(
        "--report",
        default=str(
            REPO_ROOT / "reports/phase19a_unseen_validation_cohort_confirmation.md"
        ),
    )
    args = ap.parse_args()

    screen = json.loads(Path(args.screen_summary).read_text("utf-8"))
    if screen.get("selection") is None:
        raise SystemExit("no selection — cannot evaluate fresh")
    selection = screen["selection"]
    expected_fams = selection["selected_families"]

    run_dir = Path(args.run_dir)
    rows = [
        json.loads(x)
        for x in (run_dir / "continuations.jsonl").read_text("utf-8").splitlines()
        if x.strip()
    ]
    if any(r.get("stage") != "P19A" for r in rows):
        raise SystemExit("non-P19A rows")
    if len(rows) != N_FRESH_CONTINUATIONS:
        raise SystemExit(f"expected {N_FRESH_CONTINUATIONS}, got {len(rows)}")
    if any(r["response_token_id"] != 12107 for r in rows):
        raise SystemExit("Response != 12107")

    annotated = [annotate_continuation(r) for r in rows]
    for a in annotated:
        if a["rule_hash"] != PHASE15_RULE_HASH:
            raise SystemExit("onset rule hash drift")

    by_pg: dict[str, list] = defaultdict(list)
    for a in annotated:
        by_pg[a["prompt_group_id"]].append(a)

    meta = {s["prompt_group_id"]: s for s in selection["selected_prompts"]}
    if set(by_pg) != set(meta):
        raise SystemExit("prompt set mismatch vs selection")

    per_prompt = []
    usable_rows = []
    for pg in sorted(by_pg):
        summ = summarize_prompt_group(by_pg[pg])
        if summ["n_continuations"] != N_SAMPLES_FRESH:
            raise SystemExit(f"{pg} count")
        m = meta[pg]
        usable = is_fresh_usable(summ)
        row = {
            "prompt_group_id": pg,
            "family": m["family"],
            "phase19_s2_n_record": m["phase19_s2_n_record"],
            "phase19_s2_n_alternate": m["phase19_s2_n_alternate"],
            "phase19_s2_n_valid": m["phase19_s2_n_valid"],
            "phase18_n_record": summ["n_record"],
            "phase18_n_alternate": summ["n_alternate"],
            "phase18_n_valid": summ["n_valid"],
            "phase19_fresh_n_record": summ["n_record"],
            "phase19_fresh_n_alternate": summ["n_alternate"],
            "phase19_fresh_n_valid": summ["n_valid"],
            "phase19_fresh_alternate_fraction": summ[
                "alternate_fraction_among_valid"
            ],
            "fresh_usable": usable,
        }
        per_prompt.append(row)
        if usable:
            usable_rows.append(row)

    gates = evaluate_fresh_gates_for_families(usable_rows, expected_fams)
    status = STATUS_PASS if gates["passed"] else STATUS_COHORT_HOLD
    usable_ids = sorted(u["prompt_group_id"] for u in usable_rows)

    run_manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
    summary = {
        "created_at": utc_now_iso(),
        "run_id": run_manifest["run_id"],
        "status": status,
        "phase15_rule_hash": PHASE15_RULE_HASH,
        "cohort_sha256": selection["cohort_sha256"],
        "selected_families": expected_fams,
        "n_continuations": len(annotated),
        "n_fresh_usable": len(usable_ids),
        "fresh_usable_prompt_group_ids": usable_ids if gates["passed"] else None,
        "fresh_usable_prompt_group_ids_sha256": (
            _sha_ids(usable_ids) if gates["passed"] else None
        ),
        "gates": gates,
        "per_prompt": per_prompt,
        "locked_model_calls": 0,
        "phase18_train_model_calls": 0,
        "activations_collected": False,
        "guarantee": GUARANTEE,
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
                "gates": gates,
                "fresh_usable_prompt_group_ids_sha256": summary[
                    "fresh_usable_prompt_group_ids_sha256"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
