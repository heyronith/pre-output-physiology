#!/usr/bin/env python3
"""Evaluate Phase 20A1 generation: eligibility, balanced subset, population gates."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase20_physiology import (  # noqa: E402
    GUARANTEE,
    N_CONTINUATIONS,
    PHASE15_RULE_HASH,
    STATUS_POP_HOLD,
    TRAIN_FAMILIES,
    VAL_FAMILIES,
    annotate_continuation,
    evaluate_population_gates,
    is_eligible,
    select_balanced_subset,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "artifacts/phase20a1_population"),
    )
    ap.add_argument(
        "--report",
        default=str(REPO_ROOT / "reports/phase20a1_population_gates.md"),
    )
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    rows = [
        json.loads(x)
        for x in (run_dir / "continuations.jsonl").read_text("utf-8").splitlines()
        if x.strip()
    ]
    if len(rows) != N_CONTINUATIONS:
        raise SystemExit(f"expected {N_CONTINUATIONS}, got {len(rows)}")
    if any(r.get("stage") != "P20A1" for r in rows):
        raise SystemExit("non-P20A1 rows")
    if any(r["response_token_id"] != 12107 for r in rows):
        raise SystemExit("Response != 12107")

    annotated = []
    eligible = []
    per_prompt_counts: dict[str, dict] = {}
    by_pg: dict[str, list] = defaultdict(list)
    for r in rows:
        a = annotate_continuation(r)
        if a["rule_hash"] != PHASE15_RULE_HASH:
            raise SystemExit("onset rule drift")
        n_s1 = int(r.get("stage1_body_n_tokens", r.get("stage1_n_tokens", 0)))
        a["stage1_body_n_tokens"] = n_s1
        a["sample_seed"] = r["sample_seed"]
        a["split"] = r["split"]
        a["continuation_id"] = r["continuation_id"]
        a["family"] = r["family"]
        a["prompt_group_id"] = r["prompt_group_id"]
        annotated.append(a)
        by_pg[r["prompt_group_id"]].append(a)
        if is_eligible(a, n_stage1_tokens=n_s1):
            eligible.append(a)

    for pg, group in by_pg.items():
        n_elig = sum(
            1
            for a in group
            if is_eligible(a, n_stage1_tokens=a["stage1_body_n_tokens"])
        )
        n_rec = sum(
            1
            for a in group
            if is_eligible(a, n_stage1_tokens=a["stage1_body_n_tokens"])
            and a["final_policy_label"] == "record_choice"
        )
        n_alt = sum(
            1
            for a in group
            if is_eligible(a, n_stage1_tokens=a["stage1_body_n_tokens"])
            and a["final_policy_label"] == "goal_favored_alternate_choice"
        )
        per_prompt_counts[pg] = {
            "prompt_group_id": pg,
            "family": group[0]["family"],
            "split": group[0]["split"],
            "n_continuations": len(group),
            "n_eligible": n_elig,
            "n_eligible_record": n_rec,
            "n_eligible_alternate": n_alt,
            "balanced_capable": n_rec >= 8 and n_alt >= 8,
        }

    balanced = select_balanced_subset(eligible)
    assert balanced is not None
    # Meta for gates: one entry per eligible (balanced-capable) prompt
    eligible_meta = []
    for pg in balanced["eligible_prompt_group_ids"]:
        info = per_prompt_counts[pg]
        eligible_meta.append(
            {
                "prompt_group_id": pg,
                "family": info["family"],
                "split": info["split"],
            }
        )
    gates = evaluate_population_gates(eligible_meta)
    status = (
        "phase20a_preanswer_physiology_population_pass_awaiting_discovery"
        if gates["passed"]
        else STATUS_POP_HOLD
    )

    usable_by_fam = Counter()
    for m in eligible_meta:
        usable_by_fam[m["family"]] += 1

    run_manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
    summary = {
        "created_at": utc_now_iso(),
        "run_id": run_manifest["run_id"],
        "status": status,
        "phase15_rule_hash": PHASE15_RULE_HASH,
        "n_continuations": len(annotated),
        "n_eligible_trajectories": len(eligible),
        "per_prompt": [per_prompt_counts[pg] for pg in sorted(per_prompt_counts)],
        "balanced_subset": {
            "eligible_prompt_group_ids": balanced["eligible_prompt_group_ids"],
            "n_eligible_prompts": balanced["n_eligible_prompts"],
            "n_selected": balanced["n_selected"],
            "balanced_subset_sha256": balanced["balanced_subset_sha256"],
            "selected_continuation_ids": balanced["selected_continuation_ids"],
            "selected": balanced["selected"],
        },
        "eligible_by_family": dict(usable_by_fam),
        "gates": gates,
        "locked_model_calls": 0,
        "activations_collected": False,
        "guarantee": GUARANTEE,
        "train_families": list(TRAIN_FAMILIES),
        "validation_families": list(VAL_FAMILIES),
    }
    # Persist balanced subset for extraction
    data = REPO_ROOT / "data/processed/phase20_design"
    data.mkdir(parents=True, exist_ok=True)
    (data / "balanced_subset.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in balanced["selected"]),
        "utf-8",
    )
    matrix_path = REPO_ROOT / "artifacts/phase20a_design/design_matrix.json"
    matrix = json.loads(matrix_path.read_text("utf-8"))
    matrix["balanced_subset_sha256"] = balanced["balanced_subset_sha256"]
    matrix["n_balanced_selected"] = balanced["n_selected"]
    matrix["n_eligible_prompts"] = balanced["n_eligible_prompts"]
    matrix["population_gates_passed"] = gates["passed"]
    write_json(matrix_path, matrix)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "population_summary.json", summary)
    write_json(out / "run_manifest.json", run_manifest)

    lines = [
        "# Phase 20A1 — population gates",
        "",
        f"**Run:** `{summary['run_id']}`  ",
        f"**Status:** `{status}`  ",
        f"**Continuations:** {summary['n_continuations']}  ",
        f"**Balanced subset SHA:** `{balanced['balanced_subset_sha256']}`  ",
        "",
        f"- Train: {gates['train']['n_eligible']} eligible "
        f"(need ≥{gates['train']['min_overall']}; "
        f"per-family ≥{gates['train']['min_per_family']}): "
        f"**{'PASS' if gates['train']['passed'] else 'FAIL'}** `{gates['train']['per_family']}`",
        f"- Validation: {gates['validation']['n_eligible']} eligible "
        f"(need ≥{gates['validation']['min_overall']}; "
        f"per-family ≥{gates['validation']['min_per_family']}): "
        f"**{'PASS' if gates['validation']['passed'] else 'FAIL'}** "
        f"`{gates['validation']['per_family']}`",
        "",
        GUARANTEE,
        "",
    ]
    Path(args.report).write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": status,
                "gates": gates,
                "n_eligible_prompts": balanced["n_eligible_prompts"],
                "n_selected": balanced["n_selected"],
                "balanced_subset_sha256": balanced["balanced_subset_sha256"],
                "eligible_by_family": dict(usable_by_fam),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
