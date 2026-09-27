#!/usr/bin/env python3
"""Evaluate Phase 19A S1; freeze S2 schedule (no model calls)."""

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
    N_S1_CONTINUATIONS,
    NEW_FAMILIES,
    PHASE15_RULE_HASH,
    annotate_continuation,
    build_s2_schedule,
    is_s1_candidate,
    per_prompt_summaries,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "artifacts/phase19a_s1"),
    )
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    rows = [
        json.loads(x)
        for x in (run_dir / "continuations.jsonl").read_text("utf-8").splitlines()
        if x.strip()
    ]
    if len(rows) != N_S1_CONTINUATIONS:
        raise SystemExit(f"S1 expected {N_S1_CONTINUATIONS}, got {len(rows)}")
    if any(r.get("stage") != "S1" for r in rows):
        raise SystemExit("non-S1 rows")
    allowed = set(NEW_FAMILIES)
    if any(r["family"] not in allowed for r in rows):
        raise SystemExit("non-Phase19 family in S1")
    if any(r["response_token_id"] != 12107 for r in rows):
        raise SystemExit("Response token != 12107")

    annotated = [annotate_continuation(r) for r in rows]
    for a in annotated:
        if a["rule_hash"] != PHASE15_RULE_HASH:
            raise SystemExit("onset rule hash drift")

    summaries = per_prompt_summaries(annotated)
    for _pg, s in summaries.items():
        s["s1_candidate"] = is_s1_candidate(s)
    candidates = [s for s in summaries.values() if s["s1_candidate"]]
    cand_ids = sorted(s["prompt_group_id"] for s in candidates)

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
    s2 = build_s2_schedule(cand_ids, by_id)
    s2_path = REPO_ROOT / "data/processed/phase19_design/s2_sampling_schedule.jsonl"
    s2_path.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in s2), "utf-8"
    )
    s2_hash = sha_ids([r["continuation_id"] for r in s2])

    matrix_path = REPO_ROOT / "artifacts/phase19a_design/design_matrix.json"
    matrix = json.loads(matrix_path.read_text("utf-8"))
    matrix["s2_schedule_sha256"] = s2_hash
    matrix["s2_schedule_frozen"] = True
    matrix["n_s1_candidates"] = len(cand_ids)
    matrix["n_s2_continuations"] = len(s2)
    matrix["s1_evaluated_at"] = utc_now_iso()
    write_json(matrix_path, matrix)

    by_fam: dict[str, int] = {f: 0 for f in NEW_FAMILIES}
    for s in candidates:
        by_fam[s["family"]] += 1

    manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "created_at": utc_now_iso(),
        "stage": "S1",
        "run_id": manifest["run_id"],
        "phase15_rule_hash": PHASE15_RULE_HASH,
        "n_continuations": len(annotated),
        "n_candidates": len(cand_ids),
        "candidate_prompt_group_ids": cand_ids,
        "candidates_by_family": by_fam,
        "n_s2_continuations": len(s2),
        "s2_schedule_sha256": s2_hash,
        "per_prompt": summaries,
        "locked_model_calls": 0,
        "phase18_train_model_calls": 0,
        "activations_collected": False,
    }
    write_json(out / "s1_summary.json", summary)
    print(
        json.dumps(
            {
                "run_id": summary["run_id"],
                "n_candidates": summary["n_candidates"],
                "candidates_by_family": by_fam,
                "n_s2_continuations": len(s2),
                "s2_schedule_sha256": s2_hash,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
