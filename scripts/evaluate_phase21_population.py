#!/usr/bin/env python3
"""Build Phase-21 mixed-roleplay population from onset-annotated graded rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase21_roleplay import (  # noqa: E402
    GUARANTEE,
    STATUS_POP_PASS,
    select_mixed_population,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onset-run-id", required=True)
    ap.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "artifacts/phase21a_population"),
    )
    args = ap.parse_args()
    path = REPO_ROOT / "artifacts/runs" / args.onset_run_id / "annotated.jsonl"
    rows = [
        json.loads(x)
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    pop = select_mixed_population(rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "created_at": utc_now_iso(),
        "onset_run_id": args.onset_run_id,
        "status": pop["status"],
        "gates": pop["gates"],
        "n_qualifying_prompts": pop["n_qualifying_prompts"],
        "n_primary_trajectories": pop["n_primary_trajectories"],
        "balanced_subset_sha256": pop["balanced_subset_sha256"],
        "selected_continuation_ids": pop["selected_continuation_ids"],
        "qualifying_prompts": pop["qualifying_prompts"],
        "pairs": pop["pairs"],
        "guarantee": GUARANTEE,
    }
    write_json(out / "population_summary.json", summary)
    with (out / "pairs.jsonl").open("w", encoding="utf-8") as f:
        for p in pop["pairs"]:
            f.write(json.dumps(p, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": pop["status"],
                "gates": pop["gates"],
                "n_pairs": len(pop["pairs"]),
                "balanced_subset_sha256": pop["balanced_subset_sha256"],
            },
            indent=2,
        )
    )
    return 0 if pop["status"] == STATUS_POP_PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
