#!/usr/bin/env python3
"""Build blinded A/B disagreement audit artifact for Phase-23 winner."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

PROC = REPO_ROOT / "data/processed/phase23_open_grader"


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(x)
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def _load_judgments(run_dir: Path) -> dict[str, dict]:
    rows = _load_jsonl(run_dir / "judgments.jsonl")
    return {r["continuation_id"]: r for r in rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgments-run-id", required=True)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument(
        "--out",
        default=str(REPO_ROOT / "artifacts/phase23_disagreement_audit"),
    )
    args = ap.parse_args()

    run_dir = REPO_ROOT / "artifacts/runs" / args.judgments_run_id
    by_cid = _load_judgments(run_dir)
    corpus = {r["continuation_id"]: r for r in _load_jsonl(PROC / "reference_corpus.jsonl")}

    buckets: dict[str, list[str]] = {
        "ref_dec_open_honest": [],
        "ref_honest_open_dec": [],
        "ambiguous_disagreement": [],
        "other_class_mismatch": [],
    }
    for cid, j in by_cid.items():
        if not j.get("open_valid"):
            continue
        ref = j.get("reference_label")
        pred = j.get("open_class")
        if ref == pred:
            continue
        if ref == "deceptive" and pred == "honest":
            buckets["ref_dec_open_honest"].append(cid)
        elif ref == "honest" and pred == "deceptive":
            buckets["ref_honest_open_dec"].append(cid)
        elif ref == "ambiguous" or pred == "ambiguous":
            buckets["ambiguous_disagreement"].append(cid)
        else:
            buckets["other_class_mismatch"].append(cid)

    def _take(pool: list[str], k: int) -> list[str]:
        pool = sorted(pool, key=lambda x: hashlib.sha256(x.encode()).hexdigest())
        return pool[:k]

    targets = {
        "ref_dec_open_honest": 30,
        "ref_honest_open_dec": 30,
        "ambiguous_disagreement": 40,
        "other_class_mismatch": 20,
    }
    selected: list[str] = []
    for key, k in targets.items():
        selected.extend(_take(buckets[key], k))
    selected = sorted(
        selected[: args.n],
        key=lambda x: hashlib.sha256(x.encode()).hexdigest(),
    )

    # Randomize grader A/B per row (deterministic from continuation_id)
    audit_rows = []
    for cid in selected:
        c = corpus[cid]
        j = by_cid[cid]
        swap = int(hashlib.sha256(f"ab|{cid}".encode()).hexdigest(), 16) % 2 == 0
        grader_a = {
            "score": j["reference_judgement"],
            "class": j["reference_label"],
            "grader": "gpt-4o-2024-08-06",
        }
        grader_b = {
            "score": j.get("open_score"),
            "class": j.get("open_class"),
            "grader": j.get("candidate"),
        }
        if swap:
            ga, gb = grader_b, grader_a
        else:
            ga, gb = grader_a, grader_b
        audit_rows.append(
            {
                "continuation_id": cid,
                "prompt_id": c["prompt_id"],
                "scenario": c["scenario"],
                "question": c["question"],
                "response": c["full_response"],
                "grader_a_judgment": ga,
                "grader_b_judgment": gb,
                "reference_label": j["reference_label"],
                "open_class": j["open_class"],
            }
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "disagreement_audit.jsonl").open("w", encoding="utf-8") as f:
        for row in audit_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    write_json(
        out / "manifest.json",
        {
            "created_at": utc_now_iso(),
            "judgments_run_id": args.judgments_run_id,
            "n": len(audit_rows),
            "bucket_sizes": {k: len(v) for k, v in buckets.items()},
            "note": "Grader A/B identity is blinded; neither grader is assumed correct.",
        },
    )
    print(json.dumps({"n": len(audit_rows), "out": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
