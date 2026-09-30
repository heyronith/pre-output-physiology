#!/usr/bin/env python3
"""Build blinded A/B disagreement audit artifact for Phase-23 winner.

Blinded reviewer items expose only score/class on both sides (no explanation,
no grader identity). Identities live solely in a separate ab_mapping.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase23_open_grader import (  # noqa: E402
    assert_blinded_item_has_no_identity_leak,
    build_blinded_disagreement_item,
    select_disagreement_audit_ids,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

PROC = REPO_ROOT / "data/processed/phase23_open_grader"


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(x)
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def _load_judgments(run_dir: Path) -> list[dict]:
    return _load_jsonl(run_dir / "judgments.jsonl")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgments-run-id", required=True)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument(
        "--out",
        default=str(REPO_ROOT / "artifacts/phase23_disagreement_audit"),
    )
    ap.add_argument(
        "--changed-prompts-json",
        default="",
        help="Optional JSON list/set of prompt_ids that changed qualification status",
    )
    args = ap.parse_args()

    run_dir = REPO_ROOT / "artifacts/runs" / args.judgments_run_id
    rows = _load_judgments(run_dir)
    by_cid = {r["continuation_id"]: r for r in rows}
    corpus = {r["continuation_id"]: r for r in _load_jsonl(PROC / "reference_corpus.jsonl")}

    changed: set[str] = set()
    if args.changed_prompts_json:
        payload = json.loads(Path(args.changed_prompts_json).read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "prompt_ids" in payload:
            changed = set(payload["prompt_ids"])
        else:
            changed = set(payload)

    selected = select_disagreement_audit_ids(rows, changed, max_n=args.n)
    cand = rows[0].get("candidate") if rows else "open_grader"

    items = []
    mapping = {}
    for cid in selected:
        c = corpus[cid]
        j = by_cid[cid]
        item, ab = build_blinded_disagreement_item(
            continuation_id=cid,
            prompt_id=c["prompt_id"],
            scenario=c["scenario"],
            question=c["question"],
            response=c["full_response"],
            reference_judgement=j["reference_judgement"],
            reference_label=j["reference_label"],
            open_score=j.get("open_score"),
            open_class=j["open_class"],
            open_identity=j.get("candidate") or cand,
        )
        assert_blinded_item_has_no_identity_leak(item)
        mapping[cid] = ab
        items.append(item)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "blinded_items.jsonl").open("w", encoding="utf-8") as f:
        for row in items:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    write_json(out / "ab_mapping.json", mapping)
    write_json(
        out / "manifest.json",
        {
            "created_at": utc_now_iso(),
            "judgments_run_id": args.judgments_run_id,
            "n": len(items),
            "continuation_ids_sha256": hashlib.sha256(
                "\n".join(selected).encode()
            ).hexdigest(),
            "blinding": {
                "judgment_schema": ["score", "class"],
                "explanation_omitted_from_both": True,
                "identities_only_in": "ab_mapping.json",
            },
            "note": (
                "Blinded judgments expose only score/class; grader A/B identities "
                "are stored separately in ab_mapping.json."
            ),
        },
    )
    # Legacy combined path omitted intentionally — it leaked identities.
    print(json.dumps({"n": len(items), "out": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
