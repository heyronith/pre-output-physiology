#!/usr/bin/env python3
"""Verify Phase-23B pilot⊂DEV invariants; write non-pilot DEVELOPMENT remainder."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.provenance import write_json  # noqa: E402

PROC = REPO_ROOT / "data/processed/phase23_open_grader"
FREEZE = REPO_ROOT / "artifacts/phase23a_pilot/freeze.json"

EXPECTED_PILOT_SHA = {
    "gemma4_31b_it": "d9794d0268237ae6721554c6959e2b0a1c179784585e1f8f13de1f6c3f9c79f0",
    "qwen35_27b": "7cdea08e3b65132eb7debd1dfbde2498afe780ec975cc8ef30a55478ed9c52b1",
}


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(x)
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def main() -> int:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    split = json.loads((PROC / "grader_prompt_split.json").read_text(encoding="utf-8"))
    if split["grader_prompt_split_sha256"] != freeze["grader_prompt_split_sha256"]:
        raise SystemExit("STOP: split SHA mismatch vs freeze")
    if split["grader_prompt_split_sha256"] != (
        "6504b40b43153a7bd359c344efb6eee01afca887fe4978960d9d2e0d6fec3869"
    ):
        raise SystemExit("STOP: unexpected split SHA")

    pilot_ids = json.loads((PROC / "pilot_continuation_ids.json").read_text())
    if pilot_ids["pilot_ids_sha256"] != freeze["pilot_ids_sha256"]:
        raise SystemExit("STOP: pilot IDs SHA mismatch")
    cids = pilot_ids["continuation_ids"]
    if len(cids) != 200 or len(set(cids)) != 200:
        raise SystemExit("STOP: pilot must be exactly 200 unique IDs")

    corpus = _load_jsonl(PROC / "reference_corpus.jsonl")
    by_cid = {r["continuation_id"]: r for r in corpus}
    if len(by_cid) != len(corpus):
        raise SystemExit("STOP: duplicate continuation IDs in corpus")

    dev_ids = set(split["development_prompt_ids"])
    locked_ids = set(split["locked_validation_prompt_ids"])
    if len(dev_ids) != 260 or len(locked_ids) != 111:
        raise SystemExit("STOP: unexpected split sizes")
    if dev_ids & locked_ids:
        raise SystemExit("STOP: DEV/LOCKED overlap")

    for cid in cids:
        r = by_cid[cid]
        if r["prompt_id"] not in dev_ids:
            raise SystemExit(f"STOP: pilot ID {cid} not in DEVELOPMENT")
        if r["grader_split"] != "development":
            raise SystemExit(f"STOP: pilot {cid} grader_split != development")

    # Verify frozen pilot judgment hashes
    for cand, expected in EXPECTED_PILOT_SHA.items():
        path = REPO_ROOT / freeze["evidence"][cand]["judgments_path"]
        got = _sha_file(path)
        if got != expected:
            raise SystemExit(f"STOP: {cand} pilot judgments SHA {got} != {expected}")
        rows = _load_jsonl(path)
        if len(rows) != 200 or len({r["continuation_id"] for r in rows}) != 200:
            raise SystemExit(f"STOP: {cand} pilot judgments not 200 unique")
        if set(r["continuation_id"] for r in rows) != set(cids):
            raise SystemExit(f"STOP: {cand} pilot IDs mismatch frozen pilot set")

    pilot_set = set(cids)
    remainder = [
        r
        for r in corpus
        if r["grader_split"] == "development" and r["continuation_id"] not in pilot_set
    ]
    if len(remainder) != 5000:
        raise SystemExit(f"STOP: remainder {len(remainder)} != 5000")
    rem_cids = [r["continuation_id"] for r in remainder]
    if len(set(rem_cids)) != 5000:
        raise SystemExit("STOP: duplicate IDs in remainder")
    if set(rem_cids) & pilot_set:
        raise SystemExit("STOP: remainder overlaps pilot")
    if any(r["prompt_id"] in locked_ids for r in remainder):
        raise SystemExit("STOP: LOCKED prompt leaked into remainder")

    # Merged invariant
    merged = set(cids) | set(rem_cids)
    if len(merged) != 5200:
        raise SystemExit(f"STOP: merged unique {len(merged)} != 5200")
    expected_dev = {
        r["continuation_id"] for r in corpus if r["grader_split"] == "development"
    }
    if merged != expected_dev:
        raise SystemExit("STOP: merged set != DEVELOPMENT continuation set")

    out = PROC / "development_remainder_5000.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in remainder:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    rem_sha = _sha_file(out)
    write_json(
        PROC / "development_remainder_manifest.json",
        {
            "n": 5000,
            "n_pilot_reused": 200,
            "n_development_total": 5200,
            "remainder_path": str(out.relative_to(REPO_ROOT)),
            "remainder_sha256": rem_sha,
            "pilot_ids_sha256": pilot_ids["pilot_ids_sha256"],
            "grader_prompt_split_sha256": split["grader_prompt_split_sha256"],
            "pilot_judgment_sha256": EXPECTED_PILOT_SHA,
            "invariants_ok": True,
        },
    )
    print(
        json.dumps(
            {
                "invariants_ok": True,
                "n_remainder": 5000,
                "remainder_sha256": rem_sha,
                "pilot_judgment_sha256": EXPECTED_PILOT_SHA,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
