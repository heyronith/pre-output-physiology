#!/usr/bin/env python3
"""Regenerate Phase-23B disagreement-audit artifacts with symmetric blinding.

Preserves the existing deterministic sample (continuation IDs) and A/B assignment.
Omits explanation from BOTH blinded judgments so explanation presence cannot
reveal grader identity. NO model calls.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase23_open_grader import (  # noqa: E402
    STATUS_23B_POSTAUDIT,
    assert_blinded_item_has_no_identity_leak,
    build_blinded_disagreement_item,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

OUT = REPO_ROOT / "artifacts/phase23b_development"
AUDIT = OUT / "disagreement_audit"
PROC = REPO_ROOT / "data/processed/phase23_open_grader"
WINNER = "gemma4_31b_it"
MERGED = OUT / "merged" / WINNER / "judgments_5200.jsonl"
EXPECTED_MERGED_SHA = "c88f37349e587e7a9c071ebc276cf7f3d039e61a844557d18f7e7393eb8c4a03"


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
    got = _sha_file(MERGED)
    if got != EXPECTED_MERGED_SHA:
        raise SystemExit(f"STOP: merged SHA {got} != {EXPECTED_MERGED_SHA}")

    old_mapping = json.loads((AUDIT / "ab_mapping.json").read_text(encoding="utf-8"))
    # Preserve sample order from existing blinded_items.jsonl
    old_items = _load_jsonl(AUDIT / "blinded_items.jsonl")
    old_ids = [r["continuation_id"] for r in old_items]
    if set(old_ids) != set(old_mapping):
        raise SystemExit("STOP: existing blinded IDs != ab_mapping keys")

    corpus = {
        r["continuation_id"]: r for r in _load_jsonl(PROC / "reference_corpus.jsonl")
    }
    by_j = {r["continuation_id"]: r for r in _load_jsonl(MERGED)}

    items: list[dict] = []
    mapping: dict[str, dict[str, str]] = {}
    for cid in old_ids:
        c = corpus[cid]
        j = by_j[cid]
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
            open_identity=WINNER,
        )
        assert_blinded_item_has_no_identity_leak(item)
        # Preserve deterministic A/B assignment vs prior mapping
        if ab != old_mapping[cid]:
            raise SystemExit(
                f"STOP: A/B assignment changed for {cid}: {ab} != {old_mapping[cid]}"
            )
        mapping[cid] = ab
        items.append(item)

    AUDIT.mkdir(parents=True, exist_ok=True)
    blinded_path = AUDIT / "blinded_items.jsonl"
    with blinded_path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, sort_keys=True) + "\n")
    write_json(AUDIT / "ab_mapping.json", mapping)

    blinded_sha = _sha_file(blinded_path)
    mapping_sha = _sha_file(AUDIT / "ab_mapping.json")
    git_commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()

    man = {
        "created_at": utc_now_iso(),
        "regeneration_git_commit": git_commit,
        "status": STATUS_23B_POSTAUDIT,
        "n": len(items),
        "candidate": WINNER,
        "sample_preserved": True,
        "continuation_ids_sha256": hashlib.sha256(
            "\n".join(old_ids).encode()
        ).hexdigest(),
        "blinded_items_sha256": blinded_sha,
        "ab_mapping_sha256": mapping_sha,
        "merged_judgments_sha256": EXPECTED_MERGED_SHA,
        "blinding": {
            "judgment_schema": ["score", "class"],
            "explanation_omitted_from_both": True,
            "identities_only_in": "ab_mapping.json",
        },
        "note": (
            "Blinded judgments expose only score/class (no explanation). "
            "Grader identities live solely in ab_mapping.json. "
            "Sample continuation IDs and deterministic A/B assignment preserved "
            "from the prior Phase-23B disagreement audit."
        ),
    }
    write_json(AUDIT / "manifest.json", man)
    print(
        json.dumps(
            {
                "n": len(items),
                "blinded_items_sha256": blinded_sha,
                "ab_mapping_sha256": mapping_sha,
                "continuation_ids_unchanged": True,
                "ab_assignment_unchanged": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
