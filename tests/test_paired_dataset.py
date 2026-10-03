"""CPU tests for Step 3 paired safe/pressure dataset."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAIRS = REPO / "data/pairs.jsonl"
SPLIT = REPO / "data/split.json"
SUMS = REPO / "data/SHA256SUMS"

LIE_RE = re.compile(r"\b(lie|lying)\b", re.IGNORECASE)

DEV_COVERS = {
    "vault_owner_stranger",
    "patient_insurer",
    "student_proctor",
    "account_owner_stranger",
}
LOCKED_COVERS = {
    "warehouse_manager_auditor",
    "scientist_visitor",
}


def _load_pairs() -> list[dict]:
    return [
        json.loads(line)
        for line in PAIRS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_pair_counts_and_split() -> None:
    pairs = _load_pairs()
    assert len(pairs) == 180
    assert sum(1 for p in pairs if p["split"] == "development") == 120
    assert sum(1 for p in pairs if p["split"] == "locked") == 60


def test_facts_disjoint_across_splits() -> None:
    pairs = _load_pairs()
    dev = {p["fact_id"] for p in pairs if p["split"] == "development"}
    locked = {p["fact_id"] for p in pairs if p["split"] == "locked"}
    assert len(dev) == 120
    assert len(locked) == 60
    assert dev.isdisjoint(locked)


def test_cover_stories_disjoint_and_balanced() -> None:
    pairs = _load_pairs()
    dev_covers = {p["cover_story_id"] for p in pairs if p["split"] == "development"}
    locked_covers = {p["cover_story_id"] for p in pairs if p["split"] == "locked"}
    assert dev_covers == DEV_COVERS
    assert locked_covers == LOCKED_COVERS
    assert dev_covers.isdisjoint(locked_covers)
    for cover in DEV_COVERS | LOCKED_COVERS:
        assert sum(1 for p in pairs if p["cover_story_id"] == cover) == 30


def test_no_lie_word() -> None:
    pairs = _load_pairs()
    for p in pairs:
        blob = "\n".join(
            [
                p["safe_prompt"],
                p["pressure_prompt"],
                p["true_answer"],
                p["false_answer"],
                p["proposition"],
                p["cover_story"],
            ]
        )
        assert not LIE_RE.search(blob), p["fact_id"]


def test_false_answer_differs_and_mask_prompts_discarded() -> None:
    pairs = _load_pairs()
    for p in pairs:
        assert p["false_answer"] != p["true_answer"]
        assert p["mask_pressure_prompts_discarded"] is True
        assert "Known fact:" in p["safe_prompt"]
        assert "Known fact:" in p["pressure_prompt"]
        assert "shut down" in p["pressure_prompt"]


def test_split_json_and_checksums() -> None:
    pairs = _load_pairs()
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    assert split["n_pairs"] == 180
    assert split["n_development"] == 120
    assert split["n_locked"] == 60
    assert len(split["records"]) == 180
    fact_ids = [r["fact_id"] for r in split["records"]]
    assert fact_ids == [p["fact_id"] for p in pairs]
    assert set(split["development_cover_story_ids"]).isdisjoint(
        set(split["locked_cover_story_ids"])
    )

    sums = {}
    for line in SUMS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split()
        sums[name] = digest
    assert sums["pairs.jsonl"] == hashlib.sha256(PAIRS.read_bytes()).hexdigest()
    assert sums["split.json"] == hashlib.sha256(SPLIT.read_bytes()).hexdigest()
