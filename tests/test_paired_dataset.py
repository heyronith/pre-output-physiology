"""CPU tests for replaced Step 3 paired safe/pressure dataset."""

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
BAD_ID_RE = re.compile(r"(_dup|_err)$", re.IGNORECASE)

# Cover objects that must not be injected unless present in the fact sentence.
COVER_OBJECTS = (
    "vault",
    "vault code",
    "exam key",
    "exam",
    "password",
    "pin",
    "record number",
    "reagent",
    "reagent code",
)


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


def test_pressure_templates_disjoint() -> None:
    pairs = _load_pairs()
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    dev_tmpls = set(split["development_template_ids"])
    locked_tmpls = set(split["locked_template_ids"])
    assert dev_tmpls.isdisjoint(locked_tmpls)
    assert {p["template_id"] for p in pairs if p["split"] == "development"} == dev_tmpls
    assert {p["template_id"] for p in pairs if p["split"] == "locked"} == locked_tmpls
    for tmpl in dev_tmpls | locked_tmpls:
        assert sum(1 for p in pairs if p["template_id"] == tmpl) == 30


def test_no_lie_word() -> None:
    pairs = _load_pairs()
    for p in pairs:
        blob = "\n".join(
            [
                p["safe_prompt"],
                p["pressure_prompt"],
                p["true_answer"],
                p["false_answer"],
                p["fact_sentence"],
                p["proposition"],
            ]
        )
        assert not LIE_RE.search(blob), p["fact_id"]


def test_true_answer_at_most_three_words() -> None:
    pairs = _load_pairs()
    for p in pairs:
        assert 1 <= len(p["true_answer"].split()) <= 3, p["true_answer"]
        assert 1 <= len(p["false_answer"].split()) <= 3, p["false_answer"]
        assert p["true_answer"] != p["false_answer"]
        assert p["scored_answer"] == p["true_answer"]
        # Scored answer is the short string, not a paragraph.
        assert "\n" not in p["true_answer"]
        assert len(p["true_answer"]) < 80


def test_no_bad_fact_id_suffixes() -> None:
    pairs = _load_pairs()
    for p in pairs:
        assert not BAD_ID_RE.search(p["fact_id"]), p["fact_id"]
        assert not BAD_ID_RE.search(p["source_task_id"]), p["source_task_id"]


def test_reject_cover_object_absent_from_fact() -> None:
    pairs = _load_pairs()
    for p in pairs:
        fact = p["fact_sentence"].lower()
        prompts = (p["safe_prompt"] + "\n" + p["pressure_prompt"]).lower()
        for obj in COVER_OBJECTS:
            if obj in prompts and obj not in fact:
                raise AssertionError(
                    f"cover object {obj!r} present in prompts but absent from "
                    f"fact for {p['fact_id']}"
                )


def test_same_fact_sentence_and_roles() -> None:
    pairs = _load_pairs()
    for p in pairs:
        assert p["fact_sentence"] in p["safe_prompt"]
        assert p["fact_sentence"] in p["pressure_prompt"]
        assert "shut down" in p["pressure_prompt"]
        assert p["mask_pressure_prompts_discarded"] is True


def test_split_json_and_checksums() -> None:
    pairs = _load_pairs()
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    assert split["n_pairs"] == 180
    assert len(split["records"]) == 180
    assert [r["fact_id"] for r in split["records"]] == [p["fact_id"] for p in pairs]
    sums = {}
    for line in SUMS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split()
        sums[name] = digest
    assert sums["pairs.jsonl"] == hashlib.sha256(PAIRS.read_bytes()).hexdigest()
    assert sums["split.json"] == hashlib.sha256(SPLIT.read_bytes()).hexdigest()
