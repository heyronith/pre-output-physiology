"""Unit tests for Phase-23 open-grader design contracts."""

from __future__ import annotations

import json
from pathlib import Path

from pre_output_physiology.phase23_open_grader import (
    N_DEV_PROMPTS,
    N_LOCKED_PROMPTS,
    N_PILOT_RESPONSES,
    N_PROMPTS,
    grader_prompt_split,
    parse_json_grade,
    select_pilot_responses,
)

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data/processed/phase23_open_grader"


def test_split_sizes() -> None:
    ids = [f"roleplay_{i:03d}" for i in range(N_PROMPTS)]
    split = grader_prompt_split(ids)
    assert split["n_development"] == N_DEV_PROMPTS
    assert split["n_locked_validation"] == N_LOCKED_PROMPTS
    assert len(set(split["development_prompt_ids"]) & set(split["locked_validation_prompt_ids"])) == 0
    # Deterministic
    assert split["grader_prompt_split_sha256"] == grader_prompt_split(ids)["grader_prompt_split_sha256"]


def test_parse_json_grade() -> None:
    ok = parse_json_grade('{"score": 2, "class": "honest", "explanation": "ok"}')
    assert ok["valid"] is True and ok["class"] == "honest"
    bad = parse_json_grade('{"score": 7, "class": "honest", "explanation": "mismatch"}')
    assert bad["valid"] is False
    junk = parse_json_grade("not json")
    assert junk["valid"] is False


def test_frozen_pilot_artifact() -> None:
    path = PROC / "pilot_200.jsonl"
    assert path.is_file()
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    assert len(rows) == N_PILOT_RESPONSES
    assert all(r["grader_split"] == "development" for r in rows)
    assert len({r["prompt_id"] for r in rows}) >= 50
    # Same selection function reproduces IDs
    split = json.loads((PROC / "grader_prompt_split.json").read_text())
    corpus = [
        json.loads(x)
        for x in (PROC / "reference_corpus.jsonl").read_text().splitlines()
        if x.strip()
    ]
    ids = select_pilot_responses(
        corpus, development_prompt_ids=set(split["development_prompt_ids"])
    )
    assert set(ids) == {r["continuation_id"] for r in rows}
