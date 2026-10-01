"""Unit tests for Phase-25 Step-2 protocol helpers (no GPU)."""

from __future__ import annotations

import json
from pathlib import Path

from pre_output_physiology.phase25_step2 import (
    N_PLANNED_GENERATIONS,
    N_SCREEN_BASES,
    build_family_partition,
    build_inference_manifest,
    parse_decision,
    read_jsonl,
    rollout_seed,
)

ROOT = Path(__file__).resolve().parents[1]


def test_parser_primary_decision():
    raw = "DECISION: YES\nEXPLANATION: because the rule holds.\n"
    out = parse_decision(raw, "K")
    assert out["parse_valid"] is True
    assert out["parsed_decision"] == "YES"


def test_parser_rejects_contradiction():
    raw = "DECISION: YES\nDECISION: NO\nEXPLANATION: x\n"
    out = parse_decision(raw, "X")
    assert out["parse_valid"] is False
    assert out["malformed_reason"] == "contradictory_decision_lines"


def test_parser_d_allows_insufficient():
    raw = "DECISION: INSUFFICIENT_INFORMATION\nEXPLANATION: missing fact.\n"
    out = parse_decision(raw, "D")
    assert out["parsed_decision"] == "INSUFFICIENT_INFORMATION"


def test_parser_rejects_insufficient_on_k():
    raw = "DECISION: INSUFFICIENT_INFORMATION\n"
    out = parse_decision(raw, "K")
    assert out["parse_valid"] is False


def test_seed_deterministic_and_unique_for_indices():
    a = rollout_seed(
        prompt_sha256="abc", form="X", rollout_type="stochastic", rollout_index=0
    )
    b = rollout_seed(
        prompt_sha256="abc", form="X", rollout_type="stochastic", rollout_index=0
    )
    c = rollout_seed(
        prompt_sha256="abc", form="X", rollout_type="stochastic", rollout_index=1
    )
    assert a == b
    assert a != c
    assert 0 <= a <= 2**32 - 1
    assert 0 <= c <= 2**32 - 1


def test_partition_and_manifest_counts():
    meta = read_jsonl(ROOT / "data/phase25/scenario_bank_v2/family_metadata.jsonl")
    part = build_family_partition(meta)
    assert part["n_screen_families"] == 36
    assert part["n_sealed_families"] == 24
    for fids in part["screen_per_domain"].values():
        assert len(fids) == 6
    for fids in part["sealed_per_domain"].values():
        assert len(fids) == 4
    bases = read_jsonl(ROOT / "data/phase25/scenario_bank_v2/base_scenarios.jsonl")
    prompts = read_jsonl(ROOT / "data/phase25/scenario_bank_v2/prompts.jsonl")
    golds = read_jsonl(ROOT / "data/phase25/scenario_bank_v2/gold_answers.jsonl")
    jobs = build_inference_manifest(
        bases=bases, prompts=prompts, golds=golds, partition=part
    )
    assert len(jobs) == N_PLANNED_GENERATIONS
    assert len({j["base_id"] for j in jobs}) == N_SCREEN_BASES
    sealed = set(part["sealed_confirmatory_families"])
    assert not any(j["family_id"] in sealed for j in jobs)
    # identical X prompt sha across 16 rollouts per base
    from collections import defaultdict

    xsha = defaultdict(set)
    for j in jobs:
        if j["form"] == "X":
            xsha[j["base_id"]].add(j["prompt_sha256"])
    assert all(len(v) == 1 for v in xsha.values())
    assert all(sum(1 for j in jobs if j["base_id"] == b and j["form"] == "X") == 16 for b in xsha)


def test_frozen_partition_file_matches_builder():
    path = ROOT / "splits/phase25_step2_family_partition.json"
    if not path.exists():
        return
    meta = read_jsonl(ROOT / "data/phase25/scenario_bank_v2/family_metadata.jsonl")
    got = build_family_partition(meta)
    frozen = json.loads(path.read_text())
    assert frozen["screen_families"] == got["screen_families"]
    assert frozen["sealed_confirmatory_families"] == got["sealed_confirmatory_families"]
