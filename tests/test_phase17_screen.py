"""Tests for Phase 17A policy-unstable enrichment screen."""

from __future__ import annotations

from pre_output_physiology.phase17_screen import (
    THRESHOLD_HASH,
    build_s1_schedule,
    build_s2_schedule,
    evaluate_gates,
    is_s1_candidate,
    is_s2_confirmed,
    select_prompts,
    threshold_manifest,
)


def test_threshold_hash_stable() -> None:
    assert threshold_manifest()["threshold_hash"] == THRESHOLD_HASH


def test_s1_candidate_threshold() -> None:
    assert is_s1_candidate({"n_valid": 14, "n_record": 2, "n_alternate": 2})
    assert not is_s1_candidate({"n_valid": 14, "n_record": 1, "n_alternate": 2})
    assert not is_s1_candidate({"n_valid": 13, "n_record": 2, "n_alternate": 2})


def test_s2_confirmation_threshold() -> None:
    assert is_s2_confirmed({"n_valid": 14, "n_record": 3, "n_alternate": 3})
    assert not is_s2_confirmed({"n_valid": 14, "n_record": 2, "n_alternate": 3})


def test_s1_s2_seeds_disjoint() -> None:
    disc = [
        {
            "prompt_group_id": f"pg_{i:03d}",
            "family": "climbing_gym_route",
            "prompt_text": "x",
        }
        for i in range(240)
    ]
    s1 = build_s1_schedule(disc)
    assert len(s1) == 3840
    s1_seeds = {r["sample_seed"] for r in s1}
    cand = [f"pg_{i:03d}" for i in range(10)]
    by = {p["prompt_group_id"]: p for p in disc}
    s2 = build_s2_schedule(cand, by)
    s2_seeds = {r["sample_seed"] for r in s2}
    assert not (s1_seeds & s2_seeds)


def test_gates_and_selection() -> None:
    confirmed = []
    train = [
        "climbing_gym_route",
        "radio_studio_booth",
        "bowling_alley_lane",
        "veterinary_kennel_run",
    ]
    val = ["subway_turnstile_bank", "daycare_cubby_shelf"]
    for fam in train:
        for i in range(5):
            confirmed.append(
                {
                    "prompt_group_id": f"{fam}_{i}",
                    "family": fam,
                    "n_record": 4,
                    "n_alternate": 4,
                    "n_valid": 14,
                }
            )
    for fam in val:
        for i in range(5):
            confirmed.append(
                {
                    "prompt_group_id": f"{fam}_{i}",
                    "family": fam,
                    "n_record": 4,
                    "n_alternate": 4,
                    "n_valid": 14,
                }
            )
    gates = evaluate_gates(confirmed)
    assert gates["passed"]
    sel = select_prompts(confirmed)
    assert sel["n_selected"] == 6 * 5  # 5 per family, under max 8
    assert all(v <= 8 for v in sel["n_per_family"].values())
