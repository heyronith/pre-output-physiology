"""Tests for Phase 19A unseen validation family screen."""

from __future__ import annotations

from pre_output_physiology.phase19_screen import (
    N_FRESH_CONTINUATIONS,
    N_SELECTED_PROMPTS,
    THRESHOLD_HASH,
    build_fresh_schedule,
    evaluate_fresh_gates_for_families,
    is_fresh_usable,
    is_s1_candidate,
    is_s2_confirmed,
    select_validation_cohort,
    threshold_manifest,
)


def test_threshold_hash_stable() -> None:
    assert threshold_manifest()["threshold_hash"] == THRESHOLD_HASH


def test_s1_s2_fresh_thresholds() -> None:
    assert is_s1_candidate({"n_valid": 14, "n_record": 2, "n_alternate": 2})
    assert not is_s1_candidate({"n_valid": 14, "n_record": 1, "n_alternate": 2})
    assert is_s2_confirmed({"n_valid": 14, "n_record": 3, "n_alternate": 3})
    assert not is_s2_confirmed({"n_valid": 14, "n_record": 2, "n_alternate": 3})
    assert is_fresh_usable({"n_valid": 42, "n_record": 6, "n_alternate": 6})
    assert not is_fresh_usable({"n_valid": 41, "n_record": 6, "n_alternate": 6})


def test_family_and_prompt_sha_selection() -> None:
    confirmed = []
    fams = [
        "florist_cooler_shelf",
        "hardware_pegboard_hook",
        "community_pool_lane",
        "music_school_practice_room",
        "toy_library_checkout_bin",
        "mini_golf_tee_pad",
    ]
    # Make 3 families qualify with 6+ each.
    for fam in fams[:3]:
        for i in range(7):
            confirmed.append(
                {
                    "prompt_group_id": f"p19_{fam}_{i:03d}",
                    "family": fam,
                    "n_record": 5,
                    "n_alternate": 5,
                    "n_valid": 14,
                    "alternate_fraction_among_valid": 0.5,
                }
            )
    # Others have <6.
    for fam in fams[3:]:
        for i in range(2):
            confirmed.append(
                {
                    "prompt_group_id": f"p19_{fam}_{i:03d}",
                    "family": fam,
                    "n_record": 5,
                    "n_alternate": 5,
                    "n_valid": 14,
                    "alternate_fraction_among_valid": 0.5,
                }
            )
    sel = select_validation_cohort(confirmed)
    assert sel is not None
    assert len(sel["selected_families"]) == 2
    assert sel["n_selected"] == N_SELECTED_PROMPTS == 12
    again = select_validation_cohort(confirmed)
    assert again["cohort_sha256"] == sel["cohort_sha256"]
    sched = build_fresh_schedule(sel["selected_prompts"])
    assert len(sched) == N_FRESH_CONTINUATIONS == 576


def test_select_none_when_fewer_than_two_qualify() -> None:
    confirmed = []
    for i in range(6):
        confirmed.append(
            {
                "prompt_group_id": f"p19_florist_cooler_shelf_{i:03d}",
                "family": "florist_cooler_shelf",
                "n_record": 5,
                "n_alternate": 5,
                "n_valid": 14,
            }
        )
    assert select_validation_cohort(confirmed) is None


def test_fresh_gates() -> None:
    fams = ["florist_cooler_shelf", "hardware_pegboard_hook"]
    usable = []
    for fam in fams:
        for i in range(5):
            usable.append({"prompt_group_id": f"{fam}_{i}", "family": fam})
    gates = evaluate_fresh_gates_for_families(usable, fams)
    assert gates["passed"]
    usable2 = [
        u
        for u in usable
        if not (u["family"] == fams[0] and u["prompt_group_id"].endswith("_4"))
    ]
    assert not evaluate_fresh_gates_for_families(usable2, fams)["passed"]
