"""Tests for Phase 18A enriched-cohort fresh confirmation."""

from __future__ import annotations

from pre_output_physiology.phase18_cohort import (
    N_COHORT,
    N_CONTINUATIONS,
    N_SAMPLES,
    SEED_BASE,
    THRESHOLD_HASH,
    build_fresh_schedule,
    evaluate_gates,
    is_fresh_usable,
    select_enriched_cohort,
    threshold_manifest,
)


def test_threshold_hash_stable() -> None:
    assert threshold_manifest()["threshold_hash"] == THRESHOLD_HASH


def test_fresh_usability_threshold() -> None:
    assert is_fresh_usable({"n_valid": 42, "n_record": 6, "n_alternate": 6})
    assert not is_fresh_usable({"n_valid": 41, "n_record": 6, "n_alternate": 6})
    assert not is_fresh_usable({"n_valid": 42, "n_record": 5, "n_alternate": 6})
    assert not is_fresh_usable({"n_valid": 42, "n_record": 6, "n_alternate": 5})


def test_select_six_per_qualifying_family() -> None:
    confirmed = []
    families = {
        "bowling_alley_lane": 15,
        "radio_studio_booth": 9,
        "veterinary_kennel_run": 8,
        "daycare_cubby_shelf": 6,
        "climbing_gym_route": 2,  # excluded (insufficient + excluded list)
    }
    for fam, n in families.items():
        for i in range(n):
            confirmed.append(
                {
                    "prompt_group_id": f"final_{fam}_{i:03d}",
                    "family": fam,
                    "n_record": 5,
                    "n_alternate": 5,
                    "n_valid": 14,
                    "alternate_fraction_among_valid": 0.5,
                    "s2_confirmed": True,
                }
            )
    cohort = select_enriched_cohort(confirmed)
    assert cohort["n_selected"] == N_COHORT == 24
    assert all(v == 6 for v in cohort["n_per_family_counts"].values())
    assert "climbing_gym_route" not in cohort["n_per_family_counts"]
    assert cohort["n_train"] == 18
    assert cohort["n_validation"] == 6
    # Deterministic: re-select yields same IDs/hash.
    again = select_enriched_cohort(confirmed)
    assert again["cohort_sha256"] == cohort["cohort_sha256"]
    assert again["selected_prompt_group_ids"] == cohort["selected_prompt_group_ids"]


def test_fresh_schedule_size_and_seeds() -> None:
    confirmed = []
    for fam in (
        "bowling_alley_lane",
        "radio_studio_booth",
        "veterinary_kennel_run",
        "daycare_cubby_shelf",
    ):
        for i in range(6):
            confirmed.append(
                {
                    "prompt_group_id": f"final_{fam}_{i:03d}",
                    "family": fam,
                    "n_record": 5,
                    "n_alternate": 5,
                    "n_valid": 14,
                    "alternate_fraction_among_valid": 0.5,
                    "s2_confirmed": True,
                }
            )
    cohort = select_enriched_cohort(confirmed)
    sched = build_fresh_schedule(cohort)
    assert len(sched) == N_CONTINUATIONS == 1152
    assert all(r["sample_seed"] >= SEED_BASE for r in sched)
    assert len({r["sample_seed"] for r in sched}) == N_CONTINUATIONS
    # Exactly 48 per prompt.
    from collections import Counter

    counts = Counter(r["prompt_group_id"] for r in sched)
    assert all(v == N_SAMPLES for v in counts.values())


def test_gates() -> None:
    usable = []
    for fam in ("bowling_alley_lane", "radio_studio_booth", "veterinary_kennel_run"):
        for i in range(5):
            usable.append({"prompt_group_id": f"{fam}_{i}", "family": fam})
    for i in range(5):
        usable.append(
            {"prompt_group_id": f"daycare_{i}", "family": "daycare_cubby_shelf"}
        )
    gates = evaluate_gates(usable)
    assert gates["passed"]
    # Drop one train family below 5 → fail.
    usable2 = [u for u in usable if u["family"] != "bowling_alley_lane"] + [
        {"prompt_group_id": "bowling_alley_lane_0", "family": "bowling_alley_lane"},
        {"prompt_group_id": "bowling_alley_lane_1", "family": "bowling_alley_lane"},
        {"prompt_group_id": "bowling_alley_lane_2", "family": "bowling_alley_lane"},
        {"prompt_group_id": "bowling_alley_lane_3", "family": "bowling_alley_lane"},
    ]
    assert not evaluate_gates(usable2)["passed"]
