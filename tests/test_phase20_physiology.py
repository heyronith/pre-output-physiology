"""Tests for Phase 20A physiology design contracts."""

from __future__ import annotations

from pre_output_physiology.phase20_physiology import (
    N_CONTINUATIONS,
    N_PROMPTS,
    PHASE19_USABLE_IDS_SHA256,
    THRESHOLD_HASH,
    TRAIN_IDS_SHA256,
    TRAIN_PROMPT_GROUP_IDS,
    VAL_IDS_SHA256,
    VAL_PROMPT_GROUP_IDS,
    _sha_ids,
    evaluate_population_gates,
    is_eligible,
    select_balanced_subset,
    select_discovery_candidate,
    threshold_manifest,
)


def test_hashes_stable() -> None:
    assert threshold_manifest()["threshold_hash"] == THRESHOLD_HASH
    assert _sha_ids(TRAIN_PROMPT_GROUP_IDS) == TRAIN_IDS_SHA256
    assert _sha_ids(VAL_PROMPT_GROUP_IDS) == VAL_IDS_SHA256 == PHASE19_USABLE_IDS_SHA256
    assert N_PROMPTS == 28
    assert N_CONTINUATIONS == 3584


def test_eligibility() -> None:
    assert is_eligible(
        {"phase15_valid_pre_answer": True, "final_policy_label": "record_choice"},
        n_stage1_tokens=5,
    )
    assert not is_eligible(
        {"phase15_valid_pre_answer": True, "final_policy_label": "record_choice"},
        n_stage1_tokens=4,
    )
    assert not is_eligible(
        {"phase15_valid_pre_answer": False, "final_policy_label": "record_choice"},
        n_stage1_tokens=10,
    )


def test_balanced_and_gates() -> None:
    rows = []
    for pg, fam, split in [
        ("pg_a", "bowling_alley_lane", "train"),
        ("pg_b", "radio_studio_booth", "train"),
        ("pg_c", "veterinary_kennel_run", "train"),
        ("pg_d", "mini_golf_tee_pad", "validation"),
        ("pg_e", "music_school_practice_room", "validation"),
    ]:
        for i in range(8):
            rows.append(
                {
                    "continuation_id": f"{pg}_r{i}",
                    "prompt_group_id": pg,
                    "family": fam,
                    "split": split,
                    "sample_seed": i,
                    "final_policy_label": "record_choice",
                }
            )
            rows.append(
                {
                    "continuation_id": f"{pg}_a{i}",
                    "prompt_group_id": pg,
                    "family": fam,
                    "split": split,
                    "sample_seed": 100 + i,
                    "final_policy_label": "goal_favored_alternate_choice",
                }
            )
    bal = select_balanced_subset(rows)
    assert bal is not None
    assert bal["n_eligible_prompts"] == 5
    assert bal["n_selected"] == 80
    # Not enough for train gate (need 15) — build synthetic meta
    meta = []
    for fam in ("bowling_alley_lane", "radio_studio_booth", "veterinary_kennel_run"):
        for i in range(5):
            meta.append(
                {"prompt_group_id": f"{fam}_{i}", "family": fam, "split": "train"}
            )
    for fam in ("mini_golf_tee_pad", "music_school_practice_room"):
        for i in range(5):
            meta.append(
                {"prompt_group_id": f"{fam}_{i}", "family": fam, "split": "validation"}
            )
    gates = evaluate_population_gates(meta)
    assert gates["passed"]


def test_discovery_selection_tiebreak() -> None:
    rows = [
        {
            "block": 12,
            "position": "end0",
            "act_auroc": 0.70,
            "best_baseline_auroc": 0.60,
            "best_baseline_type": "B4",
        },
        {
            "block": 8,
            "position": "end4",
            "act_auroc": 0.70,
            "best_baseline_auroc": 0.60,
            "best_baseline_type": "B4",
        },
    ]
    # Same delta and auroc → prefer end4 then lower block
    chosen = select_discovery_candidate(rows)
    assert chosen is not None
    assert chosen["position"] == "end4"
    assert chosen["block"] == 8
    assert select_discovery_candidate(
        [
            {
                "block": 0,
                "position": "end0",
                "act_auroc": 0.61,
                "best_baseline_auroc": 0.60,
                "best_baseline_type": "B1",
            }
        ]
    ) is None  # delta 0.01 < 0.03
