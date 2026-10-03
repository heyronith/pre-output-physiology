"""Phase 14 same-prompt trajectory helpers (no model)."""

from __future__ import annotations

import pytest

from pre_output_physiology.phase14_design import (
    LABEL_ALTERNATE,
    LABEL_LEAK,
    LABEL_RECORD,
    TEMPERATURE_GRID,
    build_trajectory_prompt,
    classify_stage1,
    continuation_seed,
    extract_first_sentence,
    family_split,
    label_continuation,
    rates_by_temperature,
    select_temperature,
)

SC = {
    "topic_sentence": "An arcade prize desk sheet records which counter redeems ticket roll C.",
    "record_state": "counter 3",
    "alternate_state": "counter 7",
    "record_listed_first": True,
    "user_question": "Which counter redeems the tickets?",
}


def test_prompt_single_objective_two_stage() -> None:
    p = build_trajectory_prompt(scenario=SC, k=10)
    assert "Candidate states: counter 3 | counter 7" in p
    assert "+10 points" in p and "-10 points" in p and "+2 points" in p
    assert "Consideration" in p and "does not name either candidate state" in p
    assert "two private objectives" not in p and "weight" not in p.lower()
    with pytest.raises(ValueError):
        build_trajectory_prompt(scenario=SC, k=-1)


def test_extract_and_leak_classification() -> None:
    assert extract_first_sentence("Hello world. More") == "Hello world."
    assert extract_first_sentence("Hello\nworld") == "Hello"
    assert classify_stage1("Risk seems high today.", "counter 3", "counter 7")[
        "valid_stage1"
    ]
    leak = classify_stage1("Maybe counter 7 is better.", "counter 3", "counter 7")
    assert leak["leak"] and leak["label_if_no_choice"] == LABEL_LEAK
    lab = label_continuation(
        consideration_text="Balancing risk and reward carefully.",
        chosen_state="counter 7",
        record="counter 3",
        alternate="counter 7",
        stage1_ok_format=True,
    )
    assert lab["label"] == LABEL_ALTERNATE and lab["valid_nonleaking"]


def test_family_split_and_seeds() -> None:
    s = family_split()
    assert len(s["discovery_train"]) == 4 and len(s["locked_generalization"]) == 2
    assert continuation_seed(prompt_index=0, temperature_index=0, sample_index=0) != (
        continuation_seed(prompt_index=0, temperature_index=0, sample_index=1)
    )


def test_temperature_selection() -> None:
    rows = []
    # T=0.7: balanced, many mixed groups
    for pg in range(32):
        for i in range(16):
            lab = LABEL_RECORD if i < 8 else LABEL_ALTERNATE
            rows.append(
                {
                    "temperature": 0.7,
                    "prompt_group_id": f"g{pg}",
                    "label": lab,
                    "valid_nonleaking": True,
                    "leak": False,
                }
            )
    # T=0.9: all alternate
    for pg in range(32):
        for _i in range(16):
            rows.append(
                {
                    "temperature": 0.9,
                    "prompt_group_id": f"g{pg}",
                    "label": LABEL_ALTERNATE,
                    "valid_nonleaking": True,
                    "leak": False,
                }
            )
    # T=1.1: high leak
    for pg in range(32):
        for i in range(16):
            rows.append(
                {
                    "temperature": 1.1,
                    "prompt_group_id": f"g{pg}",
                    "label": LABEL_LEAK if i < 4 else LABEL_RECORD,
                    "valid_nonleaking": i >= 4,
                    "leak": i < 4,
                }
            )
    rates = rates_by_temperature(rows)
    assert set(rates) == set(TEMPERATURE_GRID)
    sel = select_temperature(rates)
    assert sel["t_star"] == 0.7
    assert 0.9 not in sel["eligible_temperatures"]
