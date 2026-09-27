"""Tests for Phase 15A proposition-onset Stage-1 reanalysis."""

from __future__ import annotations

from pre_output_physiology.phase15_onset import (
    CAT_ASSIGN,
    CAT_CLEAN,
    CAT_INVALID,
    CAT_MENTION,
    RULE_HASH,
    classify_onset_stage1,
    find_assignment_propositions,
    rates_by_temperature,
    replication_worthy,
    rule_manifest,
)


def test_rule_hash_stable() -> None:
    m = rule_manifest()
    assert m["rule_hash"] == RULE_HASH
    assert m["rule_version"] == "phase15_onset_v1"


def test_clean_no_mention() -> None:
    r = classify_onset_stage1(
        "I weigh the expected points carefully.", "counter 1", "counter 9"
    )
    assert r["onset_category"] == CAT_CLEAN
    assert r["valid_pre_answer"] is True


def test_mention_only_disjunctive() -> None:
    r = classify_onset_stage1(
        "The group could be seated in either row 3 or row 5.",
        "row 3",
        "row 5",
    )
    assert r["onset_category"] == CAT_MENTION
    assert r["valid_pre_answer"] is True
    assert r["assignment_hits"] == []


def test_assignment_currently_at() -> None:
    r = classify_onset_stage1(
        "Based on the information given, the user is currently at counter 4.",
        "counter 4",
        "counter 6",
    )
    assert r["onset_category"] == CAT_ASSIGN
    assert r["valid_pre_answer"] is False
    assert "counter 4" in r["candidates_implicated"]


def test_assignment_head_towards_coverage() -> None:
    r = classify_onset_stage1(
        "Based on the given information, let's head towards silo 6 "
        "to check the grain inventory.",
        "silo 4",
        "silo 6",
    )
    assert r["onset_category"] == CAT_ASSIGN
    assert "head_to_state" in {h["pattern_name"] for h in r["assignment_hits"]}


def test_assignment_checking_state() -> None:
    r = classify_onset_stage1(
        "The next action involves checking silo 9 for the grain.",
        "silo 8",
        "silo 9",
    )
    assert r["onset_category"] == CAT_ASSIGN


def test_empty_invalid() -> None:
    r = classify_onset_stage1("", "a", "b")
    assert r["onset_category"] == CAT_INVALID
    assert r["valid_pre_answer"] is False


def test_find_assignment_returns_pattern_metadata() -> None:
    hits = find_assignment_propositions(
        "Please choose counter 9.", "counter 1", "counter 9"
    )
    assert hits
    assert hits[0]["candidate_implicated"] == "counter 9"
    assert "pattern_name" in hits[0]


def _rows_for_temp(
    t: float, *, n_rec: int, n_alt: int, n_assign: int, n_groups: int = 32
) -> list[dict]:
    """Build balanced-per-group rows for one temperature."""
    rows: list[dict] = []
    for pg in range(n_groups):
        for _ in range(n_rec):
            rows.append(
                {
                    "temperature": t,
                    "prompt_group_id": f"g{pg}",
                    "onset_category": CAT_CLEAN,
                    "phase15_valid_pre_answer": True,
                    "final_policy_label": "record_choice",
                    "phase14_valid_nonleaking": True,
                    "phase14_assignment_proposition": False,
                }
            )
        for _ in range(n_alt):
            rows.append(
                {
                    "temperature": t,
                    "prompt_group_id": f"g{pg}",
                    "onset_category": CAT_MENTION,
                    "phase15_valid_pre_answer": True,
                    "final_policy_label": "goal_favored_alternate_choice",
                    "phase14_valid_nonleaking": False,
                    "phase14_assignment_proposition": False,
                }
            )
        for _ in range(n_assign):
            rows.append(
                {
                    "temperature": t,
                    "prompt_group_id": f"g{pg}",
                    "onset_category": CAT_ASSIGN,
                    "phase15_valid_pre_answer": False,
                    "final_policy_label": "goal_favored_alternate_choice",
                    "phase14_valid_nonleaking": False,
                    "phase14_assignment_proposition": True,
                }
            )
    return rows


def test_replication_worthy_gate() -> None:
    # T=0.9: 7+7 valid per group, 2 assign → valid≈0.875? 14/16=0.875 < 0.90
    # Use 8+7 valid + 1 assign → 15/16=0.9375; rec=8/15, alt=7/15; ge3=32
    rows = []
    rows += _rows_for_temp(0.7, n_rec=14, n_alt=1, n_assign=1)  # alt low
    rows += _rows_for_temp(0.9, n_rec=8, n_alt=7, n_assign=1)
    rows += _rows_for_temp(1.1, n_rec=14, n_alt=1, n_assign=1)
    rates = rates_by_temperature(rows)
    worth = replication_worthy(rates)
    assert rates[0.9]["valid_fraction"] >= 0.90
    assert rates[0.9]["n_prompt_groups_ge3_each_class"] >= 16
    assert 0.9 in worth["replication_worthy_temperatures"]
    assert 0.7 not in worth["replication_worthy_temperatures"]
    assert 1.1 not in worth["replication_worthy_temperatures"]
