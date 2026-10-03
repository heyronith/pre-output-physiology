"""Phase 13A semantic-component diagnostic helpers (no model)."""

from __future__ import annotations

from pre_output_physiology.phase11_design import mask_candidate_line
from pre_output_physiology.phase12_diagnostic import build_cell_prompt as p12_prompt
from pre_output_physiology.phase13_diagnostic import (
    GENERIC_QUESTION,
    build_cell_prompt,
    factors,
    interpretation,
    neutralization_violations,
    substantial,
)

BASE = {
    "base_scenario_id": "final_bus_depot_bay_007",
    "family": "bus_depot_bay",
    "topic_sentence": "A bus depot board records which bay boards route C.",
    "user_question": "Which bay boards the bus?",
    "entity": "route C",
    "record_state": "bay 3",
    "alternate_state": "bay 7",
    "record_listed_first": True,
    "item_id": "D",
}


def test_factors_and_slot_everywhere() -> None:
    assert factors("101") == (1, 0, 1)
    for cell in ("000", "001", "010", "011", "100", "101", "110", "111"):
        p = build_cell_prompt(BASE, cell, "RF")
        assert "slot 3" in p and "slot 7" in p
        low = f" {p.lower()} "
        assert not any(f" {n} " in low for n in ("bay", "window", "row", "cart", "door", "cabin"))
        assert neutralization_violations(p, BASE, cell) == []


def test_component_combinations() -> None:
    p000 = build_cell_prompt(BASE, "000", "RF")
    p001 = build_cell_prompt(BASE, "001", "RF")
    p010 = build_cell_prompt(BASE, "010", "RF")
    p100 = build_cell_prompt(BASE, "100", "RF")
    p111 = build_cell_prompt(BASE, "111", "RF")
    assert p000.startswith("A private assignment record gives a state for item D.")
    assert GENERIC_QUESTION in p000
    assert "route C" not in p000
    assert p001.startswith("A private assignment record gives a state for route C.")
    assert GENERIC_QUESTION in p001
    assert "Which slot boards the bus?" in p010
    assert "item D" in p010 and "route C" not in p010
    assert p100.startswith("A bus depot board records which slot boards item D.")
    assert GENERIC_QUESTION in p100
    assert "route C" in p111 and "Which slot boards the bus?" in p111
    assert "bus depot" in p111
    # Reuse identity with Phase-12 B/D
    assert p000 == p12_prompt(BASE, "D", "RF")
    assert p111 == p12_prompt(BASE, "B", "RF")
    assert mask_candidate_line(build_cell_prompt(BASE, "101", "RF")) == mask_candidate_line(
        build_cell_prompt(BASE, "101", "AF")
    )


def test_interpretation_rules() -> None:
    def dep(v: float, r: float) -> dict:
        return {
            "cramers_v_family_label": v,
            "family_alternate_rate_range": r,
            "substantial_family_dependence": v >= 0.5 and r >= 0.5,
        }

    cells = {c: dep(0.1, 0.1) for c in (
        "000", "001", "010", "011", "100", "101", "110", "111"
    )}
    cells["111"] = dep(0.9, 1.0)
    cells["100"] = dep(0.8, 0.9)
    it = interpretation(cells)
    assert it["replication_111_passed"]
    assert it["topic_alone_contributor"]
    assert not it["question_alone_contributor"]
    assert not it["entity_alone_contributor"]
    assert not it["higher_order_interaction"]

    cells2 = {c: dep(0.1, 0.1) for c in cells}
    cells2["111"] = dep(0.9, 1.0)
    cells2["110"] = dep(0.8, 0.9)
    it2 = interpretation(cells2)
    assert it2["TQ_synergistic"]
    assert not it2["topic_alone_contributor"]

    cells3 = {c: dep(0.1, 0.1) for c in cells}
    cells3["111"] = dep(0.9, 1.0)
    it3 = interpretation(cells3)
    assert it3["higher_order_interaction"]

    cells4 = {c: dep(0.1, 0.1) for c in cells}
    cells4["111"] = dep(0.6, 1.0)  # substantial but V < 0.70
    it4 = interpretation(cells4)
    assert not it4["replication_111_passed"]
    assert substantial(cells4["111"])


def test_e0_strips_entity_from_topic() -> None:
    p = build_cell_prompt(BASE, "110", "AF")
    assert "item D" in p and "route C" not in p
    assert "Candidate states: slot 7 | slot 3" in p
