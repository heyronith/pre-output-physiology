"""Phase 12A family-bias source diagnostic helpers (no model)."""

from __future__ import annotations

from pre_output_physiology.phase11_design import (
    LABEL_ALTERNATE,
    LABEL_ORDER_SENSITIVE,
    LABEL_RECORD,
    mask_candidate_line,
)
from pre_output_physiology.phase12_diagnostic import (
    FAMILIES,
    build_cell_prompt,
    cramers_v,
    family_dependence,
    interpretation,
    neutralization_violations,
    select_bases,
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


def test_cells_transform_vocabulary_only() -> None:
    a = build_cell_prompt(BASE, "A", "RF")
    b = build_cell_prompt(BASE, "B", "RF")
    c = build_cell_prompt(BASE, "C", "AF")
    d = build_cell_prompt(BASE, "D", "RF")
    assert "Candidate states: bay 3 | bay 7" in a and "route C" in a
    assert "Candidate states: slot 3 | slot 7" in b and "bus depot" in b and " bay" not in b
    assert "Candidate states: bay 7 | bay 3" in c and "item D" in c and "bus" not in c
    assert "slot 3" in d and "bay" not in d and "route" not in d
    assert all("-10 points" in p for p in (a, b, c, d))
    assert mask_candidate_line(build_cell_prompt(BASE, "C", "RF")) == mask_candidate_line(c)
    for cell, p in zip("ABCD", (a, b, c, d), strict=True):
        assert neutralization_violations(p, BASE, cell) == []
    assert neutralization_violations(a, BASE, "D")


def test_selection_is_label_independent_and_deterministic() -> None:
    bases = [{"base_scenario_id": f"final_{f}_{i:03d}", "family": f, "label": "x"}
             for f in FAMILIES for i in range(30)]
    s1 = select_bases(bases)
    s2 = select_bases([{k: v for k, v in b.items() if k != "label"} for b in bases][::-1])
    assert [b["base_scenario_id"] for b in s1] == [b["base_scenario_id"] for b in s2]
    assert len(s1) == 144 and s1[0]["item_id"] == "A"


def test_cramers_v_and_dependence() -> None:
    assert cramers_v([("f1", "r")] * 10 + [("f2", "a")] * 10) == 1.0
    assert cramers_v([("f1", "r")] * 5 + [("f1", "a")] * 5 + [("f2", "r")] * 5
                     + [("f2", "a")] * 5) == 0.0
    bases = []
    for i, f in enumerate(FAMILIES):
        lab = LABEL_RECORD if i % 2 else LABEL_ALTERNATE
        bases += [{"family": f, "label": lab}] * 20 + [
            {"family": f, "label": LABEL_ORDER_SENSITIVE}] * 4
    d = family_dependence(bases)
    assert d["cramers_v_family_label"] == 1.0 and d["family_alternate_rate_range"] == 1.0
    assert d["substantial_family_dependence"]


def test_interpretation_rules() -> None:
    def dep(v: float, r: float) -> dict:
        return {"cramers_v_family_label": v, "family_alternate_rate_range": r,
                "substantial_family_dependence": v >= 0.5 and r >= 0.5}
    it = interpretation({"A": dep(0.9, 1.0), "B": dep(0.2, 0.3), "C": dep(0.8, 0.9),
                         "D": dep(0.1, 0.1)})
    assert it["state_vocabulary_contributor"] and not it["semantic_shell_contributor"]
    assert not it["mixed_contributors"] and not it["interaction_or_other_family_structure"]
    it2 = interpretation({"A": dep(0.9, 1.0), "B": dep(0.2, 0.3), "C": dep(0.3, 0.2),
                          "D": dep(0.1, 0.1)})
    assert it2["interaction_or_other_family_structure"]
    it3 = interpretation({"A": dep(0.6, 1.0), "B": dep(0.9, 0.9), "C": dep(0.9, 0.9),
                          "D": dep(0.1, 0.1)})
    assert not it3["replication_passed"] and "state_vocabulary_contributor" not in it3
