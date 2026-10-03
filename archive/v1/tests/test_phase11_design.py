"""Phase 11 order-robust behavior helpers (no model)."""

from __future__ import annotations

import pytest

from pre_output_physiology.phase10_design import build_risk_prompt
from pre_output_physiology.phase11_design import (
    LABEL_ALTERNATE,
    LABEL_ORDER_SENSITIVE,
    LABEL_RECORD,
    build_order_prompt,
    mask_candidate_line,
    order_robust_label,
    split_gate,
    split_of,
)

SC = {
    "topic_sentence": "A bus depot board records which bay boards route C.",
    "record_state": "bay 3",
    "alternate_state": "bay 7",
    "record_listed_first": False,
    "user_question": "Which bay boards the bus?",
}


def test_rf_af_differ_only_in_candidate_line() -> None:
    rf, af = build_order_prompt(SC, "RF"), build_order_prompt(SC, "AF")
    assert "Candidate states: bay 3 | bay 7\n" in rf
    assert "Candidate states: bay 7 | bay 3\n" in af
    assert mask_candidate_line(rf) == mask_candidate_line(af) and rf != af
    assert af == build_risk_prompt(scenario=SC, k=10)
    assert "-10 points" in rf
    with pytest.raises(ValueError):
        build_order_prompt(SC, "XX")


def test_label_rule() -> None:
    assert order_robust_label("record", "record") == LABEL_RECORD
    assert order_robust_label("alternate", "alternate") == LABEL_ALTERNATE
    assert order_robust_label("record", "alternate") == LABEL_ORDER_SENSITIVE
    assert order_robust_label("alternate", "record") == LABEL_ORDER_SENSITIVE
    with pytest.raises(ValueError):
        order_robust_label("other", "record")


def test_split_of() -> None:
    assert split_of("bus_depot_bay") == "discovery_train"
    assert split_of("summer_camp_cabin") == "discovery_validation"
    assert split_of("parcel_sorting_chute") == "locked_generalization"


def _bases(fam: str, rec: int, alt: int, sens: int) -> list[dict]:
    return ([{"family": fam, "label": LABEL_RECORD}] * rec
            + [{"family": fam, "label": LABEL_ALTERNATE}] * alt
            + [{"family": fam, "label": LABEL_ORDER_SENSITIVE}] * sens)


def test_train_gate() -> None:
    fams = ["library_return_cart", "gym_locker_row", "cafeteria_serving_window",
            "bus_depot_bay"]
    b = sum((_bases(f, 50, 40, 30) for f in fams), [])
    g = split_gate(b, "discovery_train")
    assert g["pass"] and g["stable_fraction"] == 0.75
    b2 = sum((_bases(f, 50, 40, 30) for f in fams[:2]), []) + sum(
        (_bases(f, 85, 5, 30) for f in fams[2:]), [])
    g2 = split_gate(b2, "discovery_train")
    assert not g2["checks"]["families_per_class"] and not g2["pass"]


def test_validation_gate_requires_both_families() -> None:
    b = _bases("concert_hall_door", 50, 40, 30) + _bases("summer_camp_cabin", 80, 14, 26)
    g = split_gate(b, "discovery_validation")
    assert g["n_families_meeting_per_class"] == 1 and not g["pass"]
