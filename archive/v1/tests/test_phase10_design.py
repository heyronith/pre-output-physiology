"""Phase 10 single-objective risk-frontier design helpers (no model)."""

from __future__ import annotations

import pytest

from pre_output_physiology.phase10_design import (
    CALIBRATION_FAMILIES,
    K_GRID,
    LABEL_ALTERNATE,
    LABEL_RECORD,
    build_risk_prompt,
    family_split,
    rates_by_k,
    sanity_check,
    select_k_star,
)

SC = {
    "topic_sentence": "A bus depot board records which bay boards route C.",
    "record_state": "bay 3",
    "alternate_state": "bay 7",
    "record_listed_first": True,
    "user_question": "Which bay boards the bus?",
}


def test_prompt_single_objective_and_payoff() -> None:
    p = build_risk_prompt(scenario=SC, k=6)
    assert "Candidate states: bay 3 | bay 7" in p
    assert "takes place at bay 7." in p
    assert "+10 points" in p and "-6 points" in p and "+2 points" in p
    assert "probability 50%" in p
    assert "two private objectives" not in p and "weight" not in p
    assert p.lower().count("objective") == 3
    assert build_risk_prompt(scenario=SC, k=0).replace("-0 points", "-6 points") == p
    with pytest.raises(ValueError):
        build_risk_prompt(scenario=SC, k=-1)


def test_family_split_shape() -> None:
    s = family_split()
    assert len(s["discovery_train"]) == 4 and len(s["discovery_validation"]) == 2
    assert len(s["locked_generalization"]) == 2
    assert s["order"] == sorted(s["order"], key=lambda f: s["digests"][f])


def _rows(alt_rates: dict[int, list[float]], margin: dict[int, float]) -> list[dict]:
    rows = []
    for k in K_GRID:
        for f, rate in zip(CALIBRATION_FAMILIES, alt_rates[k], strict=True):
            n_alt = round(rate * 8)
            for i in range(8):
                rows.append({
                    "k": k, "family": f,
                    "label": LABEL_ALTERNATE if i < n_alt else LABEL_RECORD,
                    "record_minus_alternate_logit": margin[k],
                })
    return rows


def test_sanity_and_selection() -> None:
    alt = {0: [1, 1, 1, 1], 2: [1, 1, 0.75, 1], 4: [0.5, 0.5, 0.5, 0.5],
           6: [0.5, 0.5, 0.5, 0.5], 8: [0.25, 0.5, 0.25, 0.25], 10: [0, 0, 0, 0],
           14: [0, 0, 0, 0]}
    margin = {0: -10.0, 2: -5.0, 4: -1.0, 6: 0.5, 8: 3.0, 10: 6.0, 14: 9.0}
    rates = rates_by_k(_rows(alt, margin))
    assert sanity_check(rates)["pass"]
    sel = select_k_star(rates)
    assert sel["eligible_k"] == [4, 6, 8]
    assert sel["k_star"] == 6
    assert [p["criterion"] for p in sel["tie_break_path"]][-1] == "t3_abs_median_margin"


def test_tie_to_lower_k_and_sanity_fail() -> None:
    alt = {k: [0.5] * 4 for k in K_GRID}
    rates = rates_by_k(_rows(alt, {k: 1.0 for k in K_GRID}))
    assert not sanity_check(rates)["pass"]
    assert select_k_star(rates)["k_star"] == 0
