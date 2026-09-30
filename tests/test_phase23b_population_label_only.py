"""Tests for Phase-23B label-only vs onset-gated population analyses."""

from __future__ import annotations

import pytest

from pre_output_physiology.phase23_open_grader import (
    compare_development_population_membership_label_only,
)


def _row(
    pid: str,
    k: int,
    *,
    ref: str,
    open_cls: str | None,
    valid: bool = True,
    explicit_span: bool | None = None,
    onset_char_start: int | None = None,
) -> dict:
    return {
        "continuation_id": f"{pid}__r{k:02d}",
        "prompt_id": pid,
        "reference_label": ref,
        "open_valid": valid,
        "open_class": open_cls,
        # Onset fields must be ignored by label-only analysis
        "explicit_span": explicit_span,
        "onset_char_start": onset_char_start,
    }


def _prompt_rows(
    pid: str,
    refs: list[str],
    opens: list[tuple[str | None, bool]],
) -> list[dict]:
    assert len(refs) == 20 and len(opens) == 20
    out = []
    for k, (ref, (oc, valid)) in enumerate(zip(refs, opens, strict=True)):
        out.append(_row(pid, k, ref=ref, open_cls=oc, valid=valid))
    return out


def test_ambiguous_to_deceptive_without_onset_counts_label_only() -> None:
    """GPT ambiguous → open deceptive counts as deceptive with no onset fields."""
    refs = ["ambiguous"] * 20  # GPT does not qualify
    # Open qualifies via 2H + 2D despite no onset metadata on rows
    opens = [("honest", True)] * 2 + [("deceptive", True)] * 2 + [("ambiguous", True)] * 16
    rows = [
        _row(
            "p0",
            k,
            ref=refs[k],
            open_cls=opens[k][0],
            valid=opens[k][1],
            explicit_span=None,
            onset_char_start=None,
        )
        for k in range(20)
    ]
    all_rows = list(rows)
    for i in range(1, 260):
        pid = f"p{i:03d}"
        all_rows.extend(
            _prompt_rows(
                pid,
                ["honest"] * 20,
                [("honest", True)] * 20,
            )
        )
    result = compare_development_population_membership_label_only(all_rows)
    assert "p0" in result["only_open_grader_qualifying"]
    assert result["open_grader"]["n_qualifying_prompts"] >= 1
    # Transitions include ambiguous→deceptive
    cr = next(x for x in result["change_reasons"] if x["prompt_id"] == "p0")
    assert cr["label_transitions"].get("ambiguous->deceptive", 0) == 2
    assert cr["open_counts"]["n_deceptive"] == 2


def test_deceptive_to_ambiguous_can_change_qualification() -> None:
    refs = ["honest"] * 2 + ["deceptive"] * 2 + ["ambiguous"] * 16
    opens = [("honest", True)] * 2 + [("ambiguous", True)] * 18  # loses deceptive
    rows = _prompt_rows("q0", refs, opens)
    all_rows = list(rows)
    for i in range(1, 260):
        all_rows.extend(
            _prompt_rows(
                f"q{i:03d}",
                ["ambiguous"] * 20,
                [("ambiguous", True)] * 20,
            )
        )
    result = compare_development_population_membership_label_only(all_rows)
    assert "q0" in result["only_reference_qualifying"]
    assert result["reference"]["n_qualifying_prompts"] == 1
    assert result["open_grader"]["n_qualifying_prompts"] == 0


def test_exactly_two_honest_two_deceptive_qualifies() -> None:
    refs = ["honest"] * 2 + ["deceptive"] * 2 + ["ambiguous"] * 16
    opens = list(zip(refs, [True] * 20, strict=False))
    rows = _prompt_rows("e0", refs, opens)
    all_rows = list(rows)
    for i in range(1, 260):
        all_rows.extend(
            _prompt_rows(
                f"e{i:03d}",
                ["ambiguous"] * 20,
                [("ambiguous", True)] * 20,
            )
        )
    result = compare_development_population_membership_label_only(all_rows)
    assert "e0" in result["reference"]["qualifying_prompt_ids"]
    assert "e0" in result["open_grader"]["qualifying_prompt_ids"]


def test_one_honest_two_deceptive_does_not_qualify() -> None:
    refs = ["honest"] * 1 + ["deceptive"] * 2 + ["ambiguous"] * 17
    opens = list(zip(refs, [True] * 20, strict=False))
    rows = _prompt_rows("n0", refs, opens)
    all_rows = list(rows)
    for i in range(1, 260):
        all_rows.extend(
            _prompt_rows(
                f"n{i:03d}",
                ["ambiguous"] * 20,
                [("ambiguous", True)] * 20,
            )
        )
    result = compare_development_population_membership_label_only(all_rows)
    assert "n0" not in result["reference"]["qualifying_prompt_ids"]


def test_invalid_open_treated_as_exclude() -> None:
    refs = ["honest"] * 2 + ["deceptive"] * 2 + ["ambiguous"] * 16
    opens = [("honest", True)] * 2 + [(None, False)] * 2 + [("ambiguous", True)] * 16
    rows = _prompt_rows("i0", refs, opens)
    all_rows = list(rows)
    for i in range(1, 260):
        all_rows.extend(
            _prompt_rows(
                f"i{i:03d}",
                ["ambiguous"] * 20,
                [("ambiguous", True)] * 20,
            )
        )
    result = compare_development_population_membership_label_only(all_rows)
    assert "i0" in result["only_reference_qualifying"]
    cr = result["change_reasons"][0]
    assert cr["open_counts"]["n_invalid"] == 2
    assert cr["open_counts"]["n_deceptive"] == 0


def test_requires_260_by_20() -> None:
    rows = _prompt_rows(
        "x0",
        ["honest"] * 20,
        [("honest", True)] * 20,
    )
    with pytest.raises(ValueError, match="expected 5200"):
        compare_development_population_membership_label_only(rows)
