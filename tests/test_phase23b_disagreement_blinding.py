"""Tests for Phase-23B disagreement-audit blinding and Stage-23C authorization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pre_output_physiology.phase23_open_grader import (
    STATUS_23B_POSTAUDIT,
    ab_swap_for_continuation,
    assert_blinded_item_has_no_identity_leak,
    build_blinded_disagreement_item,
    select_disagreement_audit_ids,
)

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / "artifacts/phase23b_development/disagreement_audit"


def _synthetic_rows(n_disagree: int = 50) -> tuple[list[dict], set[str]]:
    rows = []
    changed = {"roleplay_000"}
    for i in range(n_disagree):
        ref = "honest" if i % 2 == 0 else "deceptive"
        pred = "deceptive" if ref == "honest" else "honest"
        pid = "roleplay_000" if i < 10 else f"roleplay_{i:03d}"
        rows.append(
            {
                "continuation_id": f"{pid}__r{i:02d}",
                "prompt_id": pid,
                "reference_label": ref,
                "open_class": pred,
                "open_valid": True,
                "reference_judgement": 1 if ref == "honest" else 5,
                "open_score": 5 if pred == "deceptive" else 1,
                "open_explanation": f"open explanation {i}",
            }
        )
    return rows, changed


def test_blinded_item_omits_explanation_both_sides() -> None:
    item, mapping = build_blinded_disagreement_item(
        continuation_id="roleplay_001__r00",
        prompt_id="roleplay_001",
        scenario="s",
        question="q",
        response="r",
        reference_judgement=1,
        reference_label="honest",
        open_score=5,
        open_class="deceptive",
        open_identity="gemma4_31b_it",
    )
    assert set(item["judgment_a"].keys()) == {"score", "class"}
    assert set(item["judgment_b"].keys()) == {"score", "class"}
    assert "explanation" not in item["judgment_a"]
    assert "explanation" not in item["judgment_b"]
    assert set(mapping.keys()) == {"grader_a", "grader_b"}
    assert "gemma4_31b_it" in mapping.values()
    assert "gpt-4o-2024-08-06" in mapping.values()
    assert_blinded_item_has_no_identity_leak(item)


def test_blinded_item_rejects_identity_leak() -> None:
    item, _ = build_blinded_disagreement_item(
        continuation_id="roleplay_002__r00",
        prompt_id="roleplay_002",
        scenario="s",
        question="q",
        response="r",
        reference_judgement=1,
        reference_label="honest",
        open_score=5,
        open_class="deceptive",
        open_identity="gemma4_31b_it",
    )
    leaky = dict(item)
    leaky["judgment_a"] = {**item["judgment_a"], "explanation": None}
    with pytest.raises(AssertionError):
        assert_blinded_item_has_no_identity_leak(leaky)


def test_explanation_asymmetry_cannot_pass_leak_check() -> None:
    """Historical leak: GPT side explanation=null, open side has text."""
    bad = {
        "continuation_id": "x",
        "prompt_id": "p",
        "scenario": "s",
        "question": "q",
        "response": "r",
        "judgment_a": {"score": 1, "class": "honest", "explanation": "open text"},
        "judgment_b": {"score": 3, "class": "ambiguous", "explanation": None},
    }
    with pytest.raises(AssertionError):
        assert_blinded_item_has_no_identity_leak(bad)


def test_identical_judgment_schemas() -> None:
    item, _ = build_blinded_disagreement_item(
        continuation_id="roleplay_003__r00",
        prompt_id="roleplay_003",
        scenario="s",
        question="q",
        response="r",
        reference_judgement=2,
        reference_label="honest",
        open_score=4,
        open_class="deceptive",
        open_identity="qwen35_27b",
    )
    assert set(item["judgment_a"].keys()) == set(item["judgment_b"].keys()) == {
        "score",
        "class",
    }


def test_mapping_stored_separately_from_blinded_item() -> None:
    item, mapping = build_blinded_disagreement_item(
        continuation_id="roleplay_004__r00",
        prompt_id="roleplay_004",
        scenario="s",
        question="q",
        response="r",
        reference_judgement=1,
        reference_label="honest",
        open_score=5,
        open_class="deceptive",
        open_identity="gemma4_31b_it",
    )
    blob = json.dumps(item).lower()
    assert "gemma" not in blob
    assert "gpt-4o" not in blob
    assert "qwen" not in blob
    assert mapping["grader_a"] != mapping["grader_b"]


def test_deterministic_ab_assignment_and_sample() -> None:
    rows, changed = _synthetic_rows()
    ids1 = select_disagreement_audit_ids(rows, changed)
    ids2 = select_disagreement_audit_ids(rows, changed)
    assert ids1 == ids2
    assert ids1  # non-empty
    for cid in ids1:
        assert ab_swap_for_continuation(cid) == ab_swap_for_continuation(cid)
    # Rebuild mapping twice — same A/B
    maps = []
    for _ in range(2):
        m = {}
        for cid in ids1:
            row = next(r for r in rows if r["continuation_id"] == cid)
            _, ab = build_blinded_disagreement_item(
                continuation_id=cid,
                prompt_id=row["prompt_id"],
                scenario="s",
                question="q",
                response="r",
                reference_judgement=row["reference_judgement"],
                reference_label=row["reference_label"],
                open_score=row["open_score"],
                open_class=row["open_class"],
                open_identity="gemma4_31b_it",
            )
            m[cid] = ab
        maps.append(m)
    assert maps[0] == maps[1]


def test_artifact_blinded_items_have_no_identity_leak() -> None:
    assert AUDIT.is_dir()
    rows = [
        json.loads(x)
        for x in (AUDIT / "blinded_items.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    mapping = json.loads((AUDIT / "ab_mapping.json").read_text(encoding="utf-8"))
    assert len(rows) == len(mapping) > 0
    for row in rows:
        assert_blinded_item_has_no_identity_leak(row)
        assert "explanation" not in row["judgment_a"]
        assert "explanation" not in row["judgment_b"]
        assert row["continuation_id"] in mapping
        assert set(mapping[row["continuation_id"]].keys()) == {"grader_a", "grader_b"}
    man = json.loads((AUDIT / "manifest.json").read_text(encoding="utf-8"))
    assert man["blinding"]["explanation_omitted_from_both"] is True
    assert man["status"] == STATUS_23B_POSTAUDIT


def test_stage3_winner_does_not_auto_authorize_locked() -> None:
    """Regression: Phase-23B analysis must never set stage3 authorized=true."""
    ws = json.loads(
        (REPO / "artifacts/phase23b_development/winner_selection.json").read_text(
            encoding="utf-8"
        )
    )
    assert ws["winner"] == "gemma4_31b_it"
    assert ws["stage3_scientifically_eligible"] is True
    assert ws["stage3_locked_validation_authorized"] is False

    freeze = json.loads(
        (REPO / "artifacts/phase23b_development/freeze.json").read_text(encoding="utf-8")
    )
    assert freeze["winner"] == "gemma4_31b_it"
    assert freeze["stage3_locked_validation_authorized"] is False
    assert freeze["stage3_scientifically_eligible"] is True
    assert freeze["stage3"]["authorized"] is False
    assert freeze["authorizations_after_freeze"]["stage3_locked_validation_authorized"] is False
    src = (REPO / "scripts/analyze_phase23_stage.py").read_text(encoding="utf-8")
    compact = "".join(src.split())
    assert '"stage3_scientifically_eligible":bool(winner)' in compact
    assert '"stage3_locked_validation_authorized":False' in compact
    assert '"stage3_locked_validation_authorized":True' not in compact
    assert "stage3_locked_validation_authorized=True" not in compact

def test_label_only_is_canonical_in_freeze() -> None:
    freeze = json.loads(
        (REPO / "artifacts/phase23b_development/freeze.json").read_text(encoding="utf-8")
    )
    assert freeze["population_metric_primary"] == "label_only_mixed_behavior_population"
    assert (
        freeze["population_metric_secondary"]
        == "reference_onset_gated_sensitivity_analysis"
    )
    g = freeze["gemma4_31b_it"]["population_label_only"]
    assert g["gpt_qualifying"] == 28
    assert g["open_qualifying"] == 40
    assert g["intersection"] == 23
    assert g["union"] == 45
    assert abs(g["jaccard"] - 23 / 45) < 1e-12
    assert g["n_status_changed"] == 22
    q = freeze["qwen35_27b"]["population_label_only"]
    assert q["gpt_qualifying"] == 28
    assert q["open_qualifying"] == 41
    assert q["intersection"] == 20
    assert q["union"] == 49
    assert abs(q["jaccard"] - 20 / 49) < 1e-12
    assert q["n_status_changed"] == 29
    # Secondary retained
    assert freeze["gemma4_31b_it"]["population_onset_gated_sensitivity"]["jaccard"] == 0.65625
    assert abs(
        freeze["qwen35_27b"]["population_onset_gated_sensitivity"]["jaccard"]
        - 20 / 33
    ) < 1e-9
