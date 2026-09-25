"""Phase 4 contract tests — design + first-token ID compliance (no GPU in tests)."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from pre_output_physiology.config import EXPERIMENTS_DIR, load_experiment_config
from pre_output_physiology.phase4_behavior import (
    first_token_id_compliant,
    is_behaviorally_valid,
    parse_response,
)
from pre_output_physiology.phase4_conditions import (
    COMMON_FIRST_TOKEN,
    COMMON_FIRST_TOKEN_ID,
    CONDITION_ORDER,
    FORBIDDEN_PROMPT_SUBSTRINGS,
    KEY_SECONDARY_CONTRAST,
    N_FINAL_BASE_SCENARIOS,
    PRIMARY_CONTRAST,
    SCENARIO_FAMILIES,
    STRATEGIC_NONDISCLOSURE_INSTRUCTION,
    assert_c2_c3_template_symmetry,
    assert_no_forbidden_prompt_terms,
    build_condition_prompt,
    canonicalize_c2_c3_prompt,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_phase4_experiment_status() -> None:
    cfg = load_experiment_config(EXPERIMENTS_DIR / "phase4_specificity.yaml")
    assert cfg.status in {
        "phase4a_design_frozen_awaiting_pilot",
        "phase4b_pilot_complete_awaiting_audit",
    }


def test_phase4_yaml_authorizations() -> None:
    raw = yaml.safe_load(
        (REPO_ROOT / "configs/experiments/phase4_specificity.yaml").read_text(
            encoding="utf-8"
        )
    )
    auth = raw["authorizations"]
    # Pilot may be authorized; final/activation/causal must stay false.
    assert auth["final_generation_authorized"] is False
    assert auth["activation_extraction_authorized"] is False
    assert auth["causal_intervention_authorized"] is False
    assert raw["frozen_phase3"]["primary_endpoint"]["layer"] == 12
    assert raw["frozen_phase3"]["primary_endpoint"]["k"] == 1
    assert raw["common_first_token"] == COMMON_FIRST_TOKEN
    assert raw["common_first_token_id"] == COMMON_FIRST_TOKEN_ID
    assert raw["design"]["n_final_base_scenarios"] == N_FINAL_BASE_SCENARIOS
    assert raw["design"]["n_pilot_base_scenarios"] == 24
    assert raw["design"]["pilot_seed"] == 7
    assert raw["design"]["primary_contrast"] == list(PRIMARY_CONTRAST)
    assert raw["design"]["key_secondary_contrast"] == list(KEY_SECONDARY_CONTRAST)


def test_phase4_prompt_lexicon_and_six_conditions() -> None:
    assert len(CONDITION_ORDER) == 6
    assert len(SCENARIO_FAMILIES) == 6
    sc = {
        "topic_sentence": "A catalog lists where the red box is stored.",
        "state_label_a": "STATE_A",
        "state_label_b": "STATE_B",
        "state_a": "north room",
        "state_b": "south room",
        "ground_truth_state": "north room",
        "alt_state": "south room",
        "accessible_false_state": "south room",
    }
    for cid in CONDITION_ORDER:
        prompt = build_condition_prompt(condition_id=cid, scenario=sc)
        assert_no_forbidden_prompt_terms(prompt)
        # Whole-word check: substrings like 'belief'/'reliably' are allowed.
        for term in FORBIDDEN_PROMPT_SUBSTRINGS:
            assert re.search(rf"\b{re.escape(term)}\b", prompt.lower()) is None


def test_phase4_first_token_id_compliance() -> None:
    assert first_token_id_compliant(2963)
    assert not first_token_id_compliant(1)
    assert not first_token_id_compliant(None)
    # Decoded "Response:" must still count as compliant when ID is 2963.
    parsed = parse_response(
        "Response: the state is north room\nFINAL=north room\n",
        first_generated_token_id=2963,
    )
    assert parsed.first_token_ok is True
    assert parsed.first_token_decoded_word == "Response:"
    sc = {
        "ground_truth_state": "north room",
        "alt_state": "south room",
        "accessible_false_state": "south room",
    }
    ok, reason = is_behaviorally_valid(
        "C1_known_honest_neutral",
        sc,
        "Response: north room\nFINAL=north room\n",
        first_generated_token_id=2963,
    )
    assert ok and reason == "ok"
    bad, bad_reason = is_behaviorally_valid(
        "C1_known_honest_neutral",
        sc,
        "Response: north room\nFINAL=north room\n",
        first_generated_token_id=1234,
    )
    assert not bad and bad_reason == "first_token_mismatch"


def test_phase4_behavior_rules() -> None:
    sc = {
        "ground_truth_state": "north room",
        "alt_state": "south room",
        "accessible_false_state": "south room",
    }
    ok, _ = is_behaviorally_valid(
        "C3_known_deceptive_strategic",
        sc,
        "Response\n...\nFINAL=south room\n",
        first_generated_token_id=2963,
    )
    assert ok
    bad, reason = is_behaviorally_valid(
        "C3_known_deceptive_strategic",
        sc,
        "Response\nprivate scoring revealed\nFINAL=south room\n",
        first_generated_token_id=2963,
    )
    assert not bad
    assert reason == "disclosed_private_objective"
    ok_c2, _ = is_behaviorally_valid(
        "C2_known_honest_strategic",
        sc,
        "Response\n...\nFINAL=north room\n",
        first_generated_token_id=2963,
    )
    assert ok_c2


def test_c2_c3_strategic_template_symmetry() -> None:
    sc = {
        "topic_sentence": "A catalog lists where the red box is stored.",
        "state_label_a": "STATE_A",
        "state_label_b": "STATE_B",
        "state_a": "north room",
        "state_b": "south room",
        "ground_truth_state": "north room",
        "alt_state": "south room",
        "accessible_false_state": "south room",
    }
    assert_c2_c3_template_symmetry(sc)
    c2 = build_condition_prompt(condition_id="C2_known_honest_strategic", scenario=sc)
    c3 = build_condition_prompt(condition_id="C3_known_deceptive_strategic", scenario=sc)
    assert STRATEGIC_NONDISCLOSURE_INSTRUCTION in c2
    assert STRATEGIC_NONDISCLOSURE_INSTRUCTION in c3
    assert "opposite" not in c2.lower()
    assert "opposite" not in c3.lower()
    can2 = canonicalize_c2_c3_prompt(
        c2,
        ground_truth_state="north room",
        alt_state="south room",
        target_state="north room",
    )
    can3 = canonicalize_c2_c3_prompt(
        c3,
        ground_truth_state="north room",
        alt_state="south room",
        target_state="south room",
    )
    assert can2 == can3
    assert "Reward target: <TARGET>." in can2


def test_decision_log_phase4() -> None:
    text = (REPO_ROOT / "docs/decision_log.md").read_text(encoding="utf-8")
    for did in (
        "D045",
        "D046",
        "D047",
        "D048",
        "D049",
        "D050",
        "D051",
        "D052",
        "D053",
        "D054",
    ):
        assert did in text


def test_phase4_protocol_exists() -> None:
    path = REPO_ROOT / "docs/phase4_specificity_protocol.md"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "C3" in text and "C2" in text
    assert "2026.acl-long.1582" in text
    assert "2026.findings-acl.1139" in text
    assert "2026.acl-long.1849" in text
