"""Phase 6B factorial analysis, freeze pins, and secondary diagnostic (no GPU)."""

from __future__ import annotations

import numpy as np

from pre_output_physiology import phase6b_freeze as fz
from pre_output_physiology.phase6_analysis import analyze, effects_per_scenario
from pre_output_physiology.phase6_behavior import (
    is_behaviorally_valid,
    phase6b_semantic_first_line,
)
from pre_output_physiology.phase6b_report import REQUIRED_STATEMENT, render_report


def _synthetic(comm: float, op: float, inter: float, n_per_fam: int = 20, seed: int = 1):
    rng = np.random.default_rng(seed)
    fams = [f"fam{i}" for i in range(6)]
    families = [f for f in fams for _ in range(n_per_fam)]
    n = len(families)
    base = rng.normal(0, 0.05, n)
    noise = lambda: rng.normal(0, 0.01, n)  # noqa: E731
    s = {
        "F00": base + noise(),
        "F10": base + op + noise(),
        "F01": base + comm + noise(),
        "F11": base + comm + op + inter + noise(),
    }
    return s, [f"b{i:03d}" for i in range(n)], families


def test_effect_formulas_exact() -> None:
    s = {k: np.array([v]) for k, v in {"F00": 0.1, "F10": 0.2, "F01": 0.5, "F11": 0.9}.items()}
    e = effects_per_scenario(s)
    assert np.isclose(e["communication_main_effect"][0], 0.5 * ((0.5 - 0.1) + (0.9 - 0.2)))
    assert np.isclose(e["operational_main_effect"][0], 0.5 * ((0.2 - 0.1) + (0.9 - 0.5)))
    assert np.isclose(e["interaction"][0], (0.9 - 0.2) - (0.5 - 0.1))


def test_analysis_recovers_communication_dominant() -> None:
    s, ids, fams = _synthetic(comm=0.4, op=0.0, inter=0.0)
    a = analyze(s, ids, fams)
    assert a["contrasts"]["F11_minus_F10"]["paired_mean_delta_ci_low"] > 0
    assert a["contrasts"]["F11_minus_F10"]["auroc"] > 0.95
    assert a["effects"]["communication_main_effect"]["ci_low"] > 0.35
    assert a["pattern_flags"]["pattern_A_communication_dominant"]
    assert not a["pattern_flags"]["pattern_B_operational_dominant"]
    assert len(a["effects_by_family"]) == 6


def test_analysis_recovers_operational_dominant() -> None:
    s, ids, fams = _synthetic(comm=0.0, op=0.4, inter=0.0)
    a = analyze(s, ids, fams)
    assert a["pattern_flags"]["pattern_B_operational_dominant"]
    assert not a["pattern_flags"]["pattern_A_communication_dominant"]


def test_analysis_recovers_interaction() -> None:
    s, ids, fams = _synthetic(comm=0.0, op=0.0, inter=0.4)
    a = analyze(s, ids, fams)
    assert a["effects"]["interaction"]["ci_low"] > 0.3
    assert a["pattern_flags"]["pattern_C_interaction_mixed"]
    assert not a["pattern_flags"]["pattern_A_communication_dominant"]


def test_analysis_deterministic() -> None:
    s, ids, fams = _synthetic(comm=0.2, op=0.1, inter=0.0)
    assert analyze(s, ids, fams) == analyze(s, ids, fams)


def test_label_free_payload() -> None:
    rows = [{"example_id": "x__F10", "condition_id": "F10", "prompt_text": "p"}]
    assert fz.label_free_payload(rows) == [{"row_index": 0, "prompt_text": "p"}]


def test_frozen_pins() -> None:
    _p, _s, hashes = fz.load_verified_corpus()
    assert hashes["final_prompt_text_sha256"] == fz.EXPECTED_PROMPT_TEXT_SHA256
    assert fz.verify_probe_sha() == fz.EXPECTED_PROBE_SHA256


def test_semantic_diagnostic_does_not_change_original_metric() -> None:
    sc = {"record_state": "jar 1", "alternate_state": "jar 2"}
    text = "Response jar 2\nNote: this differs from the private record."
    ok, reason = is_behaviorally_valid(
        "F01", sc, text, first_generated_token_id=12107, expected_first_token_id=12107
    )
    assert not ok and reason == "not_exact_one_nonempty_line"
    d = phase6b_semantic_first_line("F01", sc, text)
    assert d["semantic_first_line_valid"] and d["trailing_disclaimer_or_comment"]
    assert not phase6b_semantic_first_line("F10", sc, "Response jar 2")["semantic_first_line_valid"]


def test_report_contains_required_statement() -> None:
    s, ids, fams = _synthetic(comm=0.2, op=0.1, inter=0.0)
    a = analyze(s, ids, fams)
    summary = {
        "status": fz.STATUS_PRIMARY_COMPLETE,
        "phase6a_outcome": fz.PHASE6A_OUTCOME,
        "pre_run_commit": "abc",
        "probe_sha256": fz.EXPECTED_PROBE_SHA256,
        "activation_integrity": {"passed": True},
        "extraction": {
            "run_id": "r",
            "final_prompt_text_sha256": "p",
            "final_scenario_text_sha256": "s",
            "final_base_scenario_ids_sha256": "i",
            "preflight_repeatability_min_cosine": 1.0,
            "wall_seconds": 1.0,
            "preflight_wall_seconds": 1.0,
            "total_estimated_cost_usd": 0.1,
        },
        "analysis_probability": a,
        "analysis_logit_secondary": a,
    }
    text = render_report(summary)
    assert REQUIRED_STATEMENT in text
    assert "not** a general deception-intent test" in text
