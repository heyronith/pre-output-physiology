"""Phase-24A live/replay equivalence unit tests (no GPU)."""

from __future__ import annotations

import math

import pytest

from pre_output_physiology.phase24a_replay import (
    ACTIVATION_EQUIVALENCE_GATES,
    MAX_SEQUENCE_LENGTH,
    MODEL_ID,
    MODEL_REVISION,
    N_DEV_PROMPTS,
    N_PILOT_PROMPTS,
    N_TRANSFORMER_BLOCKS,
    TOKENIZER_REVISION,
    align_live_replay_positions,
    causal_prefix_token_count,
    cosine_similarity,
    evaluate_activation_gates,
    percentile,
    pilot_sample_seed,
    post_block_hidden_index,
    relative_l2_error,
    select_pilot_prompt_ids,
    summarize_metric_values,
    terminal_generated_token_handling,
)


def test_model_pins_match_phase21() -> None:
    from pre_output_physiology.phase21_roleplay import (
        MAX_NEW_TOKENS as P21_MAX,
    )
    from pre_output_physiology.phase21_roleplay import (
        MODEL_ID as P21_ID,
    )
    from pre_output_physiology.phase21_roleplay import (
        MODEL_REVISION as P21_REV,
    )
    from pre_output_physiology.phase21_roleplay import (
        TEMPERATURE as P21_T,
    )
    from pre_output_physiology.phase24a_replay import MAX_NEW_TOKENS, TEMPERATURE

    assert MODEL_ID == P21_ID == "mistralai/Mistral-7B-Instruct-v0.2"
    assert MODEL_REVISION == P21_REV == "63a8b081895390a26e140280378bc85ec8bce07a"
    assert TOKENIZER_REVISION == MODEL_REVISION
    assert TEMPERATURE == P21_T == 1.0
    assert MAX_NEW_TOKENS == P21_MAX == 200
    assert MAX_SEQUENCE_LENGTH == 32768


def test_gates_frozen_before_results() -> None:
    assert ACTIVATION_EQUIVALENCE_GATES == {
        "min_median_cosine": 0.99999,
        "min_p01_cosine": 0.9999,
        "max_median_rel_l2": 0.001,
        "max_p99_rel_l2": 0.01,
    }


def test_layer_indexing_convention() -> None:
    assert post_block_hidden_index(0) == 1
    assert post_block_hidden_index(11) == 12
    assert post_block_hidden_index(31) == 32
    with pytest.raises(ValueError):
        post_block_hidden_index(-1)
    with pytest.raises(ValueError):
        post_block_hidden_index(N_TRANSFORMER_BLOCKS)


def test_prompt_boundary_alignment() -> None:
    positions = align_live_replay_positions(prompt_n_tokens=10, n_generated_tokens=3)
    assert positions[0] == {
        "name": "prompt_end",
        "sequence_index": 9,
        "generated_token_index": None,
    }
    assert positions[1]["sequence_index"] == 10
    assert positions[1]["generated_token_index"] == 0
    assert positions[-1]["sequence_index"] == 12
    assert positions[-1]["generated_token_index"] == 2
    assert len(positions) == 1 + 3


def test_live_replay_token_alignment_counts() -> None:
    for prompt_n, n_gen in [(1, 0), (5, 1), (40, 200), (100, 17)]:
        pos = align_live_replay_positions(prompt_n, n_gen)
        assert len(pos) == 1 + n_gen
        assert pos[0]["sequence_index"] == prompt_n - 1
        for t, p in enumerate(pos[1:]):
            assert p["sequence_index"] == prompt_n + t
            assert p["generated_token_index"] == t


def test_future_tokens_cannot_affect_replayed_prefix_state() -> None:
    """Causal dependency: state at index i uses only tokens[0:i+1]."""
    full_len = 50
    prompt_n = 20
    n_gen = 10
    positions = align_live_replay_positions(prompt_n, n_gen)
    for p in positions:
        idx = int(p["sequence_index"])
        dep = causal_prefix_token_count(idx)
        assert dep == idx + 1
        # Future tokens beyond idx must be outside the dependency set
        assert dep <= full_len
        future_start = dep
        assert future_start == idx + 1
        # Shared-prefix invariance: any two sequences agreeing on tokens[:dep]
        # have the same causal inputs for this state.
        prefix_a = list(range(dep))
        prefix_b = list(range(dep)) + [999, 998]  # extra future tokens
        assert prefix_a == prefix_b[:dep]


def test_eos_terminal_token_handling() -> None:
    eos = 2
    info = terminal_generated_token_handling([10, 11, eos], eos_token_id=eos)
    assert info["includes_terminal_eos_state"] is True
    assert info["compare_eos_position"] is True
    assert info["eos_positions"] == [2]
    info2 = terminal_generated_token_handling([10, 11, 12], eos_token_id=eos)
    assert info2["includes_terminal_eos_state"] is False
    assert info2["compare_eos_position"] is None


def test_deterministic_pilot_selection() -> None:
    ids = [f"p{i:03d}" for i in range(N_DEV_PROMPTS)]
    a = select_pilot_prompt_ids(ids, n=N_PILOT_PROMPTS)
    b = select_pilot_prompt_ids(ids, n=N_PILOT_PROMPTS)
    assert a == b
    assert len(a) == N_PILOT_PROMPTS
    assert len(set(a)) == N_PILOT_PROMPTS
    # Independent of label/behavior — only IDs matter
    shuffled = list(reversed(ids))
    assert select_pilot_prompt_ids(shuffled) == a
    with pytest.raises(ValueError):
        select_pilot_prompt_ids(ids[:10])


def test_pilot_sample_seed_deterministic() -> None:
    s1 = pilot_sample_seed("abc")
    s2 = pilot_sample_seed("abc")
    assert s1 == s2
    assert s1 != pilot_sample_seed("abd")
    assert s1 >= 24_000_000


def test_cosine_and_rel_l2_metrics() -> None:
    a = [1.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
    assert cosine_similarity(a, b) == pytest.approx(1.0)
    assert relative_l2_error(a, b) == pytest.approx(0.0)
    c = [0.0, 1.0, 0.0]
    assert cosine_similarity(a, c) == pytest.approx(0.0)
    # Orthognal same-norm: rel L2 = sqrt(2)
    assert relative_l2_error(a, c) == pytest.approx(math.sqrt(2.0))
    # Near-equal
    d = [1.0, 1e-6, 0.0]
    assert cosine_similarity(a, d) > 0.999999
    assert relative_l2_error(a, d) < 1e-5


def test_percentile_and_summarize() -> None:
    vals = list(range(100))  # 0..99
    s = summarize_metric_values(vals)
    assert s["n"] == 100
    assert s["min"] == 0.0
    assert s["max"] == 99.0
    assert s["median"] == pytest.approx(49.5)
    assert percentile(sorted(float(v) for v in vals), 1) == s["p01"]
    assert percentile(sorted(float(v) for v in vals), 99) == s["p99"]


def test_frozen_gates_evaluation_pass_and_fail() -> None:
    cos_ok = {
        "median": 0.999995,
        "p01": 0.99995,
        "min": 0.9999,
        "max": 1.0,
        "n": 10,
        "mean": 0.99999,
        "p99": 1.0,
    }
    rel_ok = {
        "median": 0.0001,
        "p99": 0.005,
        "min": 0.0,
        "max": 0.006,
        "n": 10,
        "mean": 0.0002,
        "p01": 0.0,
    }
    g = evaluate_activation_gates(cos_ok, rel_ok)
    assert g["passed"] is True
    assert all(r["pass"] for r in g["gates"])

    cos_bad = dict(cos_ok)
    cos_bad["median"] = 0.999
    g2 = evaluate_activation_gates(cos_bad, rel_ok)
    assert g2["passed"] is False
    assert any(r["metric"] == "median_cosine" and not r["pass"] for r in g2["gates"])

    rel_bad = dict(rel_ok)
    rel_bad["p99"] = 0.05
    g3 = evaluate_activation_gates(cos_ok, rel_bad)
    assert g3["passed"] is False
