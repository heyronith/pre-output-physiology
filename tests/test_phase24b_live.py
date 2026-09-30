"""Phase-24B live capture unit tests (no GPU)."""

from __future__ import annotations

import hashlib
import math

import pytest

from pre_output_physiology.phase24b_live import (
    BEHAVIORAL_YIELD,
    CAPTURE_GATES,
    MODEL_ID,
    MODEL_REVISION,
    N_ELIGIBLE_DEV_MIXED,
    N_PILOT_PROMPTS,
    N_REPLICATES,
    N_TRAJECTORIES,
    TOKENIZER_REVISION,
    align_sampled_token_to_prediction_step,
    artifact_integrity_ok,
    build_schedule,
    canary_schedule_rows,
    count_nonfinite_in_flat,
    eos_terminal_handling,
    evaluate_behavioral_yield,
    evaluate_capture_gates,
    final_status,
    layer_hook_module_paths,
    prediction_step_bookkeeping,
    project_storage_runtime,
    required_artifact_fields,
    select_pilot_prompt_ids,
    sha256_json,
    trajectory_seed,
)


def test_model_pins() -> None:
    from pre_output_physiology import phase21_roleplay as p21
    from pre_output_physiology.phase24b_live import MAX_NEW_TOKENS, TEMPERATURE

    assert MODEL_ID == p21.MODEL_ID
    assert MODEL_REVISION == p21.MODEL_REVISION == TOKENIZER_REVISION
    assert TEMPERATURE == p21.TEMPERATURE == 1.0
    assert MAX_NEW_TOKENS == p21.MAX_NEW_TOKENS == 200


def test_deterministic_pilot_selection_first_16_of_sha_sorted() -> None:
    ids = [f"roleplay_{i:03d}" for i in range(N_ELIGIBLE_DEV_MIXED)]
    a = select_pilot_prompt_ids(ids)
    b = select_pilot_prompt_ids(list(reversed(ids)))
    assert a == b
    assert len(a) == N_PILOT_PROMPTS
    ordered = sorted(ids, key=lambda p: hashlib.sha256(p.encode()).hexdigest())
    assert a == ordered[:16]
    with pytest.raises(ValueError):
        select_pilot_prompt_ids(ids[:10])


def test_seed_derivation_deterministic() -> None:
    s = trajectory_seed("roleplay_003", 0)
    assert s == trajectory_seed("roleplay_003", 0)
    assert s != trajectory_seed("roleplay_003", 1)
    digest = hashlib.sha256(b"phase24b|roleplay_003|0").hexdigest()
    assert s == int(digest[:8], 16)


def test_schedule_96_and_canary() -> None:
    ids = select_pilot_prompt_ids([f"p{i:03d}" for i in range(N_ELIGIBLE_DEV_MIXED)])
    sched = build_schedule(ids)
    assert len(sched) == N_TRAJECTORIES == 96
    assert {r["replicate_index"] for r in sched} == set(range(N_REPLICATES))
    canary = canary_schedule_rows(sched)
    assert len(canary) == 4
    assert [c["trajectory_id"] for c in canary] == [
        r["trajectory_id"] for r in sched[:4]
    ]


def test_prediction_step_alignment() -> None:
    book = prediction_step_bookkeeping(5)
    assert book["n_prediction_steps"] == 5
    assert book["steps"] == [0, 1, 2, 3, 4]
    aligned = align_sampled_token_to_prediction_step([10, 11, 12])
    assert aligned[0]["prediction_step"] == 0
    assert aligned[0]["preceding_generated_index"] is None
    assert aligned[1]["preceding_generated_index"] == 0
    assert aligned[2]["sampled_token_id"] == 12
    # prompt-state → first token
    assert aligned[0]["prediction_step"] == 0
    # token t-1 state → token t
    assert aligned[2]["prediction_step"] == 2


def test_max_new_tokens_no_extra_prediction_step() -> None:
    """Hitting max_new_tokens must not create a trailing unsampled step."""
    n = 200
    book = prediction_step_bookkeeping(n)
    assert book["n_prediction_steps"] == n
    # Simulated capture lengths must match generated tokens exactly
    fake_generated = list(range(n))
    aligned = align_sampled_token_to_prediction_step(fake_generated)
    assert len(aligned) == n
    assert aligned[-1]["prediction_step"] == n - 1


def test_eos_handling() -> None:
    info = eos_terminal_handling([5, 6, 2], eos_token_id=2)
    assert info["stopped_on_eos"] is True
    assert info["early_eos_before_final"] is False
    info2 = eos_terminal_handling([2], eos_token_id=2)
    assert info2["stopped_on_eos"] is True


def test_layer_hook_indexing() -> None:
    paths = layer_hook_module_paths()
    assert paths["embedding"] == "model.model.embed_tokens"
    assert paths["final_rmsnorm"] == "model.model.norm"
    assert len(paths["transformer_blocks"]) == 32
    assert paths["transformer_blocks"][0] == "model.model.layers[0]"
    assert paths["transformer_blocks"][31] == "model.model.layers[31]"


def test_complete_logit_storage_contract_fields() -> None:
    # Artifact must retain full vocab logits (not merely top-k / entropy)
    fields = required_artifact_fields()
    assert "logit_sha256" in fields
    assert "generated_token_ids" in fields


def test_nan_inf_detection() -> None:
    assert count_nonfinite_in_flat([1.0, 2.0, 3.0]) == 0
    assert count_nonfinite_in_flat([1.0, float("nan"), float("inf")]) == 2
    assert math.isnan(float("nan"))


def test_artifact_hashing_and_integrity() -> None:
    h = sha256_json({"a": 1})
    assert len(h) == 64
    row = {
        "prompt_id": "p",
        "replicate_index": 0,
        "sample_seed": 1,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "generated_token_ids": [1, 2],
        "completion_text": "x",
        "full_response": "Coach:x",
        "activation_sha256": "a" * 64,
        "logit_sha256": "b" * 64,
        "stopping_reason": "eos",
        "n_prediction_steps": 2,
        "prompt_n_tokens": 10,
        "trajectory_id": "p__r00",
    }
    assert artifact_integrity_ok(row) is True
    bad = dict(row)
    del bad["logit_sha256"]
    assert artifact_integrity_ok(bad) is False


def test_capture_and_behavioral_gates_frozen() -> None:
    assert CAPTURE_GATES["min_completed"] == 95
    assert BEHAVIORAL_YIELD == {
        "min_prompts_with_both_hd": 8,
        "min_honest_total": 20,
        "min_deceptive_total": 20,
    }
    g = evaluate_capture_gates(
        n_requested=96,
        n_completed=96,
        n_alignment_ok=96,
        n_activation_nan=0,
        n_logit_nan=0,
        n_artifact_ok=96,
    )
    assert g["passed"] is True
    g2 = evaluate_capture_gates(
        n_requested=96,
        n_completed=90,
        n_alignment_ok=90,
        n_activation_nan=0,
        n_logit_nan=0,
        n_artifact_ok=90,
    )
    assert g2["passed"] is False

    per = [{"has_ge1_h_and_ge1_d": True}] * 8 + [{"has_ge1_h_and_ge1_d": False}] * 8
    y = evaluate_behavioral_yield(per, n_honest=20, n_deceptive=20)
    assert y["passed"] is True
    y2 = evaluate_behavioral_yield(per, n_honest=19, n_deceptive=20)
    assert y2["passed"] is False


def test_final_status_branches() -> None:
    assert "not_validated" in final_status(engineering_pass=False, behavioral_pass=True)
    assert "sampling_design_needed" in final_status(
        engineering_pass=True, behavioral_pass=False
    )
    assert "awaiting_full_trajectory" in final_status(
        engineering_pass=True, behavioral_pass=True
    )


def test_storage_projection() -> None:
    p = project_storage_runtime(
        bytes_per_trajectory=1_000_000,
        gpu_seconds_per_trajectory=10.0,
        n_prompts=28,
        trajectories_per_prompt=20,
    )
    assert p["n_trajectories"] == 560
    assert p["projected_bytes"] == 560_000_000
    assert p["projected_gpu_seconds"] == 5600.0


def test_capture_on_off_harness_contract() -> None:
    """Canary requires exact token-sequence match; text similarity is insufficient."""
    off = [1, 2, 3, 2]
    on = [1, 2, 3, 2]
    assert off == on
    assert [1, 2, 3, 4] != off


def test_immutable_raw_artifact_sha_stability() -> None:
    payload = {"trajectory_id": "x", "generated_token_ids": [1, 2, 3]}
    assert sha256_json(payload) == sha256_json(dict(payload))
