"""Tests for Phase 26C primary durable checkpoint / resume (CPU only)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pre_output_physiology.phase26c_primary import (
    MODEL_REVISION,
    N_PLANNED_GENERATIONS,
    PROTOCOL_VERSION,
    build_inference_manifest,
    load_prompt_bank,
    rollout_seed,
    select_adverse_prompts,
)
from pre_output_physiology.phase26c_primary_durability import (
    CHECKPOINT_BATCH_SIZE,
    FROZEN_INFERENCE_MANIFEST_SHA256,
    RESULTS_VOLUME_NAME,
    load_checkpoints,
    reconstruct_ordered_rows,
    run_key_from_manifest_sha,
    unfinished_jobs,
    validate_checkpoint_row_against_job,
    write_checkpoint_batch,
)

ROOT = Path(__file__).resolve().parents[1]


def _mini_jobs(n: int = 5) -> list[dict]:
    bank = load_prompt_bank(ROOT / "data/phase26/production_prompts_v1.jsonl")
    selected = select_adverse_prompts(bank)
    jobs = build_inference_manifest(selected)
    return jobs[:n]


def _row_from_job(job: dict, *, technical: bool = False) -> dict:
    row = {
        "job_id": job["job_id"],
        "scenario_id": job["scenario_id"],
        "state_id": job["state_id"],
        "condition": job["condition"],
        "prompt_id": job["prompt_id"],
        "prompt_sha256": job["prompt_sha256"],
        "rollout_type": job["rollout_type"],
        "rollout_index": job["rollout_index"],
        "seed": job["seed"],
        "technical_failure": technical,
        "parse_valid": not technical,
        "parsed_decision": None if technical else "YES",
        "raw_response_text": None if technical else "REPORT: YES",
        "n_generated_tokens": 0 if technical else 5,
    }
    return row


def test_deterministic_run_key():
    key = run_key_from_manifest_sha(FROZEN_INFERENCE_MANIFEST_SHA256)
    assert key == "phase26c_primary_behavior_v1_b8f0ca6d5617"
    assert PROTOCOL_VERSION in key
    assert RESULTS_VOLUME_NAME == "phase26c-primary-results"
    assert CHECKPOINT_BATCH_SIZE == 16


def test_checkpoint_reconstruction_and_skip(tmp_path: Path):
    jobs = _mini_jobs(5)
    jobs_by_id = {j["job_id"]: j for j in jobs}
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir()
    rows = [_row_from_job(j) for j in jobs[:3]]
    write_checkpoint_batch(checkpoint_dir=ckpt, batch_index=0, rows=rows)
    completed, errors = load_checkpoints(checkpoint_dir=ckpt, jobs_by_id=jobs_by_id)
    assert not errors
    assert len(completed) == 3
    rem = unfinished_jobs(jobs, set(completed))
    assert len(rem) == 2
    assert rem[0]["job_id"] == jobs[3]["job_id"]


def test_duplicate_checkpoint_job_rejected(tmp_path: Path):
    jobs = _mini_jobs(3)
    jobs_by_id = {j["job_id"]: j for j in jobs}
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir()
    r0 = _row_from_job(jobs[0])
    write_checkpoint_batch(checkpoint_dir=ckpt, batch_index=0, rows=[r0])
    write_checkpoint_batch(checkpoint_dir=ckpt, batch_index=1, rows=[r0])
    completed, errors = load_checkpoints(checkpoint_dir=ckpt, jobs_by_id=jobs_by_id)
    assert any("duplicate" in e for e in errors)


def test_mismatched_seed_rejected(tmp_path: Path):
    jobs = _mini_jobs(2)
    jobs_by_id = {j["job_id"]: j for j in jobs}
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir()
    bad = _row_from_job(jobs[0])
    bad["seed"] = int(bad["seed"]) ^ 0xABCDEF
    write_checkpoint_batch(checkpoint_dir=ckpt, batch_index=0, rows=[bad])
    completed, errors = load_checkpoints(checkpoint_dir=ckpt, jobs_by_id=jobs_by_id)
    assert any("seed" in e for e in errors)
    assert jobs[0]["job_id"] not in completed


def test_mismatched_prompt_hash_rejected(tmp_path: Path):
    jobs = _mini_jobs(2)
    jobs_by_id = {j["job_id"]: j for j in jobs}
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir()
    bad = _row_from_job(jobs[0])
    bad["prompt_sha256"] = "0" * 64
    write_checkpoint_batch(checkpoint_dir=ckpt, batch_index=0, rows=[bad])
    _, errors = load_checkpoints(checkpoint_dir=ckpt, jobs_by_id=jobs_by_id)
    assert any("prompt_sha256" in e for e in errors)


def test_no_overwrite_existing_checkpoint(tmp_path: Path):
    jobs = _mini_jobs(2)
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir()
    write_checkpoint_batch(
        checkpoint_dir=ckpt, batch_index=0, rows=[_row_from_job(jobs[0])]
    )
    with pytest.raises(ValueError, match="refusing to overwrite"):
        write_checkpoint_batch(
            checkpoint_dir=ckpt, batch_index=0, rows=[_row_from_job(jobs[1])]
        )


def test_interrupted_resume_continues(tmp_path: Path):
    jobs = _mini_jobs(8)
    jobs_by_id = {j["job_id"]: j for j in jobs}
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir()
    write_checkpoint_batch(
        checkpoint_dir=ckpt,
        batch_index=0,
        rows=[_row_from_job(j) for j in jobs[:4]],
    )
    completed, errors = load_checkpoints(checkpoint_dir=ckpt, jobs_by_id=jobs_by_id)
    assert not errors
    rem = unfinished_jobs(jobs, set(completed))
    assert [j["job_id"] for j in rem] == [j["job_id"] for j in jobs[4:]]
    # second batch
    write_checkpoint_batch(
        checkpoint_dir=ckpt,
        batch_index=1,
        rows=[_row_from_job(j) for j in rem],
    )
    completed2, errors2 = load_checkpoints(checkpoint_dir=ckpt, jobs_by_id=jobs_by_id)
    assert not errors2
    assert len(completed2) == 8
    assert unfinished_jobs(jobs, set(completed2)) == []


def test_completion_requires_exact_2904(tmp_path: Path):
    jobs = _mini_jobs(3)
    completed = {j["job_id"]: _row_from_job(j) for j in jobs}
    with pytest.raises(ValueError, match="cannot complete"):
        reconstruct_ordered_rows(manifest_jobs=jobs, completed=completed)


def test_safe_row_rejected():
    jobs = _mini_jobs(1)
    job = dict(jobs[0])
    row = _row_from_job(job)
    row["state_id"] = "SAFE"
    errs = validate_checkpoint_row_against_job(row, job)
    assert any("SAFE" in e for e in errs)


def test_scientific_plan_and_seed_unchanged():
    assert N_PLANNED_GENERATIONS == 2904
    assert MODEL_REVISION == "63a8b081895390a26e140280378bc85ec8bce07a"
    assert rollout_seed(prompt_sha256="abc", condition="X_C1", rollout_index=0) == 1410709147
    bank = load_prompt_bank(ROOT / "data/phase26/production_prompts_v1.jsonl")
    selected = select_adverse_prompts(bank)
    assert len(selected) == 312
    assert all(p["state_id"] == "ADVERSE" for p in selected)
    jobs = build_inference_manifest(selected)
    assert len(jobs) == 2904
    assert sum(1 for j in jobs if j["condition"] == "K") == 216
    assert sum(1 for j in jobs if j["condition"].startswith("N_")) == 384
    assert sum(1 for j in jobs if j["condition"].startswith("X_")) == 1536
    assert sum(1 for j in jobs if j["condition"].startswith("E_")) == 768


def test_no_physiology_paths_in_durability_module():
    text = (
        ROOT / "src/pre_output_physiology/phase26c_primary_durability.py"
    ).read_text()
    assert "output_hidden_states" in text  # must be explicitly false in progress
    # no activation capture wiring
    assert "register_forward_hook" not in text
    assert "get_activations" not in text
    modal_text = (
        ROOT / "modal/phase26c_primary_behavioral_feasibility.py"
    ).read_text()
    assert "activation_capture\": False" in modal_text or "activation_capture=False" in modal_text
    assert "run_primary_resumable" in modal_text
    assert "Retries" in modal_text
    assert "spawn" in modal_text
