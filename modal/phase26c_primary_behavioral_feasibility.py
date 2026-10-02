"""Modal GPU: Phase 26C primary — detached, checkpointed, resumable (text only).

Durable results live on Modal Volume `phase26c-primary-results`.
Local process only validates, spawns, briefly verifies health, then exits.

GPU execution is BLOCKED until configs/...yaml sets:
  authorizations.modal_gpu_mistral_screening_authorized: true
  status: phase26c_primary_generation_authorized
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

import modal

from pre_output_physiology.phase26c_primary import (
    ATTN_IMPLEMENTATION,
    MODEL_ID,
    MODEL_REVISION,
    N_PLANNED_GENERATIONS,
    PROTOCOL_VERSION,
    TOKENIZER_REVISION,
)
from pre_output_physiology.phase26c_primary_durability import (
    CHECKPOINT_BATCH_SIZE,
    FROZEN_INFERENCE_MANIFEST_SHA256,
    FROZEN_SEED_MANIFEST_SHA256,
    FROZEN_SELECTION_MANIFEST_SHA256,
    RESULTS_MOUNT,
    RESULTS_VOLUME_NAME,
    assert_frozen_hashes,
    atomic_write_json,
    atomic_write_jsonl,
    load_checkpoints,
    reconstruct_ordered_rows,
    run_dir,
    run_key_from_manifest_sha,
    sha256_bytes,
    sha256_file,
    unfinished_jobs,
    write_checkpoint_batch,
    write_heartbeat,
    write_progress,
)

APP_NAME = "pre-output-physiology-phase26c-primary"
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/phase26c_primary_behavioral_feasibility.yaml"
MANIFEST_PATH = REPO_ROOT / "artifacts/phase26c_primary/inference_manifest.jsonl"
SEED_PATH = REPO_ROOT / "artifacts/phase26c_primary/seed_manifest.jsonl"
SELECTION_PATH = REPO_ROOT / "data/phase26/phase26c_primary_prompt_selection.jsonl"
BANK_PATH = REPO_ROOT / "data/phase26/production_prompts_v1.jsonl"
LAUNCH_RECEIPT = REPO_ROOT / "artifacts/phase26c_primary/launch_receipt.json"

A100_USD_PER_HOUR = 2.50
GPU_TYPE = "A100-80GB"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.44.2",
        "accelerate==0.33.0",
        "numpy==1.26.4",
        "huggingface_hub==0.24.6",
        "safetensors==0.4.5",
        "pyyaml==6.0.2",
    )
    .add_local_python_source("pre_output_physiology")
)
app = modal.App(APP_NAME)
mistral_vol = modal.Volume.from_name("preoutput-mistral-cache", create_if_missing=True)
results_vol = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)
MODEL_CACHE = "/vol/hf_cache"


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _dirty_ok() -> None:
    raw = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True
    ).strip()
    if not raw:
        return
    frozen_forbidden = {
        "artifacts/phase26c_primary/inference_manifest.jsonl",
        "artifacts/phase26c_primary/seed_manifest.jsonl",
        "data/phase26/phase26c_primary_prompt_selection.jsonl",
        "data/phase26/production_prompts_v1.jsonl",
        "data/phase26/template_assignment_v1.json",
    }
    allowed_exact = {"artifacts/phase26c_primary/launch_receipt.json"}
    allowed_prefixes = (
        "artifacts/phase26c_primary/launch_",
        "reports/phase26c_primary",
    )
    for line in raw.splitlines():
        path = line[3:].strip().strip('"').lstrip("./")
        if path in frozen_forbidden:
            raise SystemExit(f"STOP: frozen artifact dirty: {path}")
        if path in allowed_exact or any(path.startswith(p) for p in allowed_prefixes):
            continue
        raise SystemExit(f"STOP: dirty tree outside permitted launch outputs:\n{raw}")


def _load_auth() -> dict[str, Any]:
    _dirty_ok()
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if cfg.get("status") != "phase26c_primary_generation_authorized":
        raise SystemExit(
            f"GPU blocked: status={cfg.get('status')!r}; "
            "requires phase26c_primary_generation_authorized after audit"
        )
    auth = cfg["authorizations"]
    if auth.get("modal_gpu_mistral_screening_authorized") is not True:
        raise SystemExit("modal GPU screening not authorized")
    for key in (
        "activation_capture_authorized",
        "output_hidden_states_authorized",
        "logit_save_authorized",
        "probe_fitting_authorized",
        "sae_analysis_authorized",
        "grader_model_authorized",
        "physiology_collection_authorized",
        "safe_robustness_run_authorized",
    ):
        if auth.get(key) is not False:
            raise SystemExit(f"{key} must remain false")
    return cfg


def _verify_frozen_hashes_local() -> dict[str, str]:
    hashes = {
        "inference_manifest_sha256": sha256_file(MANIFEST_PATH),
        "seed_manifest_sha256": sha256_file(SEED_PATH),
        "selection_manifest_sha256": sha256_file(SELECTION_PATH),
        "production_prompt_bank_sha256": sha256_file(BANK_PATH),
    }
    assert_frozen_hashes(
        inference_sha=hashes["inference_manifest_sha256"],
        seed_sha=hashes["seed_manifest_sha256"],
        selection_sha=hashes["selection_manifest_sha256"],
        bank_sha=hashes["production_prompt_bank_sha256"],
    )
    return hashes


def _generate_one(
    model,
    tok,
    *,
    prompt_ids: list[int],
    seed: int,
    do_sample: bool,
    temperature: float | None,
    max_new_tokens: int,
    device: str,
) -> dict[str, Any]:
    import torch
    from transformers import set_seed

    set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    generated: list[int] = []
    stopping_reason = "max_new_tokens"
    with torch.inference_mode():
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        out = model(ids, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1, :].float()
        for _step in range(max_new_tokens):
            if do_sample:
                assert temperature is not None and temperature > 0
                probs = torch.softmax(logits / temperature, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())
            else:
                next_id = int(torch.argmax(logits).item())
            generated.append(next_id)
            if next_id == tok.eos_token_id:
                stopping_reason = "eos"
                break
            if len(generated) >= max_new_tokens:
                break
            next_t = torch.tensor([[next_id]], dtype=torch.long, device=device)
            out = model(next_t, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[0, -1, :].float()
    return {
        "generated_token_ids": generated,
        "stopping_reason": stopping_reason,
        "n_tokens": len(generated),
    }


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=60 * 60 * 5,
    volumes={MODEL_CACHE: mistral_vol, RESULTS_MOUNT: results_vol},
    memory=65536,
    retries=modal.Retries(max_retries=3, backoff_coefficient=2.0, initial_delay=5.0),
)
def run_primary_resumable(
    *,
    manifest_jsonl: str,
    manifest_sha256: str,
    authorization_commit: str,
    protocol_version: str,
    function_call_id: str | None = None,
) -> dict[str, Any]:
    """Own the full 2904-job run with durable Volume checkpoints + resume."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from pre_output_physiology.phase26_protocol import parse_report

    if protocol_version != PROTOCOL_VERSION:
        raise RuntimeError(f"protocol_version mismatch: {protocol_version}")
    if manifest_sha256 != FROZEN_INFERENCE_MANIFEST_SHA256:
        raise RuntimeError(f"manifest sha mismatch: {manifest_sha256}")

    # Prefer Modal's own function-call id when available.
    try:
        from modal import current_function_call_id as _cfid

        function_call_id = _cfid() or function_call_id
    except Exception:  # noqa: BLE001
        pass

    jobs = [json.loads(x) for x in manifest_jsonl.splitlines() if x.strip()]
    if len(jobs) != N_PLANNED_GENERATIONS:
        raise RuntimeError(f"jobs {len(jobs)} != {N_PLANNED_GENERATIONS}")
    if any(j.get("state_id") == "SAFE" for j in jobs):
        raise RuntimeError("SAFE present in manifest")
    jobs_by_id = {j["job_id"]: j for j in jobs}
    if len(jobs_by_id) != N_PLANNED_GENERATIONS:
        raise RuntimeError("duplicate job_ids in manifest")

    run_key = run_key_from_manifest_sha(manifest_sha256)
    rdir = run_dir(Path(RESULTS_MOUNT), run_key)
    ckpt_dir = rdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    complete_path = rdir / "COMPLETE.json"
    progress_path = rdir / "progress.json"
    heartbeat_path = rdir / "heartbeat.json"

    results_vol.reload()

    if complete_path.exists():
        print(f"PHASE26C_ALREADY_COMPLETE run_key={run_key}")
        return {
            "status": "already_complete",
            "run_key": run_key,
            "completed_jobs": N_PLANNED_GENERATIONS,
        }

    # Persist run_spec + manifest snapshot once
    run_spec_path = rdir / "run_spec.json"
    if not run_spec_path.exists():
        atomic_write_json(
            run_spec_path,
            {
                "protocol_version": protocol_version,
                "run_key": run_key,
                "authorization_commit": authorization_commit,
                "manifest_sha256": manifest_sha256,
                "n_jobs": N_PLANNED_GENERATIONS,
                "checkpoint_batch_size": CHECKPOINT_BATCH_SIZE,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "tokenizer_revision": TOKENIZER_REVISION,
                "attn_implementation": ATTN_IMPLEMENTATION,
                "activation_capture": False,
                "output_hidden_states": False,
                "logit_save": False,
                "physiology": False,
            },
        )
        atomic_write_jsonl(rdir / "manifest_snapshot.jsonl", jobs)
        results_vol.commit()

    started_utc = _utc_now()
    if progress_path.exists():
        try:
            prev = json.loads(progress_path.read_text(encoding="utf-8"))
            started_utc = prev.get("started_utc") or started_utc
        except Exception:  # noqa: BLE001
            pass

    completed, ckpt_errors = load_checkpoints(
        checkpoint_dir=ckpt_dir, jobs_by_id=jobs_by_id
    )
    if ckpt_errors:
        write_progress(
            progress_path,
            protocol_version=protocol_version,
            run_key=run_key,
            authorization_commit=authorization_commit,
            function_call_id=function_call_id,
            manifest_sha256=manifest_sha256,
            started_utc=started_utc,
            last_checkpoint_utc=_utc_now(),
            completed_jobs=len(completed),
            total_jobs=N_PLANNED_GENERATIONS,
            technical_failures=0,
            last_completed_job_id=None,
            state="FAILED",
        )
        results_vol.commit()
        raise RuntimeError(
            "HARD STOP: checkpoint conflict with frozen manifest:\n"
            + "\n".join(ckpt_errors[:40])
        )

    remaining = unfinished_jobs(jobs, set(completed))
    write_progress(
        progress_path,
        protocol_version=protocol_version,
        run_key=run_key,
        authorization_commit=authorization_commit,
        function_call_id=function_call_id,
        manifest_sha256=manifest_sha256,
        started_utc=started_utc,
        last_checkpoint_utc=_utc_now(),
        completed_jobs=len(completed),
        total_jobs=N_PLANNED_GENERATIONS,
        technical_failures=sum(1 for r in completed.values() if r.get("technical_failure")),
        last_completed_job_id=None,
        state="INITIALIZING",
    )
    write_heartbeat(
        heartbeat_path, utc=_utc_now(), state="INITIALIZING", completed_jobs=len(completed)
    )
    results_vol.commit()

    if not remaining:
        # All jobs checkpointed but finalization pending
        ordered = reconstruct_ordered_rows(manifest_jobs=jobs, completed=completed)
        return _finalize(
            rdir=rdir,
            ordered=ordered,
            jobs=jobs,
            run_key=run_key,
            authorization_commit=authorization_commit,
            function_call_id=function_call_id,
            manifest_sha256=manifest_sha256,
            started_utc=started_utc,
            wall_start=time.time(),
            total_tokens=sum(int(r.get("n_generated_tokens") or 0) for r in ordered),
        )

    # Load model
    tok = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=TOKENIZER_REVISION,
        cache_dir=MODEL_CACHE,
        use_fast=True,
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        cache_dir=MODEL_CACHE,
        attn_implementation=ATTN_IMPLEMENTATION,
        low_cpu_mem_usage=True,
    )
    model.to("cuda:0")
    model.eval()
    assert not any(True for _ in model._forward_hooks)  # noqa: SLF001
    print("PHASE26C_REMOTE_MODEL_LOADED", flush=True)

    write_progress(
        progress_path,
        protocol_version=protocol_version,
        run_key=run_key,
        authorization_commit=authorization_commit,
        function_call_id=function_call_id,
        manifest_sha256=manifest_sha256,
        started_utc=started_utc,
        last_checkpoint_utc=_utc_now(),
        completed_jobs=len(completed),
        total_jobs=N_PLANNED_GENERATIONS,
        technical_failures=sum(1 for r in completed.values() if r.get("technical_failure")),
        last_completed_job_id=None,
        state="MODEL_LOADED",
    )
    write_heartbeat(
        heartbeat_path, utc=_utc_now(), state="MODEL_LOADED", completed_jobs=len(completed)
    )
    results_vol.commit()

    wall_start = time.time()
    total_tokens = sum(int(r.get("n_generated_tokens") or 0) for r in completed.values())
    pending_batch: list[dict[str, Any]] = []
    next_batch_index = len(list(ckpt_dir.glob("checkpoint_*.jsonl")))
    first_ckpt_logged = next_batch_index > 0
    last_completed_job_id = None

    write_progress(
        progress_path,
        protocol_version=protocol_version,
        run_key=run_key,
        authorization_commit=authorization_commit,
        function_call_id=function_call_id,
        manifest_sha256=manifest_sha256,
        started_utc=started_utc,
        last_checkpoint_utc=_utc_now(),
        completed_jobs=len(completed),
        total_jobs=N_PLANNED_GENERATIONS,
        technical_failures=sum(1 for r in completed.values() if r.get("technical_failure")),
        last_completed_job_id=last_completed_job_id,
        state="RUNNING",
    )
    results_vol.commit()

    def flush_batch() -> None:
        nonlocal pending_batch, next_batch_index, first_ckpt_logged, last_completed_job_id
        if not pending_batch:
            return
        path = write_checkpoint_batch(
            checkpoint_dir=ckpt_dir,
            batch_index=next_batch_index,
            rows=pending_batch,
        )
        for row in pending_batch:
            completed[row["job_id"]] = row
            last_completed_job_id = row["job_id"]
        next_batch_index += 1
        pending_batch = []
        n_tech = sum(1 for r in completed.values() if r.get("technical_failure"))
        write_progress(
            progress_path,
            protocol_version=protocol_version,
            run_key=run_key,
            authorization_commit=authorization_commit,
            function_call_id=function_call_id,
            manifest_sha256=manifest_sha256,
            started_utc=started_utc,
            last_checkpoint_utc=_utc_now(),
            completed_jobs=len(completed),
            total_jobs=N_PLANNED_GENERATIONS,
            technical_failures=n_tech,
            last_completed_job_id=last_completed_job_id,
            state="RUNNING",
        )
        write_heartbeat(
            heartbeat_path,
            utc=_utc_now(),
            state="RUNNING",
            completed_jobs=len(completed),
        )
        results_vol.commit()
        if not first_ckpt_logged:
            print(f"PHASE26C_FIRST_CHECKPOINT_COMMITTED path={path.name}", flush=True)
            first_ckpt_logged = True

    for job in remaining:
        if job["job_id"] in completed:
            continue
        if job.get("state_id") != "ADVERSE":
            raise RuntimeError(f"SAFE leak in job {job.get('job_id')}")

        messages = [{"role": "user", "content": job["prompt_text"]}]
        formatted = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tok.encode(formatted, add_special_tokens=False)
        seed = int(job["seed"])
        do_sample = bool(job["generation"]["do_sample"])
        temperature = job["generation"]["temperature"]
        max_new = int(job["generation"]["max_new_tokens"])
        last_err = None
        gen = None
        n_attempts = 0
        for _attempt in range(3):
            n_attempts += 1
            try:
                gen = _generate_one(
                    model,
                    tok,
                    prompt_ids=prompt_ids,
                    seed=seed,
                    do_sample=do_sample,
                    temperature=temperature,
                    max_new_tokens=max_new,
                    device="cuda:0",
                )
                last_err = None
                break
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)

        if gen is None:
            row = {
                **{k: job[k] for k in job if k != "prompt_text"},
                "raw_response_text": None,
                "parsed_decision": None,
                "parse_valid": False,
                "malformed_reason": "technical_failure",
                "technical_failure": True,
                "technical_error": last_err,
                "n_attempts": n_attempts,
                "stopping_reason": None,
                "n_generated_tokens": 0,
                "formatted_prompt_sha256": hashlib.sha256(formatted.encode()).hexdigest(),
                "job_id": job["job_id"],
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "tokenizer_revision": TOKENIZER_REVISION,
                "generation_parameters": job["generation"],
                "activation_capture": False,
                "output_hidden_states": False,
                "logit_save": False,
            }
        else:
            raw = tok.decode(gen["generated_token_ids"], skip_special_tokens=True)
            parsed = parse_report(raw)
            total_tokens += gen["n_tokens"]
            row = {
                "protocol_version": job["protocol_version"],
                "parent_phase26b_commit": job["parent_phase26b_commit"],
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "tokenizer_revision": TOKENIZER_REVISION,
                "scenario_id": job["scenario_id"],
                "state_id": job["state_id"],
                "instance_id": job["instance_id"],
                "condition": job["condition"],
                "consequence_class": job.get("consequence_class"),
                "prompt_id": job["prompt_id"],
                "prompt_sha256": job["prompt_sha256"],
                "n_template_id": job.get("n_template_id"),
                "x_template_id": job.get("x_template_id"),
                "gold_decision": job["gold_decision"],
                "rollout_type": job["rollout_type"],
                "rollout_index": job["rollout_index"],
                "seed": seed,
                "generation_parameters": job["generation"],
                "raw_response_text": raw,
                "parsed_decision": parsed["parsed_decision"],
                "parse_valid": parsed["parse_valid"],
                "malformed_reason": parsed.get("malformed_reason"),
                "technical_failure": False,
                "technical_error": None,
                "n_attempts": n_attempts,
                "stopping_reason": gen["stopping_reason"],
                "n_generated_tokens": gen["n_tokens"],
                "formatted_prompt_sha256": hashlib.sha256(formatted.encode()).hexdigest(),
                "job_id": job["job_id"],
                "activation_capture": False,
                "output_hidden_states": False,
                "logit_save": False,
            }

        pending_batch.append(row)
        if len(pending_batch) >= CHECKPOINT_BATCH_SIZE:
            flush_batch()

    flush_batch()

    ordered = reconstruct_ordered_rows(manifest_jobs=jobs, completed=completed)
    return _finalize(
        rdir=rdir,
        ordered=ordered,
        jobs=jobs,
        run_key=run_key,
        authorization_commit=authorization_commit,
        function_call_id=function_call_id,
        manifest_sha256=manifest_sha256,
        started_utc=started_utc,
        wall_start=wall_start,
        total_tokens=total_tokens,
    )


def _finalize(
    *,
    rdir: Path,
    ordered: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    run_key: str,
    authorization_commit: str,
    function_call_id: str | None,
    manifest_sha256: str,
    started_utc: str,
    wall_start: float,
    total_tokens: int,
) -> dict[str, Any]:
    wall = time.time() - wall_start
    raw_path = rdir / "raw_generations.jsonl"
    atomic_write_jsonl(raw_path, ordered)
    raw_sha = sha256_file(raw_path)
    n_tech = sum(1 for r in ordered if r.get("technical_failure"))
    meta = {
        "protocol_version": PROTOCOL_VERSION,
        "run_key": run_key,
        "authorization_commit": authorization_commit,
        "function_call_id": function_call_id,
        "manifest_sha256": manifest_sha256,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "attn_implementation": ATTN_IMPLEMENTATION,
        "n_generations": len(ordered),
        "n_technical_failures": n_tech,
        "total_generated_tokens": total_tokens,
        "wall_seconds": wall,
        "gpu_hours": wall / 3600.0,
        "estimated_cost_usd": (wall / 3600.0) * A100_USD_PER_HOUR,
        "cost_rate_usd_per_hour": A100_USD_PER_HOUR,
        "gpu": GPU_TYPE,
        "raw_generations_sha256": raw_sha,
        "started_utc": started_utc,
        "completed_utc": _utc_now(),
        "activation_capture": False,
        "output_hidden_states": False,
        "logit_save": False,
        "grader_model": False,
        "physiology_collection": False,
        "safe_included": False,
        "checkpoint_batch_size": CHECKPOINT_BATCH_SIZE,
        "detached_durable_execution": True,
    }
    atomic_write_json(rdir / "run_metadata.json", meta)
    write_progress(
        rdir / "progress.json",
        protocol_version=PROTOCOL_VERSION,
        run_key=run_key,
        authorization_commit=authorization_commit,
        function_call_id=function_call_id,
        manifest_sha256=manifest_sha256,
        started_utc=started_utc,
        last_checkpoint_utc=_utc_now(),
        completed_jobs=N_PLANNED_GENERATIONS,
        total_jobs=N_PLANNED_GENERATIONS,
        technical_failures=n_tech,
        last_completed_job_id=ordered[-1]["job_id"],
        state="COMPLETE",
    )
    write_heartbeat(
        rdir / "heartbeat.json",
        utc=_utc_now(),
        state="COMPLETE",
        completed_jobs=N_PLANNED_GENERATIONS,
    )
    atomic_write_json(
        rdir / "COMPLETE.json",
        {
            "status": "COMPLETE",
            "n_generations": N_PLANNED_GENERATIONS,
            "raw_generations_sha256": raw_sha,
            "manifest_sha256": manifest_sha256,
            "validate_raw_against_manifest": True,
            "completed_utc": _utc_now(),
        },
    )
    results_vol.commit()
    print("PHASE26C_ALL_2904_GENERATIONS_COMPLETE", flush=True)
    return {
        "status": "complete",
        "run_key": run_key,
        "completed_jobs": N_PLANNED_GENERATIONS,
        "raw_generations_sha256": raw_sha,
        "n_technical_failures": n_tech,
    }


@app.function(
    image=image,
    volumes={RESULTS_MOUNT: results_vol},
    timeout=120,
)
def read_run_status(run_key: str) -> dict[str, Any]:
    results_vol.reload()
    rdir = run_dir(Path(RESULTS_MOUNT), run_key)
    out: dict[str, Any] = {"run_key": run_key, "exists": rdir.exists()}
    for name in ("progress.json", "heartbeat.json", "COMPLETE.json"):
        p = rdir / name
        if p.exists():
            out[name] = json.loads(p.read_text(encoding="utf-8"))
        else:
            out[name] = None
    ckpt = rdir / "checkpoints"
    out["n_checkpoint_files"] = (
        len(list(ckpt.glob("checkpoint_*.jsonl"))) if ckpt.exists() else 0
    )
    return out


@app.local_entrypoint()
def main() -> None:
    """Validate, spawn detached remote run, brief health check, write receipt, exit."""
    from pre_output_physiology.phase26c_primary import rollout_seed

    cfg = _load_auth()
    hashes = _verify_frozen_hashes_local()

    # Hard scientific gates
    jobs = [
        json.loads(x) for x in MANIFEST_PATH.read_text(encoding="utf-8").splitlines() if x.strip()
    ]
    sel = [
        json.loads(x)
        for x in SELECTION_PATH.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    if len(jobs) != 2904 or len(sel) != 312:
        raise SystemExit("count gate failed")
    if any(j["state_id"] != "ADVERSE" for j in jobs) or any(
        p["state_id"] != "ADVERSE" for p in sel
    ):
        raise SystemExit("SAFE gate failed")
    if rollout_seed(prompt_sha256="abc", condition="X_C1", rollout_index=0) != 1410709147:
        raise SystemExit("seed vector gate failed")
    if cfg["model"]["revision"] != MODEL_REVISION:
        raise SystemExit("model revision gate failed")

    auth_commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    branch = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "branch", "--show-current"], text=True
    ).strip()
    run_key = run_key_from_manifest_sha(hashes["inference_manifest_sha256"])

    # Duplicate-launch protection via Volume status
    status = read_run_status.remote(run_key)
    if status.get("COMPLETE.json"):
        raise SystemExit(
            f"STOP: run_key {run_key} already COMPLETE; refusing relaunch"
        )
    progress = status.get("progress.json") or {}
    if progress.get("state") in {"RUNNING", "MODEL_LOADED", "INITIALIZING"}:
        # Prefer resume via spawn of same run_key (remote will skip completed).
        # But if a live writer may exist, refuse second concurrent writer.
        existing_fc = progress.get("function_call_id")
        if existing_fc and progress.get("state") == "RUNNING":
            raise SystemExit(
                f"STOP: run appears active (fc={existing_fc}); refusing second writer"
            )

    manifest_text = MANIFEST_PATH.read_text(encoding="utf-8")
    call = run_primary_resumable.spawn(
        manifest_jsonl=manifest_text,
        manifest_sha256=hashes["inference_manifest_sha256"],
        authorization_commit=auth_commit,
        protocol_version=PROTOCOL_VERSION,
        function_call_id=None,  # filled after we know object_id
    )
    fc_id = getattr(call, "object_id", None) or str(call)
    # Re-spawn is not needed; pass fc via a tiny progress update after spawn by
    # reading status. Remote may have started with function_call_id=None; we
    # record fc_id in the local receipt and optionally update via status poll.

    # Brief health verification (bounded)
    model_loaded = False
    first_ckpt = False
    completed = 0
    remote_state = "SPAWNED"
    deadline = time.time() + 20 * 60  # up to ~20 min for model load + first ckpt
    while time.time() < deadline:
        time.sleep(20)
        try:
            st = read_run_status.remote(run_key)
        except Exception as exc:  # noqa: BLE001
            print(f"status poll error: {exc}", flush=True)
            continue
        prog = st.get("progress.json") or {}
        remote_state = prog.get("state") or remote_state
        completed = int(prog.get("completed_jobs") or 0)
        if remote_state in {"MODEL_LOADED", "RUNNING", "COMPLETE"}:
            model_loaded = True
        if st.get("n_checkpoint_files", 0) >= 1 or completed >= CHECKPOINT_BATCH_SIZE:
            first_ckpt = True
        print(
            f"health: state={remote_state} completed={completed}/2904 "
            f"ckpts={st.get('n_checkpoint_files')} fc={fc_id}",
            flush=True,
        )
        if model_loaded and first_ckpt:
            break
        if remote_state == "FAILED":
            raise SystemExit("STOP: remote reported FAILED during startup")
        if remote_state == "COMPLETE":
            break

    receipt = {
        "branch": branch,
        "authorization_commit_sha": auth_commit,
        "engineering_durability_commit_sha": auth_commit,  # same HEAD at launch if auth is tip
        "protocol_version": PROTOCOL_VERSION,
        "run_key": run_key,
        "modal_app_name": APP_NAME,
        "modal_function_call_id": fc_id,
        "modal_results_volume": RESULTS_VOLUME_NAME,
        "inference_manifest_sha256": hashes["inference_manifest_sha256"],
        "seed_manifest_sha256": hashes["seed_manifest_sha256"],
        "selection_manifest_sha256": hashes["selection_manifest_sha256"],
        "production_prompt_bank_sha256": hashes["production_prompt_bank_sha256"],
        "frozen_hashes_verified": {
            "inference": FROZEN_INFERENCE_MANIFEST_SHA256,
            "seed": FROZEN_SEED_MANIFEST_SHA256,
            "selection": FROZEN_SELECTION_MANIFEST_SHA256,
        },
        "launch_utc": _utc_now(),
        "startup_verification_result": {
            "model_loaded_confirmed": model_loaded,
            "first_checkpoint_confirmed": first_ckpt,
            "remote_state": remote_state,
            "completed_jobs_at_stop": completed,
            "total_jobs": 2904,
        },
        "first_checkpoint_confirmed": first_ckpt,
        "checkpoint_batch_size": CHECKPOINT_BATCH_SIZE,
        "modal_retry_policy": {
            "max_retries": 3,
            "backoff_coefficient": 2.0,
            "initial_delay_seconds": 5.0,
        },
        "detached_execution": True,
        "laptop_editor_disconnection_does_not_own_experiment_lifetime": True,
        "status": "phase26c_primary_generation_running",
        "do_not_summarize_until_complete": True,
    }
    LAUNCH_RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    LAUNCH_RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    print(
        "THE PHASE 26C PRIMARY RUN IS EXECUTING REMOTELY ON MODAL. "
        "THE LOCAL CODE-EDITOR/LAPTOP PROCESS IS NOT REQUIRED TO REMAIN CONNECTED.",
        flush=True,
    )
