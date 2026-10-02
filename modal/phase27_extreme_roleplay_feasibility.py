"""Modal GPU runner for Phase 27 extreme roleplay (DETACHED / resumable).

PREFLIGHT: this module MUST NOT be executed for generation until:
  status == phase27_extreme_roleplay_generation_authorized
  AND modal_gpu_behavior_authorized == true

This file prepares the durable architecture only.
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

from pre_output_physiology.phase27_extreme_roleplay import (
    ATTN_IMPLEMENTATION,
    CHECKPOINT_BATCH_SIZE,
    MODEL_ID,
    MODEL_REVISION,
    N_PLANNED_GENERATIONS,
    PROTOCOL_VERSION,
    TOKENIZER_REVISION,
    load_checkpoint_job_ids,
    run_key_from_manifest_sha,
    unfinished_jobs,
)

APP_NAME = "pre-output-physiology-phase27-extreme-roleplay"
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/phase27_extreme_roleplay_feasibility.yaml"
MANIFEST_PATH = REPO_ROOT / "artifacts/phase27/inference_manifest.jsonl"
RESULTS_VOLUME_NAME = "phase27-extreme-roleplay-results"
RESULTS_MOUNT = "/phase27_results"
MODEL_CACHE = "/vol/hf_cache"
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


def _load_auth() -> dict[str, Any]:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if cfg.get("status") != "phase27_extreme_roleplay_generation_authorized":
        raise SystemExit(
            f"GPU blocked: status={cfg.get('status')!r}; "
            "requires phase27_extreme_roleplay_generation_authorized"
        )
    if cfg["authorizations"].get("modal_gpu_behavior_authorized") is not True:
        raise SystemExit("modal_gpu_behavior_authorized must be true to launch")
    for key in (
        "activation_capture_authorized",
        "output_hidden_states_authorized",
        "logit_save_authorized",
        "probe_fitting_authorized",
        "sae_analysis_authorized",
        "causal_intervention_authorized",
        "physiology_collection_authorized",
    ):
        if cfg["authorizations"].get(key) is not False:
            raise SystemExit(f"{key} must remain false")
    return cfg


def _generate_one(model, tok, *, prompt_ids, seed, temperature, max_new_tokens, device):
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
        for _ in range(max_new_tokens):
            probs = torch.softmax(logits / temperature, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
            generated.append(next_id)
            if next_id == tok.eos_token_id:
                stopping_reason = "eos"
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
    timeout=60 * 60 * 3,
    volumes={MODEL_CACHE: mistral_vol, RESULTS_MOUNT: results_vol},
    memory=65536,
    retries=modal.Retries(max_retries=3, backoff_coefficient=2.0, initial_delay=5.0),
)
def run_phase27_resumable(
    *,
    manifest_jsonl: str,
    manifest_sha256: str,
    authorization_commit: str,
    protocol_version: str,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if protocol_version != PROTOCOL_VERSION:
        raise RuntimeError("protocol mismatch")
    jobs = [json.loads(x) for x in manifest_jsonl.splitlines() if x.strip()]
    if len(jobs) != N_PLANNED_GENERATIONS:
        raise RuntimeError(f"jobs {len(jobs)} != {N_PLANNED_GENERATIONS}")
    if hashlib.sha256(manifest_jsonl.encode()).hexdigest() != manifest_sha256:
        raise RuntimeError("manifest sha mismatch vs payload")

    run_key = run_key_from_manifest_sha(manifest_sha256)
    rdir = Path(RESULTS_MOUNT) / run_key
    ckpt_dir = rdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_vol.reload()
    if (rdir / "COMPLETE.json").exists():
        return {"status": "already_complete", "run_key": run_key}

    jobs_by_id = {j["job_id"]: j for j in jobs}
    completed_ids, errors = load_checkpoint_job_ids(ckpt_dir)
    if errors:
        raise RuntimeError("checkpoint conflict: " + "; ".join(errors[:20]))
    for jid in completed_ids:
        if jid not in jobs_by_id:
            raise RuntimeError(f"unknown checkpoint job {jid}")

    remaining = unfinished_jobs(jobs, completed_ids)
    tok = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=TOKENIZER_REVISION, cache_dir=MODEL_CACHE, use_fast=True
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
    print("PHASE27_REMOTE_MODEL_LOADED", flush=True)

    pending: list[dict[str, Any]] = []
    batch_index = len(list(ckpt_dir.glob("checkpoint_*.jsonl")))
    completed_rows: dict[str, dict[str, Any]] = {}

    def flush() -> None:
        nonlocal pending, batch_index
        if not pending:
            return
        path = ckpt_dir / f"checkpoint_{batch_index:05d}.jsonl"
        if path.exists():
            raise RuntimeError(f"refusing overwrite {path}")
        tmp = path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for row in pending:
                f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        tmp.replace(path)
        for row in pending:
            completed_rows[row["job_id"]] = row
            completed_ids.add(row["job_id"])
        pending = []
        batch_index += 1
        (rdir / "progress.json").write_text(
            json.dumps(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "run_key": run_key,
                    "authorization_commit": authorization_commit,
                    "manifest_sha256": manifest_sha256,
                    "completed_jobs": len(completed_ids),
                    "total_jobs": N_PLANNED_GENERATIONS,
                    "state": "RUNNING",
                    "utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        results_vol.commit()
        if batch_index == 1:
            print("PHASE27_FIRST_CHECKPOINT_COMMITTED", flush=True)

    for job in remaining:
        if job["job_id"] in completed_ids:
            continue
        if job["generation"]["model_revision"] != MODEL_REVISION:
            raise RuntimeError("model revision drift")
        messages = [
            {"role": "system", "content": job["system_content"]},
            {"role": "user", "content": job["user_question"]},
        ]
        formatted = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Append assistant character prefix so generation continues as that character.
        formatted = formatted + job["assistant_prefix"]
        prompt_ids = tok.encode(formatted, add_special_tokens=False)
        seed = int(job["seed"])
        gen = None
        last_err = None
        for _attempt in range(3):
            try:
                gen = _generate_one(
                    model,
                    tok,
                    prompt_ids=prompt_ids,
                    seed=seed,
                    temperature=float(job["generation"]["temperature"]),
                    max_new_tokens=int(job["generation"]["max_new_tokens"]),
                    device="cuda:0",
                )
                last_err = None
                break
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
        if gen is None:
            row = {
                **{k: job[k] for k in job if k not in {"system_content", "rendered_prompt_text"}},
                "raw_response_text": None,
                "technical_failure": True,
                "technical_error": last_err,
                "formatted_prompt_sha256": hashlib.sha256(formatted.encode()).hexdigest(),
            }
        else:
            raw = tok.decode(gen["generated_token_ids"], skip_special_tokens=True)
            row = {
                "job_id": job["job_id"],
                "protocol_version": PROTOCOL_VERSION,
                "scenario_id": job["scenario_id"],
                "consequence_class": job["consequence_class"],
                "ground_truth": job["ground_truth"],
                "advantageous_false": job["advantageous_false"],
                "prompt_sha256": job["prompt_sha256"],
                "system_scenario_sha256": job["system_scenario_sha256"],
                "user_question_sha256": job["user_question_sha256"],
                "assistant_prefix_sha256": job["assistant_prefix_sha256"],
                "rollout_index": job["rollout_index"],
                "seed": seed,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "tokenizer_revision": TOKENIZER_REVISION,
                "generation_parameters": job["generation"],
                "raw_response_text": raw,
                "technical_failure": False,
                "n_generated_tokens": gen["n_tokens"],
                "stopping_reason": gen["stopping_reason"],
                "formatted_prompt_sha256": hashlib.sha256(formatted.encode()).hexdigest(),
                "activation_capture": False,
                "output_hidden_states": False,
                "logit_save": False,
            }
        pending.append(row)
        if len(pending) >= CHECKPOINT_BATCH_SIZE:
            flush()
    flush()

    # Reconstruct final
    # Reload all checkpoints for canonical file
    all_rows: dict[str, dict[str, Any]] = {}
    for path in sorted(ckpt_dir.glob("checkpoint_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                all_rows[r["job_id"]] = r
    if len(all_rows) != N_PLANNED_GENERATIONS:
        raise RuntimeError(f"incomplete {len(all_rows)}")
    ordered = [all_rows[j["job_id"]] for j in jobs]
    raw_path = rdir / "raw_generations.jsonl"
    with raw_path.open("w", encoding="utf-8") as f:
        for r in ordered:
            f.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")
    raw_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    (rdir / "COMPLETE.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "n_generations": N_PLANNED_GENERATIONS,
                "raw_generations_sha256": raw_sha,
                "manifest_sha256": manifest_sha256,
                "completed_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    results_vol.commit()
    print("PHASE27_ALL_288_GENERATIONS_COMPLETE", flush=True)
    return {"status": "complete", "run_key": run_key, "raw_sha": raw_sha}


@app.local_entrypoint()
def main() -> None:
    """Refuses launch while status is preflight_ready."""
    _load_auth()
    raise SystemExit(
        "Phase 27 generation authorized path is prepared, but this entrypoint "
        "should only be used after independent audit + authorization commit."
    )
