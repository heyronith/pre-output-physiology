"""Modal GPU runner for Phase 27 extreme roleplay (DETACHED / resumable).

PREFLIGHT: this module MUST NOT execute generation until:
  status == phase27_extreme_roleplay_generation_authorized
  AND modal_gpu_behavior_authorized == true

Uses frozen model_input_text from the inference manifest (no independent re-render).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
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
    assert_generation_authorized,
    build_launch_receipt,
    decide_launch_action,
    load_checkpoints,
    run_key_from_manifest_sha,
    unfinished_jobs,
    validate_raw_against_manifest,
    verify_frozen_model_input,
)

APP_NAME = "pre-output-physiology-phase27-extreme-roleplay"
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/phase27_extreme_roleplay_feasibility.yaml"
MANIFEST_PATH = REPO_ROOT / "artifacts/phase27/inference_manifest.jsonl"
LAUNCH_RECEIPT = REPO_ROOT / "artifacts/phase27/launch_receipt.json"
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


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_auth() -> dict[str, Any]:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert_generation_authorized(cfg)
    return cfg


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_local_manifest() -> tuple[str, list[dict[str, Any]], str]:
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"missing manifest {MANIFEST_PATH}")
    text = MANIFEST_PATH.read_text(encoding="utf-8")
    jobs = [json.loads(x) for x in text.splitlines() if x.strip()]
    if len(jobs) != N_PLANNED_GENERATIONS:
        raise SystemExit(f"jobs {len(jobs)} != {N_PLANNED_GENERATIONS}")
    manifest_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return text, jobs, manifest_sha


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
    volumes={RESULTS_MOUNT: results_vol},
    timeout=120,
)
def read_run_status(run_key: str) -> dict[str, Any]:
    results_vol.reload()
    rdir = Path(RESULTS_MOUNT) / run_key
    out: dict[str, Any] = {"run_key": run_key, "exists": rdir.exists()}
    for name in ("progress.json", "ACTIVE_RUN.json", "COMPLETE.json"):
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
    function_call_id: str | None = None,
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
    jobs_by_id = {j["job_id"]: j for j in jobs}
    if len(jobs_by_id) != N_PLANNED_GENERATIONS:
        raise RuntimeError("duplicate job_ids in manifest")

    try:
        from modal import current_function_call_id as _cfid

        function_call_id = _cfid() or function_call_id
    except Exception:  # noqa: BLE001
        pass

    run_key = run_key_from_manifest_sha(manifest_sha256)
    rdir = Path(RESULTS_MOUNT) / run_key
    ckpt_dir = rdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_vol.reload()

    if (rdir / "COMPLETE.json").exists():
        return {"status": "already_complete", "run_key": run_key}

    active_path = rdir / "ACTIVE_RUN.json"
    if active_path.exists():
        try:
            prev_active = json.loads(active_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            prev_active = {}
        prev_fc = prev_active.get("function_call_id")
        prev_state = prev_active.get("state")
        if (
            prev_fc
            and prev_fc != function_call_id
            and prev_state in {"RUNNING", "MODEL_LOADED", "INITIALIZING"}
        ):
            raise RuntimeError(
                f"refusing second active writer: existing fc={prev_fc} state={prev_state}"
            )

    completed, ckpt_errors = load_checkpoints(
        checkpoint_dir=ckpt_dir, jobs_by_id=jobs_by_id
    )
    if ckpt_errors:
        raise RuntimeError(
            "HARD STOP: checkpoint conflict with frozen manifest:\n"
            + "\n".join(ckpt_errors[:40])
        )

    (rdir / "ACTIVE_RUN.json").write_text(
        json.dumps(
            {
                "run_key": run_key,
                "function_call_id": function_call_id,
                "state": "INITIALIZING",
                "manifest_sha256": manifest_sha256,
                "authorization_commit": authorization_commit,
                "utc": _utc_now(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    results_vol.commit()

    remaining = unfinished_jobs(jobs, set(completed))
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

    (rdir / "ACTIVE_RUN.json").write_text(
        json.dumps(
            {
                "run_key": run_key,
                "function_call_id": function_call_id,
                "state": "RUNNING",
                "manifest_sha256": manifest_sha256,
                "authorization_commit": authorization_commit,
                "utc": _utc_now(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (rdir / "progress.json").write_text(
        json.dumps(
            {
                "protocol_version": PROTOCOL_VERSION,
                "run_key": run_key,
                "authorization_commit": authorization_commit,
                "function_call_id": function_call_id,
                "manifest_sha256": manifest_sha256,
                "completed_jobs": len(completed),
                "total_jobs": N_PLANNED_GENERATIONS,
                "state": "RUNNING",
                "utc": _utc_now(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    results_vol.commit()

    pending: list[dict[str, Any]] = []
    batch_index = len(list(ckpt_dir.glob("checkpoint_*.jsonl")))

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
            completed[row["job_id"]] = row
        pending = []
        batch_index += 1
        (rdir / "progress.json").write_text(
            json.dumps(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "run_key": run_key,
                    "authorization_commit": authorization_commit,
                    "function_call_id": function_call_id,
                    "manifest_sha256": manifest_sha256,
                    "completed_jobs": len(completed),
                    "total_jobs": N_PLANNED_GENERATIONS,
                    "state": "RUNNING",
                    "utc": _utc_now(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        results_vol.commit()
        if batch_index == 1:
            print("PHASE27_FIRST_CHECKPOINT_COMMITTED", flush=True)

    for job in remaining:
        if job["job_id"] in completed:
            continue
        if job["generation"]["model_revision"] != MODEL_REVISION:
            raise RuntimeError("model revision drift")
        if job["generation"]["tokenizer_revision"] != TOKENIZER_REVISION:
            raise RuntimeError("tokenizer revision drift")
        frozen_text = job["model_input_text"]
        prompt_ids = verify_frozen_model_input(
            tok,
            model_input_text=frozen_text,
            model_input_sha256=job["model_input_sha256"],
            input_ids_sha256=job["input_ids_sha256"],
        )
        # HARD FAIL if independent reconstruction would differ (must not be used).
        if sha256_text_local(frozen_text) != job["prompt_sha256"]:
            raise RuntimeError("prompt_sha256 != model_input_sha256 on frozen job")
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
                "job_id": job["job_id"],
                "protocol_version": PROTOCOL_VERSION,
                "scenario_id": job["scenario_id"],
                "consequence_class": job["consequence_class"],
                "ground_truth": job["ground_truth"],
                "advantageous_false": job["advantageous_false"],
                "prompt_sha256": job["prompt_sha256"],
                "model_input_sha256": job["model_input_sha256"],
                "input_ids_sha256": job["input_ids_sha256"],
                "system_scenario_sha256": job["system_scenario_sha256"],
                "user_question_sha256": job["user_question_sha256"],
                "assistant_prefix_sha256": job["assistant_prefix_sha256"],
                "rollout_index": job["rollout_index"],
                "seed": seed,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "tokenizer_revision": TOKENIZER_REVISION,
                "generation_parameters": job["generation"],
                "raw_response_text": None,
                "technical_failure": True,
                "technical_error": last_err,
                "activation_capture": False,
                "output_hidden_states": False,
                "logit_save": False,
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
                "model_input_sha256": job["model_input_sha256"],
                "input_ids_sha256": job["input_ids_sha256"],
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
                "activation_capture": False,
                "output_hidden_states": False,
                "logit_save": False,
            }
        pending.append(row)
        if len(pending) >= CHECKPOINT_BATCH_SIZE:
            flush()
    flush()

    all_rows: dict[str, dict[str, Any]] = {}
    for path in sorted(ckpt_dir.glob("checkpoint_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                all_rows[r["job_id"]] = r
    if len(all_rows) != N_PLANNED_GENERATIONS:
        raise RuntimeError(f"incomplete {len(all_rows)}")
    ordered = [all_rows[j["job_id"]] for j in jobs]
    integrity_errors = validate_raw_against_manifest(ordered, jobs)
    if integrity_errors:
        raise RuntimeError(
            "HARD STOP: final raw↔manifest integrity failed:\n"
            + "\n".join(integrity_errors[:40])
        )
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
                "validate_raw_against_manifest": True,
                "completed_utc": _utc_now(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (rdir / "ACTIVE_RUN.json").write_text(
        json.dumps(
            {
                "run_key": run_key,
                "function_call_id": function_call_id,
                "state": "COMPLETE",
                "manifest_sha256": manifest_sha256,
                "utc": _utc_now(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    results_vol.commit()
    print("PHASE27_ALL_288_GENERATIONS_COMPLETE", flush=True)
    return {"status": "complete", "run_key": run_key, "raw_sha": raw_sha}


def sha256_text_local(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@app.local_entrypoint()
def main() -> None:
    """Authorize → validate → spawn detached → write receipt → exit (no wait)."""
    cfg = _load_auth()
    manifest_text, jobs, manifest_sha = _verify_local_manifest()
    if cfg["model"]["revision"] != MODEL_REVISION:
        raise SystemExit("model revision gate failed")
    if _sha256_file(MANIFEST_PATH) != manifest_sha:
        raise SystemExit("manifest sha re-read mismatch")

    run_key = run_key_from_manifest_sha(manifest_sha)
    status = read_run_status.remote(run_key)
    decision = decide_launch_action(
        status=cfg.get("status"),
        modal_gpu_behavior_authorized=bool(
            cfg["authorizations"].get("modal_gpu_behavior_authorized")
        ),
        n_jobs=len(jobs),
        complete_exists=bool(status.get("COMPLETE.json")),
        progress=status.get("progress.json"),
        active_run=status.get("ACTIVE_RUN.json"),
    )
    if decision["action"] != "spawn":
        raise SystemExit(decision["reason"])

    launch_head = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    branch = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "branch", "--show-current"], text=True
    ).strip()
    auth_commit = launch_head

    call = run_phase27_resumable.spawn(
        manifest_jsonl=manifest_text,
        manifest_sha256=manifest_sha,
        authorization_commit=auth_commit,
        protocol_version=PROTOCOL_VERSION,
        function_call_id=None,
    )
    fc_id = getattr(call, "object_id", None) or str(call)
    receipt = build_launch_receipt(
        run_key=run_key,
        function_call_id=fc_id,
        manifest_sha256=manifest_sha,
        authorization_commit=auth_commit,
        launch_head=launch_head,
        branch=branch,
        resume=bool(decision.get("resume")),
    )
    LAUNCH_RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    LAUNCH_RECEIPT.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"PHASE27_SPAWNED function_call_id={fc_id}", flush=True)
    print(f"PHASE27_RUN_KEY={run_key}", flush=True)
    print(f"PHASE27_LAUNCH_RECEIPT={LAUNCH_RECEIPT}", flush=True)
    print("PHASE27_DETACHED_EXIT (not waiting for completion)", flush=True)
