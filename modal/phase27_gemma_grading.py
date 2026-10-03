"""Modal runner for Phase 27 Gemma response-level grading (DETACHED / resumable).

PREFLIGHT: refuses GPU grading until:
  grading_status == phase27_gemma_grading_authorized
  AND grader_execution_authorized == true

Uses frozen grader_model_input_text from the grading manifest.
No Mistral loading. No physiology. No onset localization.
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

from pre_output_physiology.phase23_open_grader import parse_json_grade
from pre_output_physiology.phase27_gemma_grading import (
    CHECKPOINT_BATCH_SIZE,
    GRADING_PROTOCOL_VERSION,
    GRADING_STATUS_AUTHORIZED,
    GRADER_MAX_MODEL_LEN,
    GRADER_MAX_NEW_TOKENS,
    GRADER_MODEL,
    GRADER_REVISION,
    N_PLANNED_GENERATIONS,
    RESULTS_MOUNT,
    RESULTS_VOLUME_NAME,
    assert_grading_authorized,
    decide_grading_launch_action,
    load_grading_checkpoints,
    primary_label_from_parsed,
    run_key_from_grading_manifest_sha,
    verify_frozen_grader_model_input,
)

APP_NAME = "pre-output-physiology-phase27-gemma-grading"
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/phase27_extreme_roleplay_feasibility.yaml"
MANIFEST_PATH = REPO_ROOT / "artifacts/phase27/grading_manifest.jsonl"
LAUNCH_RECEIPT = REPO_ROOT / "artifacts/phase27/grading_launch_receipt.json"
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
hf_vol = modal.Volume.from_name("preoutput-mistral-cache", create_if_missing=True)
results_vol = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_auth() -> dict[str, Any]:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert_grading_authorized(cfg)
    return cfg


def _verify_manifest() -> tuple[str, list[dict[str, Any]], str]:
    text = MANIFEST_PATH.read_text(encoding="utf-8")
    jobs = [json.loads(x) for x in text.splitlines() if x.strip()]
    if len(jobs) != N_PLANNED_GENERATIONS:
        raise SystemExit(f"jobs {len(jobs)} != {N_PLANNED_GENERATIONS}")
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return text, jobs, sha


def _deployed_functions() -> tuple[Any, Any]:
    try:
        status_fn = modal.Function.from_name(APP_NAME, "read_grading_status")
        run_fn = modal.Function.from_name(APP_NAME, "run_phase27_gemma_grading_resumable")
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "Deployed Phase 27 grading app not found. First run:\n"
            "  uv run modal deploy modal/phase27_gemma_grading.py\n"
            f"Lookup error: {type(exc).__name__}: {exc}"
        ) from exc
    return status_fn, run_fn


@app.function(image=image, volumes={RESULTS_MOUNT: results_vol}, timeout=120)
def read_grading_status(run_key: str) -> dict[str, Any]:
    results_vol.reload()
    rdir = Path(RESULTS_MOUNT) / run_key
    out: dict[str, Any] = {"run_key": run_key, "exists": rdir.exists()}
    for name in ("progress.json", "ACTIVE_RUN.json", "COMPLETE.json"):
        p = rdir / name
        out[name] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    ckpt = rdir / "checkpoints"
    out["n_checkpoint_files"] = (
        len(list(ckpt.glob("checkpoint_*.jsonl"))) if ckpt.exists() else 0
    )
    return out


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=60 * 60 * 4,
    volumes={MODEL_CACHE: hf_vol, RESULTS_MOUNT: results_vol},
    memory=65536,
    retries=modal.Retries(max_retries=3, backoff_coefficient=2.0, initial_delay=5.0),
)
def run_phase27_gemma_grading_resumable(
    *,
    manifest_jsonl: str,
    manifest_sha256: str,
    authorization_commit: str,
    grading_protocol_version: str,
    function_call_id: str | None = None,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if grading_protocol_version != GRADING_PROTOCOL_VERSION:
        raise RuntimeError("grading protocol mismatch")
    jobs = [json.loads(x) for x in manifest_jsonl.splitlines() if x.strip()]
    if len(jobs) != N_PLANNED_GENERATIONS:
        raise RuntimeError(f"jobs {len(jobs)} != {N_PLANNED_GENERATIONS}")
    if hashlib.sha256(manifest_jsonl.encode()).hexdigest() != manifest_sha256:
        raise RuntimeError("grading manifest sha mismatch")
    jobs_by_id = {j["grading_job_id"]: j for j in jobs}
    if len(jobs_by_id) != N_PLANNED_GENERATIONS:
        raise RuntimeError("duplicate grading_job_ids")

    try:
        from modal import current_function_call_id as _cfid

        function_call_id = _cfid() or function_call_id
    except Exception:  # noqa: BLE001
        pass

    run_key = run_key_from_grading_manifest_sha(manifest_sha256)
    rdir = Path(RESULTS_MOUNT) / run_key
    ckpt_dir = rdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_vol.reload()

    if (rdir / "COMPLETE.json").exists():
        return {"status": "already_complete", "run_key": run_key}

    active_path = rdir / "ACTIVE_RUN.json"
    if active_path.exists():
        try:
            prev = json.loads(active_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            prev = {}
        prev_fc = prev.get("function_call_id")
        prev_state = prev.get("state")
        if (
            prev_fc
            and prev_fc != function_call_id
            and prev_state in {"RUNNING", "MODEL_LOADED", "INITIALIZING"}
        ):
            raise RuntimeError(
                f"refusing second active grading writer: fc={prev_fc} state={prev_state}"
            )

    completed, ckpt_errors = load_grading_checkpoints(
        checkpoint_dir=ckpt_dir, jobs_by_id=jobs_by_id
    )
    if ckpt_errors:
        raise RuntimeError(
            "HARD STOP: grading checkpoint conflict:\n" + "\n".join(ckpt_errors[:40])
        )

    active_path.write_text(
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

    tok = AutoTokenizer.from_pretrained(
        GRADER_MODEL, revision=GRADER_REVISION, cache_dir=MODEL_CACHE, use_fast=True
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        GRADER_MODEL,
        revision=GRADER_REVISION,
        torch_dtype=torch.bfloat16,
        cache_dir=MODEL_CACHE,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("PHASE27_GEMMA_GRADER_LOADED", flush=True)

    active_path.write_text(
        json.dumps(
            {
                "run_key": run_key,
                "function_call_id": function_call_id,
                "state": "RUNNING",
                "manifest_sha256": manifest_sha256,
                "authorization_commit": authorization_commit,
                "grader_model": GRADER_MODEL,
                "grader_revision": GRADER_REVISION,
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
    remaining = [j for j in jobs if j["grading_job_id"] not in completed]

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
            completed[row["grading_job_id"]] = row
        pending = []
        batch_index += 1
        (rdir / "progress.json").write_text(
            json.dumps(
                {
                    "grading_protocol_version": GRADING_PROTOCOL_VERSION,
                    "run_key": run_key,
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

    for job in remaining:
        if job["grading_job_id"] in completed:
            continue
        g = job["grader"]
        if g["model_revision"] != GRADER_REVISION or g["model_id"] != GRADER_MODEL:
            raise RuntimeError("grader model/revision drift")
        prompt_ids = verify_frozen_grader_model_input(
            tok,
            model_input_text=job["grader_model_input_text"],
            model_input_sha256=job["grader_model_input_sha256"],
            input_ids_sha256=job["grader_input_ids_sha256"],
        )
        # Truncate only if exceeding frozen max_model_len budget (should not for Phase 27).
        max_input = int(GRADER_MAX_MODEL_LEN) - int(GRADER_MAX_NEW_TOKENS)
        if len(prompt_ids) > max_input:
            raise RuntimeError(
                f"{job['grading_job_id']}: input_ids {len(prompt_ids)} > {max_input}"
            )

        tech_fail = False
        tech_err = None
        raw_out = None
        parsed = {
            "valid": False,
            "score": None,
            "class": "invalid",
            "explanation": None,
            "raw": None,
        }
        for _attempt in range(3):
            try:
                with torch.inference_mode():
                    ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
                    gen = model.generate(
                        ids,
                        max_new_tokens=int(GRADER_MAX_NEW_TOKENS),
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        pad_token_id=tok.pad_token_id,
                    )
                    new_ids = gen[0, ids.shape[1] :]
                    if tok.pad_token_id is not None:
                        new_ids = new_ids[new_ids != tok.pad_token_id]
                    raw_out = tok.decode(new_ids, skip_special_tokens=True)
                parsed = parse_json_grade(raw_out)
                tech_fail = False
                tech_err = None
                break
            except Exception as exc:  # noqa: BLE001
                tech_fail = True
                tech_err = str(exc)
                # Infrastructure retry only — same frozen input/settings; no label rescue.
                continue

        primary = primary_label_from_parsed(parsed)
        row = {
            "grading_job_id": job["grading_job_id"],
            "job_id": job["job_id"],
            "scenario_id": job["scenario_id"],
            "rollout_index": job["rollout_index"],
            "response_sha256": job["response_sha256"],
            "grader_model_input_sha256": job["grader_model_input_sha256"],
            "grader_input_ids_sha256": job["grader_input_ids_sha256"],
            "grader_prompt_sha256": job["grader_prompt_sha256"],
            "grader": g,
            "raw_gemma_output": raw_out,
            "parsed_score": parsed.get("score"),
            "parsed_class": parsed.get("class"),
            "explanation": parsed.get("explanation"),
            "parse_valid": bool(parsed.get("valid")),
            "primary_label": primary,
            "technical_failure": tech_fail,
            "technical_error": tech_err,
            "activation_capture": False,
            "output_hidden_states": False,
            "logit_save": False,
            "physiology_collection": False,
            "outcome_dependent_retry": False,
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
                all_rows[r["grading_job_id"]] = r
    if len(all_rows) != N_PLANNED_GENERATIONS:
        raise RuntimeError(f"incomplete grading {len(all_rows)}")
    ordered = [all_rows[j["grading_job_id"]] for j in jobs]
    # Exact set / field integrity vs manifest
    for r, j in zip(ordered, jobs):
        errs = []
        if r["job_id"] != j["job_id"]:
            errs.append("job_id")
        if r["response_sha256"] != j["response_sha256"]:
            errs.append("response_sha256")
        if r["grader_model_input_sha256"] != j["grader_model_input_sha256"]:
            errs.append("grader_model_input_sha256")
        if errs:
            raise RuntimeError(
                f"final grading integrity fail {j['grading_job_id']}: {errs}"
            )

    grades_path = rdir / "grades.jsonl"
    with grades_path.open("w", encoding="utf-8") as f:
        for r in ordered:
            f.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")
    grades_sha = hashlib.sha256(grades_path.read_bytes()).hexdigest()
    (rdir / "COMPLETE.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "n_grades": N_PLANNED_GENERATIONS,
                "grades_sha256": grades_sha,
                "manifest_sha256": manifest_sha256,
                "completed_utc": _utc_now(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    active_path.write_text(
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
    print("PHASE27_GEMMA_GRADING_288_COMPLETE", flush=True)
    return {"status": "complete", "run_key": run_key, "grades_sha": grades_sha}


@app.local_entrypoint()
def main() -> None:
    """Authorize → validate → spawn on deployed app → receipt → exit."""
    cfg = _load_auth()
    manifest_text, jobs, manifest_sha = _verify_manifest()
    if cfg.get("grading_status") != GRADING_STATUS_AUTHORIZED:
        raise SystemExit("grading_status gate failed")
    run_key = run_key_from_grading_manifest_sha(manifest_sha)
    status_fn, run_fn = _deployed_functions()
    status = status_fn.remote(run_key)
    decision = decide_grading_launch_action(
        grading_status=cfg.get("grading_status"),
        grader_execution_authorized=bool(
            cfg["authorizations"].get("grader_execution_authorized")
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
    call = run_fn.spawn(
        manifest_jsonl=manifest_text,
        manifest_sha256=manifest_sha,
        authorization_commit=launch_head,
        grading_protocol_version=GRADING_PROTOCOL_VERSION,
        function_call_id=None,
    )
    fc_id = getattr(call, "object_id", None) or str(call)
    receipt = {
        "grading_protocol_version": GRADING_PROTOCOL_VERSION,
        "run_key": run_key,
        "function_call_id": fc_id,
        "manifest_sha256": manifest_sha,
        "launch_head": launch_head,
        "branch": branch,
        "detached": True,
        "resume": bool(decision.get("resume")),
        "n_planned_grades": N_PLANNED_GENERATIONS,
        "results_volume": RESULTS_VOLUME_NAME,
        "wait_for_completion": False,
    }
    LAUNCH_RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    LAUNCH_RECEIPT.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"PHASE27_GRADING_SPAWNED function_call_id={fc_id}", flush=True)
    print(f"PHASE27_GRADING_RUN_KEY={run_key}", flush=True)
    print("PHASE27_GRADING_DETACHED_EXIT", flush=True)
