"""Modal GPU runner for Phase 26C primary (ADVERSE-only; text generation only).

GPU execution is BLOCKED until configs/...yaml sets:
  authorizations.modal_gpu_mistral_screening_authorized: true
  status: phase26c_primary_generation_authorized

Do not run this module during preflight preparation.
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

APP_NAME = "pre-output-physiology-phase26c-primary"
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/phase26c_primary_behavioral_feasibility.yaml"
MANIFEST_PATH = REPO_ROOT / "artifacts/phase26c_primary/inference_manifest.jsonl"
OUT_RAW = REPO_ROOT / "data/phase26/behavioral_feasibility_primary/raw_generations.jsonl"
OUT_META = REPO_ROOT / "artifacts/phase26c_primary/run_metadata.json"

A100_USD_PER_HOUR = 2.50
GPU_TYPE = "A100-80GB"
CHUNK_SIZE = 2904

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
MODEL_CACHE = "/vol/hf_cache"


def _dirty_ok() -> None:
    raw = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True
    ).strip()
    if not raw:
        return
    allowed = (
        "data/phase26/behavioral_feasibility_primary/",
        "artifacts/phase26c_primary/",
        "reports/phase26c_primary",
    )
    for line in raw.splitlines():
        path = line[3:].strip().strip('"').lstrip("./")
        if any(path.startswith(p) for p in allowed):
            continue
        raise SystemExit(f"STOP: dirty tree outside Phase-26C primary outputs:\n{raw}")


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
    timeout=60 * 60 * 4,
    volumes={MODEL_CACHE: mistral_vol},
    memory=65536,
)
def run_chunk(rows_json: str) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from pre_output_physiology.phase26_protocol import parse_report
    from pre_output_physiology.phase26c_primary import (
        ATTN_IMPLEMENTATION,
        MODEL_ID,
        MODEL_REVISION,
        TOKENIZER_REVISION,
    )

    rows = json.loads(rows_json)
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

    out_rows: list[dict[str, Any]] = []
    t0 = time.time()
    total_tokens = 0
    for r in rows:
        if r.get("state_id") != "ADVERSE":
            raise RuntimeError(f"SAFE leak in job {r.get('job_id')}")
        messages = [{"role": "user", "content": r["prompt_text"]}]
        formatted = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tok.encode(formatted, add_special_tokens=False)
        seed = int(r["seed"])
        do_sample = bool(r["generation"]["do_sample"])
        temperature = r["generation"]["temperature"]
        max_new = int(r["generation"]["max_new_tokens"])
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
            out_rows.append(
                {
                    **{k: r[k] for k in r if k != "prompt_text"},
                    "raw_response_text": None,
                    "parsed_decision": None,
                    "parse_valid": False,
                    "malformed_reason": "technical_failure",
                    "technical_failure": True,
                    "technical_error": last_err,
                    "n_attempts": n_attempts,
                    "stopping_reason": None,
                    "n_generated_tokens": 0,
                    "formatted_prompt_sha256": hashlib.sha256(
                        formatted.encode()
                    ).hexdigest(),
                    "job_id": r["job_id"],
                }
            )
            continue
        raw = tok.decode(gen["generated_token_ids"], skip_special_tokens=True)
        parsed = parse_report(raw)
        total_tokens += gen["n_tokens"]
        out_rows.append(
            {
                "protocol_version": r["protocol_version"],
                "parent_phase26b_commit": r["parent_phase26b_commit"],
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "tokenizer_revision": TOKENIZER_REVISION,
                "scenario_id": r["scenario_id"],
                "state_id": r["state_id"],
                "instance_id": r["instance_id"],
                "condition": r["condition"],
                "consequence_class": r.get("consequence_class"),
                "prompt_id": r["prompt_id"],
                "prompt_sha256": r["prompt_sha256"],
                "n_template_id": r.get("n_template_id"),
                "x_template_id": r.get("x_template_id"),
                "gold_decision": r["gold_decision"],
                "rollout_type": r["rollout_type"],
                "rollout_index": r["rollout_index"],
                "seed": seed,
                "generation_parameters": r["generation"],
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
                "job_id": r["job_id"],
                "activation_capture": False,
                "output_hidden_states": False,
                "logit_save": False,
            }
        )
    return {
        "rows": out_rows,
        "wall_seconds": time.time() - t0,
        "total_generated_tokens": total_tokens,
        "n_rows": len(out_rows),
    }


@app.local_entrypoint()
def main() -> None:
    from pre_output_physiology.phase26c_primary import N_PLANNED_GENERATIONS, write_jsonl

    cfg = _load_auth()
    jobs = [
        json.loads(x)
        for x in MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    if len(jobs) != N_PLANNED_GENERATIONS:
        raise SystemExit(f"manifest size {len(jobs)} != {N_PLANNED_GENERATIONS}")
    if any(j["state_id"] != "ADVERSE" for j in jobs):
        raise SystemExit("SAFE present in inference manifest")
    if OUT_RAW.exists():
        raise SystemExit(f"refusing to overwrite existing raw file: {OUT_RAW}")

    chunks = [jobs[i : i + CHUNK_SIZE] for i in range(0, len(jobs), CHUNK_SIZE)]
    print(f"launching {len(chunks)} chunks on {GPU_TYPE}")
    t0 = time.time()
    all_rows: list[dict[str, Any]] = []
    total_tokens = 0
    chunk_meta = []
    for i, chunk in enumerate(chunks):
        print(f"chunk {i + 1}/{len(chunks)} n={len(chunk)}", flush=True)
        result = run_chunk.remote(json.dumps(chunk))
        all_rows.extend(result["rows"])
        total_tokens += int(result["total_generated_tokens"])
        chunk_meta.append(
            {
                "chunk_index": i,
                "n_rows": result["n_rows"],
                "wall_seconds": result["wall_seconds"],
                "total_generated_tokens": result["total_generated_tokens"],
            }
        )
    wall = time.time() - t0
    if len(all_rows) != N_PLANNED_GENERATIONS:
        raise SystemExit(f"row count {len(all_rows)} != {N_PLANNED_GENERATIONS}")

    OUT_RAW.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_RAW, all_rows)
    raw_sha = hashlib.sha256(OUT_RAW.read_bytes()).hexdigest()
    meta = {
        "run_id": f"phase26c_primary_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
        "protocol_version": cfg["protocol_version"],
        "git_head": subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "parent_phase26b_commit": cfg["parent_phase26b_commit"],
        "model_id": cfg["model"]["model_id"],
        "model_revision": cfg["model"]["revision"],
        "tokenizer_revision": cfg["model"]["tokenizer_revision"],
        "attn_implementation": cfg["model"]["attention_implementation"],
        "n_generations": len(all_rows),
        "n_technical_failures": sum(1 for r in all_rows if r.get("technical_failure")),
        "total_generated_tokens": total_tokens,
        "wall_seconds": wall,
        "gpu_hours": wall / 3600.0,
        "estimated_cost_usd": (wall / 3600.0) * A100_USD_PER_HOUR,
        "cost_rate_usd_per_hour": A100_USD_PER_HOUR,
        "gpu": GPU_TYPE,
        "raw_generations_path": str(OUT_RAW),
        "raw_generations_sha256": raw_sha,
        "chunk_meta": chunk_meta,
        "activation_capture": False,
        "output_hidden_states": False,
        "logit_save": False,
        "grader_model": False,
        "physiology_collection": False,
        "safe_included": False,
    }
    OUT_META.parent.mkdir(parents=True, exist_ok=True)
    OUT_META.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: meta[k] for k in meta if k != "chunk_meta"}, indent=2))
