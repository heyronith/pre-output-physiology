"""Modal GPU: Phase 22B additional RoleplayDeception generation (3710 new rollouts r10–r19).

Apollo settings: do_sample=True, temperature=1.0, max_new_tokens=200, BF16, batch=1.
No activations during generation.

Usage (clean tree):

    uv run modal run modal/phase22b_generate.py
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

APP_NAME = "pre-output-physiology-phase22b-generate"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
L40S_USD_PER_HOUR = 1.95

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_PATH = REPO_ROOT / "data/processed/phase22b_k20/prompts.jsonl"
SCHEDULE_PATH = REPO_ROOT / "data/processed/phase22b_k20/generation_schedule.jsonl"
CONFIG_PATH = REPO_ROOT / "configs/experiments/phase22b_k20_sampling_validation.yaml"
MATRIX_PATH = REPO_ROOT / "artifacts/phase22b_design/design_matrix.json"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.44.2",
        "accelerate==0.33.0",
        "numpy==1.26.4",
        "huggingface_hub==0.24.6",
        "pyyaml==6.0.2",
    )
    .add_local_python_source("pre_output_physiology")
)
app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name("preoutput-mistral-cache", create_if_missing=True)
MODEL_CACHE_DIR = "/vol/hf_cache"


def _local_setup() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    from pre_output_physiology.phase21_roleplay import TEMPERATURE, schedule_sha
    from pre_output_physiology.phase22b_sampling import (
        N_ADDITIONAL as N_TOTAL,
    )
    from pre_output_physiology.phase22b_sampling import (
        STATUS_AUTHORIZED,
    )

    dirty = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise SystemExit(f"STOP: working tree dirty:\n{dirty}")

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if cfg.get("status") != STATUS_AUTHORIZED:
        raise SystemExit(f"requires authorized, got {cfg.get('status')}")
    auth = cfg["authorizations"]
    if auth.get("generation_model_calls_authorized") is not True:
        raise SystemExit("generation not authorized")
    if auth.get("modal_gpu_authorized") is not True:
        raise SystemExit("modal GPU not authorized")
    for key in (
        "grading_api_authorized",
        "onset_annotation_authorized",
        "activation_extraction_authorized",
        "probe_fitting_authorized",
        "prompt_changes_authorized",
        "layer_search_authorized",
        "test_set_probe_selection_authorized",
        "threshold_changes_authorized",
        "causal_intervention_authorized",
        "phase20_stage2_authorized",
    ):
        if auth.get(key) is not False:
            raise SystemExit(f"{key} must remain false during generation")

    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    prompts = [
        json.loads(x)
        for x in PROMPTS_PATH.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    schedule = [
        json.loads(x)
        for x in SCHEDULE_PATH.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    if len(schedule) != N_TOTAL:
        raise SystemExit(f"schedule {len(schedule)} != {N_TOTAL}")
    if schedule_sha(schedule) != matrix["generation_schedule_sha256"]:
        raise SystemExit("generation_schedule_sha256 mismatch")
    if float(matrix["temperature"]) != float(TEMPERATURE):
        raise SystemExit("temperature mismatch")

    prompt_by_id = {p["prompt_id"]: p for p in prompts}
    payload = []
    for row in schedule:
        p = prompt_by_id[row["prompt_id"]]
        payload.append(
            {
                **row,
                "scenario": p["scenario"],
                "question": p["question"],
                "answer_prefix": p["answer_prefix"],
            }
        )
    meta = {
        "git_commit": subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "design_matrix": matrix,
        "n": len(payload),
    }
    return meta, prompts, payload


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 6,
    volumes={MODEL_CACHE_DIR: model_volume},
    memory=65536,
)
def generate_all(payload_json: str) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    from pre_output_physiology.phase21_roleplay import (
        MAX_NEW_TOKENS,
        TEMPERATURE,
        format_prompt_for_generation,
    )
    from pre_output_physiology.phase21_roleplay import (
        MODEL_ID as MID,
    )
    from pre_output_physiology.phase21_roleplay import (
        MODEL_REVISION as MREV,
    )

    rows = json.loads(payload_json)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(
        MID, revision=MREV, cache_dir=MODEL_CACHE_DIR, use_fast=True
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MID,
        revision=MREV,
        torch_dtype=torch.bfloat16,
        cache_dir=MODEL_CACHE_DIR,
        low_cpu_mem_usage=True,
    )
    model.to("cuda:0")
    model.eval()

    out_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for i, r in enumerate(rows):
            prompt_text = format_prompt_for_generation(
                tok, r["scenario"], r["question"], r["answer_prefix"]
            )
            input_ids = tok.encode(prompt_text, add_special_tokens=False)
            ids = torch.tensor([input_ids], dtype=torch.long, device="cuda:0")
            set_seed(int(r["sample_seed"]))
            torch.manual_seed(int(r["sample_seed"]))
            torch.cuda.manual_seed_all(int(r["sample_seed"]))
            gen = model.generate(
                ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                pad_token_id=tok.eos_token_id,
            )
            new_ids = gen[0, ids.shape[1] :].tolist()
            completion = tok.decode(new_ids, skip_special_tokens=True)
            # Apollo postprocess: strip trailing spaces/newlines; lstrip newlines only
            completion = completion.rstrip("\n ").lstrip("\n")
            full_response = r["answer_prefix"] + completion
            out_rows.append(
                {
                    "continuation_id": r["continuation_id"],
                    "prompt_id": r["prompt_id"],
                    "scenario_index": r["scenario_index"],
                    "rollout_index": r["rollout_index"],
                    "sample_seed": r["sample_seed"],
                    "split": r["split"],
                    "temperature": TEMPERATURE,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "do_sample": True,
                    "answer_prefix": r["answer_prefix"],
                    "completion": completion,
                    "full_response": full_response,
                    "n_new_tokens": len(new_ids),
                    "prompt_n_tokens": len(input_ids),
                    "activation_extracted": False,
                    "graded": False,
                }
            )
            if (i + 1) % 100 == 0:
                print(f"generated {i + 1}/{len(rows)}")

    wall = time.time() - t0
    return {
        "rows": out_rows,
        "wall_seconds": wall,
        "estimated_cost_usd": wall / 3600.0 * L40S_USD_PER_HOUR,
        "n": len(out_rows),
    }


@app.local_entrypoint()
def main() -> None:
    meta, _prompts, payload = _local_setup()
    print(f"Generating n={len(payload)}")
    result = generate_all.remote(json.dumps(payload))
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    short = hashlib.sha256(meta["git_commit"].encode()).hexdigest()[:8]
    run_id = f"phase22b_gen_{ts}_{short}"
    out_dir = REPO_ROOT / "artifacts/runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "continuations.jsonl").open("w", encoding="utf-8") as f:
        for row in result["rows"]:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": meta["git_commit"],
        "n_continuations": result["n"],
        "wall_seconds": result["wall_seconds"],
        "estimated_cost_usd": result["estimated_cost_usd"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dtype": "bfloat16",
        "batch_size": 1,
        "temperature": 1.0,
        "max_new_tokens": 200,
        "do_sample": True,
        "generation_schedule_sha256": meta["design_matrix"]["generation_schedule_sha256"],
        "prompt_split_sha256": meta["design_matrix"]["prompt_split_sha256"],
        "activations_collected": False,
        "locked_model_calls": 0,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    # Update config status pointer locally for next stage (committed separately)
    summary = {
        "run_id": run_id,
        "n": result["n"],
        "wall_seconds": result["wall_seconds"],
        "estimated_cost_usd": result["estimated_cost_usd"],
        "manifest": str(out_dir / "manifest.json"),
    }
    print(json.dumps(summary, indent=2))
