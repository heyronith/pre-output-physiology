"""Modal GPU entrypoint: Phase 4B pilot generation only (no activations).

Usage (from repo root, clean tree required):

  uv run modal run modal/phase4_pilot_generate.py

Generates greedy BF16 responses for the frozen 144 pilot prompts.
Does not score probes. Does not generate final scenarios. Does not extract activations.
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

APP_NAME = "pre-output-physiology-phase4-pilot"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
COMMON_FIRST_TOKEN_ID = 2963
MAX_NEW_TOKENS = 128
BATCH_SIZE = 1
L40S_USD_PER_HOUR = 1.95
REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT_PROMPTS_PATH = REPO_ROOT / "data/processed/phase4_design/pilot_candidate_prompts.jsonl"
FINAL_PROMPTS_PATH = REPO_ROOT / "data/processed/phase4_design/final_candidate_prompts.jsonl"
CONFIG_PATH = REPO_ROOT / "configs/experiments/phase4_specificity.yaml"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.44.2",
        "accelerate==0.33.0",
        "safetensors==0.4.4",
        "numpy==1.26.4",
        "pyyaml==6.0.2",
        "huggingface_hub==0.24.6",
    )
)

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name("preoutput-mistral-cache", create_if_missing=True)
MODEL_CACHE_DIR = "/vol/hf_cache"


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_text(text: str) -> str:
    return _sha_bytes(text.encode("utf-8"))


def _sha_ids(ids: list[int]) -> str:
    return _sha_text(",".join(str(i) for i in ids))


def _collect_local_provenance() -> dict[str, Any]:
    dirty = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise SystemExit(
            "STOP: working tree dirty — commit before Modal pilot launch:\n" f"{dirty}"
        )
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    auth = cfg.get("authorizations", {})
    if auth.get("pilot_generation_authorized") is not True:
        raise SystemExit("pilot_generation_authorized must be true")
    if auth.get("final_generation_authorized") is not False:
        raise SystemExit("final_generation_authorized must remain false")
    if auth.get("activation_extraction_authorized") is not False:
        raise SystemExit("activation_extraction_authorized must remain false")
    if auth.get("causal_intervention_authorized") is not False:
        raise SystemExit("causal_intervention_authorized must remain false")
    if auth.get("modal_gpu_authorized") is not True:
        raise SystemExit("modal_gpu_authorized must be true for pilot")

    git_commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    generator = Path(__file__).resolve()
    return {
        "git_commit": git_commit,
        "working_tree_clean": True,
        "generator_path": str(generator.relative_to(REPO_ROOT)),
        "generator_sha256": _sha_bytes(generator.read_bytes()),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "precision": "bf16",
        "quantization": None,
        "batch_size": BATCH_SIZE,
        "do_sample": False,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": None,
        "common_first_token_id_expected": COMMON_FIRST_TOKEN_ID,
        "force_first_token": False,
        "activations_authorized": False,
        "probe_scoring_authorized": False,
        "final_generation_authorized": False,
    }


def _load_pilot_rows() -> list[dict[str, Any]]:
    if not PILOT_PROMPTS_PATH.is_file():
        raise SystemExit(f"missing pilot prompts: {PILOT_PROMPTS_PATH}")
    rows = []
    with PILOT_PROMPTS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != 144:
        raise SystemExit(f"expected 144 pilot prompts, got {len(rows)}")
    if any(r.get("split") != "pilot" for r in rows):
        raise SystemExit("non-pilot split found in pilot prompt file")
    # Contamination check: never load final prompts into this job.
    if not FINAL_PROMPTS_PATH.is_file():
        raise SystemExit("final prompt file missing (needed for contamination guard)")
    return rows


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 90,
    volumes={MODEL_CACHE_DIR: model_volume},
    memory=65536,
)
def generate_pilot_responses(
    prompts_jsonl: str,
    *,
    provenance: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    """Greedy BF16 generation for pilot prompts only. No activations."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(line) for line in prompts_jsonl.splitlines() if line.strip()]
    if len(rows) != 144:
        raise ValueError(f"expected 144 rows, got {len(rows)}")
    if any(r.get("split") != "pilot" for r in rows):
        raise ValueError("contamination: non-pilot row in generation payload")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=MODEL_CACHE_DIR,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        cache_dir=MODEL_CACHE_DIR,
        low_cpu_mem_usage=True,
    )
    model.to("cuda:0")
    model.eval()
    model_load_s = time.time() - t0

    commit_hash = getattr(model.config, "_commit_hash", None)
    outputs: list[dict[str, Any]] = []
    gen_t0 = time.time()
    with torch.inference_mode():
        for row in rows:
            prompt_text = row["prompt_text"]
            messages = [{"role": "user", "content": prompt_text}]
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            enc = tokenizer(formatted, return_tensors="pt", add_special_tokens=False)
            input_ids = enc["input_ids"][0].tolist()
            enc = {k: v.to("cuda:0") for k, v in enc.items()}
            gen = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            new_tokens = gen[0, enc["input_ids"].shape[1] :]
            gen_ids = [int(x) for x in new_tokens.tolist()]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            if gen_ids and gen_ids[-1] == tokenizer.eos_token_id:
                stop_reason = "eos_token"
            elif len(gen_ids) >= MAX_NEW_TOKENS:
                stop_reason = "max_new_tokens"
            else:
                stop_reason = "other"
            first_id = gen_ids[0] if gen_ids else None
            outputs.append(
                {
                    "example_id": row["example_id"],
                    "base_scenario_id": row["base_scenario_id"],
                    "condition_id": row["condition_id"],
                    "split": "pilot",
                    "family": row.get("family"),
                    "ground_truth_state": row["ground_truth_state"],
                    "alt_state": row["alt_state"],
                    "accessible_false_state": row["accessible_false_state"],
                    "formatted_prompt": formatted,
                    "formatted_prompt_sha256": _sha_text(formatted),
                    "input_token_ids": input_ids,
                    "input_token_ids_sha256": _sha_ids(input_ids),
                    "output_text": text,
                    "generated_token_ids": gen_ids,
                    "first_generated_token_id": first_id,
                    "n_generated_tokens": len(gen_ids),
                    "stop_reason": stop_reason,
                    "activation_extracted": False,
                    "probe_scored": False,
                }
            )
    gen_s = time.time() - gen_t0
    wall_s = time.time() - t0
    model_volume.commit()
    # Drop bulky formatted prompt / full input ids from return if needed?
    # Keep them — raw run is gitignored; needed for provenance.
    return {
        "run_id": run_id,
        "n_outputs": len(outputs),
        "outputs": outputs,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "model_config_commit_hash": commit_hash,
        "precision": "bf16",
        "do_sample": False,
        "max_new_tokens": MAX_NEW_TOKENS,
        "batch_size": BATCH_SIZE,
        "gpu_type": "L40S",
        "model_load_seconds": model_load_s,
        "generation_seconds": gen_s,
        "wall_seconds": wall_s,
        "estimated_cost_usd": (wall_s / 3600.0) * L40S_USD_PER_HOUR,
        "l40s_usd_per_hour_assumed": L40S_USD_PER_HOUR,
        "activations_collected": False,
        "probe_scores_computed": False,
        "final_scenarios_generated": False,
        "provenance": provenance,
    }


@app.local_entrypoint()
def main() -> None:
    prov = _collect_local_provenance()
    rows = _load_pilot_rows()
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"phase4b_pilot_{ts}_{prov['git_commit'][:8]}"
    prompts_jsonl = "\n".join(json.dumps(r, sort_keys=True) for r in rows)

    print(f"Launching pilot generation run_id={run_id} n={len(rows)}")
    result = generate_pilot_responses.remote(
        prompts_jsonl,
        provenance=prov,
        run_id=run_id,
    )
    if result["n_outputs"] != 144:
        raise SystemExit(f"expected 144 outputs, got {result['n_outputs']}")

    run_dir = REPO_ROOT / "artifacts/runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / "pilot_outputs.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in result["outputs"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    # Slim rows for local eval (drop bulky token lists from committed path later)
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "git_commit": prov["git_commit"],
        "working_tree_clean": True,
        "generator_sha256": prov["generator_sha256"],
        "generator_path": prov["generator_path"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "model_config_commit_hash": result.get("model_config_commit_hash"),
        "precision": "bf16",
        "quantization": None,
        "batch_size": BATCH_SIZE,
        "do_sample": False,
        "max_new_tokens": MAX_NEW_TOKENS,
        "gpu_type": "L40S",
        "n_pilot_outputs": result["n_outputs"],
        "n_final_outputs": 0,
        "raw_outputs_path": str(raw_path.relative_to(REPO_ROOT)),
        "raw_outputs_sha256": _sha_bytes(raw_path.read_bytes()),
        "model_load_seconds": result["model_load_seconds"],
        "generation_seconds": result["generation_seconds"],
        "wall_seconds": result["wall_seconds"],
        "estimated_cost_usd": result["estimated_cost_usd"],
        "l40s_usd_per_hour_assumed": L40S_USD_PER_HOUR,
        "activations_collected": False,
        "probe_scores_computed": False,
        "final_scenarios_generated": False,
        "force_first_token": False,
        "common_first_token_id_expected": COMMON_FIRST_TOKEN_ID,
        "first_generated_token_id_counts": {},
    }
    # Token-ID histogram for quick inspection
    counts: dict[str, int] = {}
    for row in result["outputs"]:
        key = str(row["first_generated_token_id"])
        counts[key] = counts.get(key, 0) + 1
    manifest["first_generated_token_id_counts"] = counts

    (run_dir / "pilot_generation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: manifest[k] for k in (
        "run_id",
        "n_pilot_outputs",
        "wall_seconds",
        "estimated_cost_usd",
        "raw_outputs_path",
        "first_generated_token_id_counts",
    )}, indent=2))
