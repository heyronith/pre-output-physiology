"""Modal GPU: Phase 5A behavior-only pilot (S2/S3; no activations).

Usage (clean tree required):

  uv run modal run modal/phase5_pilot_generate.py
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

APP_NAME = "pre-output-physiology-phase5a-pilot"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
MAX_NEW_TOKENS = 32
BATCH_SIZE = 1
L40S_USD_PER_HOUR = 1.95
N_EXPECTED = 64

REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT_PROMPTS_PATH = REPO_ROOT / "data/processed/phase5_design/pilot_candidate_prompts.jsonl"
CONFIG_PATH = REPO_ROOT / "configs/experiments/phase5_strategic_discovery.yaml"
MATRIX_PATH = REPO_ROOT / "artifacts/phase5a_design/condition_matrix.json"

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


def _sha_prompt_texts(rows: list[dict[str, Any]]) -> str:
    ordered = sorted(rows, key=lambda r: r["example_id"])
    payload = "\n".join(f"{r['example_id']}\t{r['prompt_text']}" for r in ordered)
    return _sha_text(payload)


def _collect_local_provenance() -> dict[str, Any]:
    dirty = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise SystemExit(f"STOP: working tree dirty:\n{dirty}")
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if cfg.get("status") != "phase5a_design_frozen_pilot_authorized":
        raise SystemExit(f"unexpected status {cfg.get('status')}")
    auth = cfg.get("authorizations", {})
    if auth.get("pilot_generation_authorized") is not True:
        raise SystemExit("pilot_generation_authorized must be true")
    if auth.get("activation_extraction_authorized") is not False:
        raise SystemExit("activations must remain unauthorized")
    if auth.get("probe_fitting_authorized") is not False:
        raise SystemExit("probe fitting must remain unauthorized")
    if auth.get("causal_intervention_authorized") is not False:
        raise SystemExit("causal must remain unauthorized")
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    expected_pilot_hash = matrix["pilot_prompt_text_sha256"]
    git_commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    generator = Path(__file__).resolve()
    return {
        "git_commit": git_commit,
        "working_tree_clean": True,
        "generator_sha256": _sha_bytes(generator.read_bytes()),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "expected_pilot_prompt_text_sha256": expected_pilot_hash,
        "neutral_prefix_token_id": matrix["neutral_prefix_token_id"],
        "prompt_template_revision": int(
            cfg.get("design", {}).get("prompt_template_revision", 2)
        ),
        "activations_authorized": False,
        "probe_fitting_authorized": False,
        "locked_families_run": False,
    }


def _load_pilot_rows(expected_hash: str) -> list[dict[str, Any]]:
    rows = []
    with PILOT_PROMPTS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != N_EXPECTED:
        raise SystemExit(f"expected {N_EXPECTED} pilot prompts, got {len(rows)}")
    if any(r.get("split") != "pilot" for r in rows):
        raise SystemExit("non-pilot rows present")
    if any(r.get("pool") == "locked" and r.get("split") != "pilot" for r in rows):
        pass  # pilot may include locked-family pilot scenarios (disjoint IDs)
    digest = _sha_prompt_texts(rows)
    if digest != expected_hash:
        raise SystemExit(f"pilot prompt hash mismatch: {digest}")
    return rows


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 40,
    volumes={MODEL_CACHE_DIR: model_volume},
    memory=65536,
)
def generate_pilot(
    prompts_jsonl: str,
    *,
    provenance: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(line) for line in prompts_jsonl.splitlines() if line.strip()]
    if len(rows) != N_EXPECTED:
        raise ValueError(f"expected {N_EXPECTED}, got {len(rows)}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=MODEL_CACHE_DIR, use_fast=True
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

    outputs = []
    gen_t0 = time.time()
    with torch.inference_mode():
        for row in rows:
            messages = [{"role": "user", "content": row["prompt_text"]}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            enc = tokenizer(formatted, return_tensors="pt")
            enc = {k: v.to("cuda:0") for k, v in enc.items()}
            prompt_len = int(enc["input_ids"].shape[1])
            gen = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            full_ids = [int(x) for x in gen[0].tolist()]
            new_ids = full_ids[prompt_len:]
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
            outputs.append(
                {
                    "example_id": row["example_id"],
                    "base_scenario_id": row["base_scenario_id"],
                    "condition_id": row["condition_id"],
                    "family": row["family"],
                    "split": "pilot",
                    "pool": row["pool"],
                    "record_state": row["record_state"],
                    "alternate_state": row["alternate_state"],
                    "objective_target": row["objective_target"],
                    "generated_token_ids": new_ids,
                    "first_generated_token_id": new_ids[0] if new_ids else None,
                    "n_generated_tokens": len(new_ids),
                    "output_text": text,
                    "prompt_template_revision": provenance.get(
                        "prompt_template_revision", 2
                    ),
                    "activation_extracted": False,
                    "probe_scored": False,
                    "model_revision": MODEL_REVISION,
                }
            )
    gen_s = time.time() - gen_t0
    wall_s = time.time() - t0
    model_volume.commit()
    return {
        "run_id": run_id,
        "n_outputs": len(outputs),
        "outputs": outputs,
        "model_revision": MODEL_REVISION,
        "do_sample": False,
        "max_new_tokens": MAX_NEW_TOKENS,
        "batch_size": BATCH_SIZE,
        "gpu_type": "L40S",
        "model_load_seconds": model_load_s,
        "generation_seconds": gen_s,
        "wall_seconds": wall_s,
        "estimated_cost_usd": (wall_s / 3600.0) * L40S_USD_PER_HOUR,
        "activations_collected": False,
        "probes_fit_or_scored": False,
        "locked_final_families_run": False,
        "provenance": provenance,
    }


@app.local_entrypoint()
def main() -> None:
    from collections import Counter

    prov = _collect_local_provenance()
    rows = _load_pilot_rows(prov["expected_pilot_prompt_text_sha256"])
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"phase5a_pilot_{ts}_{prov['git_commit'][:8]}"
    prompts_jsonl = "\n".join(json.dumps(r, sort_keys=True) for r in rows)
    print(f"Launching Phase 5A pilot run_id={run_id} n={len(rows)}")
    result = generate_pilot.remote(prompts_jsonl, provenance=prov, run_id=run_id)
    if result["n_outputs"] != N_EXPECTED:
        raise SystemExit(f"expected {N_EXPECTED}, got {result['n_outputs']}")
    counts = Counter(r["condition_id"] for r in result["outputs"])
    if counts.get("S2_strategic_honesty") != 32 or counts.get("S3_strategic_deception") != 32:
        raise SystemExit(f"condition counts {dict(counts)}")

    run_dir = REPO_ROOT / "artifacts/runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / "pilot_outputs.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in result["outputs"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "git_commit": prov["git_commit"],
        "working_tree_clean": True,
        "generator_sha256": prov["generator_sha256"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "do_sample": False,
        "max_new_tokens": MAX_NEW_TOKENS,
        "batch_size": BATCH_SIZE,
        "gpu_type": "L40S",
        "n_pilot_outputs": N_EXPECTED,
        "condition_counts": dict(counts),
        "pilot_prompt_text_sha256": prov["expected_pilot_prompt_text_sha256"],
        "neutral_prefix_token_id": prov["neutral_prefix_token_id"],
        "raw_outputs_path": str(raw_path.relative_to(REPO_ROOT)),
        "raw_outputs_sha256": _sha_bytes(raw_path.read_bytes()),
        "wall_seconds": result["wall_seconds"],
        "estimated_cost_usd": result["estimated_cost_usd"],
        "activations_collected": False,
        "probes_fit_or_scored": False,
        "locked_final_families_run": False,
    }
    (run_dir / "pilot_generation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: manifest[k] for k in (
        "run_id", "n_pilot_outputs", "condition_counts", "wall_seconds", "estimated_cost_usd"
    )}, indent=2))
