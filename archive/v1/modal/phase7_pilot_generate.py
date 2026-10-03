"""Modal GPU: Phase 7A policy-choice behavior-only pilot (no activations, no probe).

Usage (clean tree required):

  uv run modal run modal/phase7_pilot_generate.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

APP_NAME = "pre-output-physiology-phase7a-pilot"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
MAX_NEW_TOKENS = 32
BATCH_SIZE = 1
L40S_USD_PER_HOUR = 1.95
N_EXPECTED = 128
N_PER_CONTEXT = 32
CONTEXTS = ("inf_low__ver_low", "inf_low__ver_high", "inf_high__ver_low", "inf_high__ver_high")

REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT_PROMPTS_PATH = REPO_ROOT / "data/processed/phase7_design/pilot_candidate_prompts.jsonl"
CONFIG_PATH = REPO_ROOT / "configs/experiments/phase7_policy_choice.yaml"
MATRIX_PATH = REPO_ROOT / "artifacts/phase7a_design/condition_matrix.json"

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.4.1",
    "transformers==4.44.2",
    "accelerate==0.33.0",
    "safetensors==0.4.4",
    "numpy==1.26.4",
    "pyyaml==6.0.2",
    "huggingface_hub==0.24.6",
)

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name("preoutput-mistral-cache", create_if_missing=True)
MODEL_CACHE_DIR = "/vol/hf_cache"


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_prompt_texts(rows: list[dict[str, Any]]) -> str:
    ordered = sorted(rows, key=lambda r: r["example_id"])
    payload = "\n".join(f"{r['example_id']}\t{r['prompt_text']}" for r in ordered)
    return _sha_bytes(payload.encode("utf-8"))


def _collect_local_provenance() -> dict[str, Any]:
    import yaml

    dirty = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise SystemExit(f"STOP: working tree dirty:\n{dirty}")
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if cfg.get("status") != "phase7a_policy_choice_design_frozen_pilot_authorized":
        raise SystemExit(f"unexpected status {cfg.get('status')}")
    auth = cfg.get("authorizations", {})
    if auth.get("pilot_generation_authorized") is not True:
        raise SystemExit("pilot_generation_authorized must be true")
    for key in (
        "final_generation_authorized",
        "activation_extraction_authorized",
        "probe_fitting_authorized",
        "phase5_probe_scoring_authorized",
        "causal_intervention_authorized",
    ):
        if auth.get(key) is not False:
            raise SystemExit(f"{key} must remain false")
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    if matrix["pilot_prompt_text_sha256"] != cfg["design"]["pilot_prompt_text_sha256"]:
        raise SystemExit("config/matrix pilot hash mismatch")
    git_commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    return {
        "git_commit": git_commit,
        "generator_sha256": _sha_bytes(Path(__file__).read_bytes()),
        "expected_pilot_prompt_text_sha256": matrix["pilot_prompt_text_sha256"],
        "prompt_template_revision": int(matrix["prompt_template_revision"]),
    }


def _load_pilot_rows(expected_hash: str) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in PILOT_PROMPTS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != N_EXPECTED or any(r.get("split") != "pilot" for r in rows):
        raise SystemExit("pilot rows invalid")
    digest = _sha_prompt_texts(rows)
    if digest != expected_hash:
        raise SystemExit(f"pilot prompt hash mismatch: {digest}")
    return sorted(rows, key=lambda r: r["example_id"])


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 40,
    volumes={MODEL_CACHE_DIR: model_volume},
    memory=65536,
)
def generate_pilot(payload_jsonl: str, *, run_id: str) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(line) for line in payload_jsonl.splitlines() if line.strip()]
    if len(rows) != N_EXPECTED or any(set(r) != {"example_id", "prompt_text"} for r in rows):
        raise ValueError("payload invalid")
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
    outputs = []
    with torch.inference_mode():
        for row in rows:
            messages = [{"role": "user", "content": row["prompt_text"]}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            ids = tokenizer.encode(formatted, add_special_tokens=False)
            enc = torch.tensor([ids], dtype=torch.long, device="cuda:0")
            gen = model.generate(
                input_ids=enc,
                attention_mask=torch.ones_like(enc),
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            new_ids = [int(x) for x in gen[0].tolist()[len(ids) :]]
            outputs.append(
                {
                    "example_id": row["example_id"],
                    "generated_token_ids": new_ids,
                    "first_generated_token_id": new_ids[0] if new_ids else None,
                    "n_generated_tokens": len(new_ids),
                    "output_text": tokenizer.decode(new_ids, skip_special_tokens=True),
                }
            )
    wall_s = time.time() - t0
    model_volume.commit()
    return {
        "run_id": run_id,
        "outputs": outputs,
        "wall_seconds": wall_s,
        "estimated_cost_usd": (wall_s / 3600.0) * L40S_USD_PER_HOUR,
    }


@app.local_entrypoint()
def main() -> None:
    prov = _collect_local_provenance()
    rows = _load_pilot_rows(prov["expected_pilot_prompt_text_sha256"])
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"phase7a_pilot_r{prov['prompt_template_revision']}_{ts}_{prov['git_commit'][:8]}"
    payload = "\n".join(
        json.dumps({"example_id": r["example_id"], "prompt_text": r["prompt_text"]}) for r in rows
    )
    print(f"Launching Phase 7A pilot run_id={run_id} n={len(rows)}")
    result = generate_pilot.remote(payload, run_id=run_id)
    by_id = {o["example_id"]: o for o in result["outputs"]}
    if set(by_id) != {r["example_id"] for r in rows}:
        raise SystemExit("output ids mismatch")
    merged = []
    for r in rows:
        merged.append(
            {
                **{
                    k: r[k]
                    for k in (
                        "example_id",
                        "base_scenario_id",
                        "context_id",
                        "influence",
                        "verification",
                        "family",
                        "split",
                        "record_state",
                        "alternate_state",
                        "record_listed_first",
                        "prompt_template_revision",
                    )
                },
                **{k: v for k, v in by_id[r["example_id"]].items() if k != "example_id"},
                "model_revision": MODEL_REVISION,
                "activation_extracted": False,
                "probe_scored": False,
            }
        )
    counts = Counter(r["context_id"] for r in merged)
    if any(counts.get(c) != N_PER_CONTEXT for c in CONTEXTS):
        raise SystemExit(f"context counts {dict(counts)}")
    run_dir = REPO_ROOT / "artifacts/runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / "pilot_outputs.jsonl"
    raw_path.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in merged), encoding="utf-8"
    )
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
        "dtype": "bfloat16",
        "quantization": None,
        "gpu_type": "L40S",
        "n_pilot_outputs": len(merged),
        "context_counts": dict(counts),
        "pilot_prompt_text_sha256": prov["expected_pilot_prompt_text_sha256"],
        "prompt_template_revision": prov["prompt_template_revision"],
        "raw_outputs_path": str(raw_path.relative_to(REPO_ROOT)),
        "raw_outputs_sha256": _sha_bytes(raw_path.read_bytes()),
        "wall_seconds": result["wall_seconds"],
        "estimated_cost_usd": result["estimated_cost_usd"],
        "activations_collected": False,
        "probe_scored": False,
        "final_scenarios_run": False,
    }
    (run_dir / "pilot_generation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in ("run_id", "n_pilot_outputs", "wall_seconds", "estimated_cost_usd")
            },
            indent=2,
        )
    )
