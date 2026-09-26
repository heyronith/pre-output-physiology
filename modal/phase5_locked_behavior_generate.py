"""Modal GPU: Phase 5D locked-family behavior generation (after primary physiology freeze).

Usage (requires primary physiology artifact already written):

  uv run modal run modal/phase5_locked_behavior_generate.py
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

APP_NAME = "pre-output-physiology-phase5d-locked-behavior"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
MAX_NEW_TOKENS = 32
N_EXPECTED = 640
L40S_USD_PER_HOUR = 1.95
PROMPT_TEMPLATE_REVISION = 2
EXPECTED_LOCKED_PROMPT_TEXT_SHA256 = (
    "3a0ac9929e0971fd333bcc3d5a17b8b5b9d8b81dd6a75a8d5fe437dcd85a5097"
)
LOCKED = {"harbor_dock_slip", "trail_marker_post"}

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/experiments/phase5_strategic_discovery.yaml"
FINAL_PROMPTS_PATH = REPO_ROOT / "data/processed/phase5_design/final_candidate_prompts.jsonl"
PRIMARY_PATH = (
    REPO_ROOT / "artifacts/phase5d_locked_test/locked_primary_physiology.json"
)

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


def _sha_prompt_texts(rows: list[dict[str, Any]]) -> str:
    ordered = sorted(rows, key=lambda r: r["example_id"])
    payload = "\n".join(f"{r['example_id']}\t{r['prompt_text']}" for r in ordered)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _collect_local_provenance() -> dict[str, Any]:
    dirty = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise SystemExit(f"STOP: working tree dirty:\n{dirty}")
    if not PRIMARY_PATH.is_file():
        raise SystemExit("STOP: primary locked physiology must be frozen first")
    primary = json.loads(PRIMARY_PATH.read_text(encoding="utf-8"))
    if primary.get("behavior_conditioned") is not False:
        raise SystemExit("primary must not be behavior-conditioned")
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if cfg.get("status") != "phase5d_locked_test_authorized":
        raise SystemExit(f"unexpected status {cfg.get('status')}")
    auth = cfg.get("authorizations", {})
    if auth.get("locked_final_generation_authorized") is not True:
        raise SystemExit("locked generation must be authorized")
    if auth.get("causal_intervention_authorized") is not False:
        raise SystemExit("causal must remain false")
    git_commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    return {
        "git_commit": git_commit,
        "working_tree_clean": True,
        "generator_sha256": _sha_bytes(Path(__file__).read_bytes()),
        "primary_physiology_path": str(PRIMARY_PATH.relative_to(REPO_ROOT)),
        "primary_created_at": primary.get("created_at"),
        "locked_prompt_text_sha256": EXPECTED_LOCKED_PROMPT_TEXT_SHA256,
        "prompt_template_revision": PROMPT_TEMPLATE_REVISION,
        "causal_intervention_authorized": False,
    }


def _load_locked_rows() -> list[dict[str, Any]]:
    rows = []
    with FINAL_PROMPTS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("pool") != "locked" or row.get("split") != "final":
                continue
            if row["family"] not in LOCKED:
                raise SystemExit("unexpected family")
            if row.get("prompt_template_revision") != PROMPT_TEMPLATE_REVISION:
                raise SystemExit("revision drift")
            rows.append(row)
    if len(rows) != N_EXPECTED:
        raise SystemExit(f"expected {N_EXPECTED}")
    if _sha_prompt_texts(rows) != EXPECTED_LOCKED_PROMPT_TEXT_SHA256:
        raise SystemExit("locked prompt hash mismatch")
    return sorted(rows, key=lambda r: r["example_id"])


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 120,
    volumes={MODEL_CACHE_DIR: model_volume},
    memory=65536,
)
def generate_locked(
    prompts_jsonl: str,
    *,
    provenance: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(line) for line in prompts_jsonl.splitlines() if line.strip()]
    if len(rows) != N_EXPECTED:
        raise ValueError(f"expected {N_EXPECTED}")
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
            new_ids = [int(x) for x in gen[0].tolist()[prompt_len:]]
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
            outputs.append(
                {
                    "example_id": row["example_id"],
                    "base_scenario_id": row["base_scenario_id"],
                    "condition_id": row["condition_id"],
                    "family": row["family"],
                    "pool": "locked",
                    "split": "final",
                    "record_state": row["record_state"],
                    "alternate_state": row["alternate_state"],
                    "objective_target": row["objective_target"],
                    "generated_token_ids": new_ids,
                    "first_generated_token_id": new_ids[0] if new_ids else None,
                    "output_text": text,
                    "prompt_template_revision": PROMPT_TEMPLATE_REVISION,
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
        "wall_seconds": wall_s,
        "generation_seconds": gen_s,
        "model_load_seconds": model_load_s,
        "estimated_cost_usd": (wall_s / 3600.0) * L40S_USD_PER_HOUR,
        "provenance": provenance,
    }


@app.local_entrypoint()
def main() -> None:
    from collections import Counter

    prov = _collect_local_provenance()
    rows = _load_locked_rows()
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"phase5d_locked_behavior_{ts}_{prov['git_commit'][:8]}"
    prompts_jsonl = "\n".join(json.dumps(r, sort_keys=True) for r in rows)
    print(f"Launching locked behavior run_id={run_id} n={len(rows)}")
    result = generate_locked.remote(prompts_jsonl, provenance=prov, run_id=run_id)
    if result["n_outputs"] != N_EXPECTED:
        raise SystemExit("output count mismatch")
    counts = Counter(r["condition_id"] for r in result["outputs"])
    if counts.get("S2_strategic_honesty") != 320 or counts.get("S3_strategic_deception") != 320:
        raise SystemExit(f"condition counts {dict(counts)}")

    run_dir = REPO_ROOT / "artifacts/runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    raw = run_dir / "locked_outputs.jsonl"
    with raw.open("w", encoding="utf-8") as handle:
        for row in sorted(result["outputs"], key=lambda r: r["example_id"]):
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "git_commit": prov["git_commit"],
        "working_tree_clean": True,
        "generator_sha256": prov["generator_sha256"],
        "n_outputs": N_EXPECTED,
        "condition_counts": dict(counts),
        "locked_prompt_text_sha256": EXPECTED_LOCKED_PROMPT_TEXT_SHA256,
        "prompt_template_revision": PROMPT_TEMPLATE_REVISION,
        "raw_outputs_path": str(raw.relative_to(REPO_ROOT)),
        "raw_outputs_sha256": _sha_bytes(raw.read_bytes()),
        "wall_seconds": result["wall_seconds"],
        "estimated_cost_usd": result["estimated_cost_usd"],
        "primary_physiology_preceded_behavior": True,
        "primary_physiology_path": prov["primary_physiology_path"],
        "causal_interventions_performed": False,
    }
    (run_dir / "locked_behavior_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in (
                    "run_id",
                    "n_outputs",
                    "wall_seconds",
                    "estimated_cost_usd",
                )
            },
            indent=2,
        )
    )
