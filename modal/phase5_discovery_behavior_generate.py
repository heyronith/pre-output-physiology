"""Modal GPU: Phase 5B discovery-family behavior generation (no activations).

Generates greedy outputs for six discovery families only (1920 prompts).
Locked families are never loaded. Clean tree + authorized status required.

Usage:

  uv run modal run modal/phase5_discovery_behavior_generate.py
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

APP_NAME = "pre-output-physiology-phase5b-discovery"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
MAX_NEW_TOKENS = 32
BATCH_SIZE = 1
L40S_USD_PER_HOUR = 1.95
N_EXPECTED = 1920
PROMPT_TEMPLATE_REVISION = 2
EXPECTED_FINAL_PROMPT_TEXT_SHA256 = (
    "cdf2003ac0d568eee28a0feffa7b07e0ef208e637de40665aa9a5b1e8bc2920e"
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FINAL_PROMPTS_PATH = REPO_ROOT / "data/processed/phase5_design/final_candidate_prompts.jsonl"
CONFIG_PATH = REPO_ROOT / "configs/experiments/phase5_strategic_discovery.yaml"
SPLIT_PATH = REPO_ROOT / "artifacts/phase5b_discovery_split/family_split.json"
PRIMARY_PATH = (
    REPO_ROOT / "artifacts/phase5b_discovery_split/primary_all_pair_populations.json"
)
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
    if cfg.get("status") != "phase5b_discovery_behavior_authorized":
        raise SystemExit(f"unexpected status {cfg.get('status')}")
    auth = cfg.get("authorizations", {})
    if auth.get("discovery_generation_authorized") is not True:
        raise SystemExit("discovery_generation_authorized must be true")
    if auth.get("locked_final_generation_authorized") is not False:
        raise SystemExit("locked generation must remain unauthorized")
    if auth.get("activation_extraction_authorized") is not False:
        raise SystemExit("activations must remain unauthorized")
    if auth.get("probe_fitting_authorized") is not False:
        raise SystemExit("probe fitting must remain unauthorized")
    if auth.get("probe_scoring_authorized") is not False:
        raise SystemExit("probe scoring must remain unauthorized")
    if auth.get("causal_intervention_authorized") is not False:
        raise SystemExit("causal must remain unauthorized")

    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    primary = json.loads(PRIMARY_PATH.read_text(encoding="utf-8"))
    if split.get("model_generation_performed") is not False:
        raise SystemExit("family_split must be pre-generation freeze")
    if primary["train"]["n_pairs"] != 640 or primary["validation"]["n_pairs"] != 320:
        raise SystemExit("primary all-pair populations size drift")
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    if matrix.get("final_prompt_text_sha256") != EXPECTED_FINAL_PROMPT_TEXT_SHA256:
        raise SystemExit("matrix prompt hash drift")
    if int(cfg.get("design", {}).get("prompt_template_revision", 0)) != 2:
        raise SystemExit("prompt_template_revision must remain 2")

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
        "prompt_template_revision": PROMPT_TEMPLATE_REVISION,
        "expected_final_prompt_text_sha256": EXPECTED_FINAL_PROMPT_TEXT_SHA256,
        "discovery_prompt_text_sha256": split["discovery_prompt_text_sha256"],
        "train_families": split["train_families"],
        "validation_families": split["validation_families"],
        "train_all_pair_sha256": primary["train"]["pair_ids_sha256"],
        "validation_all_pair_sha256": primary["validation"]["pair_ids_sha256"],
        "neutral_prefix_token_id": matrix["neutral_prefix_token_id"],
        "activations_authorized": False,
        "probe_fitting_authorized": False,
        "locked_families_run": False,
    }


def _load_discovery_rows(provenance: dict[str, Any]) -> list[dict[str, Any]]:
    train = set(provenance["train_families"])
    val = set(provenance["validation_families"])
    allowed = train | val
    locked = {"harbor_dock_slip", "trail_marker_post"}
    rows: list[dict[str, Any]] = []
    with FINAL_PROMPTS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("pool") != "discovery" or row.get("split") != "final":
                continue
            if row["family"] in locked:
                raise SystemExit(f"locked family in discovery load: {row['family']}")
            if row["family"] not in allowed:
                raise SystemExit(f"unexpected discovery family: {row['family']}")
            if row.get("prompt_template_revision") != PROMPT_TEMPLATE_REVISION:
                raise SystemExit("prompt revision drift")
            rows.append(row)
    if len(rows) != N_EXPECTED:
        raise SystemExit(f"expected {N_EXPECTED} discovery prompts, got {len(rows)}")
    digest = _sha_prompt_texts(rows)
    if digest != provenance["discovery_prompt_text_sha256"]:
        raise SystemExit(f"discovery prompt hash mismatch: {digest}")
    # Also verify full final corpus hash unchanged.
    all_rows = []
    with FINAL_PROMPTS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                all_rows.append(json.loads(line))
    if _sha_prompt_texts(all_rows) != EXPECTED_FINAL_PROMPT_TEXT_SHA256:
        raise SystemExit("full final prompt corpus hash changed")
    n_s2 = sum(1 for r in rows if r["condition_id"] == "S2_strategic_honesty")
    n_s3 = sum(1 for r in rows if r["condition_id"] == "S3_strategic_deception")
    if n_s2 != 960 or n_s3 != 960:
        raise SystemExit(f"condition counts {n_s2}/{n_s3}")
    return rows


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 120,
    volumes={MODEL_CACHE_DIR: model_volume},
    memory=65536,
)
def generate_discovery(
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
    locked = {"harbor_dock_slip", "trail_marker_post"}
    if any(r["family"] in locked for r in rows):
        raise ValueError("locked family present in remote payload")

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
                    "split_role": (
                        "train"
                        if row["family"] in provenance["train_families"]
                        else "validation"
                    ),
                    "pool": "discovery",
                    "split": "final",
                    "record_state": row["record_state"],
                    "alternate_state": row["alternate_state"],
                    "objective_target": row["objective_target"],
                    "generated_token_ids": new_ids,
                    "first_generated_token_id": new_ids[0] if new_ids else None,
                    "n_generated_tokens": len(new_ids),
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
    rows = _load_discovery_rows(prov)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"phase5b_discovery_{ts}_{prov['git_commit'][:8]}"
    # Stable order: by example_id for reproducibility of write order.
    rows = sorted(rows, key=lambda r: r["example_id"])
    prompts_jsonl = "\n".join(json.dumps(r, sort_keys=True) for r in rows)
    print(f"Launching Phase 5B discovery run_id={run_id} n={len(rows)}")
    result = generate_discovery.remote(prompts_jsonl, provenance=prov, run_id=run_id)
    if result["n_outputs"] != N_EXPECTED:
        raise SystemExit(f"expected {N_EXPECTED}, got {result['n_outputs']}")
    counts = Counter(r["condition_id"] for r in result["outputs"])
    if counts.get("S2_strategic_honesty") != 960 or counts.get("S3_strategic_deception") != 960:
        raise SystemExit(f"condition counts {dict(counts)}")
    if any(r["family"] in {"harbor_dock_slip", "trail_marker_post"} for r in result["outputs"]):
        raise SystemExit("locked family appeared in outputs")
    role_counts = Counter(r["split_role"] for r in result["outputs"])
    if role_counts.get("train") != 1280 or role_counts.get("validation") != 640:
        raise SystemExit(f"split_role counts {dict(role_counts)}")

    run_dir = REPO_ROOT / "artifacts/runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / "discovery_outputs.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in sorted(result["outputs"], key=lambda r: r["example_id"]):
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
        "n_discovery_outputs": N_EXPECTED,
        "condition_counts": dict(counts),
        "split_role_counts": dict(role_counts),
        "train_families": prov["train_families"],
        "validation_families": prov["validation_families"],
        "final_prompt_text_sha256": EXPECTED_FINAL_PROMPT_TEXT_SHA256,
        "discovery_prompt_text_sha256": prov["discovery_prompt_text_sha256"],
        "prompt_template_revision": PROMPT_TEMPLATE_REVISION,
        "neutral_prefix_token_id": prov["neutral_prefix_token_id"],
        "raw_outputs_path": str(raw_path.relative_to(REPO_ROOT)),
        "raw_outputs_sha256": _sha_bytes(raw_path.read_bytes()),
        "wall_seconds": result["wall_seconds"],
        "generation_seconds": result["generation_seconds"],
        "model_load_seconds": result["model_load_seconds"],
        "estimated_cost_usd": result["estimated_cost_usd"],
        "activations_collected": False,
        "probes_fit_or_scored": False,
        "locked_final_families_run": False,
    }
    (run_dir / "discovery_generation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in (
                    "run_id",
                    "n_discovery_outputs",
                    "condition_counts",
                    "split_role_counts",
                    "wall_seconds",
                    "estimated_cost_usd",
                )
            },
            indent=2,
        )
    )
