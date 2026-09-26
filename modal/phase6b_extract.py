"""Modal GPU: Phase 6B frozen-probe factorial extraction (L12 controlled-prefix k1 only).

Input per prompt: chat-formatted prompt ids + [12107]; last-token layer-12 residual.
GPU payload carries only an opaque row index and the unchanged prompt text (no
example ids, condition ids, factor labels, or targets as metadata). No generation.

Usage:

  uv run modal run modal/phase6b_extract.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

import modal

APP_NAME = "pre-output-physiology-phase6b-extract"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
LAYER = 12
CONTROLLED_PREFIX_TOKEN_ID = 12107
COSINE_MIN = 0.9999
L40S_USD_PER_HOUR = 1.95
N_EXPECTED = 2880
N_PREFLIGHT = 8

REPO_ROOT = Path(__file__).resolve().parents[1]

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
artifact_volume = modal.Volume.from_name("preoutput-phase5-artifacts", create_if_missing=True)
MODEL_CACHE_DIR = "/vol/hf_cache"
ART_DIR = "/vol/phase5_artifacts"


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def _collect_local_provenance() -> tuple[
    dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]
]:
    import yaml

    from pre_output_physiology import phase6b_freeze as fz

    dirty = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise SystemExit(f"STOP: working tree dirty:\n{dirty}")
    cfg = yaml.safe_load(fz.CONFIG_PATH.read_text(encoding="utf-8"))
    if cfg.get("status") != fz.STATUS_AUTHORIZED:
        raise SystemExit(f"unexpected status {cfg.get('status')}")
    if cfg.get("phase6a_outcome") != fz.PHASE6A_OUTCOME:
        raise SystemExit("phase6a_outcome must remain HOLD")
    auth = cfg.get("authorizations", {})
    if auth.get("activation_extraction_authorized") is not True:
        raise SystemExit("activation_extraction_authorized must be true")
    for key in (
        "final_generation_authorized",
        "probe_fitting_authorized",
        "causal_intervention_authorized",
        "pilot_generation_authorized",
    ):
        if auth.get(key) is not False:
            raise SystemExit(f"{key} must be false")
    prompts, _scenarios, hashes = fz.load_verified_corpus()
    probe_sha = fz.verify_probe_sha()
    payload = fz.label_free_payload(prompts)
    git_commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    prov = {
        "git_commit": git_commit,
        "working_tree_clean": True,
        "extractor_sha256": _sha_bytes(Path(__file__).read_bytes()),
        "probe_sha256": probe_sha,
        **hashes,
        "payload_sha256": _sha_bytes(
            "\n".join(json.dumps(r, sort_keys=True) for r in payload).encode()
        ),
        "layer": LAYER,
        "endpoint": "k1",
        "controlled_prefix_token_id": CONTROLLED_PREFIX_TOKEN_ID,
        "future_response_tokens_present": False,
        "condition_labels_in_payload": False,
    }
    return prov, prompts, payload


def _load_model(cache_dir: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=cache_dir, use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
    )
    model.to("cuda:0")
    model.eval()
    return tokenizer, model


def _register_hook(model, layer: int):
    captured: dict[str, Any] = {}

    def hook(_m, _inp, output):  # noqa: ANN001
        captured["act"] = output[0] if isinstance(output, tuple) else output

    handle = model.model.layers[layer].register_forward_hook(hook)
    return captured, handle


def _prompt_ids(tokenizer, prompt_text: str) -> list[int]:
    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return list(tokenizer.encode(formatted, add_special_tokens=False))


def _forward_k1(model, captured, prompt_ids: list[int]) -> np.ndarray:
    import torch

    ids_k1 = list(prompt_ids) + [CONTROLLED_PREFIX_TOKEN_ID]
    ids = torch.tensor([ids_k1], dtype=torch.long, device="cuda:0")
    captured.clear()
    model(input_ids=ids, attention_mask=torch.ones_like(ids))
    vec = captured["act"][0, -1].float().detach().cpu().numpy().astype(np.float32)
    if not np.isfinite(vec).all():
        raise RuntimeError("NaN/Inf activation")
    return vec


def _check_payload(rows: list[dict[str, Any]]) -> None:
    for r in rows:
        if set(r.keys()) != {"row_index", "prompt_text"}:
            raise RuntimeError(f"payload fields not label-free: {sorted(r.keys())}")


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 30,
    volumes={MODEL_CACHE_DIR: model_volume, ART_DIR: artifact_volume},
    memory=65536,
)
def run_repeatability_preflight(rows_jsonl: str, *, run_id: str) -> dict[str, Any]:
    import torch

    rows = [json.loads(line) for line in rows_jsonl.splitlines() if line.strip()]
    _check_payload(rows)
    sample = rows[:N_PREFLIGHT]
    t0 = time.time()
    tokenizer, model = _load_model(MODEL_CACHE_DIR)
    captured, handle = _register_hook(model, LAYER)
    cosines = []
    with torch.inference_mode():
        for row in sample:
            ids = _prompt_ids(tokenizer, row["prompt_text"])
            a1 = _forward_k1(model, captured, ids)
            a2 = _forward_k1(model, captured, ids)
            cosines.append(_cosine(a1, a2))
    handle.remove()
    wall_s = time.time() - t0
    min_cos = float(min(cosines))
    return {
        "run_id": run_id,
        "passed": min_cos >= COSINE_MIN,
        "cosine_min_required": COSINE_MIN,
        "repeatability_min_cosine": min_cos,
        "cosines": cosines,
        "n_samples": len(sample),
        "wall_seconds": wall_s,
        "estimated_cost_usd": (wall_s / 3600.0) * L40S_USD_PER_HOUR,
    }


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 120,
    volumes={MODEL_CACHE_DIR: model_volume, ART_DIR: artifact_volume},
    memory=65536,
)
def extract_l12_k1(rows_jsonl: str, *, run_id: str, preflight_passed: bool) -> dict[str, Any]:
    import torch
    from safetensors.numpy import save_file

    if not preflight_passed:
        raise RuntimeError("STOP: preflight failed")
    rows = [json.loads(line) for line in rows_jsonl.splitlines() if line.strip()]
    _check_payload(rows)
    if len(rows) != N_EXPECTED:
        raise ValueError(f"expected {N_EXPECTED}, got {len(rows)}")
    if [r["row_index"] for r in rows] != list(range(N_EXPECTED)):
        raise ValueError("row_index not contiguous")

    t0 = time.time()
    tokenizer, model = _load_model(MODEL_CACHE_DIR)
    model_load_s = time.time() - t0
    captured, handle = _register_hook(model, LAYER)
    acts = np.zeros((len(rows), int(model.config.hidden_size)), dtype=np.float32)
    n_prompt_tokens = []
    gen_t0 = time.time()
    with torch.inference_mode():
        for i, row in enumerate(rows):
            ids = _prompt_ids(tokenizer, row["prompt_text"])
            acts[i] = _forward_k1(model, captured, ids)
            n_prompt_tokens.append(len(ids))
            if (i + 1) % 250 == 0:
                print(f"extracted {i + 1}/{len(rows)}")
    handle.remove()
    extract_s = time.time() - gen_t0
    wall_s = time.time() - t0
    if not np.isfinite(acts).all():
        raise RuntimeError("non-finite activations")

    out_dir = Path(ART_DIR) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    act_path = out_dir / "activations_l12_k1.safetensors"
    save_file(
        {
            "activations_l12_k1": acts,
            "row_index": np.arange(len(rows), dtype=np.int32),
            "n_prompt_tokens": np.asarray(n_prompt_tokens, dtype=np.int32),
            "layer": np.asarray([LAYER], dtype=np.int32),
            "k1_prefix_token_id": np.asarray([CONTROLLED_PREFIX_TOKEN_ID], dtype=np.int32),
        },
        str(act_path),
    )
    act_sha = _sha_bytes(act_path.read_bytes())
    artifact_volume.commit()
    return {
        "run_id": run_id,
        "n_rows": len(rows),
        "hidden_size": int(acts.shape[1]),
        "activations_sha256": act_sha,
        "wall_seconds": wall_s,
        "model_load_seconds": model_load_s,
        "extraction_seconds": extract_s,
        "estimated_cost_usd": (wall_s / 3600.0) * L40S_USD_PER_HOUR,
        "dtype_compute": "bfloat16",
        "dtype_storage": "float32",
        "batch_size": 1,
        "quantization": None,
    }


@app.local_entrypoint()
def main() -> None:
    prov, prompts, payload = _collect_local_provenance()
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"phase6b_extract_{ts}_{prov['git_commit'][:8]}"
    rows_jsonl = "\n".join(json.dumps(r, sort_keys=True) for r in payload)
    local_out = REPO_ROOT / "artifacts/runs" / run_id
    local_out.mkdir(parents=True, exist_ok=True)
    (local_out / "row_index_map.json").write_text(
        json.dumps(
            {
                "row_index_to_example_id": [r["example_id"] for r in prompts],
                "note": "Local-only mapping; never sent to the GPU.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Launching Phase 6B L12/k1 preflight run_id={run_id}")
    preflight = run_repeatability_preflight.remote(rows_jsonl, run_id=f"{run_id}_preflight")
    (local_out / "preflight_repeatability.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: preflight[k] for k in ("passed", "repeatability_min_cosine")}))
    if not preflight["passed"]:
        raise SystemExit(
            f"STOP: preflight min cosine {preflight['repeatability_min_cosine']} < {COSINE_MIN}"
        )

    print(f"Launching Phase 6B L12/k1 extraction n={len(payload)}")
    result = extract_l12_k1.remote(rows_jsonl, run_id=run_id, preflight_passed=True)
    subprocess.run(
        ["modal", "volume", "get", "preoutput-phase5-artifacts", run_id, str(local_out), "--force"],
        check=True,
        cwd=str(REPO_ROOT),
    )
    nested = local_out / run_id
    if nested.is_dir():
        for p in nested.iterdir():
            dest = local_out / p.name
            if dest.exists():
                shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
            shutil.move(str(p), str(dest))
        nested.rmdir()
    act_path = local_out / "activations_l12_k1.safetensors"
    if _sha_bytes(act_path.read_bytes()) != result["activations_sha256"]:
        raise SystemExit("activation sha mismatch")

    manifest = {
        **result,
        **prov,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "preflight_repeatability_min_cosine": preflight["repeatability_min_cosine"],
        "preflight_passed": True,
        "preflight_wall_seconds": preflight["wall_seconds"],
        "preflight_estimated_cost_usd": preflight["estimated_cost_usd"],
        "total_estimated_cost_usd": preflight["estimated_cost_usd"] + result["estimated_cost_usd"],
        "activations_path_gitignored": str(act_path.relative_to(REPO_ROOT)),
        "generation_performed": False,
        "probe_retrained": False,
        "probe_recalibrated": False,
        "layer_reselected": False,
        "causal_interventions_performed": False,
    }
    (local_out / "extraction_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in (
                    "run_id",
                    "n_rows",
                    "wall_seconds",
                    "total_estimated_cost_usd",
                    "activations_sha256",
                    "preflight_repeatability_min_cosine",
                )
            },
            indent=2,
        )
    )
