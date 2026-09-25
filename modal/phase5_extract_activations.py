"""Modal GPU: Phase 5C discovery activation extraction (k0 + controlled k1).

Layers [0,4,8,12,16,20,24,28,31]. Discovery families only. Label-free payload.
Repeatability preflight (min cosine >= 0.9999) required before full extract.

Usage (clean tree required):

  uv run modal run modal/phase5_extract_activations.py
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
import yaml

import modal

APP_NAME = "pre-output-physiology-phase5c-extract"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31]
CONTROLLED_PREFIX_TOKEN_ID = 12107
COSINE_MIN = 0.9999
L40S_USD_PER_HOUR = 1.95
N_EXPECTED = 1920
BATCH_SIZE = 1
COMPUTE_DTYPE = "bfloat16"
ACTIVATION_STORAGE_DTYPE = "float32"
EXPECTED_FINAL_PROMPT_TEXT_SHA256 = (
    "cdf2003ac0d568eee28a0feffa7b07e0ef208e637de40665aa9a5b1e8bc2920e"
)
EXPECTED_DISCOVERY_PROMPT_TEXT_SHA256 = (
    "2eb22deeeb520d712fee000644fa958f35137e029a08ba0aa4b58b89327fe464"
)
EXPECTED_TRAIN_ALL_PAIR_SHA256 = (
    "e2b5a559399025dacbeacfb4e50ef16d1745aa1a1b3f336775749ae3202b7641"
)
EXPECTED_VAL_ALL_PAIR_SHA256 = (
    "6b9f6385cdcd651d86fa27483deab6842cbbdc96e03bc9104fb628c50eff07ab"
)
LOCKED = {"harbor_dock_slip", "trail_marker_post"}

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/experiments/phase5_strategic_discovery.yaml"
FINAL_PROMPTS_PATH = REPO_ROOT / "data/processed/phase5_design/final_candidate_prompts.jsonl"
SPLIT_PATH = REPO_ROOT / "artifacts/phase5b_discovery_split/family_split.json"
PRIMARY_PATH = (
    REPO_ROOT / "artifacts/phase5b_discovery_split/primary_all_pair_populations.json"
)
CONTRACT_PATH = (
    REPO_ROOT / "artifacts/phase5c_physiology_freeze/physiology_contract.json"
)
ANALYSIS_SCRIPT = REPO_ROOT / "scripts/analyze_phase5c_discovery.py"

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
artifact_volume = modal.Volume.from_name(
    "preoutput-phase5-artifacts", create_if_missing=True
)
MODEL_CACHE_DIR = "/vol/hf_cache"
ART_DIR = "/vol/phase5_artifacts"


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_text(text: str) -> str:
    return _sha_bytes(text.encode("utf-8"))


def _sha_prompt_texts(rows: list[dict[str, Any]]) -> str:
    ordered = sorted(rows, key=lambda r: r["example_id"])
    payload = "\n".join(f"{r['example_id']}\t{r['prompt_text']}" for r in ordered)
    return _sha_text(payload)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def _collect_local_provenance() -> dict[str, Any]:
    dirty = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise SystemExit(f"STOP: working tree dirty:\n{dirty}")
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if cfg.get("status") != "phase5c_discovery_physiology_authorized":
        raise SystemExit(f"unexpected status {cfg.get('status')}")
    auth = cfg.get("authorizations", {})
    if auth.get("activation_extraction_authorized") is not True:
        raise SystemExit("activation_extraction_authorized must be true")
    if auth.get("probe_fitting_authorized") is not True:
        raise SystemExit("probe_fitting_authorized must be true")
    if auth.get("locked_final_generation_authorized") is not False:
        raise SystemExit("locked generation must remain false")
    if auth.get("causal_intervention_authorized") is not False:
        raise SystemExit("causal must remain false")
    if not CONTRACT_PATH.is_file():
        raise SystemExit("missing physiology contract freeze")
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    primary = json.loads(PRIMARY_PATH.read_text(encoding="utf-8"))
    if primary["train"]["pair_ids_sha256"] != EXPECTED_TRAIN_ALL_PAIR_SHA256:
        raise SystemExit("train all-pair hash drift")
    if primary["validation"]["pair_ids_sha256"] != EXPECTED_VAL_ALL_PAIR_SHA256:
        raise SystemExit("validation all-pair hash drift")
    if split["discovery_prompt_text_sha256"] != EXPECTED_DISCOVERY_PROMPT_TEXT_SHA256:
        raise SystemExit("discovery prompt hash drift")
    git_commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    extractor = Path(__file__).resolve()
    return {
        "git_commit": git_commit,
        "working_tree_clean": True,
        "extractor_sha256": _sha_bytes(extractor.read_bytes()),
        "analysis_script_sha256": _sha_bytes(ANALYSIS_SCRIPT.read_bytes()),
        "contract_sha256": _sha_bytes(CONTRACT_PATH.read_bytes()),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "compute_dtype": COMPUTE_DTYPE,
        "activation_storage_dtype": ACTIVATION_STORAGE_DTYPE,
        "extraction_mode": "truncated_prefix_single_example",
        "batch_size": BATCH_SIZE,
        "layers": LAYERS,
        "controlled_prefix_token_id": CONTROLLED_PREFIX_TOKEN_ID,
        "future_response_tokens_present": False,
        "condition_labels_in_extractor": False,
        "train_families": split["train_families"],
        "validation_families": split["validation_families"],
        "train_all_pair_sha256": EXPECTED_TRAIN_ALL_PAIR_SHA256,
        "validation_all_pair_sha256": EXPECTED_VAL_ALL_PAIR_SHA256,
        "discovery_prompt_text_sha256": EXPECTED_DISCOVERY_PROMPT_TEXT_SHA256,
        "final_prompt_text_sha256": EXPECTED_FINAL_PROMPT_TEXT_SHA256,
        "locked_families_run": False,
        "causal_intervention_authorized": False,
    }


def _load_discovery_label_free_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with FINAL_PROMPTS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("pool") != "discovery" or row.get("split") != "final":
                continue
            if row["family"] in LOCKED:
                raise SystemExit(f"locked family in discovery load: {row['family']}")
            # Label-free extract payload (no condition / objective_target).
            rows.append(
                {
                    "example_id": row["example_id"],
                    "base_scenario_id": row["base_scenario_id"],
                    "family": row["family"],
                    "prompt_text": row["prompt_text"],
                    "record_state": row["record_state"],
                    "alternate_state": row["alternate_state"],
                }
            )
    if len(rows) != N_EXPECTED:
        raise SystemExit(f"expected {N_EXPECTED}, got {len(rows)}")
    # Verify full final corpus hash unchanged via separate full load.
    all_rows = []
    with FINAL_PROMPTS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                all_rows.append(json.loads(line))
    if _sha_prompt_texts(all_rows) != EXPECTED_FINAL_PROMPT_TEXT_SHA256:
        raise SystemExit("final prompt corpus hash changed")
    disc = [r for r in all_rows if r.get("pool") == "discovery" and r.get("split") == "final"]
    if _sha_prompt_texts(disc) != EXPECTED_DISCOVERY_PROMPT_TEXT_SHA256:
        raise SystemExit("discovery prompt hash mismatch")
    for r in rows:
        for banned in ("condition_id", "objective_target", "label", "binary_label"):
            if banned in r:
                raise SystemExit(f"label leaked: {banned}")
    return sorted(rows, key=lambda r: r["example_id"])


def _load_model(cache_dir: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=cache_dir, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
    )
    model.to("cuda:0")
    model.eval()
    if model.dtype != torch.bfloat16:
        raise RuntimeError(f"expected bf16, got {model.dtype}")
    return tokenizer, model


def _register_hooks(model, layer_ids: list[int]):
    import torch

    captured: dict[int, torch.Tensor] = {}
    handles = []
    for li in layer_ids:

        def make_hook(layer_i: int):
            def hook(_m, _inp, output):  # noqa: ANN001
                act = output[0] if isinstance(output, tuple) else output
                captured[layer_i] = act

            return hook

        handles.append(model.model.layers[li].register_forward_hook(make_hook(li)))
    return captured, handles


def _forward_final_token(model, captured, input_ids_list: list[int], layers: list[int]):
    import torch

    ids = torch.tensor([input_ids_list], dtype=torch.long, device="cuda:0")
    mask = torch.ones_like(ids)
    captured.clear()
    out = model(input_ids=ids, attention_mask=mask)
    if set(captured) != set(layers):
        raise RuntimeError(f"hook miss: got {sorted(captured)} want {layers}")
    acts = []
    for li in layers:
        vec = captured[li][0, -1].float().detach().cpu().numpy()
        if not np.isfinite(vec).all():
            raise RuntimeError("NaN/Inf in activation")
        acts.append(vec.astype(np.float32))
    logits = out.logits[0, -1].float()
    return np.stack(acts, axis=0).astype(np.float32), logits


def _state_first_token_id(tokenizer, state: str) -> int:
    ids = tokenizer.encode(" " + state, add_special_tokens=False)
    if not ids:
        raise RuntimeError(f"empty tokenization for state {state!r}")
    return int(ids[0])


def _logit_features(logits, tokenizer, record_state: str, alternate_state: str) -> np.ndarray:
    import torch

    id_r = _state_first_token_id(tokenizer, record_state)
    id_a = _state_first_token_id(tokenizer, alternate_state)
    probs = torch.softmax(logits, dim=-1)
    lr = float(logits[id_r].item())
    la = float(logits[id_a].item())
    pr = float(probs[id_r].item())
    pa = float(probs[id_a].item())
    return np.asarray([lr, la, lr - la, pr, pa, pr - pa], dtype=np.float32)


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 40,
    volumes={MODEL_CACHE_DIR: model_volume, ART_DIR: artifact_volume},
    memory=65536,
)
def run_repeatability_preflight(
    rows_jsonl: str,
    *,
    provenance: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    """Repeat truncated k0/k1 forwards twice; require min cosine >= 0.9999."""
    import torch

    rows = [json.loads(line) for line in rows_jsonl.splitlines() if line.strip()]
    # Fixed sample of 8 discovery prompts for preflight.
    sample = rows[:8]
    t0 = time.time()
    tokenizer, model = _load_model(MODEL_CACHE_DIR)
    model_load_s = time.time() - t0
    test_layers = [0, 12, 31]
    captured, handles = _register_hooks(model, test_layers)
    cosines: list[float] = []
    per_example = []
    with torch.inference_mode():
        for row in sample:
            messages = [{"role": "user", "content": row["prompt_text"]}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = list(tokenizer.encode(formatted, add_special_tokens=False))
            ids_k0 = list(prompt_ids)
            ids_k1 = list(prompt_ids) + [CONTROLLED_PREFIX_TOKEN_ID]
            a1, _ = _forward_final_token(model, captured, ids_k0, test_layers)
            a2, _ = _forward_final_token(model, captured, ids_k0, test_layers)
            b1, _ = _forward_final_token(model, captured, ids_k1, test_layers)
            b2, _ = _forward_final_token(model, captured, ids_k1, test_layers)
            row_cos = []
            for li in range(len(test_layers)):
                c0 = _cosine(a1[li], a2[li])
                c1 = _cosine(b1[li], b2[li])
                cosines.extend([c0, c1])
                row_cos.append({"layer": test_layers[li], "k0": c0, "k1": c1})
            per_example.append({"example_id": row["example_id"], "cosines": row_cos})
    for h in handles:
        h.remove()
    min_cos = float(min(cosines)) if cosines else 0.0
    passed = min_cos >= COSINE_MIN
    wall_s = time.time() - t0
    model_volume.commit()
    return {
        "run_id": run_id,
        "passed": passed,
        "cosine_min_required": COSINE_MIN,
        "repeatability_min_cosine": min_cos,
        "n_comparisons": len(cosines),
        "n_samples": len(sample),
        "test_layers": test_layers,
        "per_example": per_example,
        "model_load_seconds": model_load_s,
        "wall_seconds": wall_s,
        "estimated_cost_usd": (wall_s / 3600.0) * L40S_USD_PER_HOUR,
        "provenance": provenance,
    }


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 240,
    volumes={MODEL_CACHE_DIR: model_volume, ART_DIR: artifact_volume},
    memory=65536,
)
def extract_discovery_activations(
    rows_jsonl: str,
    *,
    provenance: dict[str, Any],
    run_id: str,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from safetensors.numpy import save_file

    if not preflight.get("passed"):
        raise RuntimeError("STOP: repeatability preflight failed")
    rows = [json.loads(line) for line in rows_jsonl.splitlines() if line.strip()]
    if len(rows) != N_EXPECTED:
        raise ValueError(f"expected {N_EXPECTED}, got {len(rows)}")
    if any(r["family"] in LOCKED for r in rows):
        raise RuntimeError("locked family in extract payload")
    for r in rows:
        for banned in ("condition_id", "objective_target", "label", "binary_label"):
            if banned in r:
                raise RuntimeError(f"label in extract payload: {banned}")

    t0 = time.time()
    tokenizer, model = _load_model(MODEL_CACHE_DIR)
    model_load_s = time.time() - t0
    captured, handles = _register_hooks(model, LAYERS)
    hidden = int(model.config.hidden_size)
    n_layers = len(LAYERS)
    acts_k0 = np.zeros((len(rows), n_layers, hidden), dtype=np.float32)
    acts_k1 = np.zeros((len(rows), n_layers, hidden), dtype=np.float32)
    logit_feats = np.zeros((len(rows), 6), dtype=np.float32)
    example_ids: list[str] = []
    n_prompt_tokens: list[int] = []
    formatted_shas: list[str] = []

    gen_t0 = time.time()
    with torch.inference_mode():
        for i, row in enumerate(rows):
            messages = [{"role": "user", "content": row["prompt_text"]}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = list(tokenizer.encode(formatted, add_special_tokens=False))
            ids_k0 = list(prompt_ids)
            ids_k1 = list(prompt_ids) + [CONTROLLED_PREFIX_TOKEN_ID]
            if ids_k1[-1] != CONTROLLED_PREFIX_TOKEN_ID:
                raise RuntimeError("k1 prefix identity failure")
            if len(ids_k1) != len(ids_k0) + 1:
                raise RuntimeError("k1 length mismatch")
            a0, _ = _forward_final_token(model, captured, ids_k0, LAYERS)
            a1, logits = _forward_final_token(model, captured, ids_k1, LAYERS)
            acts_k0[i] = a0
            acts_k1[i] = a1
            logit_feats[i] = _logit_features(
                logits, tokenizer, row["record_state"], row["alternate_state"]
            )
            example_ids.append(row["example_id"])
            n_prompt_tokens.append(len(ids_k0))
            formatted_shas.append(_sha_text(formatted))
            if (i + 1) % 100 == 0:
                print(f"extracted {i + 1}/{len(rows)}")

    for h in handles:
        h.remove()
    extract_s = time.time() - gen_t0
    wall_s = time.time() - t0

    if not np.isfinite(acts_k0).all() or not np.isfinite(acts_k1).all():
        raise RuntimeError("non-finite activations")
    if not np.isfinite(logit_feats).all():
        raise RuntimeError("non-finite logit features")

    out_dir = Path(ART_DIR) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    act_path = out_dir / "activations_k0_k1.safetensors"
    save_file(
        {
            "activations_k0": acts_k0,
            "activations_k1": acts_k1,
            "logit_features_k1": logit_feats,
            "n_prompt_tokens": np.asarray(n_prompt_tokens, dtype=np.int32),
            "layers": np.asarray(LAYERS, dtype=np.int32),
            "k1_prefix_token_id": np.asarray([CONTROLLED_PREFIX_TOKEN_ID], dtype=np.int32),
        },
        str(act_path),
    )
    meta = {
        "example_ids": example_ids,
        "formatted_prompt_sha256": formatted_shas,
        "base_scenario_ids": [r["base_scenario_id"] for r in rows],
        "families": [r["family"] for r in rows],
    }
    (out_dir / "extraction_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    act_sha = _sha_bytes(act_path.read_bytes())
    artifact_volume.commit()
    model_volume.commit()

    return {
        "run_id": run_id,
        "n_rows": len(rows),
        "layers": LAYERS,
        "controlled_prefix_token_id": CONTROLLED_PREFIX_TOKEN_ID,
        "activations_volume_path": f"{run_id}/activations_k0_k1.safetensors",
        "meta_volume_path": f"{run_id}/extraction_meta.json",
        "activations_sha256": act_sha,
        "future_response_tokens_present": False,
        "condition_labels_present": False,
        "locked_families_present": False,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "compute_dtype": COMPUTE_DTYPE,
        "activation_storage_dtype": ACTIVATION_STORAGE_DTYPE,
        "batch_size": BATCH_SIZE,
        "extraction_mode": "truncated_prefix_single_example",
        "gpu_type": "L40S",
        "model_load_seconds": model_load_s,
        "extraction_seconds": extract_s,
        "wall_seconds": wall_s,
        "estimated_cost_usd": (wall_s / 3600.0) * L40S_USD_PER_HOUR,
        "preflight": {
            "passed": preflight["passed"],
            "repeatability_min_cosine": preflight["repeatability_min_cosine"],
        },
        "provenance": provenance,
        "causal_interventions_performed": False,
    }


@app.local_entrypoint()
def main() -> None:
    prov = _collect_local_provenance()
    rows = _load_discovery_label_free_rows()
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"phase5c_extract_{ts}_{prov['git_commit'][:8]}"
    rows_jsonl = "\n".join(json.dumps(r, sort_keys=True) for r in rows)

    print(f"Launching repeatability preflight run_id={run_id}")
    preflight = run_repeatability_preflight.remote(
        rows_jsonl, provenance=prov, run_id=f"{run_id}_preflight"
    )
    print(
        json.dumps(
            {
                "preflight_passed": preflight["passed"],
                "repeatability_min_cosine": preflight["repeatability_min_cosine"],
            },
            indent=2,
        )
    )
    if not preflight["passed"]:
        raise SystemExit(
            f"STOP: repeatability min cosine "
            f"{preflight['repeatability_min_cosine']} < {COSINE_MIN}"
        )

    print(f"Launching full discovery extract n={len(rows)}")
    result = extract_discovery_activations.remote(
        rows_jsonl, provenance=prov, run_id=run_id, preflight=preflight
    )

    local_out = REPO_ROOT / "artifacts/runs" / run_id
    local_out.mkdir(parents=True, exist_ok=True)
    (local_out / "preflight_repeatability.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Downloading volume artifacts {run_id} ...")
    subprocess.run(
        [
            "modal",
            "volume",
            "get",
            "preoutput-phase5-artifacts",
            run_id,
            str(local_out),
            "--force",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )
    nested = local_out / run_id
    if nested.is_dir():
        for p in nested.iterdir():
            dest = local_out / p.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(p), str(dest))
        nested.rmdir()

    act_path = local_out / "activations_k0_k1.safetensors"
    if not act_path.is_file():
        raise SystemExit("activations file missing after download")
    local_sha = _sha_bytes(act_path.read_bytes())
    if local_sha != result["activations_sha256"]:
        raise SystemExit("activation sha mismatch after download")

    manifest = {
        **{k: v for k, v in result.items() if k != "provenance"},
        "git_commit": prov["git_commit"],
        "working_tree_clean": True,
        "extractor_sha256": prov["extractor_sha256"],
        "analysis_script_sha256": prov["analysis_script_sha256"],
        "contract_sha256": prov["contract_sha256"],
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "preflight_repeatability_min_cosine": preflight["repeatability_min_cosine"],
        "preflight_passed": True,
        "activations_path_gitignored": str(act_path.relative_to(REPO_ROOT)),
        "train_all_pair_sha256": EXPECTED_TRAIN_ALL_PAIR_SHA256,
        "validation_all_pair_sha256": EXPECTED_VAL_ALL_PAIR_SHA256,
        "final_prompt_text_sha256": EXPECTED_FINAL_PROMPT_TEXT_SHA256,
        "discovery_prompt_text_sha256": EXPECTED_DISCOVERY_PROMPT_TEXT_SHA256,
        "locked_final_families_run": False,
        "probes_fit_on_gpu": False,
    }
    (local_out / "extraction_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "n_rows": result["n_rows"],
                "wall_seconds": result["wall_seconds"],
                "estimated_cost_usd": result["estimated_cost_usd"],
                "activations_sha256": result["activations_sha256"],
                "preflight_min_cosine": preflight["repeatability_min_cosine"],
            },
            indent=2,
        )
    )
