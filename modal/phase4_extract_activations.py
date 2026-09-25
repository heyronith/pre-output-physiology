"""Modal GPU: Phase 4E activation extraction (L12 k0 + controlled-prefix k1).

Compatibility preflight reproduces canonical Phase-3B1 L12/k0 and L12/k1
activations on a fixed sample (cosine >= 0.9999) before any Phase-4 extract.

Extraction has no condition labels. Labels are joined only during local scoring.

Usage (clean tree required):

  uv run modal run modal/phase4_extract_activations.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import modal

APP_NAME = "pre-output-physiology-phase4e-extract"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
LAYER = 12
CONTROLLED_PREFIX_TOKEN_ID = 12107
COSINE_MIN = 0.9999
L40S_USD_PER_HOUR = 1.95
COMPUTE_DTYPE = "bfloat16"
ACTIVATION_STORAGE_DTYPE = "float32"
BATCH_SIZE = 1
CANONICAL_B1_RUN = "phase3b1_extract_20260925T154005Z_39505f42"
PHASE4D_RUN = "phase4d_final_20260925T174548Z_4d93730c"
FROZEN_PROBE_L12_K1 = "3bbf0c78e97d8ff14b028592e6e6ab3215651e973e0be1f004ce66e2b9b7aa65"
FROZEN_PROBE_L12_K0 = "dc14569a4e229cadc55fd3f6e6fef4e33f4eac4f58d00e83916238d5274f4f29"
# Fixed B1 sample for compatibility preflight (eligible for k>=1).
B1_COMPAT_EXAMPLE_IDS: tuple[str, ...] = (
    "92_8",
    "86_1",
    "51_4",
    "223_1",
    "364_2",
    "243_0",
    "185_6",
    "2_18",
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/experiments/phase4_specificity.yaml"
FINAL_PROMPTS_PATH = REPO_ROOT / "data/processed/phase4_design/final_candidate_prompts.jsonl"
FINAL_OUTPUTS_PATH = (
    REPO_ROOT / "artifacts/runs" / PHASE4D_RUN / "final_outputs.jsonl"
)
B1_RUN_DIR = REPO_ROOT / "artifacts/runs" / CANONICAL_B1_RUN
PHASE3_TRAIN_META = (
    REPO_ROOT / "data/processed/phase3_roleplay/phase3_train_metadata.jsonl"
)
PHASE3_VAL_META = (
    REPO_ROOT / "data/processed/phase3_roleplay/phase3_validation_metadata.jsonl"
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


def _sha_text(text: str) -> str:
    return _sha_bytes(text.encode("utf-8"))


def _sha_ids(ids: list[int]) -> str:
    return _sha_text(",".join(str(i) for i in ids))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def _collect_local_provenance() -> dict[str, Any]:
    dirty = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise SystemExit(
            "STOP: working tree dirty — commit before Modal launch:\n" f"{dirty}"
        )
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if cfg.get("status") != "phase4e_specificity_extraction_authorized":
        raise SystemExit(
            "status must be phase4e_specificity_extraction_authorized, "
            f"got {cfg.get('status')}"
        )
    auth = cfg.get("authorizations", {})
    if auth.get("activation_extraction_authorized") is not True:
        raise SystemExit("activation_extraction_authorized must be true")
    if auth.get("probe_scoring_authorized") is not True:
        raise SystemExit("probe_scoring_authorized must be true")
    if auth.get("causal_intervention_authorized") is not False:
        raise SystemExit("causal_intervention_authorized must remain false")
    for name, expected in (
        ("probe_l12_k1.npz", FROZEN_PROBE_L12_K1),
        ("probe_l12_k0.npz", FROZEN_PROBE_L12_K0),
    ):
        path = REPO_ROOT / "artifacts/phase4_models" / name
        digest = _sha_bytes(path.read_bytes())
        if digest != expected:
            raise SystemExit(f"frozen probe drift {name}: {digest}")
    git_commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    extractor = Path(__file__).resolve()
    return {
        "git_commit": git_commit,
        "working_tree_clean": True,
        "extractor_path": str(extractor.relative_to(REPO_ROOT)),
        "extractor_sha256": _sha_bytes(extractor.read_bytes()),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "compute_dtype": COMPUTE_DTYPE,
        "activation_storage_dtype": ACTIVATION_STORAGE_DTYPE,
        "extraction_mode": "truncated_prefix_single_example",
        "batch_size": BATCH_SIZE,
        "layer": LAYER,
        "k_values": [0, 1],
        "controlled_prefix_token_id": CONTROLLED_PREFIX_TOKEN_ID,
        "future_response_tokens_present": False,
        "condition_labels_in_extractor": False,
        "canonical_b1_run_id": CANONICAL_B1_RUN,
        "phase4d_run_id": PHASE4D_RUN,
        "frozen_probe_l12_k1_sha256": FROZEN_PROBE_L12_K1,
        "frozen_probe_l12_k0_sha256": FROZEN_PROBE_L12_K0,
        "causal_intervention_authorized": False,
    }


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
    model(input_ids=ids, attention_mask=mask)
    if set(captured) != set(layers):
        raise RuntimeError(f"hook miss: got {sorted(captured)} want {layers}")
    acts = []
    for li in layers:
        vec = captured[li][0, -1].float().detach().cpu().numpy()
        if not np.isfinite(vec).all():
            raise RuntimeError("NaN/Inf in activation")
        acts.append(vec.astype(np.float32))
    return acts[0] if len(acts) == 1 else np.stack(acts, axis=0).astype(np.float32)


def _analyze_boundary(prompt: str, response: str, tokenizer) -> dict[str, Any]:
    full = prompt + response
    boundary = len(prompt)
    prompt_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    full_ids = list(enc["input_ids"])
    offsets = [(int(s), int(e)) for s, e in enc["offset_mapping"]]
    n = len(prompt_ids)
    prefix_exact = full_ids[:n] == prompt_ids
    straddle = any(s < boundary < e for s, e in offsets if s != e)
    suffix = full_ids[n:]
    return {
        "prompt_ids": prompt_ids,
        "response_suffix_ids": suffix,
        "prompt_prefix_exact": prefix_exact,
        "boundary_token_straddle": straddle,
    }


def _prepare_b1_compat_payload() -> list[dict[str, Any]]:
    """Load fixed B1 examples + reference L12/k0 and L12/k1 activations."""
    from safetensors.numpy import load_file

    from pre_output_physiology.trajectory import (  # local import
        COARSE_TRANSFORMER_BLOCKS,
        PREFIX_LENGTHS_K,
    )

    by_id: dict[str, dict] = {}
    for path in (PHASE3_TRAIN_META, PHASE3_VAL_META):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    by_id[row["example_id"]] = row

    manifest = json.loads((B1_RUN_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("run_id") != CANONICAL_B1_RUN:
        raise SystemExit("B1 run_id mismatch")

    k0 = load_file(str(B1_RUN_DIR / "k0_prompt_groups.safetensors"))
    k0_acts = np.asarray(k0["activations"], dtype=np.float32)
    group_map = json.loads((B1_RUN_DIR / "k0_group_map.json").read_text(encoding="utf-8"))
    group_index = {g["prompt_sha256"]: g["group_index"] for g in group_map}

    layer_index = {li: i for i, li in enumerate(COARSE_TRANSFORMER_BLOCKS)}
    k_index = {k: i for i, k in enumerate(PREFIX_LENGTHS_K)}
    lj = layer_index[LAYER]
    k1i = k_index[1]

    # Index B1 rows
    id_to_ref: dict[str, dict[str, Any]] = {}
    for sm in manifest["trajectory_shards"]:
        meta = json.loads((B1_RUN_DIR / sm["meta"]).read_text(encoding="utf-8"))
        tens = load_file(str(B1_RUN_DIR / sm["shard"]))
        acts = np.asarray(tens["activations"], dtype=np.float32)
        valid = tens["valid_mask"].astype(bool)
        resp = tens["response_n_tokens"]
        for i, eid in enumerate(meta["example_ids"]):
            if eid not in B1_COMPAT_EXAMPLE_IDS:
                continue
            ph = meta["prompt_sha256"][i]
            gi = group_index[ph]
            ref_k0 = k0_acts[gi, lj]
            ref_k1 = acts[i, lj, k1i]
            if not valid[i, k1i]:
                raise SystemExit(f"B1 sample {eid} invalid at k1")
            if int(resp[i]) <= 1:
                raise SystemExit(f"B1 sample {eid} resp_len={resp[i]} not >1")
            id_to_ref[eid] = {
                "example_id": eid,
                "input_formatted": by_id[eid]["input_formatted"],
                "model_outputs": by_id[eid]["model_outputs"],
                "prompt_sha256": ph,
                "ref_l12_k0": ref_k0.astype(np.float32).tolist(),
                "ref_l12_k1": ref_k1.astype(np.float32).tolist(),
            }
    missing = [eid for eid in B1_COMPAT_EXAMPLE_IDS if eid not in id_to_ref]
    if missing:
        raise SystemExit(f"B1 compat samples missing: {missing}")
    return [id_to_ref[eid] for eid in B1_COMPAT_EXAMPLE_IDS]


def _prepare_phase4_extract_rows() -> list[dict[str, Any]]:
    """Build label-free extraction rows from design prompts + Phase4D hashes."""
    prompts: dict[str, dict] = {}
    with FINAL_PROMPTS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("condition_id") == "C5_uncertain_honest":
                continue
            prompts[row["example_id"]] = row
    outs: list[dict] = []
    with FINAL_OUTPUTS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                outs.append(json.loads(line))
    if len(outs) != 1200:
        raise SystemExit(f"expected 1200 final outputs, got {len(outs)}")
    if any(o.get("condition_id") == "C5_uncertain_honest" for o in outs):
        raise SystemExit("C5 present in final outputs")
    if any(o.get("first_generated_token_id") != CONTROLLED_PREFIX_TOKEN_ID for o in outs):
        raise SystemExit("prefix identity failure in final outputs")

    rows: list[dict[str, Any]] = []
    for o in outs:
        p = prompts.get(o["example_id"])
        if p is None:
            raise SystemExit(f"missing design prompt for {o['example_id']}")
        # Label-free payload only
        rows.append(
            {
                "example_id": o["example_id"],
                "prompt_text": p["prompt_text"],
                "formatted_prompt_sha256": o["formatted_prompt_sha256"],
                "input_token_ids_sha256": o["input_token_ids_sha256"],
                "expected_prefix_token_id": CONTROLLED_PREFIX_TOKEN_ID,
            }
        )
    # Ensure no condition keys leaked
    for r in rows:
        for banned in ("condition_id", "ground_truth_state", "alt_state", "label"):
            if banned in r:
                raise SystemExit(f"label leaked into extract payload: {banned}")
    return rows


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 90,
    volumes={MODEL_CACHE_DIR: model_volume},
    memory=65536,
)
def run_b1_compat_preflight(
    samples_json: str,
    *,
    provenance: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    """Re-extract B1 L12/k0 and L12/k1; require cosine >= 0.9999."""
    import torch

    samples = json.loads(samples_json)
    t0 = time.time()
    tokenizer, model = _load_model(MODEL_CACHE_DIR)
    model_load_s = time.time() - t0
    captured, handles = _register_hooks(model, [LAYER])
    rows_out = []
    cos_k0: list[float] = []
    cos_k1: list[float] = []
    with torch.inference_mode():
        for sample in samples:
            b = _analyze_boundary(
                sample["input_formatted"], sample["model_outputs"], tokenizer
            )
            if not b["prompt_prefix_exact"] or b["boundary_token_straddle"]:
                raise RuntimeError(f"boundary fail {sample['example_id']}")
            if len(b["response_suffix_ids"]) < 1:
                raise RuntimeError(f"empty suffix {sample['example_id']}")
            ids_k0 = list(b["prompt_ids"])
            ids_k1 = list(b["prompt_ids"]) + [int(b["response_suffix_ids"][0])]
            act_k0 = _forward_final_token(model, captured, ids_k0, [LAYER])
            act_k1 = _forward_final_token(model, captured, ids_k1, [LAYER])
            ref_k0 = np.asarray(sample["ref_l12_k0"], dtype=np.float32)
            ref_k1 = np.asarray(sample["ref_l12_k1"], dtype=np.float32)
            c0 = _cosine(act_k0, ref_k0)
            c1 = _cosine(act_k1, ref_k1)
            cos_k0.append(c0)
            cos_k1.append(c1)
            rows_out.append(
                {
                    "example_id": sample["example_id"],
                    "cosine_l12_k0": c0,
                    "cosine_l12_k1": c1,
                    "n_prompt_tokens": len(ids_k0),
                    "k1_token_id": int(b["response_suffix_ids"][0]),
                }
            )
    for h in handles:
        h.remove()
    min_k0 = float(min(cos_k0))
    min_k1 = float(min(cos_k1))
    min_all = float(min(min_k0, min_k1))
    passed = min_all >= COSINE_MIN
    wall_s = time.time() - t0
    model_volume.commit()
    return {
        "run_id": run_id,
        "passed": passed,
        "cosine_min_required": COSINE_MIN,
        "min_cosine_l12_k0": min_k0,
        "min_cosine_l12_k1": min_k1,
        "min_cosine_all": min_all,
        "n_samples": len(samples),
        "example_ids": [s["example_id"] for s in samples],
        "per_example": rows_out,
        "model_load_seconds": model_load_s,
        "wall_seconds": wall_s,
        "estimated_cost_usd": (wall_s / 3600.0) * L40S_USD_PER_HOUR,
        "provenance": provenance,
        "layer": LAYER,
        "canonical_b1_run_id": CANONICAL_B1_RUN,
    }


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 120,
    volumes={MODEL_CACHE_DIR: model_volume},
    memory=65536,
)
def extract_phase4_activations(
    rows_jsonl: str,
    *,
    provenance: dict[str, Any],
    run_id: str,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    """Extract L12/k0 and controlled-prefix L12/k1 for Phase-4 final rows."""
    import torch
    from safetensors.numpy import save_file

    if not preflight.get("passed"):
        raise RuntimeError("refusing extract: B1 compatibility preflight failed")
    rows = [json.loads(line) for line in rows_jsonl.splitlines() if line.strip()]
    if len(rows) != 1200:
        raise ValueError(f"expected 1200 rows, got {len(rows)}")
    for r in rows:
        for banned in ("condition_id", "ground_truth_state", "alt_state", "label"):
            if banned in r:
                raise RuntimeError(f"label present in extract payload: {banned}")

    t0 = time.time()
    tokenizer, model = _load_model(MODEL_CACHE_DIR)
    model_load_s = time.time() - t0
    captured, handles = _register_hooks(model, [LAYER])
    hidden = int(model.config.hidden_size)
    acts_k0 = np.zeros((len(rows), hidden), dtype=np.float32)
    acts_k1 = np.zeros((len(rows), hidden), dtype=np.float32)
    example_ids: list[str] = []
    prefix_ids: list[int] = []
    n_prompt_tokens: list[int] = []
    n_k1_tokens: list[int] = []
    formatted_shas: list[str] = []
    input_shas: list[str] = []

    gen_t0 = time.time()
    with torch.inference_mode():
        for i, row in enumerate(rows):
            messages = [{"role": "user", "content": row["prompt_text"]}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)
            fsha = _sha_text(formatted)
            isha = _sha_ids(prompt_ids)
            if fsha != row["formatted_prompt_sha256"]:
                raise RuntimeError(f"formatted hash mismatch {row['example_id']}")
            if isha != row["input_token_ids_sha256"]:
                raise RuntimeError(f"input ids hash mismatch {row['example_id']}")
            if row["expected_prefix_token_id"] != CONTROLLED_PREFIX_TOKEN_ID:
                raise RuntimeError("unexpected prefix token expectation")
            ids_k0 = list(prompt_ids)
            ids_k1 = list(prompt_ids) + [CONTROLLED_PREFIX_TOKEN_ID]
            if ids_k1[-1] != CONTROLLED_PREFIX_TOKEN_ID:
                raise RuntimeError("k1 prefix identity failure")
            if len(ids_k1) != len(ids_k0) + 1:
                raise RuntimeError("k1 must be exactly prompt + one token")
            # No future tokens beyond the controlled prefix.
            act_k0 = _forward_final_token(model, captured, ids_k0, [LAYER])
            act_k1 = _forward_final_token(model, captured, ids_k1, [LAYER])
            acts_k0[i] = act_k0
            acts_k1[i] = act_k1
            example_ids.append(row["example_id"])
            prefix_ids.append(CONTROLLED_PREFIX_TOKEN_ID)
            n_prompt_tokens.append(len(ids_k0))
            n_k1_tokens.append(len(ids_k1))
            formatted_shas.append(fsha)
            input_shas.append(isha)
            if (i + 1) % 100 == 0:
                print(f"extracted {i + 1}/{len(rows)}")

    for h in handles:
        h.remove()
    extract_s = time.time() - gen_t0
    wall_s = time.time() - t0
    model_volume.commit()

    if not all(p == CONTROLLED_PREFIX_TOKEN_ID for p in prefix_ids):
        raise RuntimeError("prefix identity incomplete")
    if not np.isfinite(acts_k0).all() or not np.isfinite(acts_k1).all():
        raise RuntimeError("non-finite activations")

    # Pack tensors for return via Modal (base64 safetensors bytes)
    import base64

    tmp = Path("/tmp") / f"{run_id}_activations.safetensors"
    save_file(
        {
            "activations_l12_k0": acts_k0,
            "activations_l12_k1": acts_k1,
            "k1_prefix_token_ids": np.asarray(prefix_ids, dtype=np.int32),
            "n_prompt_tokens": np.asarray(n_prompt_tokens, dtype=np.int32),
            "n_k1_input_tokens": np.asarray(n_k1_tokens, dtype=np.int32),
        },
        str(tmp),
    )
    blob = tmp.read_bytes()
    tmp.unlink(missing_ok=True)

    return {
        "run_id": run_id,
        "n_rows": len(rows),
        "example_ids": example_ids,
        "formatted_prompt_sha256": formatted_shas,
        "input_token_ids_sha256": input_shas,
        "activations_safetensors_b64": base64.b64encode(blob).decode("ascii"),
        "activations_sha256": _sha_bytes(blob),
        "layer": LAYER,
        "k_values": [0, 1],
        "controlled_prefix_token_id": CONTROLLED_PREFIX_TOKEN_ID,
        "prefix_identity_count": len(prefix_ids),
        "future_response_tokens_present": False,
        "condition_labels_present": False,
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
        "l40s_usd_per_hour_assumed": L40S_USD_PER_HOUR,
        "preflight": {
            "passed": preflight["passed"],
            "min_cosine_all": preflight["min_cosine_all"],
            "min_cosine_l12_k0": preflight["min_cosine_l12_k0"],
            "min_cosine_l12_k1": preflight["min_cosine_l12_k1"],
        },
        "provenance": provenance,
        "causal_interventions_performed": False,
        "probes_retrained": False,
    }


@app.local_entrypoint()
def main() -> None:
    import base64
    import sys

    sys.path.insert(0, str(REPO_ROOT / "src"))

    prov = _collect_local_provenance()
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"phase4e_extract_{ts}_{prov['git_commit'][:8]}"

    print("Preparing B1 compatibility sample…")
    samples = _prepare_b1_compat_payload()
    print(f"Launching B1 compat preflight n={len(samples)} run_id={run_id}")
    preflight = run_b1_compat_preflight.remote(
        json.dumps(samples),
        provenance=prov,
        run_id=f"{run_id}_preflight",
    )
    print(
        json.dumps(
            {
                "preflight_passed": preflight["passed"],
                "min_cosine_all": preflight["min_cosine_all"],
                "min_cosine_l12_k0": preflight["min_cosine_l12_k0"],
                "min_cosine_l12_k1": preflight["min_cosine_l12_k1"],
            },
            indent=2,
        )
    )
    if not preflight["passed"]:
        raise SystemExit(
            f"STOP: B1 compatibility preflight failed "
            f"(min cosine {preflight['min_cosine_all']} < {COSINE_MIN})"
        )

    rows = _prepare_phase4_extract_rows()
    rows_jsonl = "\n".join(json.dumps(r, sort_keys=True) for r in rows)
    print(f"Launching Phase-4 extract n={len(rows)}")
    result = extract_phase4_activations.remote(
        rows_jsonl,
        provenance=prov,
        run_id=run_id,
        preflight=preflight,
    )
    if result["n_rows"] != 1200:
        raise SystemExit(f"expected 1200 rows, got {result['n_rows']}")
    if result["prefix_identity_count"] != 1200:
        raise SystemExit("prefix identity incomplete")

    run_dir = REPO_ROOT / "artifacts/runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    act_path = run_dir / "activations_l12_k0_k1.safetensors"
    act_path.write_bytes(base64.b64decode(result["activations_safetensors_b64"]))
    if _sha_bytes(act_path.read_bytes()) != result["activations_sha256"]:
        raise SystemExit("activation blob hash mismatch after write")

    meta = {
        "example_ids": result["example_ids"],
        "formatted_prompt_sha256": result["formatted_prompt_sha256"],
        "input_token_ids_sha256": result["input_token_ids_sha256"],
        "layer": LAYER,
        "k_values": [0, 1],
        "controlled_prefix_token_id": CONTROLLED_PREFIX_TOKEN_ID,
        "condition_labels_present": False,
        "future_response_tokens_present": False,
    }
    (run_dir / "activation_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    preflight_path = run_dir / "b1_compatibility_preflight.json"
    preflight_path.write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    total_wall = float(preflight["wall_seconds"]) + float(result["wall_seconds"])
    total_cost = float(preflight["estimated_cost_usd"]) + float(
        result["estimated_cost_usd"]
    )
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "git_commit": prov["git_commit"],
        "working_tree_clean": True,
        "extractor_sha256": prov["extractor_sha256"],
        "phase4d_run_id": PHASE4D_RUN,
        "canonical_b1_run_id": CANONICAL_B1_RUN,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "compute_dtype": COMPUTE_DTYPE,
        "activation_storage_dtype": ACTIVATION_STORAGE_DTYPE,
        "batch_size": BATCH_SIZE,
        "extraction_mode": "truncated_prefix_single_example",
        "layer": LAYER,
        "k_values": [0, 1],
        "controlled_prefix_token_id": CONTROLLED_PREFIX_TOKEN_ID,
        "n_rows": 1200,
        "n_c5_rows": 0,
        "prefix_identity_count": 1200,
        "future_response_tokens_present": False,
        "condition_labels_in_extractor": False,
        "activations_path": str(act_path.relative_to(REPO_ROOT)),
        "activations_sha256": result["activations_sha256"],
        "preflight_path": str(preflight_path.relative_to(REPO_ROOT)),
        "preflight_passed": True,
        "preflight_min_cosine_all": preflight["min_cosine_all"],
        "preflight_min_cosine_l12_k0": preflight["min_cosine_l12_k0"],
        "preflight_min_cosine_l12_k1": preflight["min_cosine_l12_k1"],
        "preflight_wall_seconds": preflight["wall_seconds"],
        "extraction_wall_seconds": result["wall_seconds"],
        "wall_seconds_total": total_wall,
        "estimated_cost_usd_total": total_cost,
        "l40s_usd_per_hour_assumed": L40S_USD_PER_HOUR,
        "frozen_probe_l12_k1_sha256": FROZEN_PROBE_L12_K1,
        "frozen_probe_l12_k0_sha256": FROZEN_PROBE_L12_K0,
        "probes_retrained": False,
        "probes_recalibrated": False,
        "eligibility_changed": False,
        "causal_interventions_performed": False,
        "c5_status": "HOLD",
    }
    (run_dir / "extraction_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in (
                    "run_id",
                    "n_rows",
                    "prefix_identity_count",
                    "preflight_min_cosine_all",
                    "wall_seconds_total",
                    "estimated_cost_usd_total",
                    "activations_path",
                )
            },
            indent=2,
        )
    )
