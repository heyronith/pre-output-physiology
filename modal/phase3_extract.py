"""Modal GPU entrypoint: Phase 3B1 coarse trajectory extraction (train+val only).

Usage:
  uv run modal run modal/phase3_extract.py --mode preflight
  uv run modal run modal/phase3_extract.py --mode extract

Never loads locked-test rows. Never runs Regime C. Never intervenes.
"""

from __future__ import annotations

import json
import time
from datetime import UTC
from pathlib import Path
from typing import Any

import modal

APP_NAME = "pre-output-physiology-phase3"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31]
K_VALUES = [0, 1, 2, 4, 8, 16]
K_GT0 = [1, 2, 4, 8, 16]
COSINE_MIN = 0.9999
L40S_USD_PER_HOUR = 1.95
REPO_ROOT = Path(__file__).resolve().parents[1]
SHARD_SIZE = 400

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.44.2",
        "accelerate==0.33.0",
        "safetensors==0.4.4",
        "numpy==1.26.4",
        "huggingface_hub==0.24.6",
    )
)

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name("preoutput-mistral-cache", create_if_missing=True)
artifact_volume = modal.Volume.from_name(
    "preoutput-phase3-artifacts", create_if_missing=True
)
MODEL_CACHE_DIR = "/vol/hf_cache"
ART_DIR = "/vol/phase3_artifacts"


def _cosine(a, b) -> float:  # noqa: ANN001
    import numpy as np

    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def _rel_l2(a, b) -> float:  # noqa: ANN001
    import numpy as np

    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))


def _analyze_boundary(prompt: str, response: str, tokenizer) -> dict[str, Any]:
    """Inline canonical boundary (mirrors trajectory.analyze_prompt_response_boundary)."""
    full = prompt + response
    boundary = len(prompt)
    prompt_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    standalone = list(tokenizer(response, add_special_tokens=False)["input_ids"])
    enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    full_ids = list(enc["input_ids"])
    offsets = [(int(s), int(e)) for s, e in enc["offset_mapping"]]
    n = len(prompt_ids)
    prefix_exact = full_ids[:n] == prompt_ids
    straddle_idx = None
    for i, (s, e) in enumerate(offsets):
        if s == e:
            continue
        if s < boundary < e:
            straddle_idx = i
            break
    suffix = full_ids[n:]
    resp_offsets = [(s - boundary, e - boundary) for s, e in offsets[n:]]
    return {
        "prompt_ids": prompt_ids,
        "full_ids": full_ids,
        "response_suffix_ids": suffix,
        "response_start_token_index": n,
        "full_offsets": offsets,
        "response_offsets_in_model_outputs": resp_offsets,
        "prompt_prefix_exact": prefix_exact,
        "boundary_token_straddle": straddle_idx is not None,
        "straddling_token_index": straddle_idx,
        "standalone_response_ids": standalone,
        "standalone_equals_suffix": standalone == suffix,
        "boundary_char": boundary,
    }


def _load_model(cache_dir: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=cache_dir,
        use_fast=True,
    )
    tokenizer.padding_side = "right"
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
    commit = getattr(model.config, "_commit_hash", None) or MODEL_REVISION
    if commit != MODEL_REVISION and getattr(model.config, "_commit_hash", None):
        raise RuntimeError(f"commit {commit} != {MODEL_REVISION}")
    return tokenizer, model, commit


def _right_pad(id_lists: list[list[int]], pad_id: int):
    import torch

    max_len = max(len(x) for x in id_lists)
    batch = []
    masks = []
    for ids in id_lists:
        pad = max_len - len(ids)
        batch.append(ids + [pad_id] * pad)
        masks.append([1] * len(ids) + [0] * pad)
    return (
        torch.tensor(batch, dtype=torch.long, device="cuda:0"),
        torch.tensor(masks, dtype=torch.long, device="cuda:0"),
    )


def _logit_summary(logits_1d):  # noqa: ANN001
    import torch

    probs = torch.softmax(logits_1d.float(), dim=-1)
    entropy = float(-(probs * (probs + 1e-12).log()).sum().item())
    top2 = torch.topk(probs, k=2).values
    max_p = float(top2[0].item())
    margin = float((top2[0] - top2[1]).item())
    return [entropy, max_p, margin]


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 45,
    volumes={MODEL_CACHE_DIR: model_volume, ART_DIR: artifact_volume},
    memory=65536,
)
def run_preflight(examples_jsonl: str, *, run_id: str) -> dict[str, Any]:
    """Causal full-vs-truncated and batch-vs-single equivalence gates."""
    import hashlib

    import numpy as np
    import torch

    t0 = time.time()
    examples = [json.loads(line) for line in examples_jsonl.splitlines() if line.strip()]
    tokenizer, model, commit = _load_model(MODEL_CACHE_DIR)
    model_load_s = time.time() - t0
    hidden = int(model.config.hidden_size)
    pad_id = int(tokenizer.pad_token_id)

    rng = np.random.default_rng(42)
    prepared = []
    for row in examples:
        b = _analyze_boundary(row["input_formatted"], row["model_outputs"], tokenizer)
        if not b["prompt_prefix_exact"] or b["boundary_token_straddle"]:
            raise RuntimeError(f"boundary fail {row['example_id']}")
        prepared.append((row, b))
    long_enough = [p for p in prepared if len(p[1]["response_suffix_ids"]) > 16]
    if len(long_enough) < 4:
        raise RuntimeError("Need >=4 train examples with response length >16")
    idx = rng.choice(len(long_enough), size=min(8, len(long_enough)), replace=False)
    sample = [long_enough[i] for i in sorted(idx.tolist())]

    test_layers = [0, 12, 31]
    test_ks = [0, 1, 4, 16]

    captured: dict[int, torch.Tensor] = {}
    handles = []
    for li in test_layers:

        def make_hook(layer_i: int):
            def hook(_m, _inp, output):  # noqa: ANN001
                act = output[0] if isinstance(output, tuple) else output
                captured[layer_i] = act

            return hook

        handles.append(model.model.layers[li].register_forward_hook(make_hook(li)))

    causal_rows = []
    batch_rows = []
    t_causal0 = time.time()
    try:
        with torch.inference_mode():
            for row, b in sample:
                full_ids = b["full_ids"]
                prompt_ids = b["prompt_ids"]
                start = b["response_start_token_index"]

                ids_a = torch.tensor([full_ids], device="cuda:0")
                mask_a = torch.ones_like(ids_a)
                captured.clear()
                _ = model(input_ids=ids_a, attention_mask=mask_a)
                acts_a = {li: captured[li][0].float().cpu() for li in test_layers}

                for k in test_ks:
                    if k == 0:
                        trunc_ids = prompt_ids
                        pos_full = len(prompt_ids) - 1
                        pos_trunc = len(prompt_ids) - 1
                    else:
                        if len(b["response_suffix_ids"]) <= k:
                            continue
                        trunc_ids = full_ids[: start + k]
                        pos_full = start + k - 1
                        pos_trunc = len(trunc_ids) - 1

                    ids_b = torch.tensor([trunc_ids], device="cuda:0")
                    mask_b = torch.ones_like(ids_b)
                    captured.clear()
                    _ = model(input_ids=ids_b, attention_mask=mask_b)
                    acts_b = {li: captured[li][0].float().cpu() for li in test_layers}

                    for li in test_layers:
                        va = acts_a[li][pos_full].numpy()
                        vb = acts_b[li][pos_trunc].numpy()
                        cos = _cosine(va, vb)
                        mad = float(np.max(np.abs(va - vb)))
                        rl2 = _rel_l2(va, vb)
                        causal_rows.append(
                            {
                                "example_id": row["example_id"],
                                "layer": li,
                                "k": k,
                                "cosine": cos,
                                "max_abs_diff": mad,
                                "rel_l2": rl2,
                            }
                        )

                # Single unpadded
                captured.clear()
                _ = model(input_ids=ids_a, attention_mask=mask_a)
                acts_single = {li: captured[li][0].float().cpu() for li in test_layers}

                shorter = prompt_ids
                ids_batch, mask_batch = _right_pad([full_ids, shorter], pad_id)
                captured.clear()
                _ = model(input_ids=ids_batch, attention_mask=mask_batch)
                acts_batch = {li: captured[li][0].float().cpu() for li in test_layers}

                for k in test_ks:
                    if k == 0:
                        pos = len(prompt_ids) - 1
                    else:
                        if len(b["response_suffix_ids"]) <= k:
                            continue
                        pos = start + k - 1
                    for li in test_layers:
                        vs = acts_single[li][pos].numpy()
                        vb = acts_batch[li][pos].numpy()
                        cos = _cosine(vs, vb)
                        mad = float(np.max(np.abs(vs - vb)))
                        rl2 = _rel_l2(vs, vb)
                        batch_rows.append(
                            {
                                "example_id": row["example_id"],
                                "layer": li,
                                "k": k,
                                "cosine": cos,
                                "max_abs_diff": mad,
                                "rel_l2": rl2,
                            }
                        )
    finally:
        for h in handles:
            h.remove()

    causal_s = time.time() - t_causal0
    min_causal = min(r["cosine"] for r in causal_rows) if causal_rows else 0.0
    min_batch = min(r["cosine"] for r in batch_rows) if batch_rows else 0.0
    causal_fail = [r for r in causal_rows if r["cosine"] < COSINE_MIN]
    batch_fail = [r for r in batch_rows if r["cosine"] < COSINE_MIN]
    wall = time.time() - t0
    result = {
        "run_id": run_id,
        "mode": "preflight",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "resolved_commit_hash": commit,
        "hidden_size": hidden,
        "padding_side": "right",
        "cosine_min_required": COSINE_MIN,
        "n_sample_examples": len(sample),
        "sample_example_ids": [r["example_id"] for r, _ in sample],
        "test_layers": test_layers,
        "test_ks": test_ks,
        "causal_comparisons": causal_rows,
        "batch_comparisons": batch_rows,
        "causal_failures": causal_fail,
        "batch_failures": batch_fail,
        "n_causal_failures": len(causal_fail),
        "n_batch_failures": len(batch_fail),
        "causal_min_cosine": min_causal,
        "batch_min_cosine": min_batch,
        "causal_pass": min_causal >= COSINE_MIN and len(causal_rows) > 0,
        "batch_pass": min_batch >= COSINE_MIN and len(batch_rows) > 0,
        "model_load_seconds": model_load_s,
        "preflight_seconds": causal_s,
        "wall_seconds": wall,
        "estimated_cost_usd": (wall / 3600.0) * L40S_USD_PER_HOUR,
        "l40s_usd_per_hour_assumed": L40S_USD_PER_HOUR,
        "gpu_type": "L40S",
        "locked_test_used": False,
        "full_extraction_authorized": False,
    }
    out_dir = Path(ART_DIR) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    (out_dir / "preflight.json").write_text(payload, encoding="utf-8")
    result["preflight_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    artifact_volume.commit()
    model_volume.commit()
    return result


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 90,
    volumes={MODEL_CACHE_DIR: model_volume, ART_DIR: artifact_volume},
    memory=65536,
)
def extract_dev(
    examples_jsonl: str,
    *,
    run_id: str,
    batch_size: int = 4,
) -> dict[str, Any]:
    """Extract k=0 prompt-group + k>0 trajectory grid for train/val only."""
    import hashlib

    import numpy as np
    import torch
    from safetensors.numpy import save_file

    t0 = time.time()
    examples = [json.loads(line) for line in examples_jsonl.splitlines() if line.strip()]
    for row in examples:
        if row.get("split") == "locked_test":
            raise RuntimeError("locked_test row leaked into Phase 3B1 payload")

    tokenizer, model, commit = _load_model(MODEL_CACHE_DIR)
    model_load_s = time.time() - t0
    hidden = int(model.config.hidden_size)
    pad_id = int(tokenizer.pad_token_id)
    out_dir = Path(ART_DIR) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    prepared = []
    for row in examples:
        b = _analyze_boundary(row["input_formatted"], row["model_outputs"], tokenizer)
        if not b["prompt_prefix_exact"] or b["boundary_token_straddle"]:
            raise RuntimeError(
                f"STOP: boundary failure {row['example_id']} "
                f"prefix={b['prompt_prefix_exact']} "
                f"straddle={b['boundary_token_straddle']}"
            )
        prepared.append({"row": row, "b": b})

    t_k0 = time.time()
    groups: dict[str, dict] = {}
    for item in prepared:
        ph = item["row"]["prompt_sha256"]
        if ph not in groups:
            groups[ph] = {
                "prompt_sha256": ph,
                "prompt_ids": item["b"]["prompt_ids"],
                "input_formatted": item["row"]["input_formatted"],
                "example_ids": [],
            }
        groups[ph]["example_ids"].append(item["row"]["example_id"])
    group_list = list(groups.values())
    n_groups = len(group_list)
    k0_acts = np.zeros((n_groups, len(LAYERS), hidden), dtype=np.float16)
    k0_logits = np.zeros((n_groups, 3), dtype=np.float32)

    captured: dict[int, torch.Tensor] = {}
    handles = []
    for li in LAYERS:

        def make_hook(layer_i: int):
            def hook(_m, _inp, output):  # noqa: ANN001
                act = output[0] if isinstance(output, tuple) else output
                captured[layer_i] = act

            return hook

        handles.append(model.model.layers[li].register_forward_hook(make_hook(li)))

    try:
        with torch.inference_mode():
            for gi in range(0, n_groups, batch_size):
                batch = group_list[gi : gi + batch_size]
                id_lists = [g["prompt_ids"] for g in batch]
                input_ids, attention_mask = _right_pad(id_lists, pad_id)
                captured.clear()
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                for bi, g in enumerate(batch):
                    pos = len(g["prompt_ids"]) - 1
                    for lj, layer_i in enumerate(LAYERS):
                        vec = captured[layer_i][bi, pos].float().cpu().numpy()
                        k0_acts[gi + bi, lj] = vec.astype(np.float16)
                    k0_logits[gi + bi] = np.asarray(
                        _logit_summary(out.logits[bi, pos]), dtype=np.float32
                    )
                    if not np.isfinite(k0_acts[gi + bi].astype(np.float32)).all():
                        raise RuntimeError(
                            f"NaN/Inf in k0 acts for {g['prompt_sha256']}"
                        )
        k0_s = time.time() - t_k0

        k0_path = out_dir / "k0_prompt_groups.safetensors"
        save_file(
            {"activations": k0_acts, "logit_summaries": k0_logits},
            str(k0_path),
        )
        group_map = [
            {
                "prompt_sha256": g["prompt_sha256"],
                "group_index": i,
                "example_ids": g["example_ids"],
                "n_prompt_tokens": len(g["prompt_ids"]),
            }
            for i, g in enumerate(group_list)
        ]
        (out_dir / "k0_group_map.json").write_text(
            json.dumps(group_map, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        t_tr = time.time()
        n = len(prepared)
        layer_index = {li: i for i, li in enumerate(LAYERS)}
        k_index = {k: i for i, k in enumerate(K_VALUES)}
        shard_meta = []
        total_tokens = 0

        for start in range(0, n, SHARD_SIZE):
            shard = prepared[start : start + SHARD_SIZE]
            sn = len(shard)
            acts = np.zeros((sn, len(LAYERS), len(K_VALUES), hidden), dtype=np.float16)
            valid = np.zeros((sn, len(K_VALUES)), dtype=np.bool_)
            logits = np.zeros((sn, len(K_VALUES), 3), dtype=np.float32)
            example_ids = []
            labels = []
            splits = []
            prompt_hashes = []
            resp_lens = []

            for bi0 in range(0, sn, batch_size):
                batch = shard[bi0 : bi0 + batch_size]
                id_lists = [it["b"]["full_ids"] for it in batch]
                input_ids, attention_mask = _right_pad(id_lists, pad_id)
                captured.clear()
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                for bi, item in enumerate(batch):
                    row = item["row"]
                    b = item["b"]
                    global_i = bi0 + bi
                    example_ids.append(row["example_id"])
                    labels.append(int(row["eventual_deception"]))
                    splits.append(row["split"])
                    prompt_hashes.append(row["prompt_sha256"])
                    resp_len = len(b["response_suffix_ids"])
                    resp_lens.append(resp_len)
                    total_tokens += len(b["full_ids"])
                    start_tok = b["response_start_token_index"]
                    valid[global_i, k_index[0]] = False

                    for k in K_GT0:
                        if resp_len <= k:
                            continue
                        pos = start_tok + k - 1
                        ki = k_index[k]
                        valid[global_i, ki] = True
                        for layer_i in LAYERS:
                            lj = layer_index[layer_i]
                            vec = captured[layer_i][bi, pos].float().cpu().numpy()
                            acts[global_i, lj, ki] = vec.astype(np.float16)
                        logits[global_i, ki] = np.asarray(
                            _logit_summary(out.logits[bi, pos]), dtype=np.float32
                        )
                    for ki in range(len(K_VALUES)):
                        if not valid[global_i, ki]:
                            continue
                        if not np.isfinite(
                            acts[global_i, :, ki].astype(np.float32)
                        ).all():
                            raise RuntimeError(
                                f"NaN/Inf activations for {row['example_id']} "
                                f"k={K_VALUES[ki]}"
                            )

            shard_name = f"trajectory_shard_{start:05d}_{start + sn - 1:05d}.safetensors"
            shard_path = out_dir / shard_name
            save_file(
                {
                    "activations": acts,
                    "valid_mask": valid.astype(np.uint8),
                    "logit_summaries": logits,
                    "labels": np.asarray(labels, dtype=np.int64),
                    "response_n_tokens": np.asarray(resp_lens, dtype=np.int64),
                },
                str(shard_path),
            )
            meta = {
                "shard": shard_name,
                "n_rows": sn,
                "example_ids": example_ids,
                "splits": splits,
                "prompt_sha256": prompt_hashes,
                "example_ids_sha256": hashlib.sha256(
                    "\n".join(example_ids).encode()
                ).hexdigest(),
                "layers": LAYERS,
                "k_values": K_VALUES,
                "dtype": "bfloat16_stored_as_float16",
                "model_revision": MODEL_REVISION,
            }
            meta_path = out_dir / shard_name.replace(".safetensors", "_meta.json")
            meta_path.write_text(
                json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            blob = shard_path.read_bytes()
            shard_meta.append(
                {
                    "shard": shard_name,
                    "n_rows": sn,
                    "sha256": hashlib.sha256(blob).hexdigest(),
                    "bytes": len(blob),
                    "example_ids_sha256": meta["example_ids_sha256"],
                    "meta": shard_name.replace(".safetensors", "_meta.json"),
                }
            )

        traj_s = time.time() - t_tr
    finally:
        for h in handles:
            h.remove()

    wall = time.time() - t0
    k0_blob = (out_dir / "k0_prompt_groups.safetensors").read_bytes()
    manifest = {
        "run_id": run_id,
        "mode": "extract_dev",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "resolved_commit_hash": commit,
        "dataset_revision": "bf93584916fbd23121eca6f2017017df0ef3184f",
        "padding_side": "right",
        "layers": LAYERS,
        "k_values": K_VALUES,
        "hidden_size": hidden,
        "n_examples": n,
        "n_prompt_groups": n_groups,
        "splits_present": sorted({p["row"]["split"] for p in prepared}),
        "locked_test_present": False,
        "regime_c_run": False,
        "causal_interventions": False,
        "batch_size": batch_size,
        "shard_size": SHARD_SIZE,
        "model_load_seconds": model_load_s,
        "k0_extract_seconds": k0_s,
        "trajectory_extract_seconds": traj_s,
        "wall_seconds": wall,
        "total_tokens": total_tokens,
        "examples_per_sec": n / traj_s if traj_s > 0 else None,
        "tokens_per_sec": total_tokens / traj_s if traj_s > 0 else None,
        "gpu_type": "L40S",
        "num_gpus": 1,
        "estimated_cost_usd": (wall / 3600.0) * L40S_USD_PER_HOUR,
        "l40s_usd_per_hour_assumed": L40S_USD_PER_HOUR,
        "k0_artifact": {
            "path": "k0_prompt_groups.safetensors",
            "sha256": hashlib.sha256(k0_blob).hexdigest(),
            "bytes": len(k0_blob),
            "shape": list(k0_acts.shape),
            "group_map": "k0_group_map.json",
        },
        "trajectory_shards": shard_meta,
        "total_artifact_bytes": len(k0_blob) + sum(s["bytes"] for s in shard_meta),
    }
    man_path = out_dir / "run_manifest.json"
    man_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifact_volume.commit()
    model_volume.commit()
    return manifest


@app.local_entrypoint()
def main(
    mode: str = "preflight",
    batch_size: int = 4,
    out_dir: str = "",
) -> None:
    import shutil
    import subprocess
    import uuid
    from datetime import datetime

    data_dir = REPO_ROOT / "data" / "processed" / "phase3_roleplay"
    train_path = data_dir / "phase3_train_metadata.jsonl"
    val_path = data_dir / "phase3_validation_metadata.jsonl"
    if not train_path.is_file() or not val_path.is_file():
        raise SystemExit("Missing phase3 train/val metadata")

    def slim_lines(path: Path, split: str) -> list[str]:
        out = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("split") == "locked_test":
                    raise SystemExit("locked_test leaked")
                slim = {
                    "example_id": row["example_id"],
                    "split": split,
                    "prompt_sha256": row["prompt_sha256"],
                    "input_formatted": row["input_formatted"],
                    "model_outputs": row["model_outputs"],
                    "eventual_deception": int(row["eventual_deception"]),
                }
                out.append(json.dumps(slim, sort_keys=True))
        return out

    train_lines = slim_lines(train_path, "phase3_train")
    val_lines = slim_lines(val_path, "phase3_validation")

    run_id = (
        f"phase3b1_{mode}_"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_"
        f"{uuid.uuid4().hex[:8]}"
    )
    local_out = (
        Path(out_dir) if out_dir else (REPO_ROOT / "artifacts" / "runs" / run_id)
    )
    local_out.mkdir(parents=True, exist_ok=True)

    if mode == "preflight":
        payload = "\n".join(train_lines) + "\n"
        print(f"Launching preflight run_id={run_id}")
        result = run_preflight.remote(payload, run_id=run_id)
        (local_out / "preflight.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "causal_pass": result["causal_pass"],
                    "batch_pass": result["batch_pass"],
                    "causal_min_cosine": result["causal_min_cosine"],
                    "batch_min_cosine": result["batch_min_cosine"],
                    "estimated_cost_usd": result["estimated_cost_usd"],
                    "out_dir": str(local_out),
                },
                indent=2,
            )
        )
        if not (result["causal_pass"] and result["batch_pass"]):
            raise SystemExit("PREFLIGHT FAILED — full extraction forbidden")
        return

    if mode != "extract":
        raise SystemExit(f"Unknown mode {mode}")

    payload = "\n".join(train_lines + val_lines) + "\n"
    if '"split": "locked_test"' in payload or '"split":"locked_test"' in payload:
        raise SystemExit("locked_test in payload")
    print(f"Launching extract run_id={run_id} n={len(train_lines) + len(val_lines)}")
    result = extract_dev.remote(payload, run_id=run_id, batch_size=batch_size)
    (local_out / "run_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"Downloading Modal volume artifacts for {run_id} ...")
    subprocess.run(
        [
            "modal",
            "volume",
            "get",
            "preoutput-phase3-artifacts",
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

    print(
        json.dumps(
            {
                "run_id": run_id,
                "out_dir": str(local_out),
                "n_examples": result["n_examples"],
                "n_prompt_groups": result["n_prompt_groups"],
                "estimated_cost_usd": result["estimated_cost_usd"],
                "total_artifact_bytes": result["total_artifact_bytes"],
                "locked_test_present": result["locked_test_present"],
            },
            indent=2,
        )
    )
