"""Modal GPU: Phase 3B1 truncated-prefix, batch-size-1 extraction (train+val only).

Canonical scientific path (D035–D037):
  - k=0: prompt_ids only (once per prompt_sha256)
  - k>0: prompt_ids + response_suffix_ids[:k]  (no future tokens)
  - batch_size = 1
  - Original 0.9999 full-sequence/batch gates NOT relaxed (historical HOLD)

Usage:
  uv run modal run modal/phase3_extract.py --mode preflight_truncated
  uv run modal run modal/phase3_extract.py --mode benchmark
  uv run modal run modal/phase3_extract.py --mode extract

Never loads locked-test. Never Regime C. Never interventions.
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
SOFT_BUDGET_USD = 20.0
REPO_ROOT = Path(__file__).resolve().parents[1]
SHARD_SIZE = 250
EXPECTED_ELIGIBLE = {
    0: (2361, 639),
    1: (2360, 639),
    2: (2317, 629),
    4: (2312, 620),
    8: (2227, 599),
    16: (1578, 492),
}
FREEZE_SHA256 = "b25c5095c1cba440103343f90408c4caac0b59785bdc28b57a46abea405bcd00"

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
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def _rel_l2(a, b) -> float:  # noqa: ANN001
    import numpy as np

    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))


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


def _truncated_ids(prompt_ids: list[int], suffix: list[int], k: int) -> list[int]:
    if k == 0:
        return list(prompt_ids)
    if k > len(suffix):
        raise ValueError(f"k={k} > response length {len(suffix)}")
    return list(prompt_ids) + list(suffix[:k])


def _eligible(resp_len: int, k: int) -> bool:
    return True if k == 0 else resp_len > k


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
    commit = getattr(model.config, "_commit_hash", None) or MODEL_REVISION
    if model.dtype != torch.bfloat16:
        raise RuntimeError(f"expected bf16, got {model.dtype}")
    return tokenizer, model, commit


def _logit_summary(logits_1d):  # noqa: ANN001
    import torch

    probs = torch.softmax(logits_1d.float(), dim=-1)
    entropy = float(-(probs * (probs + 1e-12).log()).sum().item())
    top2 = torch.topk(probs, k=2).values
    return [entropy, float(top2[0].item()), float((top2[0] - top2[1]).item())]


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
    """Single-example forward; return (acts[9,H] float32 cpu, logit_summary[3])."""
    import numpy as np
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
        if vec.shape[-1] != int(model.config.hidden_size):
            raise RuntimeError("hidden dim mismatch")
        acts.append(vec)
    logits = _logit_summary(out.logits[0, -1])
    return np.stack(acts, axis=0).astype(np.float32), logits


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 40,
    volumes={MODEL_CACHE_DIR: model_volume, ART_DIR: artifact_volume},
    memory=65536,
)
def run_preflight_truncated(examples_jsonl: str, *, run_id: str) -> dict[str, Any]:
    """Repeatability + truncated-input integrity checks (batch_size=1)."""
    import hashlib

    import numpy as np
    import torch

    t0 = time.time()
    examples = [json.loads(x) for x in examples_jsonl.splitlines() if x.strip()]
    tokenizer, model, commit = _load_model(MODEL_CACHE_DIR)
    model_load_s = time.time() - t0
    hidden = int(model.config.hidden_size)

    rng = np.random.default_rng(42)
    prepared = []
    for row in examples:
        b = _analyze_boundary(row["input_formatted"], row["model_outputs"], tokenizer)
        if not b["prompt_prefix_exact"] or b["boundary_token_straddle"]:
            raise RuntimeError(f"boundary fail {row['example_id']}")
        prepared.append((row, b))
    long_enough = [p for p in prepared if len(p[1]["response_suffix_ids"]) > 16]
    if len(long_enough) < 4:
        raise RuntimeError("need >=4 long examples")
    # Prefer same sample IDs from prior hold if present
    prefer = {"5_6", "95_1", "135_12", "148_11"}
    preferred = [p for p in long_enough if p[0]["example_id"] in prefer]
    pool = preferred if len(preferred) >= 4 else long_enough
    idx = rng.choice(len(pool), size=min(8, len(pool)), replace=False)
    sample = [pool[i] for i in sorted(idx.tolist())]

    test_layers = [0, 12, 31]
    test_ks = [0, 1, 4, 16]
    captured, handles = _register_hooks(model, test_layers)
    rows = []
    try:
        with torch.inference_mode():
            for row, b in sample:
                for k in test_ks:
                    if k > 0 and len(b["response_suffix_ids"]) <= k:
                        continue
                    ids = _truncated_ids(b["prompt_ids"], b["response_suffix_ids"], k)
                    # Integrity: truncated input equals exact prefix; no future tokens by length
                    if k == 0:
                        if ids != b["prompt_ids"]:
                            raise RuntimeError("k0 input != prompt_ids")
                    else:
                        expected = b["prompt_ids"] + b["response_suffix_ids"][:k]
                        if ids != expected:
                            raise RuntimeError(f"k={k} truncated ids mismatch")
                        if len(ids) != len(b["prompt_ids"]) + k:
                            raise RuntimeError("truncated length mismatch")
                        if ids[-1] != b["response_suffix_ids"][k - 1]:
                            raise RuntimeError("final token is not response[k-1]")
                        # Future tokens absent by construction (not present in input list)
                        if len(ids) > len(b["prompt_ids"]) + k:
                            raise RuntimeError("future tokens present in input")

                    a1, _lg1 = _forward_final_token(model, captured, ids, test_layers)
                    a2, _lg2 = _forward_final_token(model, captured, ids, test_layers)
                    for j, li in enumerate(test_layers):
                        cos = _cosine(a1[j], a2[j])
                        mad = float(np.max(np.abs(a1[j] - a2[j])))
                        rl2 = _rel_l2(a1[j], a2[j])
                        rows.append(
                            {
                                "example_id": row["example_id"],
                                "layer": li,
                                "k": k,
                                "n_input_tokens": len(ids),
                                "cosine": cos,
                                "max_abs_diff": mad,
                                "rel_l2": rl2,
                                "bf16": True,
                                "quantization": None,
                                "batch_size": 1,
                            }
                        )
                        if cos < COSINE_MIN:
                            raise RuntimeError(
                                f"repeatability FAIL {row['example_id']} "
                                f"layer={li} k={k} cosine={cos}"
                            )
    finally:
        for h in handles:
            h.remove()

    wall = time.time() - t0
    min_cos = min(r["cosine"] for r in rows)
    result = {
        "run_id": run_id,
        "mode": "preflight_truncated",
        "extraction_mode": "truncated_prefix_single_example",
        "batch_size": 1,
        "future_response_tokens_present": False,
        "full_sequence_teacher_forced_primary": False,
        "original_preflight_gate_relaxed": False,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "resolved_commit_hash": commit,
        "hidden_size": hidden,
        "dtype": "bfloat16",
        "quantization": None,
        "cosine_min_required": COSINE_MIN,
        "n_comparisons": len(rows),
        "comparisons": rows,
        "repeatability_min_cosine": min_cos,
        "repeatability_pass": min_cos >= COSINE_MIN,
        "sample_example_ids": [r["example_id"] for r, _ in sample],
        "test_layers": test_layers,
        "test_ks": test_ks,
        "model_load_seconds": model_load_s,
        "wall_seconds": wall,
        "estimated_cost_usd": (wall / 3600.0) * L40S_USD_PER_HOUR,
        "l40s_usd_per_hour_assumed": L40S_USD_PER_HOUR,
        "gpu_type": "L40S",
        "locked_test_used": False,
        "surface_baseline_freeze_sha256": FREEZE_SHA256,
    }
    out = Path(ART_DIR) / run_id
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    (out / "preflight_truncated.json").write_text(text, encoding="utf-8")
    result["artifact_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    artifact_volume.commit()
    model_volume.commit()
    return result


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 30,
    volumes={MODEL_CACHE_DIR: model_volume, ART_DIR: artifact_volume},
    memory=65536,
)
def run_benchmark(examples_jsonl: str, *, run_id: str, n_forwards: int = 100) -> dict[str, Any]:
    """Bounded throughput benchmark (~100 truncated forwards) for cost projection."""
    import numpy as np
    import torch

    t0 = time.time()
    examples = [json.loads(x) for x in examples_jsonl.splitlines() if x.strip()]
    tokenizer, model, commit = _load_model(MODEL_CACHE_DIR)
    model_load_s = time.time() - t0

    rng = np.random.default_rng(42)
    jobs: list[tuple[list[int], int]] = []
    for row in examples:
        b = _analyze_boundary(row["input_formatted"], row["model_outputs"], tokenizer)
        if not b["prompt_prefix_exact"] or b["boundary_token_straddle"]:
            raise RuntimeError(f"boundary fail {row['example_id']}")
        for k in K_VALUES:
            if not _eligible(len(b["response_suffix_ids"]), k):
                continue
            jobs.append((_truncated_ids(b["prompt_ids"], b["response_suffix_ids"], k), k))
    if len(jobs) < n_forwards:
        raise RuntimeError(f"only {len(jobs)} candidate forwards")
    idx = rng.choice(len(jobs), size=n_forwards, replace=False)
    selected = [jobs[i] for i in idx.tolist()]

    captured, handles = _register_hooks(model, LAYERS)
    total_tokens = 0
    t_fwd0 = time.time()
    try:
        with torch.inference_mode():
            for ids, _k in selected:
                _acts, _lg = _forward_final_token(model, captured, ids, LAYERS)
                total_tokens += len(ids)
    finally:
        for h in handles:
            h.remove()
    fwd_s = time.time() - t_fwd0
    wall = time.time() - t0
    forwards_per_sec = n_forwards / fwd_s if fwd_s > 0 else None
    # Projected scientific forwards:
    # k0 groups ≈ 306; k>0 = sum eligible train+val across k>0
    n_k0 = 306
    n_kgt0 = sum(
        EXPECTED_ELIGIBLE[k][0] + EXPECTED_ELIGIBLE[k][1] for k in K_GT0
    )
    n_scientific = n_k0 + n_kgt0
    projected_extract_s = (n_scientific / forwards_per_sec) if forwards_per_sec else None
    # include one model load equivalent
    projected_wall_s = (projected_extract_s or 0) + model_load_s
    projected_cost = (projected_wall_s / 3600.0) * L40S_USD_PER_HOUR

    result = {
        "run_id": run_id,
        "mode": "benchmark",
        "extraction_mode": "truncated_prefix_single_example",
        "batch_size": 1,
        "n_benchmark_forwards": n_forwards,
        "benchmark_forward_seconds": fwd_s,
        "model_load_seconds": model_load_s,
        "wall_seconds": wall,
        "forwards_per_sec": forwards_per_sec,
        "tokens_per_sec": (total_tokens / fwd_s) if fwd_s > 0 else None,
        "total_tokens": total_tokens,
        "projected_n_scientific_forwards": n_scientific,
        "projected_k0_forwards": n_k0,
        "projected_kgt0_forwards": n_kgt0,
        "projected_extract_seconds": projected_extract_s,
        "projected_wall_seconds": projected_wall_s,
        "projected_cost_usd": projected_cost,
        "soft_budget_usd": SOFT_BUDGET_USD,
        "within_soft_budget": projected_cost <= SOFT_BUDGET_USD,
        "model_revision": MODEL_REVISION,
        "resolved_commit_hash": commit,
        "estimated_benchmark_cost_usd": (wall / 3600.0) * L40S_USD_PER_HOUR,
        "l40s_usd_per_hour_assumed": L40S_USD_PER_HOUR,
        "gpu_type": "L40S",
        "locked_test_used": False,
        "original_preflight_gate_relaxed": False,
        "surface_baseline_freeze_sha256": FREEZE_SHA256,
    }
    out = Path(ART_DIR) / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifact_volume.commit()
    model_volume.commit()
    return result


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 4,
    volumes={MODEL_CACHE_DIR: model_volume, ART_DIR: artifact_volume},
    memory=65536,
)
def extract_truncated_dev(
    examples_jsonl: str,
    *,
    run_id: str,
) -> dict[str, Any]:
    """Full train/val truncated-prefix batch-size-1 extraction."""
    import hashlib

    import numpy as np
    import torch
    from safetensors.numpy import save_file

    t0 = time.time()
    examples = [json.loads(x) for x in examples_jsonl.splitlines() if x.strip()]
    for row in examples:
        if row.get("split") == "locked_test":
            raise RuntimeError("locked_test leaked")

    tokenizer, model, commit = _load_model(MODEL_CACHE_DIR)
    model_load_s = time.time() - t0
    hidden = int(model.config.hidden_size)
    out_dir = Path(ART_DIR) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    prepared = []
    for row in examples:
        b = _analyze_boundary(row["input_formatted"], row["model_outputs"], tokenizer)
        if not b["prompt_prefix_exact"] or b["boundary_token_straddle"]:
            raise RuntimeError(f"STOP boundary {row['example_id']}")
        prepared.append({"row": row, "b": b})

    # Eligibility counts must match freeze
    for split_name in ("phase3_train", "phase3_validation"):
        rows_s = [p for p in prepared if p["row"]["split"] == split_name]
        for k in K_VALUES:
            n = sum(
                1
                for p in rows_s
                if _eligible(len(p["b"]["response_suffix_ids"]), k)
            )
            expect = EXPECTED_ELIGIBLE[k][0 if split_name == "phase3_train" else 1]
            if n != expect:
                raise RuntimeError(
                    f"eligibility mismatch {split_name} k={k}: got {n} expected {expect}"
                )

    captured, handles = _register_hooks(model, LAYERS)
    n_forwards = 0
    total_tokens = 0

    try:
        with torch.inference_mode():
            # ---- k0 once per prompt group ----
            t_k0 = time.time()
            groups: dict[str, dict] = {}
            for item in prepared:
                ph = item["row"]["prompt_sha256"]
                if ph not in groups:
                    groups[ph] = {
                        "prompt_sha256": ph,
                        "prompt_ids": item["b"]["prompt_ids"],
                        "splits": set(),
                        "example_ids": [],
                    }
                groups[ph]["example_ids"].append(item["row"]["example_id"])
                groups[ph]["splits"].add(item["row"]["split"])
            group_list = list(groups.values())
            n_groups = len(group_list)
            if n_groups != 306:
                raise RuntimeError(f"expected 306 prompt groups, got {n_groups}")
            k0_acts = np.zeros((n_groups, len(LAYERS), hidden), dtype=np.float16)
            k0_logits = np.zeros((n_groups, 3), dtype=np.float32)
            for gi, g in enumerate(group_list):
                acts, lg = _forward_final_token(
                    model, captured, g["prompt_ids"], LAYERS
                )
                k0_acts[gi] = acts.astype(np.float16)
                k0_logits[gi] = np.asarray(lg, dtype=np.float32)
                n_forwards += 1
                total_tokens += len(g["prompt_ids"])
            k0_s = time.time() - t_k0
            save_file(
                {"activations": k0_acts, "logit_summaries": k0_logits},
                str(out_dir / "k0_prompt_groups.safetensors"),
            )
            group_map = [
                {
                    "prompt_sha256": g["prompt_sha256"],
                    "group_index": i,
                    "example_ids": g["example_ids"],
                    "splits": sorted(g["splits"]),
                    "n_prompt_tokens": len(g["prompt_ids"]),
                }
                for i, g in enumerate(group_list)
            ]
            (out_dir / "k0_group_map.json").write_text(
                json.dumps(group_map, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            # ---- k>0 row/k truncated forwards ----
            t_tr = time.time()
            n = len(prepared)
            k_index = {k: i for i, k in enumerate(K_VALUES)}
            shard_meta = []
            for start in range(0, n, SHARD_SIZE):
                shard = prepared[start : start + SHARD_SIZE]
                sn = len(shard)
                acts = np.zeros(
                    (sn, len(LAYERS), len(K_VALUES), hidden), dtype=np.float16
                )
                valid = np.zeros((sn, len(K_VALUES)), dtype=np.uint8)
                logits = np.zeros((sn, len(K_VALUES), 3), dtype=np.float32)
                example_ids = []
                labels = []
                splits = []
                prompt_hashes = []
                resp_lens = []

                for bi, item in enumerate(shard):
                    row = item["row"]
                    b = item["b"]
                    example_ids.append(row["example_id"])
                    labels.append(int(row["eventual_deception"]))
                    splits.append(row["split"])
                    prompt_hashes.append(row["prompt_sha256"])
                    resp_len = len(b["response_suffix_ids"])
                    resp_lens.append(resp_len)
                    # k0 filled later from group artifact; mark invalid here
                    valid[bi, k_index[0]] = 0
                    for k in K_GT0:
                        if not _eligible(resp_len, k):
                            continue
                        ids = _truncated_ids(b["prompt_ids"], b["response_suffix_ids"], k)
                        a, lg = _forward_final_token(model, captured, ids, LAYERS)
                        ki = k_index[k]
                        acts[bi, :, ki, :] = a.astype(np.float16)
                        logits[bi, ki] = np.asarray(lg, dtype=np.float32)
                        valid[bi, ki] = 1
                        n_forwards += 1
                        total_tokens += len(ids)

                shard_name = (
                    f"trajectory_shard_{start:05d}_{start + sn - 1:05d}.safetensors"
                )
                save_file(
                    {
                        "activations": acts,
                        "valid_mask": valid,
                        "logit_summaries": logits,
                        "labels": np.asarray(labels, dtype=np.int64),
                        "response_n_tokens": np.asarray(resp_lens, dtype=np.int64),
                    },
                    str(out_dir / shard_name),
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
                    "dtype": "float16_storage_of_bf16_compute",
                    "model_revision": MODEL_REVISION,
                    "extraction_mode": "truncated_prefix_single_example",
                    "batch_size": 1,
                }
                meta_name = shard_name.replace(".safetensors", "_meta.json")
                (out_dir / meta_name).write_text(
                    json.dumps(meta, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                blob = (out_dir / shard_name).read_bytes()
                shard_meta.append(
                    {
                        "shard": shard_name,
                        "meta": meta_name,
                        "n_rows": sn,
                        "sha256": hashlib.sha256(blob).hexdigest(),
                        "bytes": len(blob),
                        "example_ids_sha256": meta["example_ids_sha256"],
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
        "mode": "extract_truncated_dev",
        "extraction_mode": "truncated_prefix_single_example",
        "batch_size": 1,
        "future_response_tokens_present": False,
        "full_sequence_teacher_forced_primary": False,
        "original_preflight_gate_relaxed": False,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "resolved_commit_hash": commit,
        "dataset_revision": "bf93584916fbd23121eca6f2017017df0ef3184f",
        "dtype": "bfloat16",
        "quantization": None,
        "layers": LAYERS,
        "k_values": K_VALUES,
        "hidden_size": hidden,
        "n_examples": len(prepared),
        "n_prompt_groups": n_groups,
        "n_scientific_forwards": n_forwards,
        "splits_present": sorted({p["row"]["split"] for p in prepared}),
        "locked_test_present": False,
        "regime_c_run": False,
        "causal_interventions": False,
        "surface_baseline_freeze_sha256": FREEZE_SHA256,
        "model_load_seconds": model_load_s,
        "k0_extract_seconds": k0_s,
        "trajectory_extract_seconds": traj_s,
        "wall_seconds": wall,
        "total_tokens": total_tokens,
        "forwards_per_sec": n_forwards / (k0_s + traj_s) if (k0_s + traj_s) else None,
        "tokens_per_sec": total_tokens / (k0_s + traj_s) if (k0_s + traj_s) else None,
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
        "software_versions": {
            "torch": "2.4.1",
            "transformers": "4.44.2",
            "accelerate": "0.33.0",
            "safetensors": "0.4.4",
            "python": "3.11",
        },
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifact_volume.commit()
    model_volume.commit()
    return manifest


@app.local_entrypoint()
def main(mode: str = "preflight_truncated", out_dir: str = "") -> None:
    import shutil
    import subprocess
    import uuid
    from datetime import datetime

    data_dir = REPO_ROOT / "data" / "processed" / "phase3_roleplay"
    freeze = REPO_ROOT / "artifacts/phase3b_dev/surface_baseline_freeze.json"
    import hashlib

    freeze_sha = hashlib.sha256(freeze.read_bytes()).hexdigest()
    if freeze_sha != FREEZE_SHA256:
        raise SystemExit(f"surface freeze hash drift: {freeze_sha}")

    def slim_lines(path: Path, split: str) -> list[str]:
        out = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("split") == "locked_test":
                    raise SystemExit("locked_test leaked")
                out.append(
                    json.dumps(
                        {
                            "example_id": row["example_id"],
                            "split": split,
                            "prompt_sha256": row["prompt_sha256"],
                            "input_formatted": row["input_formatted"],
                            "model_outputs": row["model_outputs"],
                            "eventual_deception": int(row["eventual_deception"]),
                        },
                        sort_keys=True,
                    )
                )
        return out

    train_lines = slim_lines(data_dir / "phase3_train_metadata.jsonl", "phase3_train")
    val_lines = slim_lines(
        data_dir / "phase3_validation_metadata.jsonl", "phase3_validation"
    )
    run_id = (
        f"phase3b1_{mode}_"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_"
        f"{uuid.uuid4().hex[:8]}"
    )
    local_out = (
        Path(out_dir) if out_dir else (REPO_ROOT / "artifacts" / "runs" / run_id)
    )
    local_out.mkdir(parents=True, exist_ok=True)

    if mode == "preflight_truncated":
        payload = "\n".join(train_lines) + "\n"
        print(f"Launching truncated preflight run_id={run_id}")
        result = run_preflight_truncated.remote(payload, run_id=run_id)
        (local_out / "preflight_truncated.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        # Also copy commit-safe summary
        slim = {
            k: result[k]
            for k in result
            if k != "comparisons"
        }
        (REPO_ROOT / "artifacts/phase3b_dev/preflight_truncated_summary.json").write_text(
            json.dumps(slim, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({
            "run_id": run_id,
            "repeatability_pass": result["repeatability_pass"],
            "repeatability_min_cosine": result["repeatability_min_cosine"],
            "estimated_cost_usd": result["estimated_cost_usd"],
            "out_dir": str(local_out),
        }, indent=2))
        if not result["repeatability_pass"]:
            raise SystemExit("TRUNCATED PREFLIGHT FAILED")
        return

    if mode == "benchmark":
        payload = "\n".join(train_lines) + "\n"
        print(f"Launching benchmark run_id={run_id}")
        result = run_benchmark.remote(payload, run_id=run_id, n_forwards=100)
        (local_out / "benchmark.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (REPO_ROOT / "artifacts/phase3b_dev/benchmark_summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({
            "run_id": run_id,
            "forwards_per_sec": result["forwards_per_sec"],
            "projected_cost_usd": result["projected_cost_usd"],
            "within_soft_budget": result["within_soft_budget"],
            "projected_n_scientific_forwards": result["projected_n_scientific_forwards"],
        }, indent=2))
        if not result["within_soft_budget"]:
            raise SystemExit(
                f"PROJECTED COST {result['projected_cost_usd']} > soft budget "
                f"{SOFT_BUDGET_USD} — STOP before full extraction"
            )
        return

    if mode != "extract":
        raise SystemExit(f"Unknown mode {mode}")

    # Require prior benchmark within budget if present
    bench = REPO_ROOT / "artifacts/phase3b_dev/benchmark_summary.json"
    if bench.is_file():
        b = json.loads(bench.read_text(encoding="utf-8"))
        if not b.get("within_soft_budget", False):
            raise SystemExit("benchmark exceeded soft budget — refuse extract")

    payload = "\n".join(train_lines + val_lines) + "\n"
    print(f"Launching extract run_id={run_id} n={len(train_lines)+len(val_lines)}")
    result = extract_truncated_dev.remote(payload, run_id=run_id)
    (local_out / "run_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Downloading volume artifacts {run_id} ...")
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

    # Commit-safe manifest pointer
    (REPO_ROOT / "artifacts/phase3b_dev/latest_extract_manifest.json").write_text(
        json.dumps(
            {
                "run_id": result["run_id"],
                "local_dir": str(local_out),
                "extraction_mode": result["extraction_mode"],
                "batch_size": result["batch_size"],
                "n_scientific_forwards": result["n_scientific_forwards"],
                "estimated_cost_usd": result["estimated_cost_usd"],
                "wall_seconds": result["wall_seconds"],
                "total_artifact_bytes": result["total_artifact_bytes"],
                "k0_sha256": result["k0_artifact"]["sha256"],
                "shards": [
                    {"shard": s["shard"], "sha256": s["sha256"]}
                    for s in result["trajectory_shards"]
                ],
                "locked_test_present": False,
                "original_preflight_gate_relaxed": False,
                "surface_baseline_freeze_sha256": FREEZE_SHA256,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "run_id": run_id,
        "out_dir": str(local_out),
        "n_examples": result["n_examples"],
        "n_scientific_forwards": result["n_scientific_forwards"],
        "estimated_cost_usd": result["estimated_cost_usd"],
        "total_artifact_bytes": result["total_artifact_bytes"],
    }, indent=2))
