"""Modal GPU entrypoint: Phase 2 block-12 mean activation extraction.

Usage (from repo root, after `uv sync --extra modal` or with modal installed):

  uv run modal run modal/phase2_extract.py --mode preflight
  uv run modal run modal/phase2_extract.py --mode full

Does not train probes. Does not download published activation pickles.
"""

from __future__ import annotations

import json
import time
from datetime import UTC
from pathlib import Path
from typing import Any

import modal

APP_NAME = "pre-output-physiology-phase2"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
TRANSFORMER_BLOCK_INDEX = 12
HOOK_MODULE_PATH = "model.model.layers[12]"
REPO_ROOT = Path(__file__).resolve().parents[1]

# Approximate Modal L40S on-demand rate used for cost tracking (USD / hour).
# Update if Modal pricing changes; recorded transparently in the run manifest.
L40S_USD_PER_HOUR = 1.10

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


def _load_examples(jsonl_text: str) -> list[dict[str, Any]]:
    rows = []
    for line in jsonl_text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60,
    volumes={MODEL_CACHE_DIR: model_volume},
    memory=65536,
)
def extract_block12_means(
    examples_jsonl: str,
    *,
    run_id: str,
    mode: str,
    batch_size: int = 8,
    model_id: str = MODEL_ID,
    model_revision: str = MODEL_REVISION,
    reproducibility_check_n: int = 2,
) -> dict[str, Any]:
    """Teacher-forced forward passes; return mean-pooled float32 activations + metadata."""
    import hashlib

    import numpy as np
    import torch
    from safetensors.numpy import save as safetensors_save
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    examples = _load_examples(examples_jsonl)
    if not examples:
        raise ValueError("No examples provided")

    cache_dir = MODEL_CACHE_DIR
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=model_revision,
        cache_dir=cache_dir,
        use_fast=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=model_revision,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
    )
    model.to("cuda:0")
    model.eval()
    model_load_s = time.time() - t0

    # Verify weights came from the pinned revision via cache path / config metadata.
    commit_hash = getattr(model.config, "_commit_hash", None)
    if commit_hash is None:
        # Fall back: ensure revision directory exists in the HF cache for this pin.
        repo_cache = f"models--{model_id.replace('/', '--')}"
        rev_dir = Path(cache_dir) / repo_cache / "snapshots" / model_revision
        if not rev_dir.exists():
            raise RuntimeError(
                f"Pinned revision snapshot missing in cache: {rev_dir}"
            )
        commit_hash = model_revision
    elif commit_hash != model_revision:
        raise RuntimeError(
            f"Loaded model commit {commit_hash} != requested {model_revision}"
        )

    # Confirm block path
    layers = model.model.layers
    if TRANSFORMER_BLOCK_INDEX >= len(layers):
        raise RuntimeError("transformer_block_index out of range")
    block = layers[TRANSFORMER_BLOCK_INDEX]
    hidden_size = int(model.config.hidden_size)

    hook_hits = {"count": 0}
    captured: dict[str, torch.Tensor] = {}

    def hook_fn(_module, _inp, output):  # noqa: ANN001
        hook_hits["count"] += 1
        act = output[0] if isinstance(output, tuple) else output
        captured["act"] = act

    handle = block.register_forward_hook(hook_fn)

    means: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []
    total_tokens = 0
    t_extract0 = time.time()

    try:
        with torch.inference_mode():
            for start in range(0, len(examples), batch_size):
                batch = examples[start : start + batch_size]
                texts = [row["teacher_forced_text"] for row in batch]
                encoded = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                    add_special_tokens=False,
                )
                input_ids = encoded["input_ids"].to("cuda:0")
                attention_mask = encoded["attention_mask"].to("cuda:0")

                # Sanity: first token should be BOS for Mistral formatted strings
                # (input_formatted already includes <s>).
                captured.clear()
                before = hook_hits["count"]
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
                after = hook_hits["count"]
                if after - before != 1:
                    raise RuntimeError(
                        f"Expected exactly 1 hook fire per forward, got {after - before}"
                    )
                acts = captured["act"]  # [B, T, H] bf16
                if acts.shape[-1] != hidden_size:
                    raise RuntimeError(
                        f"Hidden dim mismatch: {acts.shape[-1]} vs config {hidden_size}"
                    )
                mask = attention_mask.to(dtype=acts.dtype)
                masked = acts * mask.unsqueeze(-1)
                denom = mask.sum(dim=1).clamp(min=1).unsqueeze(1)
                pooled = (masked.sum(dim=1) / denom).float().cpu().numpy()
                if not np.isfinite(pooled).all():
                    raise RuntimeError("NaN/Inf in pooled activations")

                for i, row in enumerate(batch):
                    n_tok = int(attention_mask[i].sum().item())
                    total_tokens += n_tok
                    means.append(pooled[i].astype(np.float32, copy=False))
                    meta_rows.append(
                        {
                            "example_id": row["example_id"],
                            "split": row.get("split", mode),
                            "source_filename": row.get("source_filename"),
                            "text_sha256": row["text_sha256"],
                            "binary_label": int(row["binary_label"]),
                            "scale_label": row.get("scale_label"),
                            "token_count": n_tok,
                        }
                    )
        extract_s = time.time() - t_extract0

        # Determinism check while the hook is still registered.
        repro = []
        with torch.inference_mode():
            for row in examples[:reproducibility_check_n]:
                enc = tokenizer(
                    row["teacher_forced_text"],
                    return_tensors="pt",
                    add_special_tokens=False,
                )
                input_ids = enc["input_ids"].to("cuda:0")
                attention_mask = enc["attention_mask"].to("cuda:0")
                outs = []
                for _ in range(2):
                    captured.clear()
                    _ = model(input_ids=input_ids, attention_mask=attention_mask)
                    acts = captured["act"]
                    mask = attention_mask.to(dtype=acts.dtype)
                    pooled = (
                        (acts * mask.unsqueeze(-1)).sum(dim=1)
                        / mask.sum(dim=1).clamp(min=1).unsqueeze(1)
                    ).float().cpu().numpy()[0]
                    outs.append(pooled)
                max_abs = float(np.max(np.abs(outs[0] - outs[1])))
                repro.append({"example_id": row["example_id"], "max_abs_diff": max_abs})
                if max_abs > 1e-5:
                    raise RuntimeError(
                        f"Non-reproducible activations for {row['example_id']}: "
                        f"max_abs_diff={max_abs}"
                    )
    finally:
        handle.remove()

    activation_matrix = np.stack(means, axis=0).astype(np.float32)
    labels = np.asarray([r["binary_label"] for r in meta_rows], dtype=np.int64)

    buf_bytes = safetensors_save(
        {
            "activations": activation_matrix,
            "labels": labels,
        }
    )
    blob = buf_bytes if isinstance(buf_bytes, (bytes, bytearray)) else bytes(buf_bytes)
    act_sha = hashlib.sha256(blob).hexdigest()

    wall_s = time.time() - t0
    gpu_billable_s = model_load_s + extract_s
    est_cost = (gpu_billable_s / 3600.0) * L40S_USD_PER_HOUR

    model_volume.commit()

    return {
        "run_id": run_id,
        "mode": mode,
        "model_id": model_id,
        "model_revision": model_revision,
        "resolved_commit_hash": commit_hash or model_revision,
        "dtype": "bfloat16",
        "quantization": None,
        "transformer_block_index": TRANSFORMER_BLOCK_INDEX,
        "hook_module_path": HOOK_MODULE_PATH,
        "hidden_size": hidden_size,
        "n_examples": len(examples),
        "activation_shape": list(activation_matrix.shape),
        "batch_size": batch_size,
        "total_tokens": total_tokens,
        "model_load_seconds": model_load_s,
        "extract_seconds": extract_s,
        "wall_seconds": wall_s,
        "examples_per_sec": len(examples) / extract_s if extract_s > 0 else None,
        "tokens_per_sec": total_tokens / extract_s if extract_s > 0 else None,
        "gpu_type": "L40S",
        "num_gpus": 1,
        "estimated_cost_usd": est_cost,
        "l40s_usd_per_hour_assumed": L40S_USD_PER_HOUR,
        "reproducibility_checks": repro,
        "reproducibility_tolerance_max_abs": 1e-5,
        "activations_safetensors_b64": __import__("base64").b64encode(blob).decode("ascii"),
        "activations_sha256": act_sha,
        "metadata_rows": meta_rows,
        "hook_fire_count": hook_hits["count"],
        "resolved_name_or_path": str(getattr(model, "name_or_path", model_id)),
    }


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 20,
    volumes={MODEL_CACHE_DIR: model_volume},
    memory=65536,
)
def generation_smoke(
    prompts_jsonl: str,
    *,
    max_new_tokens: int = 32,
    model_id: str = MODEL_ID,
    model_revision: str = MODEL_REVISION,
) -> dict[str, Any]:
    """Optional tiny generation path check. Not scientific evidence."""
    import json
    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = [json.loads(line) for line in prompts_jsonl.splitlines() if line.strip()]
    if len(prompts) > 16:
        raise ValueError("Smoke test limited to 16 prompts")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=model_revision, cache_dir=MODEL_CACHE_DIR, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=model_revision,
        torch_dtype=torch.bfloat16,
        cache_dir=MODEL_CACHE_DIR,
        low_cpu_mem_usage=True,
    )
    model.to("cuda:0")
    model.eval()
    outputs = []
    with torch.inference_mode():
        for row in prompts:
            enc = tokenizer(row["prompt"], return_tensors="pt", add_special_tokens=False)
            enc = {k: v.to("cuda:0") for k, v in enc.items()}
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            new_tokens = gen[0, enc["input_ids"].shape[1] :]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            outputs.append(
                {
                    "example_id": row["example_id"],
                    "n_new_tokens": int(new_tokens.numel()),
                    "output_preview": text[:200],
                }
            )
    wall = time.time() - t0
    model_volume.commit()
    return {
        "n_prompts": len(prompts),
        "max_new_tokens": max_new_tokens,
        "temperature": 0,
        "wall_seconds": wall,
        "estimated_cost_usd": (wall / 3600.0) * L40S_USD_PER_HOUR,
        "outputs": outputs,
        "scientific_evidence": False,
    }


@app.local_entrypoint()
def main(
    mode: str = "preflight",
    batch_size: int = 8,
    examples_path: str = "",
    out_dir: str = "",
) -> None:
    import base64
    import uuid
    from datetime import datetime

    if mode not in {"preflight", "full", "train", "test", "smoke"}:
        raise SystemExit(f"Unknown mode {mode}")

    data_dir = REPO_ROOT / "data" / "processed" / "phase2_roleplay"

    if mode == "smoke":
        prompts_path = data_dir / "smoke_prompts.jsonl"
        result = generation_smoke.remote(prompts_path.read_text(encoding="utf-8"))
        out_root = (
            Path(out_dir)
            if out_dir
            else (
                REPO_ROOT
                / "artifacts"
                / "runs"
                / f"phase2_smoke_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
            )
        )
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "smoke_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({
            "mode": "smoke",
            "out_dir": str(out_root),
            "n_prompts": result["n_prompts"],
            "estimated_cost_usd": result["estimated_cost_usd"],
            "scientific_evidence": False,
        }, indent=2))
        return

    if not examples_path:
        if mode == "preflight":
            examples_path = str(data_dir / "preflight_examples.jsonl")
        elif mode == "full":
            # Concatenate train+test for a single canonical extraction run, tagged by split.
            train = (data_dir / "train_examples.jsonl").read_text(encoding="utf-8")
            test = (data_dir / "test_examples.jsonl").read_text(encoding="utf-8")
            examples_jsonl = train + test
        elif mode in {"train", "test"}:
            examples_path = str(data_dir / f"{mode}_examples.jsonl")
        else:
            raise SystemExit(mode)
    if mode != "full":
        examples_jsonl = Path(examples_path).read_text(encoding="utf-8")

    run_id = (
        f"phase2_{mode}_"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_"
        f"{uuid.uuid4().hex[:8]}"
    )
    print(f"Launching Modal extraction run_id={run_id} mode={mode}")

    result = extract_block12_means.remote(
        examples_jsonl,
        run_id=run_id,
        mode=mode,
        batch_size=batch_size,
    )

    out_root = Path(out_dir) if out_dir else (REPO_ROOT / "artifacts" / "runs" / run_id)
    out_root.mkdir(parents=True, exist_ok=True)

    blob = base64.b64decode(result.pop("activations_safetensors_b64"))
    act_path = out_root / "activations.safetensors"
    act_path.write_bytes(blob)

    meta_path = out_root / "metadata.jsonl"
    with meta_path.open("w", encoding="utf-8") as handle:
        for row in result["metadata_rows"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    # Persist a compact result JSON without the huge b64 field
    compact = {k: v for k, v in result.items() if k != "metadata_rows"}
    compact["metadata_rows_n"] = len(result["metadata_rows"])
    compact["artifacts"] = {
        "activations": str(act_path),
        "metadata": str(meta_path),
    }
    (out_root / "gpu_result.json").write_text(
        json.dumps(compact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(json.dumps({
        "run_id": run_id,
        "out_dir": str(out_root),
        "activation_shape": result["activation_shape"],
        "estimated_cost_usd": result["estimated_cost_usd"],
        "extract_seconds": result["extract_seconds"],
        "reproducibility_checks": result["reproducibility_checks"],
        "activations_sha256": result["activations_sha256"],
    }, indent=2))
