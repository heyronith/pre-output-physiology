"""Modal: Phase-24D primary live K=16 collection + Gemma grading.

Capture loop (_generate_one / _bf16_u16) is byte-identical to Phase 24B.
Canary not re-run when source SHA matches.

Usage:

    uv run modal run modal/phase24d_live_collection.py
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

APP_NAME = "pre-output-physiology-phase24d-live"
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/experiments/phase24d_primary_live_collection.yaml"
SCHEDULE_PATH = REPO_ROOT / "artifacts/phase24d_collection/schedule.json"
OUT_ROOT = REPO_ROOT / "artifacts/phase24d_collection"

A100_USD_PER_HOUR = 2.50
GPU_TYPE = "A100-80GB"
CHUNK_SIZE = 66  # 528 / 66 = 8 sequential chunks

mistral_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.44.2",
        "accelerate==0.33.0",
        "numpy==1.26.4",
        "huggingface_hub==0.24.6",
        "safetensors==0.4.5",
        "pyyaml==6.0.2",
    )
    .add_local_python_source("pre_output_physiology")
)
gemma_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.7.1",
        "transformers==5.17.0",
        "accelerate==1.15.0",
        "numpy==1.26.4",
        "huggingface_hub==1.5.0",
        "safetensors==0.8.0",
        "pyyaml==6.0.2",
    )
    .add_local_python_source("pre_output_physiology")
)
app = modal.App(APP_NAME)
mistral_vol = modal.Volume.from_name("preoutput-mistral-cache", create_if_missing=True)
gemma_vol = modal.Volume.from_name("preoutput-gemma4-cache", create_if_missing=True)
MODEL_CACHE = "/vol/hf_cache"
CAPTURE_VOL_ROOT = "/vol/hf_cache/phase24d_captures"


def _dirty_ok() -> None:
    raw = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True
    ).strip()
    if not raw:
        return
    for line in raw.splitlines():
        path = line[3:].strip().strip('"')
        if "phase24" in path.replace("\\", "/") or path.startswith(
            ("docs/decision_log.md", "reports/phase24d")
        ):
            continue
        raise SystemExit(f"STOP: dirty tree:\n{raw}")


def _load_auth() -> dict[str, Any]:
    _dirty_ok()
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if cfg.get("status") != "phase24d_primary_live_collection_authorized":
        raise SystemExit(f"requires authorized, got {cfg.get('status')}")
    auth = cfg["authorizations"]
    for key in (
        "probe_fitting_authorized",
        "physiology_authorized",
        "sae_analysis_authorized",
        "causal_intervention_authorized",
        "onset_grading_authorized",
        "openai_grading_api_authorized",
        "test_scientific_analysis_authorized",
        "auroc_analysis_authorized",
        "layer_token_search_authorized",
    ):
        if auth.get(key) is not False:
            raise SystemExit(f"{key} must be false")
    if auth.get("modal_gpu_mistral_live_capture_authorized") is not True:
        raise SystemExit("mistral capture not authorized")
    return cfg


def _split_dir(split: str) -> str:
    return {"train": "TRAIN", "validation": "VALIDATION", "test": "LOCKED_TEST"}[split]


def _bf16_u16(t) -> np.ndarray:
    import torch

    return t.detach().to(torch.bfloat16).cpu().contiguous().view(torch.uint16).numpy()

def _generate_one(
    model,
    tok,
    *,
    prompt_ids: list[int],
    seed: int,
    capture: bool,
    temperature: float,
    max_new_tokens: int,
    device: str,
) -> dict[str, Any]:
    """Explicit cached AR loop; optional activation/logit capture via hooks."""
    import torch
    from transformers import set_seed

    from pre_output_physiology.phase24b_live import N_TRANSFORMER_BLOCKS

    set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    captured: dict[str, Any] = {
        "embedding": [],
        "post_block": [],
        "final_norm": [],
        "logits": [],
        "sampled_logprobs": [],
    }
    hooks = []

    buf: dict[str, Any] = {}

    def _last_pos(out_t):
        if isinstance(out_t, tuple):
            out_t = out_t[0]
        return out_t[0, -1, :]

    if capture:
        def emb_hook(_m, _i, o):
            buf["embedding"] = _last_pos(o).detach()

        def make_layer_hook(idx: int):
            def hook(_m, _i, o):
                buf.setdefault("layers", {})[idx] = _last_pos(o).detach()

            return hook

        def norm_hook(_m, _i, o):
            buf["final_norm"] = _last_pos(o).detach()

        hooks.append(model.model.embed_tokens.register_forward_hook(emb_hook))
        for i in range(N_TRANSFORMER_BLOCKS):
            hooks.append(model.model.layers[i].register_forward_hook(make_layer_hook(i)))
        hooks.append(model.model.norm.register_forward_hook(norm_hook))

    def record_step(logits_1d: torch.Tensor) -> None:
        if not capture:
            return
        # layers must be complete
        layer_stack = torch.stack(
            [buf["layers"][i] for i in range(N_TRANSFORMER_BLOCKS)], dim=0
        )
        for name, tensor in (
            ("embedding", buf["embedding"]),
            ("post_block", layer_stack),
            ("final_norm", buf["final_norm"]),
            ("logits", logits_1d),
        ):
            if not torch.isfinite(tensor.float()).all():
                captured.setdefault("nonfinite", []).append(name)
        captured["embedding"].append(_bf16_u16(buf["embedding"]))
        captured["post_block"].append(_bf16_u16(layer_stack))
        captured["final_norm"].append(_bf16_u16(buf["final_norm"]))
        captured["logits"].append(
            logits_1d.detach().to(torch.float16).cpu().numpy()
        )
        buf.clear()

    generated: list[int] = []
    stopping_reason = "max_new_tokens"
    peak_mem = 0

    with torch.inference_mode():
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        out = model(ids, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1, :].float()
        record_step(logits)

        for _step in range(max_new_tokens):
            if device.startswith("cuda"):
                peak_mem = max(peak_mem, int(torch.cuda.max_memory_allocated()))
            probs = torch.softmax(logits / temperature, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
            logp = float(torch.log(probs[next_id]).item())
            if capture:
                captured["sampled_logprobs"].append(logp)
            generated.append(next_id)
            if next_id == tok.eos_token_id:
                stopping_reason = "eos"
                break
            # Do not forward after the final allowed token: that would create a
            # prediction_step with no corresponding sampled token (off-by-one).
            if len(generated) >= max_new_tokens:
                break
            next_t = torch.tensor([[next_id]], dtype=torch.long, device=device)
            out = model(next_t, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[0, -1, :].float()
            record_step(logits)

    for h in hooks:
        h.remove()

    n_gen = len(generated)
    result: dict[str, Any] = {
        "generated_token_ids": generated,
        "stopping_reason": stopping_reason,
        "n_prediction_steps": n_gen,
        "prompt_n_tokens": len(prompt_ids),
        "peak_memory_bytes": peak_mem,
    }
    if capture:
        if n_gen == 0:
            raise RuntimeError("capture produced zero tokens")
        # sampled_logprobs length == n_gen; activation arrays length == n_gen
        emb = np.stack(captured["embedding"], axis=0)
        post = np.stack(captured["post_block"], axis=0)
        fnorm = np.stack(captured["final_norm"], axis=0)
        logits_arr = np.stack(captured["logits"], axis=0)
        result["arrays"] = {
            "embedding_bf16_u16": emb,
            "post_block_bf16_u16": post,
            "final_norm_bf16_u16": fnorm,
            "logits_f16": logits_arr,
            "sampled_logprobs_f32": np.asarray(
                captured["sampled_logprobs"], dtype=np.float32
            ),
            "generated_token_ids": np.asarray(generated, dtype=np.int32),
            "prompt_token_ids": np.asarray(prompt_ids, dtype=np.int32),
        }
        # integrity
        result["n_activation_nonfinite"] = len(
            [x for x in captured.get("nonfinite", []) if x != "logits"]
        )
        result["n_logit_nonfinite"] = int(
            np.sum(~np.isfinite(logits_arr.astype(np.float32)))
        ) + sum(1 for x in captured.get("nonfinite", []) if x == "logits")
    return result


@app.function(
    image=mistral_image,
    gpu=GPU_TYPE,
    timeout=60 * 60 * 4,
    volumes={MODEL_CACHE: mistral_vol},
    memory=65536,
)
def run_capture_chunk(rows_json: str) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from pre_output_physiology.phase21_roleplay import format_prompt_for_generation
    from pre_output_physiology.phase24b_live import (
        ATTN_IMPLEMENTATION,
        MAX_NEW_TOKENS,
        MODEL_ID,
        MODEL_REVISION,
        TEMPERATURE,
        align_sampled_token_to_prediction_step,
        artifact_integrity_ok,
        prediction_step_bookkeeping,
    )

    rows = json.loads(rows_json)
    tok = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=MODEL_CACHE, use_fast=True
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        cache_dir=MODEL_CACHE,
        attn_implementation=ATTN_IMPLEMENTATION,
        low_cpu_mem_usage=True,
    )
    model.to("cuda:0")
    model.eval()

    meta_rows: list[dict[str, Any]] = []
    t0 = time.time()
    total_tokens = 0
    peak_mem = 0

    for i, r in enumerate(rows):
        torch.cuda.reset_peak_memory_stats()
        split_folder = _split_dir(r["split"])
        out_dir = Path(CAPTURE_VOL_ROOT) / split_folder
        out_dir.mkdir(parents=True, exist_ok=True)
        traj_path = out_dir / f"{r['trajectory_id']}.npz"

        # Resume: if completed artifact exists, load metadata-only skip regen
        # (we still require meta rebuild from arrays when resuming mid-chunk)
        prompt_text = format_prompt_for_generation(
            tok, r["scenario"], r["question"], r["answer_prefix"]
        )
        prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
        seed = int(r["sample_seed"])
        t_traj = time.time()
        last_err = None
        gen = None
        for _attempt in range(3):
            try:
                gen = _generate_one(
                    model,
                    tok,
                    prompt_ids=prompt_ids,
                    seed=seed,
                    capture=True,
                    temperature=TEMPERATURE,
                    max_new_tokens=MAX_NEW_TOKENS,
                    device="cuda:0",
                )
                last_err = None
                break
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
                print(f"retry {r['trajectory_id']}: {exc}", flush=True)
        if gen is None:
            meta_rows.append(
                {
                    "trajectory_id": r["trajectory_id"],
                    "prompt_id": r["prompt_id"],
                    "replicate_index": r["replicate_index"],
                    "sample_seed": seed,
                    "scheduled_seed": seed,
                    "split": r["split"],
                    "sealed": bool(r.get("sealed")),
                    "completed": False,
                    "error": last_err,
                }
            )
            continue

        arrays = gen["arrays"]
        np.savez_compressed(traj_path, **arrays)
        act_bytes = (
            arrays["embedding_bf16_u16"].tobytes()
            + arrays["post_block_bf16_u16"].tobytes()
            + arrays["final_norm_bf16_u16"].tobytes()
        )
        logit_bytes = arrays["logits_f16"].tobytes()
        act_sha = hashlib.sha256(act_bytes).hexdigest()
        logit_sha = hashlib.sha256(logit_bytes).hexdigest()
        file_sha = hashlib.sha256(traj_path.read_bytes()).hexdigest()

        completion = tok.decode(gen["generated_token_ids"], skip_special_tokens=True)
        completion = completion.rstrip("\n ").lstrip("\n")
        full_response = r["answer_prefix"] + completion
        book = prediction_step_bookkeeping(len(gen["generated_token_ids"]))
        aligned = align_sampled_token_to_prediction_step(gen["generated_token_ids"])
        alignment_ok = (
            gen["n_prediction_steps"] == book["n_prediction_steps"]
            and len(arrays["logits_f16"]) == len(gen["generated_token_ids"])
            and all(
                aligned[t]["sampled_token_id"] == gen["generated_token_ids"][t]
                for t in range(len(gen["generated_token_ids"]))
            )
        )
        wall_traj = time.time() - t_traj
        peak_mem = max(peak_mem, int(gen["peak_memory_bytes"]))
        total_tokens += len(gen["generated_token_ids"])

        row_out = {
            "trajectory_id": r["trajectory_id"],
            "prompt_id": r["prompt_id"],
            "replicate_index": r["replicate_index"],
            "sample_seed": seed,
            "scheduled_seed": seed,
            "seed_unchanged": True,
            "completed": True,
            "split": r["split"],
            "sealed": bool(r.get("sealed")),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "tokenizer_revision": MODEL_REVISION,
            "attn_implementation": ATTN_IMPLEMENTATION,
            "dtype": "bfloat16",
            "prompt_token_ids": prompt_ids,
            "generated_token_ids": gen["generated_token_ids"],
            "completion_text": completion,
            "full_response": full_response,
            "answer_prefix": r["answer_prefix"],
            "scenario": r["scenario"],
            "question": r["question"],
            "stopping_reason": gen["stopping_reason"],
            "n_prediction_steps": gen["n_prediction_steps"],
            "prompt_n_tokens": gen["prompt_n_tokens"],
            "activation_sha256": act_sha,
            "logit_sha256": logit_sha,
            "artifact_sha256": file_sha,
            "artifact_path": str(traj_path),
            "artifact_bytes": int(traj_path.stat().st_size),
            "n_activation_nonfinite": gen["n_activation_nonfinite"],
            "n_logit_nonfinite": gen["n_logit_nonfinite"],
            "alignment_ok": alignment_ok,
            "wall_seconds": wall_traj,
            "source": "phase24d_new",
        }
        row_out["artifact_ok"] = artifact_integrity_ok(row_out)
        meta_rows.append(row_out)
        print(
            f"[{i + 1}/{len(rows)}] {r['trajectory_id']} split={r['split']} "
            f"n={len(gen['generated_token_ids'])} align={alignment_ok} "
            f"bytes={row_out['artifact_bytes']}",
            flush=True,
        )

    mistral_vol.commit()
    completed = [m for m in meta_rows if m.get("completed")]
    return {
        "meta_rows": meta_rows,
        "wall_seconds": time.time() - t0,
        "n_requested": len(rows),
        "n_completed": len(completed),
        "total_generated_tokens": total_tokens,
        "peak_memory_bytes": peak_mem,
        "capture_dir": CAPTURE_VOL_ROOT,
    }


@app.function(
    image=gemma_image,
    gpu="A100-80GB",
    timeout=60 * 60 * 4,
    volumes={MODEL_CACHE: gemma_vol},
    memory=131072,
)
def grade_gemma_chunk(rows_json: str) -> dict[str, Any]:
    import os

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from pre_output_physiology.phase23_open_grader import (
        INFERENCE,
        build_grader_messages,
        parse_json_grade,
    )
    from pre_output_physiology.phase24b_live import GEMMA_HF_ID, GEMMA_REVISION

    os.environ["HF_HOME"] = MODEL_CACHE
    rows = json.loads(rows_json)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(
        GEMMA_HF_ID,
        revision=GEMMA_REVISION,
        cache_dir=MODEL_CACHE,
        trust_remote_code=True,
    )
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        GEMMA_HF_ID,
        revision=GEMMA_REVISION,
        torch_dtype=torch.bfloat16,
        cache_dir=MODEL_CACHE,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    out_rows = []
    max_new = int(INFERENCE["max_new_tokens"])
    with torch.inference_mode():
        for i, r in enumerate(rows):
            messages = build_grader_messages(
                r["scenario"], r["question"], r["full_response"]
            )
            try:
                text = tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                text = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            enc = tok(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=int(INFERENCE["max_model_len"]) - max_new,
            )
            enc = {k: v.to(model.device) for k, v in enc.items()}
            gen = model.generate(
                **enc,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
            new_ids = gen[0, enc["input_ids"].shape[1] :]
            decoded = tok.decode(new_ids, skip_special_tokens=True)
            parsed = parse_json_grade(decoded)
            out_rows.append(
                {
                    "trajectory_id": r["trajectory_id"],
                    "prompt_id": r["prompt_id"],
                    "replicate_index": r["replicate_index"],
                    "split": r["split"],
                    "sealed": bool(r.get("sealed")),
                    "open_valid": parsed["valid"],
                    "open_score": parsed.get("score"),
                    "open_class": parsed.get("class"),
                    "open_explanation": parsed.get("explanation"),
                    "grader": GEMMA_HF_ID,
                    "grader_revision": GEMMA_REVISION,
                }
            )
            if (i + 1) % 8 == 0:
                print(f"graded {i + 1}/{len(rows)}", flush=True)
    return {"rows": out_rows, "wall_seconds": time.time() - t0}


@app.local_entrypoint()
def main() -> None:
    from pre_output_physiology.phase24c_design import sha256_file
    from pre_output_physiology.phase24d_collection import (
        GUARANTEE,
        N_NEW_TRAJECTORIES,
        evaluate_integrity_gates_528,
        evaluate_realized_yield,
        final_status,
        realized_yield_counts,
        seal_test_summary,
        verify_phase24c_artifact_hashes,
    )
    from pre_output_physiology.provenance import utc_now_iso, write_json

    cfg = _load_auth()
    ver = verify_phase24c_artifact_hashes(REPO_ROOT)
    if not ver["verified"]:
        raise SystemExit("STOP: Phase-24C hashes failed at GPU launch")

    schedule = json.loads(SCHEDULE_PATH.read_text(encoding="utf-8"))
    split = json.loads(
        (REPO_ROOT / "artifacts/phase24c_design/split_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    new_rows = schedule["new_rows"]
    if len(new_rows) != N_NEW_TRAJECTORIES:
        raise SystemExit(f"STOP: schedule n_new={len(new_rows)}")

    # Capture in sequential chunks (same A100 class; resume-friendly)
    all_meta: list[dict[str, Any]] = []
    capture_wall = 0.0
    total_tokens = 0
    peak_mem = 0
    chunks = [
        new_rows[i : i + CHUNK_SIZE] for i in range(0, len(new_rows), CHUNK_SIZE)
    ]
    print(f"Phase-24D capture: {len(new_rows)} traj in {len(chunks)} chunks", flush=True)
    for ci, chunk in enumerate(chunks):
        print(f"=== capture chunk {ci + 1}/{len(chunks)} n={len(chunk)} ===", flush=True)
        # Drop bulky text fields not needed? Keep all for grading.
        result = run_capture_chunk.remote(json.dumps(chunk))
        all_meta.extend(result["meta_rows"])
        capture_wall += float(result["wall_seconds"])
        total_tokens += int(result["total_generated_tokens"])
        peak_mem = max(peak_mem, int(result["peak_memory_bytes"]))
        write_json(OUT_ROOT / f"capture_chunk_{ci:02d}.json", {"n": len(result["meta_rows"])})

    completed = [m for m in all_meta if m.get("completed")]
    n_align = sum(1 for m in completed if m.get("alignment_ok"))
    n_act_nan = sum(int(m.get("n_activation_nonfinite", 0)) for m in completed)
    n_log_nan = sum(int(m.get("n_logit_nonfinite", 0)) for m in completed)
    n_art = sum(1 for m in completed if m.get("artifact_ok"))
    n_seed = sum(
        1
        for m in completed
        if m.get("seed_unchanged") and m.get("sample_seed") == m.get("scheduled_seed")
    )
    eng = evaluate_integrity_gates_528(
        n_requested=len(new_rows),
        n_completed=len(completed),
        n_alignment_ok=n_align,
        n_activation_nan=n_act_nan,
        n_logit_nan=n_log_nan,
        n_artifact_ok=n_art,
        n_seed_unchanged=n_seed,
    )
    write_json(
        OUT_ROOT / "capture_meta.json",
        {"meta_rows": all_meta, "wall_seconds": capture_wall},
    )

    grade_rows: list[dict[str, Any]] = []
    grade_wall = 0.0
    if eng["passed"] and cfg["authorizations"].get("modal_gpu_gemma_grading_authorized"):
        # Grade only newly completed (exclude sealed scientific use later)
        grade_inputs = completed
        gchunks = [
            grade_inputs[i : i + CHUNK_SIZE]
            for i in range(0, len(grade_inputs), CHUNK_SIZE)
        ]
        print(f"Phase-24D Gemma grade: {len(grade_inputs)} in {len(gchunks)} chunks", flush=True)
        for gi, gchunk in enumerate(gchunks):
            print(f"=== grade chunk {gi + 1}/{len(gchunks)} ===", flush=True)
            # Slim payload for grader
            slim = [
                {
                    "trajectory_id": r["trajectory_id"],
                    "prompt_id": r["prompt_id"],
                    "replicate_index": r["replicate_index"],
                    "split": r["split"],
                    "sealed": r["sealed"],
                    "scenario": r["scenario"],
                    "question": r["question"],
                    "full_response": r["full_response"],
                }
                for r in gchunk
            ]
            gres = grade_gemma_chunk.remote(json.dumps(slim))
            grade_rows.extend(gres["rows"])
            grade_wall += float(gres["wall_seconds"])
        write_json(OUT_ROOT / "grades_new.json", {"rows": grade_rows, "wall_seconds": grade_wall})
    else:
        print("SKIP grading — engineering failed or not authorized", flush=True)

    # Merge labels for new traj
    by_tid = {g["trajectory_id"]: g for g in grade_rows}
    labeled_new = [{**m, **by_tid.get(m["trajectory_id"], {})} for m in completed]

    # Attach Phase-24B reused TRAIN labels (provenance retained)
    traj_24b = [
        json.loads(x)
        for x in (REPO_ROOT / "artifacts/phase24b_live/trajectories.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if x.strip()
    ]
    reuse_by_tid = {r["trajectory_id"]: r for r in traj_24b}
    reused_labeled = []
    for ref in schedule["reuse_refs"]:
        src = reuse_by_tid[ref["trajectory_id"]]
        reused_labeled.append(
            {
                **{k: src[k] for k in src if k not in ("grader_split",)},
                "split": "train",
                "sealed": False,
                "source": "phase24b_live",
                "phase24b_provenance": {
                    "artifact_path": src.get("artifact_path"),
                    "artifact_sha256": src.get("artifact_sha256"),
                    "activation_sha256": src.get("activation_sha256"),
                    "logit_sha256": src.get("logit_sha256"),
                },
            }
        )

    # Full labeled set for TRAIN/VAL analysis only
    train_ids = split["train_prompt_ids"]
    val_ids = split["validation_prompt_ids"]
    test_ids = split["test_prompt_ids"]

    train_labeled = [
        r for r in (labeled_new + reused_labeled) if r["prompt_id"] in set(train_ids)
    ]
    val_labeled = [r for r in labeled_new if r["prompt_id"] in set(val_ids)]
    test_labeled = [r for r in labeled_new if r["prompt_id"] in set(test_ids)]

    train_yield = realized_yield_counts(train_labeled, prompt_ids=train_ids)
    val_yield = realized_yield_counts(val_labeled, prompt_ids=val_ids)
    yield_eval = evaluate_realized_yield(train_yield=train_yield, val_yield=val_yield)

    status = final_status(
        engineering_pass=bool(eng["passed"]),
        yield_pass=bool(yield_eval["passed"]),
    )

    # Write split-isolated artifacts (TEST labels sealed file — hashed but not summarized)
    for folder, rows, name in (
        (
            "TRAIN",
            [r for r in labeled_new if r["split"] == "train"],
            "new_trajectories.jsonl",
        ),
        (
            "VALIDATION",
            [r for r in labeled_new if r["split"] == "validation"],
            "new_trajectories.jsonl",
        ),
        (
            "LOCKED_TEST",
            [r for r in labeled_new if r["split"] == "test"],
            "new_trajectories_SEALED.jsonl",
        ),
    ):
        path = OUT_ROOT / "by_split" / folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(json.dumps(r, sort_keys=True) for r in rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )

    # TRAIN reuse index (pointers only)
    write_json(
        OUT_ROOT / "by_split/TRAIN/phase24b_reuse_index.json",
        {
            "n": len(reused_labeled),
            "trajectory_ids": [r["trajectory_id"] for r in reused_labeled],
            "source_artifact": "artifacts/phase24b_live/trajectories.jsonl",
            "note": "Original Phase-24B provenance retained; not re-copied as new data",
        },
    )

    # Labels files: TRAIN/VAL open; TEST sealed blob
    write_json(
        OUT_ROOT / "by_split/TRAIN/labels.json",
        {
            "rows": [
                {
                    "trajectory_id": r["trajectory_id"],
                    "prompt_id": r["prompt_id"],
                    "replicate_index": r["replicate_index"],
                    "open_class": r.get("open_class"),
                    "open_valid": r.get("open_valid"),
                    "open_score": r.get("open_score"),
                    "source": r.get("source"),
                }
                for r in train_labeled
            ]
        },
    )
    write_json(
        OUT_ROOT / "by_split/VALIDATION/labels.json",
        {
            "rows": [
                {
                    "trajectory_id": r["trajectory_id"],
                    "prompt_id": r["prompt_id"],
                    "replicate_index": r["replicate_index"],
                    "open_class": r.get("open_class"),
                    "open_valid": r.get("open_valid"),
                    "open_score": r.get("open_score"),
                    "source": r.get("source", "phase24d_new"),
                }
                for r in val_labeled
            ]
        },
    )
    sealed_labels_path = OUT_ROOT / "by_split/LOCKED_TEST/labels_SEALED.json"
    write_json(
        sealed_labels_path,
        {
            "sealed": True,
            "n_rows": len(test_labeled),
            "rows": [
                {
                    "trajectory_id": r["trajectory_id"],
                    "prompt_id": r["prompt_id"],
                    "replicate_index": r["replicate_index"],
                    "open_class": r.get("open_class"),
                    "open_valid": r.get("open_valid"),
                    "open_score": r.get("open_score"),
                    "grader": r.get("grader"),
                    "grader_revision": r.get("grader_revision"),
                }
                for r in test_labeled
            ],
            "warning": "DO NOT OPEN for scientific analysis until Phase 24E candidate freeze",
        },
    )

    test_hashes = {
        "new_trajectories_SEALED.jsonl": sha256_file(
            str(OUT_ROOT / "by_split/LOCKED_TEST/new_trajectories_SEALED.jsonl")
        ),
        "labels_SEALED.json": sha256_file(str(sealed_labels_path)),
    }
    # Per-trajectory artifact hashes for TEST (no labels)
    test_hashes["trajectory_artifact_sha256s"] = {
        r["trajectory_id"]: r["artifact_sha256"]
        for r in completed
        if r["split"] == "test"
    }

    test_summary = seal_test_summary(
        n_requested=sum(1 for r in new_rows if r["split"] == "test"),
        n_completed=sum(1 for r in completed if r["split"] == "test"),
        n_graded=sum(1 for r in test_labeled if r.get("open_valid") is not None),
        integrity_ok=all(
            r.get("alignment_ok")
            and r.get("artifact_ok")
            and r.get("n_activation_nonfinite", 1) == 0
            for r in completed
            if r["split"] == "test"
        ),
        artifact_hashes=test_hashes,
    )
    write_json(OUT_ROOT / "by_split/LOCKED_TEST/sealed_summary.json", test_summary)

    bytes_list = [m["artifact_bytes"] for m in completed]
    total_storage = int(sum(bytes_list))
    cost = (capture_wall + grade_wall) / 3600.0 * A100_USD_PER_HOUR
    git_commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"phase24d_live_{ts}_{hashlib.sha256(git_commit.encode()).hexdigest()[:8]}"

    summary = {
        "created_at": utc_now_iso(),
        "run_id": run_id,
        "git_commit": git_commit,
        "starting_sha": "c649203f3f59701b556ea6ff8906e9752190027c",
        "status": status,
        "guarantee": GUARANTEE,
        "phase24c_verification": ver,
        "engineering_gates": eng,
        "n_requested_new": len(new_rows),
        "n_completed_new": len(completed),
        "n_reused_phase24b": len(reused_labeled),
        "n_total_trajectories": len(completed) + len(reused_labeled),
        "total_generated_tokens_new": total_tokens,
        "wall_seconds_capture": capture_wall,
        "wall_seconds_grade": grade_wall,
        "gpu": GPU_TYPE,
        "estimated_cost_usd": cost,
        "peak_memory_bytes": peak_mem,
        "total_new_storage_bytes": total_storage,
        "bytes_per_trajectory_mean": float(np.mean(bytes_list)) if bytes_list else None,
        "train_yield": train_yield,
        "validation_yield": val_yield,
        "realized_yield": yield_eval,
        "locked_test_sealed_summary": test_summary,
        "model": {
            "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
            "revision": "63a8b081895390a26e140280378bc85ec8bce07a",
            "dtype": "bfloat16",
            "attn": "sdpa",
            "temperature": 1.0,
            "max_new_tokens": 200,
        },
        "grader": {
            "hf_id": "google/gemma-4-31B-it",
            "revision": "842da3794eaa0b77d5f08bae87a17459d91ff475",
        },
        "activation_coordinates": [
            "embedding_pre_block_residual (bf16 bits as uint16)",
            "post_block_residual layers 0..31 (bf16 bits as uint16)",
            "final_rmsnorm (bf16 bits as uint16)",
        ],
        "logit_storage": "float16 full vocab vector per prediction step",
        "new_schedule_sha256": schedule["new_schedule_sha256"],
    }
    # Drop accidental TEST H/A/D if any code path leaked
    assert "test_yield" not in summary
    # Sealed summary may list omission keys; forbid live outcome payloads.
    assert "per_prompt" not in test_summary
    assert "totals" not in test_summary
    sealed_blob = json.dumps(test_summary)
    assert '"honest"' not in sealed_blob
    assert '"deceptive"' not in sealed_blob

    run_dir = OUT_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "summary.json", summary)
    write_json(OUT_ROOT / "summary.json", summary)

    # Consume authorizations
    text = CONFIG_PATH.read_text(encoding="utf-8")
    text = text.replace(
        "status: phase24d_primary_live_collection_authorized",
        f"status: {status}",
        1,
    )
    text = text.replace(
        "modal_gpu_mistral_live_capture_authorized: true",
        "modal_gpu_mistral_live_capture_authorized: false",
        1,
    )
    text = text.replace(
        "modal_gpu_gemma_grading_authorized: true",
        "modal_gpu_gemma_grading_authorized: false",
        1,
    )
    CONFIG_PATH.write_text(text, encoding="utf-8")

    write_json(
        OUT_ROOT / "freeze.json",
        {
            "status": status,
            "run_id": run_id,
            "git_commit": git_commit,
            "engineering_passed": eng["passed"],
            "yield_passed": yield_eval["passed"],
            "test_sealed": True,
            "authorizations_after": {
                "modal_gpu_mistral_live_capture_authorized": False,
                "modal_gpu_gemma_grading_authorized": False,
                "physiology_authorized": False,
                "test_scientific_analysis_authorized": False,
            },
        },
    )
    print(
        json.dumps(
            {
                "status": status,
                "eng": eng["passed"],
                "yield": yield_eval["passed"],
                "cost": cost,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
