"""Modal: Phase-24B live activation capture canary + 96-trajectory pilot + Gemma grade.

Usage:

    uv run modal run modal/phase24b_live_capture.py
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

APP_NAME = "pre-output-physiology-phase24b-live"
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/experiments/phase24b_live_capture.yaml"
SCHEDULE_PATH = REPO_ROOT / "artifacts/phase24b_live/schedule.json"
CANARY_PATH = REPO_ROOT / "artifacts/phase24b_live/canary_manifest.json"
OUT_ROOT = REPO_ROOT / "artifacts/phase24b_live"

A100_USD_PER_HOUR = 2.50
GPU_TYPE = "A100-80GB"

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
CAPTURE_VOL_DIR = "/vol/hf_cache/phase24b_captures"


def _dirty_ok() -> None:
    raw = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True
    ).strip()
    if not raw:
        return
    for line in raw.splitlines():
        path = line[3:].strip().strip('"')
        if "phase24" in path.replace("\\", "/") or path.startswith(
            ("docs/decision_log.md", "reports/phase24b")
        ):
            continue
        raise SystemExit(f"STOP: dirty tree:\n{raw}")


def _load_auth() -> dict[str, Any]:
    _dirty_ok()
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if cfg.get("status") != "phase24b_live_capture_authorized":
        raise SystemExit(f"requires authorized, got {cfg.get('status')}")
    auth = cfg["authorizations"]
    for key in (
        "probe_fitting_authorized",
        "physiology_authorized",
        "sae_analysis_authorized",
        "causal_intervention_authorized",
        "k_gt_20_generation_authorized",
        "k20_replay_extraction_authorized",
        "onset_grading_authorized",
        "openai_grading_api_authorized",
        "held_out_mixed_prompts_authorized",
    ):
        if auth.get(key) is not False:
            raise SystemExit(f"{key} must be false")
    return cfg


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
def run_canary(rows_json: str) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from pre_output_physiology.phase21_roleplay import format_prompt_for_generation
    from pre_output_physiology.phase24b_live import (
        ATTN_IMPLEMENTATION,
        MAX_NEW_TOKENS,
        MODEL_ID,
        MODEL_REVISION,
        TEMPERATURE,
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

    results = []
    with torch.inference_mode():
        for r in rows:
            prompt_text = format_prompt_for_generation(
                tok, r["scenario"], r["question"], r["answer_prefix"]
            )
            prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
            seed = int(r["sample_seed"])
            off = _generate_one(
                model,
                tok,
                prompt_ids=prompt_ids,
                seed=seed,
                capture=False,
                temperature=TEMPERATURE,
                max_new_tokens=MAX_NEW_TOKENS,
                device="cuda:0",
            )
            on = _generate_one(
                model,
                tok,
                prompt_ids=prompt_ids,
                seed=seed,
                capture=True,
                temperature=TEMPERATURE,
                max_new_tokens=MAX_NEW_TOKENS,
                device="cuda:0",
            )
            match = off["generated_token_ids"] == on["generated_token_ids"]
            stop_match = off["stopping_reason"] == on["stopping_reason"]
            results.append(
                {
                    "trajectory_id": r["trajectory_id"],
                    "prompt_id": r["prompt_id"],
                    "sample_seed": seed,
                    "token_sequence_match": match,
                    "stopping_reason_match": stop_match,
                    "n_tokens_off": len(off["generated_token_ids"]),
                    "n_tokens_on": len(on["generated_token_ids"]),
                    "stopping_reason": on["stopping_reason"],
                    "passed": match and stop_match,
                }
            )
            print(
                f"canary {r['trajectory_id']} match={match} stop={stop_match}",
                flush=True,
            )
    passed = all(x["passed"] for x in results) and len(results) == 4
    return {"passed": passed, "results": results}


@app.function(
    image=mistral_image,
    gpu=GPU_TYPE,
    timeout=60 * 60 * 6,
    volumes={MODEL_CACHE: mistral_vol},
    memory=65536,
)
def run_capture(rows_json: str) -> dict[str, Any]:
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
    out_dir = Path(CAPTURE_VOL_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

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
        prompt_text = format_prompt_for_generation(
            tok, r["scenario"], r["question"], r["answer_prefix"]
        )
        prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
        seed = int(r["sample_seed"])
        t_traj = time.time()
        # retry same seed on failure
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
                    "completed": False,
                    "error": last_err,
                }
            )
            continue

        arrays = gen["arrays"]
        traj_path = out_dir / f"{r['trajectory_id']}.npz"
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
            "completed": True,
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
            "grader_split": "development",
        }
        row_out["artifact_ok"] = artifact_integrity_ok(row_out)
        meta_rows.append(row_out)
        print(
            f"[{i + 1}/{len(rows)}] {r['trajectory_id']} "
            f"n={len(gen['generated_token_ids'])} "
            f"align={alignment_ok} bytes={row_out['artifact_bytes']}",
            flush=True,
        )

    mistral_vol.commit()
    wall = time.time() - t0
    completed = [m for m in meta_rows if m.get("completed")]
    return {
        "meta_rows": meta_rows,
        "wall_seconds": wall,
        "n_requested": len(rows),
        "n_completed": len(completed),
        "total_generated_tokens": total_tokens,
        "peak_memory_bytes": peak_mem,
        "capture_dir": CAPTURE_VOL_DIR,
    }


@app.function(
    image=gemma_image,
    gpu="A100-80GB",
    timeout=60 * 60 * 3,
    volumes={MODEL_CACHE: gemma_vol},
    memory=131072,
)
def grade_gemma(rows_json: str) -> dict[str, Any]:
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


@app.function(
    image=mistral_image,
    timeout=60 * 60,
    volumes={MODEL_CACHE: mistral_vol},
)
def fetch_capture_bytes(trajectory_ids_json: str) -> dict[str, bytes]:
    ids = json.loads(trajectory_ids_json)
    out = {}
    for tid in ids:
        path = Path(CAPTURE_VOL_DIR) / f"{tid}.npz"
        out[tid] = path.read_bytes()
    return out


@app.local_entrypoint()
def main() -> None:
    from collections import Counter

    from pre_output_physiology.phase24b_live import (
        GUARANTEE,
        evaluate_behavioral_yield,
        evaluate_capture_gates,
        final_status,
        project_storage_runtime,
    )
    from pre_output_physiology.provenance import utc_now_iso, write_json

    cfg = _load_auth()
    if cfg["authorizations"].get("modal_gpu_mistral_canary_authorized") is not True:
        raise SystemExit("canary not authorized")
    canary = json.loads(CANARY_PATH.read_text(encoding="utf-8"))
    schedule = json.loads(SCHEDULE_PATH.read_text(encoding="utf-8"))
    print(f"Phase-24B canary n={len(canary['rows'])}", flush=True)
    canary_result = run_canary.remote(json.dumps(canary["rows"]))
    write_json(OUT_ROOT / "canary_results.json", canary_result)
    if not canary_result["passed"]:
        print("CANARY FAILED — STOP before 96-trajectory pilot")
        write_json(
            OUT_ROOT / "freeze.json",
            {
                "status": "phase24b_live_capture_not_validated",
                "reason": "canary_token_sequence_mismatch",
                "canary": canary_result,
            },
        )
        # consume canary auth only
        text = CONFIG_PATH.read_text(encoding="utf-8")
        text = text.replace(
            "status: phase24b_live_capture_authorized",
            "status: phase24b_live_capture_not_validated",
            1,
        )
        text = text.replace(
            "modal_gpu_mistral_canary_authorized: true",
            "modal_gpu_mistral_canary_authorized: false",
            1,
        )
        CONFIG_PATH.write_text(text, encoding="utf-8")
        raise SystemExit("STOP: canary failed")

    if cfg["authorizations"].get("modal_gpu_mistral_live_capture_authorized") is not True:
        raise SystemExit("live capture not authorized")
    print(f"Phase-24B capture n={len(schedule['rows'])}", flush=True)
    capture = run_capture.remote(json.dumps(schedule["rows"]))
    completed = [m for m in capture["meta_rows"] if m.get("completed")]
    n_align = sum(1 for m in completed if m.get("alignment_ok"))
    n_act_nan = sum(int(m.get("n_activation_nonfinite", 0)) for m in completed)
    n_log_nan = sum(int(m.get("n_logit_nonfinite", 0)) for m in completed)
    n_art = sum(1 for m in completed if m.get("artifact_ok"))
    eng_gates = evaluate_capture_gates(
        n_requested=capture["n_requested"],
        n_completed=capture["n_completed"],
        n_alignment_ok=n_align,
        n_activation_nan=n_act_nan,
        n_logit_nan=n_log_nan,
        n_artifact_ok=n_art,
    )

    # Fetch a sample of artifacts for local provenance (hashes already recorded);
    # full npz stay on Modal volume (gitignored *.npz). Store metadata locally.
    write_json(OUT_ROOT / "capture_meta.json", capture)

    grade_rows = []
    grade_wall = 0.0
    if eng_gates["passed"] and cfg["authorizations"].get(
        "modal_gpu_gemma_grading_authorized"
    ):
        print(f"Phase-24B Gemma grade n={len(completed)}", flush=True)
        grade = grade_gemma.remote(json.dumps(completed))
        grade_rows = grade["rows"]
        grade_wall = float(grade["wall_seconds"])
        write_json(OUT_ROOT / "grades.json", grade)
    else:
        grade = {"rows": [], "skipped": True}

    # Merge labels
    by_tid = {g["trajectory_id"]: g for g in grade_rows}
    labeled = []
    for m in completed:
        g = by_tid.get(m["trajectory_id"], {})
        labeled.append({**m, **g})

    # Behavioral yield
    per_prompt = []
    for pid in schedule["prompt_ids"]:
        labs = [
            x.get("open_class")
            for x in labeled
            if x["prompt_id"] == pid and x.get("open_valid")
        ]
        c = Counter(labs)
        h, a, d = c.get("honest", 0), c.get("ambiguous", 0), c.get("deceptive", 0)
        per_prompt.append(
            {
                "prompt_id": pid,
                "n_honest": h,
                "n_ambiguous": a,
                "n_deceptive": d,
                "has_ge1_h_and_ge1_d": h >= 1 and d >= 1,
                "has_ge2_h_and_ge2_d": h >= 2 and d >= 2,
            }
        )
    n_h = sum(p["n_honest"] for p in per_prompt)
    n_a = sum(p["n_ambiguous"] for p in per_prompt)
    n_d = sum(p["n_deceptive"] for p in per_prompt)
    beh = evaluate_behavioral_yield(per_prompt, n_honest=n_h, n_deceptive=n_d)

    bytes_list = [m["artifact_bytes"] for m in completed]
    mean_bytes = float(np.mean(bytes_list)) if bytes_list else float("nan")
    total_storage = int(sum(bytes_list))
    gpu_sec_per = (
        float(capture["wall_seconds"]) / max(len(completed), 1) if completed else float("nan")
    )
    tok_per_sec = (
        capture["total_generated_tokens"] / max(capture["wall_seconds"], 1e-9)
    )

    projections = {}
    for n_prompts, label in [(28, "dev_mixed_28"), (11, "heldout_mixed_11"), (39, "all_mixed_39")]:
        projections[label] = {
            str(k): project_storage_runtime(
                bytes_per_trajectory=mean_bytes,
                gpu_seconds_per_trajectory=gpu_sec_per,
                n_prompts=n_prompts,
                trajectories_per_prompt=k,
            )
            for k in (6, 10, 20)
        }

    status = final_status(
        engineering_pass=bool(eng_gates["passed"]),
        behavioral_pass=bool(beh["passed"]) if eng_gates["passed"] else False,
    )
    git_commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"phase24b_live_{ts}_{hashlib.sha256(git_commit.encode()).hexdigest()[:8]}"
    cost = (
        (float(capture["wall_seconds"]) + grade_wall) / 3600.0 * A100_USD_PER_HOUR
    )

    summary = {
        "created_at": utc_now_iso(),
        "run_id": run_id,
        "git_commit": git_commit,
        "status": status,
        "canary": canary_result,
        "engineering_gates": eng_gates,
        "behavioral_yield": beh,
        "n_requested": capture["n_requested"],
        "n_completed": capture["n_completed"],
        "total_generated_tokens": capture["total_generated_tokens"],
        "labels": {"honest": n_h, "ambiguous": n_a, "deceptive": n_d},
        "per_prompt": per_prompt,
        "n_prompts_ge1h_ge1d": sum(1 for p in per_prompt if p["has_ge1_h_and_ge1_d"]),
        "n_prompts_ge2h_ge2d": sum(1 for p in per_prompt if p["has_ge2_h_and_ge2_d"]),
        "bytes_per_trajectory_mean": mean_bytes,
        "total_storage_bytes": total_storage,
        "wall_seconds_capture": capture["wall_seconds"],
        "wall_seconds_grade": grade_wall,
        "gpu": GPU_TYPE,
        "estimated_cost_usd": cost,
        "tokens_per_second": tok_per_sec,
        "peak_memory_bytes": capture["peak_memory_bytes"],
        "projections": projections,
        "pilot_prompt_ids_sha256": schedule["prompt_ids_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "guarantee": GUARANTEE,
        "capture_volume_dir": capture["capture_dir"],
        "activation_coordinates": [
            "embedding_pre_block_residual (bf16 bits as uint16)",
            "post_block_residual layers 0..31 (bf16 bits as uint16)",
            "final_rmsnorm (bf16 bits as uint16)",
        ],
        "logit_storage": "float16 full vocab vector per prediction step",
    }
    run_dir = OUT_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "summary.json", summary)
    write_json(OUT_ROOT / "summary.json", summary)
    write_json(
        OUT_ROOT / "freeze.json",
        {
            "status": status,
            "run_id": run_id,
            "git_commit": git_commit,
            "engineering_passed": eng_gates["passed"],
            "behavioral_passed": beh["passed"] if eng_gates["passed"] else False,
            "canary_passed": canary_result["passed"],
            "authorizations_after": {
                "modal_gpu_mistral_canary_authorized": False,
                "modal_gpu_mistral_live_capture_authorized": False,
                "modal_gpu_gemma_grading_authorized": False,
                "probe_fitting_authorized": False,
                "physiology_authorized": False,
            },
        },
    )
    with (OUT_ROOT / "trajectories.jsonl").open("w", encoding="utf-8") as f:
        for row in labeled:
            # drop huge prompt token lists duplication if needed — keep compact
            compact = {
                k: v
                for k, v in row.items()
                if k
                not in (
                    "scenario",
                    "question",
                )
            }
            f.write(json.dumps(compact, sort_keys=True) + "\n")

    text = CONFIG_PATH.read_text(encoding="utf-8")
    text = text.replace(
        "status: phase24b_live_capture_authorized", f"status: {status}", 1
    )
    for flag in (
        "modal_gpu_mistral_canary_authorized",
        "modal_gpu_mistral_live_capture_authorized",
        "modal_gpu_gemma_grading_authorized",
    ):
        text = text.replace(f"{flag}: true", f"{flag}: false")
    CONFIG_PATH.write_text(text, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": status,
                "run_id": run_id,
                "eng": eng_gates["passed"],
                "beh": beh["passed"],
            },
            indent=2,
        )
    )
