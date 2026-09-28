"""Modal: Phase-21 layer-12 activation extraction at k0 and onset-aligned cutoffs.

Extracts only for mixed-population primary trajectories.
Primary: k1 = t-1; secondary k2/k4/k8; k0 prompt-end NC.

Usage (clean tree; activation authorized; population PASS):

    uv run modal run modal/phase21_extract.py \\
        --gen-run-id <gen> --onset-run-id <onset>
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

APP_NAME = "pre-output-physiology-phase21-extract"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
L40S_USD_PER_HOUR = 1.95
BLOCK = 12

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/experiments/phase21_roleplay_preoutput_physiology.yaml"
PROMPTS_PATH = REPO_ROOT / "data/processed/phase21_roleplay/prompts.jsonl"
POP_PATH = REPO_ROOT / "artifacts/phase21a_population/population_summary.json"

image = (
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
app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name("preoutput-mistral-cache", create_if_missing=True)
MODEL_CACHE_DIR = "/vol/hf_cache"


def _local_setup(
    gen_run_id: str, onset_run_id: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from pre_output_physiology.phase21_roleplay import (
        OFFSET_TOKENS,
        STATUS_POP_PASS,
        cutoff_token_index_for_offset,
    )

    dirty = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise SystemExit(f"STOP: dirty:\n{dirty}")
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if cfg.get("status") != STATUS_POP_PASS:
        raise SystemExit(f"requires {STATUS_POP_PASS}, got {cfg.get('status')}")
    if cfg["authorizations"].get("activation_extraction_authorized") is not True:
        raise SystemExit("activation extraction not authorized")

    pop = json.loads(POP_PATH.read_text(encoding="utf-8"))
    if not pop["gates"]["passed"]:
        raise SystemExit("population gates failed")
    annotated = {
        json.loads(x)["continuation_id"]: json.loads(x)
        for x in (
            REPO_ROOT / "artifacts/runs" / onset_run_id / "annotated.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()
        if x.strip()
    }
    gen = {
        json.loads(x)["continuation_id"]: json.loads(x)
        for x in (
            REPO_ROOT / "artifacts/runs" / gen_run_id / "continuations.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()
        if x.strip()
    }
    prompts = {
        json.loads(x)["prompt_id"]: json.loads(x)
        for x in PROMPTS_PATH.read_text(encoding="utf-8").splitlines()
        if x.strip()
    }

    payload = []
    for pair in pop["pairs"]:
        for role, cid_key in (
            ("honest", "honest_continuation_id"),
            ("deceptive", "deceptive_continuation_id"),
        ):
            cid = pair[cid_key]
            a = annotated[cid]
            g = gen[cid]
            p = prompts[pair["prompt_id"]]
            onset_tok = pair.get("onset_token_index")
            if onset_tok is None:
                onset_tok = a.get("onset_token_index")
            cutoffs: dict[str, int | None] = {}
            for name in OFFSET_TOKENS:
                cutoffs[name] = (
                    cutoff_token_index_for_offset(int(onset_tok), name)
                    if onset_tok is not None
                    else None
                )
            payload.append(
                {
                    "continuation_id": cid,
                    "pair_id": pair["pair_id"],
                    "prompt_id": pair["prompt_id"],
                    "split": pair["split"],
                    "role": role,
                    "label_deceptive": 1 if role == "deceptive" else 0,
                    "scenario": p["scenario"],
                    "question": p["question"],
                    "answer_prefix": g["answer_prefix"],
                    "completion": g["completion"],
                    "onset_token_index": onset_tok,
                    "cutoffs": cutoffs,
                }
            )
    meta = {
        "gen_run_id": gen_run_id,
        "onset_run_id": onset_run_id,
        "n": len(payload),
        "balanced_subset_sha256": pop["balanced_subset_sha256"],
        "git_commit": subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    return meta, payload


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 4,
    volumes={MODEL_CACHE_DIR: model_volume},
    memory=65536,
)
def extract_all(payload_json: str) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from pre_output_physiology.phase21_roleplay import (
        OFFSET_TOKENS,
        format_prompt_for_generation,
    )

    rows = json.loads(payload_json)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=MODEL_CACHE_DIR, use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        cache_dir=MODEL_CACHE_DIR,
        low_cpu_mem_usage=True,
    )
    model.to("cuda:0")
    model.eval()

    hidden = model.config.hidden_size
    n = len(rows)
    acts = {name: np.full((n, hidden), np.nan, dtype=np.float32) for name in ["k0", *OFFSET_TOKENS]}
    logits_at = {
        name: np.full((n, model.config.vocab_size), np.nan, dtype=np.float32)
        for name in OFFSET_TOKENS
    }
    texts_at = {name: [""] * n for name in OFFSET_TOKENS}
    meta_rows: list[dict[str, Any]] = []
    k0_by_prompt: dict[str, np.ndarray] = {}
    k0_ok = True

    with torch.inference_mode():
        for i, r in enumerate(rows):
            prompt_text = format_prompt_for_generation(
                tok, r["scenario"], r["question"], r["answer_prefix"]
            )
            prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
            # Response tokens = prefix + completion
            resp_ids = tok.encode(
                r["answer_prefix"] + r["completion"], add_special_tokens=False
            )
            # Full sequence for truncated prefixes: prompt (with answer_prefix in chat) +
            # additional completion tokens beyond the prefix already in prompt_text.
            # prompt_text already ends with answer_prefix; append only completion tokens.
            completion_ids = tok.encode(r["completion"], add_special_tokens=False)
            # Prefer reconstructing response-relative indexing via resp_ids length.
            # Map response token index -> absolute index in prompt+extra_completion.
            # Since answer_prefix is inside prompt_text, absolute index for response
            # token t is: len(prompt_ids) - len(prefix_ids) + t
            prefix_ids = tok.encode(r["answer_prefix"], add_special_tokens=False)
            prefix_len_in_prompt = len(prefix_ids)
            resp_start_abs = len(prompt_ids) - prefix_len_in_prompt
            if resp_start_abs < 0:
                raise RuntimeError("answer_prefix not aligned with prompt_ids")

            captured: dict[str, torch.Tensor] = {}

            def make_hook(cap: dict[str, torch.Tensor] = captured):
                def hook(_m, _inp, output, cap: dict[str, torch.Tensor] = cap):  # noqa: ANN001
                    act = output[0] if isinstance(output, tuple) else output
                    cap["act"] = act

                return hook

            handle = model.model.layers[BLOCK].register_forward_hook(make_hook())

            def forward_prefix(
                abs_len: int,
                prompt_ids: list[int] = prompt_ids,
                completion_ids: list[int] = completion_ids,
                captured: dict[str, torch.Tensor] = captured,
            ) -> tuple[np.ndarray, np.ndarray]:
                need_resp_tokens = max(0, abs_len - len(prompt_ids))
                ids_list = list(prompt_ids) + list(completion_ids[:need_resp_tokens])
                ids_list = ids_list[:abs_len]
                ids = torch.tensor([ids_list], dtype=torch.long, device="cuda:0")
                captured.clear()
                out = model(input_ids=ids, attention_mask=torch.ones_like(ids))
                vec = captured["act"][0, -1].float().detach().cpu().numpy()
                logit = out.logits[0, -1].float().detach().cpu().numpy()
                return vec.astype(np.float32), logit.astype(np.float32)

            # k0: end of prompt (before any stochastic completion token)
            k0_vec, _ = forward_prefix(len(prompt_ids))
            acts["k0"][i] = k0_vec
            pid = r["prompt_id"]
            if pid in k0_by_prompt:
                if not np.allclose(k0_by_prompt[pid], k0_vec, rtol=0, atol=1e-5):
                    k0_ok = False
            else:
                k0_by_prompt[pid] = k0_vec.copy()

            avail = {}
            for name, cut in r["cutoffs"].items():
                if cut is None:
                    avail[name] = False
                    continue
                # Need at least cut+1 response tokens present
                if cut >= len(resp_ids):
                    avail[name] = False
                    continue
                abs_len = resp_start_abs + cut + 1
                # Generated prefix text through cutoff (response tokens 0..cut inclusive)
                # Decode resp_ids[:cut+1] for text baseline
                prefix_resp = tok.decode(resp_ids[: cut + 1], skip_special_tokens=True)
                texts_at[name][i] = prefix_resp
                vec, logit = forward_prefix(abs_len)
                acts[name][i] = vec
                logits_at[name][i] = logit
                avail[name] = True

            handle.remove()
            meta_rows.append(
                {
                    "continuation_id": r["continuation_id"],
                    "pair_id": r["pair_id"],
                    "prompt_id": r["prompt_id"],
                    "split": r["split"],
                    "role": r["role"],
                    "label_deceptive": r["label_deceptive"],
                    "onset_token_index": r["onset_token_index"],
                    "cutoffs": r["cutoffs"],
                    "available": avail,
                    "n_response_tokens": len(resp_ids),
                }
            )
            if (i + 1) % 20 == 0:
                print(f"extract {i + 1}/{n}")

    wall = time.time() - t0
    return {
        "activations": {k: v for k, v in acts.items()},
        "logits": {k: v for k, v in logits_at.items()},
        "texts": texts_at,
        "meta_rows": meta_rows,
        "k0_identity_ok": k0_ok,
        "wall_seconds": wall,
        "estimated_cost_usd": wall / 3600.0 * L40S_USD_PER_HOUR,
        "hidden_size": hidden,
        "vocab_size": int(model.config.vocab_size),
    }


@app.local_entrypoint()
def main(gen_run_id: str, onset_run_id: str) -> None:
    from safetensors.numpy import save_file

    meta, payload = _local_setup(gen_run_id, onset_run_id)
    print(f"Extracting n={len(payload)}")
    result = extract_all.remote(json.dumps(payload))
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    short = hashlib.sha256(meta["git_commit"].encode()).hexdigest()[:8]
    run_id = f"phase21_extract_{ts}_{short}"
    out_dir = REPO_ROOT / "artifacts/runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    tensors = {}
    for name, arr in result["activations"].items():
        tensors[f"act_{name}"] = np.asarray(arr, dtype=np.float32)
    for name, arr in result["logits"].items():
        tensors[f"logits_{name}"] = np.asarray(arr, dtype=np.float32)
    labels = np.asarray([r["label_deceptive"] for r in result["meta_rows"]], dtype=np.int64)
    tensors["labels_deceptive"] = labels
    save_file(tensors, str(out_dir / "activations.safetensors"))

    np.savez_compressed(
        out_dir / "prefix_texts.npz",
        **{k: np.asarray(v, dtype=object) for k, v in result["texts"].items()},
    )
    meta_out = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        **meta,
        "k0_identity_ok": result["k0_identity_ok"],
        "wall_seconds": result["wall_seconds"],
        "estimated_cost_usd": result["estimated_cost_usd"],
        "block": BLOCK,
        "rows": result["meta_rows"],
        "continuation_ids": [r["continuation_id"] for r in result["meta_rows"]],
    }
    (out_dir / "extract_meta.json").write_text(
        json.dumps(meta_out, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "n": len(payload),
                "k0_identity_ok": result["k0_identity_ok"],
                "wall_seconds": result["wall_seconds"],
                "estimated_cost_usd": result["estimated_cost_usd"],
            },
            indent=2,
        )
    )
