"""Modal: Phase-23D onset annotation validation for frozen winner.

Usage (after 23C pass + Stage-4 authorized):

    uv run modal run modal/phase23_onset_pilot.py --candidate <winner>
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

import modal

APP_NAME = "pre-output-physiology-phase23-onset"
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/experiments/phase23_open_grader_validation.yaml"
SAMPLE_PATH = REPO_ROOT / "artifacts/phase23_onset_sample/onset_validation.jsonl"
SAMPLE_MANIFEST_PATH = REPO_ROOT / "artifacts/phase23_onset_sample/manifest.json"
DESIGN_PATH = REPO_ROOT / "artifacts/phase23_design/design_matrix.json"
EXPECTED_SAMPLE_SHA = "5a439094f23cbb22ec121a7f2c074466a112a77bac737f36ce7dc6e7ddd357ca"
STAGE4_CANDIDATE = "gemma4_31b_it"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers==5.17.0",
        "accelerate==1.15.0",
        "huggingface_hub==1.5.0",
        "sentencepiece==0.2.0",
        "protobuf==5.29.4",
        "numpy==1.26.4",
        "pyyaml==6.0.2",
        "safetensors==0.8.0",
    )
    .add_local_python_source("pre_output_physiology")
)
app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name("preoutput-open-grader-cache", create_if_missing=True)
CACHE_DIR = "/vol/hf_cache"


def _extract_json(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None


def _dirty_ok() -> None:
    raw = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True
    ).strip()
    if not raw:
        return
    for line in raw.splitlines():
        path = line[3:].strip().strip('"')
        if "phase23" in path.replace("\\", "/"):
            continue
        if path.startswith(
            (
                "artifacts/phase23",
                "reports/phase23",
            )
        ):
            continue
        raise SystemExit(f"STOP: dirty tree:\n{raw}")


@app.function(
    image=image,
    timeout=60 * 60 * 4,
    volumes={CACHE_DIR: model_volume},
    memory=131072,
    secrets=[modal.Secret.from_name("huggingface")],
)
def onset_remote(candidate: str, rows_json: str, gpu: str) -> dict[str, Any]:
    import os

    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    from pre_output_physiology.phase23_open_grader import (
        CANDIDATES,
        INFERENCE,
        build_onset_messages,
    )

    os.environ["HF_HOME"] = CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
    os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE_DIR

    spec = CANDIDATES[candidate]
    rows = json.loads(rows_json)
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(
        spec["hf_id"],
        revision=spec["revision"],
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
    )
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16
    load_kw = dict(
        revision=spec["revision"],
        cache_dir=CACHE_DIR,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(spec["hf_id"], **load_kw)
    except ValueError:
        model = AutoModel.from_pretrained(spec["hf_id"], **load_kw)
    model.eval()

    out_rows: list[dict[str, Any]] = []
    n_in = 0
    n_out = 0
    max_new = int(INFERENCE["max_new_tokens"])

    with torch.inference_mode():
        for i, r in enumerate(rows):
            messages = build_onset_messages(
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
            n_in += int(enc["input_ids"].numel())
            gen = model.generate(
                **enc,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
            new_ids = gen[0, enc["input_ids"].shape[1] :]
            n_out += int(new_ids.numel())
            decoded = tok.decode(new_ids, skip_special_tokens=True)
            parsed = _extract_json(decoded) or {}
            quote = str(parsed.get("quote") or "")
            explicit = parsed.get("explicit_span")
            if isinstance(explicit, str):
                explicit = explicit.lower() in ("true", "1", "yes")
            start = end = None
            if quote and quote in (r.get("full_response") or ""):
                start = r["full_response"].find(quote)
                end = start + len(quote) if start >= 0 else None
            out_rows.append(
                {
                    "continuation_id": r["continuation_id"],
                    "prompt_id": r["prompt_id"],
                    "reference_explicit_span": r.get("explicit_span"),
                    "reference_onset_quote": r.get("onset_quote"),
                    "reference_onset_char_start": r.get("onset_char_start"),
                    "reference_onset_char_end": r.get("onset_char_end"),
                    "reference_onset_token_index": r.get("onset_token_index"),
                    "open_explicit_span": explicit,
                    "open_onset_quote": quote,
                    "open_onset_char_start": start,
                    "open_onset_char_end": end,
                    "open_onset_reason": parsed.get("reason"),
                    "open_raw": decoded,
                    "candidate": candidate,
                }
            )
            if (i + 1) % 10 == 0:
                print(f"onset {i + 1}/{len(rows)}", flush=True)

    return {
        "rows": out_rows,
        "wall_seconds": time.time() - t0,
        "gpu": gpu,
        "n_input_tokens": n_in,
        "n_output_tokens": n_out,
        "model_id": spec["hf_id"],
        "revision": spec["revision"],
    }


@app.local_entrypoint()
def main(candidate: str) -> None:
    from pre_output_physiology.phase23_open_grader import (
        CANDIDATES,
        ONSET_TOKENIZER_HF_ID,
        ONSET_TOKENIZER_REVISION,
        CostReport,
        _sha_json,
    )

    _dirty_ok()
    if candidate != STAGE4_CANDIDATE:
        raise SystemExit(
            f"Stage-23D requires {STAGE4_CANDIDATE}, got {candidate}"
        )
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    auth = cfg.get("authorizations", {})
    if auth.get("stage4_onset_validation_authorized") is not True:
        raise SystemExit("Stage-4 onset validation not authorized")
    if auth.get("modal_gpu_open_grader_inference_authorized") is not True:
        raise SystemExit("open-grader Modal inference not authorized")
    for key in (
        "k_gt_20_generation_authorized",
        "mistral_roleplay_generation_authorized",
        "openai_grading_api_authorized",
        "openai_onset_api_authorized",
        "activation_extraction_authorized",
        "physiology_authorized",
    ):
        if auth.get(key) is not False:
            raise SystemExit(f"{key} must be false")

    sample_man = json.loads(SAMPLE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if sample_man.get("onset_sample_sha256") != EXPECTED_SAMPLE_SHA:
        raise SystemExit("STOP: frozen onset sample SHA mismatch")
    if int(sample_man.get("n", -1)) != 80:
        raise SystemExit("STOP: onset sample n != 80")
    if _sha_json(sample_man["continuation_ids"]) != EXPECTED_SAMPLE_SHA:
        raise SystemExit("STOP: continuation_ids SHA mismatch")

    rows = [
        json.loads(x)
        for x in SAMPLE_PATH.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    if len(rows) != 80:
        raise SystemExit(f"STOP: onset jsonl has {len(rows)} rows, expected 80")
    if {r["continuation_id"] for r in rows} != set(sample_man["continuation_ids"]):
        raise SystemExit("STOP: onset jsonl IDs != frozen manifest IDs")

    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    spec = CANDIDATES[candidate]
    if spec["revision"] != "842da3794eaa0b77d5f08bae87a17459d91ff475":
        raise SystemExit("STOP: Gemma revision mismatch")
    gpu = spec["gpu"]
    print(f"Phase-23D onset: candidate={candidate} gpu={gpu} n={len(rows)}", flush=True)

    result = onset_remote.with_options(gpu=gpu).remote(
        candidate, json.dumps(rows), gpu
    )
    cost = CostReport(
        gpu_type=gpu,
        wall_seconds=float(result["wall_seconds"]),
        usd_per_hour=float(spec["usd_per_hour"]),
        n_responses=len(result["rows"]),
        n_input_tokens=int(result["n_input_tokens"]),
        n_output_tokens=int(result["n_output_tokens"]),
    )
    git_commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    short = hashlib.sha256(git_commit.encode()).hexdigest()[:8]
    run_id = f"phase23d_onset_{candidate}_{ts}_{short}"
    out_dir = REPO_ROOT / "artifacts/runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "onset_judgments.jsonl").open("w", encoding="utf-8") as f:
        for row in result["rows"]:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    judgments_sha = hashlib.sha256(
        (out_dir / "onset_judgments.jsonl").read_bytes()
    ).hexdigest()
    manifest = {
        "run_id": run_id,
        "stage": "23D_onset",
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "candidate": candidate,
        "hf_id": result["model_id"],
        "revision": result["revision"],
        "design_sha256": design["design_sha256"],
        "onset_sample_sha256": EXPECTED_SAMPLE_SHA,
        "metric_tokenizer_hf_id": ONSET_TOKENIZER_HF_ID,
        "metric_tokenizer_revision": ONSET_TOKENIZER_REVISION,
        "n": len(result["rows"]),
        "n_valid_json": sum(
            1 for r in result["rows"] if (r.get("open_raw") or "").strip()
        ),
        "judgments_sha256": judgments_sha,
        "cost": cost.as_dict(),
        "inference": {
            "temperature": 0.0,
            "do_sample": False,
            "max_new_tokens": 256,
            "max_model_len": 8192,
            "engine": "transformers_generate_temp0",
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
