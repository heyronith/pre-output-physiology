"""Modal: Phase-23A open-grader pilot (200 frozen DEVELOPMENT responses).

Uses HuggingFace Transformers generate (temperature=0). vLLM is preferred where
an image supports the candidate architecture; Transformers is the frozen pilot
engine for cross-model comparability on novel checkpoints.

Usage (clean tree; design frozen):

    uv run modal run modal/phase23_grade_pilot.py --candidate gpt_oss_20b
    uv run modal run modal/phase23_grade_pilot.py --candidate qwen35_27b
    uv run modal run modal/phase23_grade_pilot.py --candidate gemma4_31b_it
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

import modal

APP_NAME = "pre-output-physiology-phase23-grade-pilot"
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/experiments/phase23_open_grader_validation.yaml"
DESIGN_PATH = REPO_ROOT / "artifacts/phase23_design/design_matrix.json"
PILOT_PATH = REPO_ROOT / "data/processed/phase23_open_grader/pilot_200.jsonl"
CORPUS_PATH = REPO_ROOT / "data/processed/phase23_open_grader/reference_corpus.jsonl"

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


# Local analysis artifacts must not block Modal grading runs.
_DIRTY_ALLOW_PREFIXES = (
    "artifacts/phase23a_pilot/",
    "artifacts/phase23b_development/",
    "artifacts/phase23c_locked/",
    "artifacts/phase23d_onset/",
    "reports/phase23a_pilot.md",
    "reports/phase23b_development.md",
    "reports/phase23c_locked.md",
    "reports/phase23d_onset.md",
)


def _dirty_tree_blockers() -> str:
    raw = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True
    ).strip()
    if not raw:
        return ""
    lines = []
    for line in raw.splitlines():
        path = line[3:].strip().strip('"')
        if path.startswith(_DIRTY_ALLOW_PREFIXES):
            continue
        # In-flight Phase-23 implementation (Modal mounts local sources).
        if "phase23" in path.replace("\\", "/"):
            continue
        lines.append(line)
    return "\n".join(lines)


def _load_corpus_rows(grader_split: str) -> list[dict[str, Any]]:
    rows = [
        json.loads(x)
        for x in CORPUS_PATH.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    if grader_split == "pilot":
        pilot = [
            json.loads(x)
            for x in PILOT_PATH.read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]
        if len(pilot) != 200:
            raise SystemExit(f"pilot has {len(pilot)} rows, expected 200")
        return pilot
    filtered = [r for r in rows if r["grader_split"] == grader_split]
    if grader_split == "development" and len(filtered) != 5200:
        raise SystemExit(f"development has {len(filtered)} rows, expected 5200")
    if grader_split == "locked_validation" and len(filtered) != 2220:
        raise SystemExit(f"locked has {len(filtered)} rows, expected 2220")
    return filtered


def _local_setup(
    candidate: str, *, grader_split: str = "pilot"
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from pre_output_physiology.phase23_open_grader import CANDIDATES

    dirty = _dirty_tree_blockers()
    if dirty:
        raise SystemExit(f"STOP: dirty tree:\n{dirty}")

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    auth = cfg.get("authorizations", {})
    if auth.get("modal_gpu_open_grader_inference_authorized") is not True:
        raise SystemExit("open-grader Modal inference not authorized")
    for key in (
        "mistral_roleplay_generation_authorized",
        "openai_grading_api_authorized",
        "openai_onset_api_authorized",
        "activation_extraction_authorized",
        "physiology_authorized",
        "k_gt_20_generation_authorized",
    ):
        if auth.get(key) is not False:
            raise SystemExit(f"{key} must be false")
    if candidate not in CANDIDATES:
        raise SystemExit(f"unknown candidate {candidate}")

    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    rows = _load_corpus_rows(grader_split)

    meta = {
        "candidate": candidate,
        "grader_split": grader_split,
        "candidate_spec": CANDIDATES[candidate],
        "design_sha256": design["design_sha256"],
        "grader_prompt_split_sha256": design["grader_prompt_split_sha256"],
        "pilot_ids_sha256": design["pilot"]["pilot_ids_sha256"],
        "git_commit": subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "n": len(rows),
    }
    return meta, rows


@app.function(
    image=image,
    timeout=60 * 60 * 10,
    volumes={CACHE_DIR: model_volume},
    memory=131072,
    secrets=[modal.Secret.from_name("huggingface")],
)
def grade_pilot_remote(candidate: str, rows_json: str, gpu: str) -> dict[str, Any]:
    import os

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

    from pre_output_physiology.phase23_open_grader import (
        CANDIDATES,
        INFERENCE,
        build_grader_messages,
        parse_json_grade,
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
    batch_size = 4
    max_new = int(INFERENCE["max_new_tokens"])

    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            texts: list[str] = []
            for r in batch:
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
                texts.append(text)

            enc = tok(
                texts,
                return_tensors="pt",
                padding=True,
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
            in_len = enc["input_ids"].shape[1]
            for i, r in enumerate(batch):
                new_ids = gen[i, in_len:]
                # trim pads
                if tok.pad_token_id is not None:
                    new_ids = new_ids[new_ids != tok.pad_token_id]
                n_out += int(new_ids.numel())
                decoded = tok.decode(new_ids, skip_special_tokens=True)
                parsed = parse_json_grade(decoded)
                out_rows.append(
                    {
                        "continuation_id": r["continuation_id"],
                        "prompt_id": r["prompt_id"],
                        "grader_split": r["grader_split"],
                        "reference_label": r["reference_label"],
                        "reference_judgement": r["reference_judgement"],
                        "candidate": candidate,
                        "hf_id": spec["hf_id"],
                        "revision": spec["revision"],
                        "open_valid": parsed["valid"],
                        "open_score": parsed.get("score"),
                        "open_class": parsed.get("class"),
                        "open_explanation": parsed.get("explanation"),
                        "open_raw": parsed.get("raw"),
                    }
                )
            print(f"graded {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)

    wall = time.time() - t0
    return {
        "rows": out_rows,
        "wall_seconds": wall,
        "gpu": gpu,
        "n_input_tokens": n_in,
        "n_output_tokens": n_out,
        "model_id": spec["hf_id"],
        "revision": spec["revision"],
        "dtype": spec["dtype"],
        "engine": "transformers_generate_temp0_batch4",
    }


@app.local_entrypoint()
def main(candidate: str, grader_split: str = "pilot") -> None:
    from pre_output_physiology.phase23_open_grader import CANDIDATES, CostReport

    if grader_split not in ("pilot", "development", "locked_validation"):
        raise SystemExit("grader_split must be pilot|development|locked_validation")
    meta, rows = _local_setup(candidate, grader_split=grader_split)
    spec = CANDIDATES[candidate]
    gpu = spec["gpu"]
    stage = {"pilot": "23A_pilot", "development": "23B_development", "locked_validation": "23C_locked"}[
        grader_split
    ]
    print(
        f"Phase-23 grade: stage={stage} candidate={candidate} gpu={gpu} n={len(rows)}",
        flush=True,
    )

    remote = grade_pilot_remote.with_options(gpu=gpu)
    result = remote.remote(candidate, json.dumps(rows), gpu)

    cost = CostReport(
        gpu_type=gpu,
        wall_seconds=float(result["wall_seconds"]),
        usd_per_hour=float(spec["usd_per_hour"]),
        n_responses=len(result["rows"]),
        n_input_tokens=int(result["n_input_tokens"]),
        n_output_tokens=int(result["n_output_tokens"]),
    )

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    short = hashlib.sha256(meta["git_commit"].encode()).hexdigest()[:8]
    prefix = {
        "pilot": "phase23a_pilot",
        "development": "phase23b_dev",
        "locked_validation": "phase23c_locked",
    }[grader_split]
    run_id = f"{prefix}_{candidate}_{ts}_{short}"
    out_dir = REPO_ROOT / "artifacts/runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "judgments.jsonl").open("w", encoding="utf-8") as f:
        for row in result["rows"]:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "run_id": run_id,
        "stage": stage,
        "grader_split": grader_split,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": meta["git_commit"],
        "candidate": candidate,
        "hf_id": result["model_id"],
        "revision": result["revision"],
        "dtype": result["dtype"],
        "engine": result.get("engine"),
        "design_sha256": meta["design_sha256"],
        "grader_prompt_split_sha256": meta["grader_prompt_split_sha256"],
        "pilot_ids_sha256": meta["pilot_ids_sha256"],
        "n": len(result["rows"]),
        "n_valid": sum(1 for r in result["rows"] if r["open_valid"]),
        "cost": cost.as_dict(),
        "inference": {
            "temperature": 0.0,
            "do_sample": False,
            "max_new_tokens": 256,
            "max_model_len": 8192,
            "batch_size": 4,
            "engine": result.get("engine"),
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
