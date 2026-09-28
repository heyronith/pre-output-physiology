"""Modal: Phase-20 B2 decoder features at truncated Stage-1 positions (no label leakage)."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

import modal

APP_NAME = "pre-output-physiology-phase20a2-b2-features"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
L40S_USD_PER_HOUR = 1.95
POSITIONS = ["end0", "end2", "end4"]
POS_OFFSET = {"end0": 0, "end2": 2, "end4": 4}

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_PATH = REPO_ROOT / "data/processed/phase20_design/physiology_prompts.jsonl"
BALANCED_PATH = REPO_ROOT / "data/processed/phase20_design/balanced_subset.jsonl"
POP_SUMMARY = REPO_ROOT / "artifacts/phase20a1_population/population_summary.json"
CONFIG_PATH = REPO_ROOT / "configs/experiments/phase20_preanswer_physiology.yaml"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.44.2",
        "accelerate==0.33.0",
        "numpy==1.26.4",
        "huggingface_hub==0.24.6",
        "pyyaml==6.0.2",
    )
    .add_local_python_source("pre_output_physiology")
)
app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name("preoutput-mistral-cache", create_if_missing=True)
MODEL_CACHE_DIR = "/vol/hf_cache"


def _local_setup() -> list[dict]:
    import yaml

    dirty = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise SystemExit(f"STOP: dirty:\n{dirty}")
    _ = yaml.safe_load(CONFIG_PATH.read_text())
    pop = json.loads(POP_SUMMARY.read_text())
    prompts = {
        json.loads(x)["prompt_group_id"]: json.loads(x)
        for x in PROMPTS_PATH.read_text().splitlines()
        if x.strip()
    }
    balanced = [
        json.loads(x)
        for x in BALANCED_PATH.read_text().splitlines()
        if x.strip()
    ]
    train_bal = [r for r in balanced if r["split"] == "train"]
    conts = {
        json.loads(x)["continuation_id"]: json.loads(x)
        for x in (
            REPO_ROOT / "artifacts/runs" / pop["run_id"] / "continuations.jsonl"
        )
        .read_text()
        .splitlines()
        if x.strip()
    }
    payload = []
    for i, b in enumerate(sorted(train_bal, key=lambda r: r["continuation_id"])):
        c = conts[b["continuation_id"]]
        p = prompts[b["prompt_group_id"]]
        payload.append(
            {
                "row_index": i,
                "continuation_id": b["continuation_id"],
                "prompt_text": p["prompt_text"],
                "consideration_text": c["consideration_text"],
                "record_state": p["record_state"],
                "alternate_state": p["alternate_state"],
            }
        )
    return payload


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 180,
    volumes={MODEL_CACHE_DIR: model_volume},
    memory=65536,
)
def compute_b2(payload_json: str) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from pre_output_physiology.phase8_design import (
        CONTROLLED_PREFIX_TOKEN_ID,
        constrained_choice,
        format_chat,
        validate_candidates,
    )
    from pre_output_physiology.phase14_design import (
        CONSIDERATION_PREFIX,
        RESPONSE_CONTINUATION,
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

    n = len(rows)
    out = {pos: np.zeros((n, 3), dtype=np.float64) for pos in POSITIONS}
    n_forwards = 0

    with torch.inference_mode():
        for r in rows:
            i = int(r["row_index"])
            fmt = format_chat(tok, r["prompt_text"])
            full_text = fmt + CONSIDERATION_PREFIX + r["consideration_text"]
            full_ids = tok.encode(full_text, add_special_tokens=False)

            for pos, off in POS_OFFSET.items():
                trunc_ids = full_ids if off == 0 else full_ids[:-off]
                # Rebuild text that tokenizes to trunc_ids prefix of full_text
                # Use decode of trunc_ids for Response append (same as generation path)
                trunc_text = tok.decode(trunc_ids, skip_special_tokens=False)
                with_r = tok.encode(
                    trunc_text + RESPONSE_CONTINUATION, add_special_tokens=False
                )
                if not with_r or with_r[-1] != CONTROLLED_PREFIX_TOKEN_ID:
                    raise RuntimeError(
                        f"Response != 12107 at {pos} for {r['continuation_id']}"
                    )
                stage2_prefix = list(with_r)

                # Candidate order: record then alternate (feature definition fixed)
                cands = []
                for state in (r["record_state"], r["alternate_state"]):
                    full_c = tok.encode(
                        trunc_text + RESPONSE_CONTINUATION + " " + state,
                        add_special_tokens=False,
                    )
                    if full_c[: len(stage2_prefix)] != stage2_prefix:
                        raise RuntimeError("candidate broke prefix")
                    cands.append(list(full_c[len(stage2_prefix) :]))
                validate_candidates(cands)

                pref = list(stage2_prefix)

                def seq_lp(cand: list[int], pref: list[int] = pref) -> float:
                    nonlocal n_forwards
                    lp = 0.0
                    for t, token in enumerate(cand):
                        ids = torch.tensor(
                            [pref + cand[:t]],
                            dtype=torch.long,
                            device="cuda:0",
                        )
                        n_forwards += 1
                        logits = model(
                            input_ids=ids, attention_mask=torch.ones_like(ids)
                        ).logits[0, -1].float()
                        lp += float(torch.log_softmax(logits, dim=-1)[token].item())
                    return lp

                lp_r = seq_lp(cands[0])
                lp_a = seq_lp(cands[1])
                seq_margin = lp_r - lp_a

                ids0 = torch.tensor([pref], dtype=torch.long, device="cuda:0")
                n_forwards += 1
                logits0 = model(
                    input_ids=ids0, attention_mask=torch.ones_like(ids0)
                ).logits[0, -1].float()
                logp0 = torch.log_softmax(logits0, dim=-1)
                first_div = float((logp0[cands[0][0]] - logp0[cands[1][0]]).item())

                def next_logits(generated_c: list[int], pref: list[int] = pref):
                    nonlocal n_forwards
                    ids = torch.tensor(
                        [pref + generated_c], dtype=torch.long, device="cuda:0"
                    )
                    n_forwards += 1
                    return (
                        model(input_ids=ids, attention_mask=torch.ones_like(ids))
                        .logits[0, -1]
                        .float()
                        .cpu()
                        .numpy()
                    )

                res = constrained_choice(cands, next_logits)
                pred_alt = 1.0 if int(res["chosen_index"]) == 1 else 0.0
                out[pos][i] = [seq_margin, first_div, pred_alt]

    wall = time.time() - t0
    return {
        "features": {pos: out[pos] for pos in POSITIONS},
        "continuation_ids": [
            r["continuation_id"] for r in sorted(rows, key=lambda x: x["row_index"])
        ],
        "n_forwards": n_forwards,
        "wall_seconds": wall,
        "estimated_cost_usd": wall / 3600.0 * L40S_USD_PER_HOUR,
    }


@app.local_entrypoint()
def main() -> None:
    payload = _local_setup()
    print(f"Computing B2 features n={len(payload)}")
    result = compute_b2.remote(json.dumps(payload))
    out_dir = REPO_ROOT / "artifacts/phase20a2_discovery"
    out_dir.mkdir(parents=True, exist_ok=True)
    feats = {
        pos: np.asarray(result["features"][pos], dtype=np.float64) for pos in POSITIONS
    }
    np.savez_compressed(
        out_dir / "b2_features_train.npz",
        end0=feats["end0"],
        end2=feats["end2"],
        end4=feats["end4"],
        continuation_ids=np.asarray(result["continuation_ids"]),
    )
    print(
        json.dumps(
            {
                "n": len(payload),
                "wall_seconds": result["wall_seconds"],
                "estimated_cost_usd": result["estimated_cost_usd"],
                "n_forwards": result["n_forwards"],
            },
            indent=2,
        )
    )
