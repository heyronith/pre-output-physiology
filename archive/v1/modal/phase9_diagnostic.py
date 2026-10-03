"""Modal GPU: Phase 9A assay diagnostic (framing P7/P8 x decoder FREE/CONSTRAINED).

64 prompts (32 bases x 2 framings); each prompt yields both decoder cells (128 evaluations).
Per prompt: natural first-token argmax before the shared prefix, the Phase-8 two-candidate
constrained decoder after 12107 (unchanged), and deterministic greedy free generation after
12107. No activations, no probes. GPU payload carries prompt text and the two candidates in
visible order only (no record/alternate labels).

Usage (clean tree required):

    uv run modal run modal/phase9_diagnostic.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

APP_NAME = "pre-output-physiology-phase9a-diagnostic"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
L40S_USD_PER_HOUR = 1.95
N_EXPECTED = 64
FREE_MAX_NEW_TOKENS = 48

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_PATH = REPO_ROOT / "data/processed/phase9_design/diagnostic_prompts.jsonl"
CONFIG_PATH = REPO_ROOT / "configs/experiments/phase9_policy_flip_diagnostic.yaml"
MATRIX_PATH = REPO_ROOT / "artifacts/phase9a_design/design_matrix.json"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.44.2",
        "accelerate==0.33.0",
        "numpy==1.26.4",
        "huggingface_hub==0.24.6",
    )
    .add_local_python_source("pre_output_physiology")
)
app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name("preoutput-mistral-cache", create_if_missing=True)
MODEL_CACHE_DIR = "/vol/hf_cache"


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _local_setup() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    import yaml
    from transformers import AutoTokenizer

    from pre_output_physiology.phase8_design import candidate_order, candidate_token_ids

    dirty = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise SystemExit(f"STOP: working tree dirty:\n{dirty}")
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if cfg.get("status") != "phase9a_policy_flip_diagnostic_authorized":
        raise SystemExit(f"unexpected status {cfg.get('status')}")
    auth = cfg["authorizations"]
    if auth.get("diagnostic_model_calls_authorized") is not True:
        raise SystemExit("diagnostic not authorized")
    for key in (
        "activation_extraction_authorized",
        "probe_fitting_authorized",
        "phase5_probe_scoring_authorized",
        "causal_intervention_authorized",
        "final_dataset_construction_authorized",
    ):
        if auth.get(key) is not False:
            raise SystemExit(f"{key} must remain false")
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    rows = [
        json.loads(x) for x in PROMPTS_PATH.read_text(encoding="utf-8").splitlines() if x.strip()
    ]
    ordered = sorted(rows, key=lambda r: r["example_id"])
    for f in ("P7", "P8"):
        sub = [r for r in ordered if r["framing"] == f]
        digest = _sha_bytes(
            "\n".join(f"{r['example_id']}\t{r['prompt_text']}" for r in sub).encode()
        )
        key = f"{f.lower()}_prompt_text_sha256"
        if digest != cfg["design"][key] or digest != matrix[key]:
            raise SystemExit(f"{f} prompt hash mismatch {digest}")
    if len(ordered) != N_EXPECTED:
        raise SystemExit("prompt count invalid")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True)
    payload = []
    for i, r in enumerate(ordered):
        c1, c2 = candidate_order(r)
        payload.append(
            {
                "row_index": i,
                "prompt_text": r["prompt_text"],
                "candidate_states": [c1, c2],
                "candidate_ids": [
                    candidate_token_ids(tok, r["prompt_text"], c1),
                    candidate_token_ids(tok, r["prompt_text"], c2),
                ],
            }
        )
    git_commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    prov = {
        "git_commit": git_commit,
        "runner_sha256": _sha_bytes(Path(__file__).read_bytes()),
        "p7_prompt_text_sha256": cfg["design"]["p7_prompt_text_sha256"],
        "p8_prompt_text_sha256": cfg["design"]["p8_prompt_text_sha256"],
    }
    return prov, ordered, payload


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 40,
    volumes={MODEL_CACHE_DIR: model_volume},
    memory=65536,
)
def run_diagnostic(payload_json: str) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from pre_output_physiology.phase8_design import (
        CONTROLLED_PREFIX_TOKEN_ID,
        candidate_token_ids,
        constrained_choice,
        prompt_prefix_ids,
    )

    rows = json.loads(payload_json)
    if len(rows) != N_EXPECTED:
        raise ValueError("payload size")
    for r in rows:
        if set(r) != {"row_index", "prompt_text", "candidate_states", "candidate_ids"}:
            raise ValueError("payload fields not label-free")
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
    out = []
    n_forwards = 0

    def logits_after(ids_list: list[int]) -> Any:
        nonlocal n_forwards
        ids = torch.tensor([ids_list], dtype=torch.long, device="cuda:0")
        n_forwards += 1
        return model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0, -1].float()

    with torch.inference_mode():
        for r in rows:
            prefix = prompt_prefix_ids(tok, r["prompt_text"])
            if prefix[-1] != CONTROLLED_PREFIX_TOKEN_ID:
                raise RuntimeError("prefix does not end in 12107")
            cands = [candidate_token_ids(tok, r["prompt_text"], s) for s in r["candidate_states"]]
            if cands != r["candidate_ids"]:
                raise RuntimeError("remote candidate tokenization differs from local")

            nat = logits_after(prefix[:-1])
            nat_top = int(torch.argmax(nat).item())
            nat_rank_12107 = int((nat > nat[CONTROLLED_PREFIX_TOKEN_ID]).sum().item()) + 1

            def next_logits(generated: list[int], prefix: list[int] = prefix) -> Any:
                return logits_after(prefix + generated).cpu().numpy()

            res = constrained_choice(cands, next_logits)

            ids = torch.tensor([prefix], dtype=torch.long, device="cuda:0")
            gen = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=FREE_MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                temperature=None,
                top_p=None,
                pad_token_id=tok.eos_token_id,
            )
            new_ids = [int(t) for t in gen[0, len(prefix) :].tolist()]
            free_text = tok.decode(
                [CONTROLLED_PREFIX_TOKEN_ID] + new_ids, skip_special_tokens=True
            )
            out.append(
                {
                    "row_index": r["row_index"],
                    "natural_first_token_id": nat_top,
                    "natural_first_token_text": tok.decode([nat_top]),
                    "natural_rank_of_12107": nat_rank_12107,
                    "constrained_chosen_visible_index": res["chosen_index"],
                    "constrained_generated_ids": res["generated"],
                    "constrained_steps": res["steps"],
                    "free_generated_ids": new_ids,
                    "free_first_token_id": new_ids[0] if new_ids else None,
                    "free_raw_text": free_text,
                    "free_hit_max_new_tokens": len(new_ids) >= FREE_MAX_NEW_TOKENS
                    and (not new_ids or new_ids[-1] != tok.eos_token_id),
                }
            )
    wall = time.time() - t0
    return {
        "results": out,
        "n_forwards_excluding_generate": n_forwards,
        "wall_seconds": wall,
        "estimated_cost_usd": wall / 3600.0 * L40S_USD_PER_HOUR,
    }


@app.local_entrypoint()
def main() -> None:
    prov, rows, payload = _local_setup()
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"phase9a_diagnostic_{ts}_{prov['git_commit'][:8]}"
    print(f"Launching Phase 9A policy-flip diagnostic run_id={run_id}")
    result = run_diagnostic.remote(json.dumps(payload))
    by_idx = {r["row_index"]: r for r in result["results"]}
    merged = []
    for i, r in enumerate(rows):
        res = by_idx[i]
        merged.append(
            {
                **{
                    k: r[k]
                    for k in (
                        "example_id",
                        "base_scenario_id",
                        "family",
                        "framing",
                        "record_state",
                        "alternate_state",
                        "record_listed_first",
                    )
                },
                "candidate_states_visible_order": payload[i]["candidate_states"],
                "candidate_ids_visible_order": payload[i]["candidate_ids"],
                **{k: v for k, v in res.items() if k != "row_index"},
                "constrained_chosen_state": payload[i]["candidate_states"][
                    res["constrained_chosen_visible_index"]
                ],
                "activation_extracted": False,
                "probe_scored": False,
            }
        )
    run_dir = REPO_ROOT / "artifacts/runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    raw = run_dir / "diagnostic_outputs.jsonl"
    raw.write_text("".join(json.dumps(m, sort_keys=True) + "\n" for m in merged), "utf-8")
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        **prov,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dtype": "bfloat16",
        "quantization": None,
        "batch_size": 1,
        "decoders": {
            "CONSTRAINED": "phase8 two_candidate_constrained_greedy_after_12107 (unchanged)",
            "FREE": f"greedy do_sample=False num_beams=1 after 12107, max_new_tokens="
            f"{FREE_MAX_NEW_TOKENS}",
        },
        "n_prompts": len(merged),
        "n_evaluations": 2 * len(merged),
        "n_forwards_excluding_generate": result["n_forwards_excluding_generate"],
        "raw_outputs_sha256": _sha_bytes(raw.read_bytes()),
        "wall_seconds": result["wall_seconds"],
        "estimated_cost_usd": result["estimated_cost_usd"],
        "activations_collected": False,
        "probe_scored": False,
        "phase5_probe_scored": False,
        "causal_intervention": False,
    }
    (run_dir / "diagnostic_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", "utf-8"
    )
    print(json.dumps({k: manifest[k] for k in (
        "run_id", "n_evaluations", "wall_seconds", "estimated_cost_usd")}, indent=2))
