"""Modal GPU: Phase 12A family-bias source diagnostic (cells B/C/D x RF/AF; 864 choices).

144 deterministic Phase-11 discovery bases; unchanged two-candidate constrained decoder
after `Response` (12107) at K=10. Cell A reuses Phase-11 results (no calls). Locked
families never sent. No activations, no probes. Payload is label-free.

Usage (clean tree required):

    uv run modal run modal/phase12_diagnostic.py
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

APP_NAME = "pre-output-physiology-phase12a-family-source"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
L40S_USD_PER_HOUR = 1.95
N_EXPECTED = 864

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_PATH = REPO_ROOT / "data/processed/phase12_design/new_cell_prompts.jsonl"
CONFIG_PATH = REPO_ROOT / "configs/experiments/phase12_family_bias_source.yaml"
MATRIX_PATH = REPO_ROOT / "artifacts/phase12a_design/design_matrix.json"

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

    from pre_output_physiology.phase8_design import (
        candidate_order,
        candidate_token_ids,
    )

    dirty = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise SystemExit(f"STOP: working tree dirty:\n{dirty}")
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if cfg.get("status") != "phase12a_family_bias_source_diagnostic_authorized":
        raise SystemExit(f"unexpected status {cfg.get('status')}")
    auth = cfg["authorizations"]
    if auth.get("diagnostic_model_calls_authorized") is not True:
        raise SystemExit("diagnostic calls not authorized")
    for key in (
        "locked_family_model_calls_authorized",
        "activation_extraction_authorized",
        "probe_fitting_authorized",
        "phase5_probe_scoring_authorized",
        "causal_intervention_authorized",
    ):
        if auth.get(key) is not False:
            raise SystemExit(f"{key} must remain false")
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    rows = [
        json.loads(x) for x in PROMPTS_PATH.read_text(encoding="utf-8").splitlines() if x.strip()
    ]
    ordered = sorted(rows, key=lambda r: r["example_id"])
    digest = _sha_bytes(
        "\n".join(f"{r['example_id']}\t{r['prompt_text']}" for r in ordered).encode()
    )
    if digest != cfg["design"]["new_prompt_text_sha256"] or digest != matrix[
            "new_prompt_text_sha256"]:
        raise SystemExit(f"new prompt hash mismatch {digest}")
    locked = set(cfg["design"]["locked_families"])
    if (
        len(ordered) != N_EXPECTED
        or any(r["family"] in locked for r in ordered)
        or any(r["cell"] not in ("B", "C", "D") for r in ordered)
    ):
        raise SystemExit("diagnostic rows invalid")
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
        "new_prompt_text_sha256": digest,
    }
    return prov, ordered, payload


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 40,
    volumes={MODEL_CACHE_DIR: model_volume},
    memory=65536,
)
def run_constrained_choices(payload_json: str) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from pre_output_physiology.phase8_design import (
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
    with torch.inference_mode():
        for r in rows:
            prefix = prompt_prefix_ids(tok, r["prompt_text"])
            cands = [candidate_token_ids(tok, r["prompt_text"], s) for s in r["candidate_states"]]
            if cands != r["candidate_ids"]:
                raise RuntimeError("remote candidate tokenization differs from local")

            def next_logits(generated: list[int], prefix: list[int] = prefix) -> Any:
                nonlocal n_forwards
                ids = torch.tensor([prefix + generated], dtype=torch.long, device="cuda:0")
                n_forwards += 1
                logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
                return logits[0, -1].float().cpu().numpy()

            res = constrained_choice(cands, next_logits)
            out.append(
                {
                    "row_index": r["row_index"],
                    "chosen_visible_index": res["chosen_index"],
                    "generated_ids": res["generated"],
                    "steps": res["steps"],
                }
            )
    wall = time.time() - t0
    return {
        "results": out,
        "n_forwards": n_forwards,
        "wall_seconds": wall,
        "estimated_cost_usd": wall / 3600.0 * L40S_USD_PER_HOUR,
    }


@app.local_entrypoint()
def main() -> None:
    prov, rows, payload = _local_setup()
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"phase12a_diagnostic_{ts}_{prov['git_commit'][:8]}"
    print(f"Launching Phase 12A family-bias source diagnostic run_id={run_id}")
    result = run_constrained_choices.remote(json.dumps(payload))
    by_idx = {r["row_index"]: r for r in result["results"]}
    merged = []
    for i, r in enumerate(rows):
        res = by_idx[i]
        chosen_state = payload[i]["candidate_states"][res["chosen_visible_index"]]
        merged.append(
            {
                **{
                    k: r[k]
                    for k in (
                        "example_id",
                        "base_scenario_id",
                        "family",
                        "cell",
                        "order",
                        "record_state",
                        "alternate_state",
                        "record_listed_first",
                    )
                },
                "candidate_ids_visible_order": payload[i]["candidate_ids"],
                "chosen_state": chosen_state,
                "generated_ids": res["generated_ids"],
                "steps": res["steps"],
                "decoder": "two_candidate_constrained_greedy_after_12107",
                "activation_extracted": False,
                "probe_scored": False,
            }
        )
    run_dir = REPO_ROOT / "artifacts/runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    raw = run_dir / "diagnostic_choices.jsonl"
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
        "decoder": "two_candidate_constrained_greedy_after_12107",
        "n_evaluations": len(merged),
        "n_forwards": result["n_forwards"],
        "raw_choices_sha256": _sha_bytes(raw.read_bytes()),
        "wall_seconds": result["wall_seconds"],
        "estimated_cost_usd": result["estimated_cost_usd"],
        "locked_families_run": False,
        "free_generation": False,
        "activations_collected": False,
        "probe_scored": False,
    }
    (run_dir / "diagnostic_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", "utf-8"
    )
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in (
                    "run_id",
                    "n_evaluations",
                    "n_forwards",
                    "wall_seconds",
                    "estimated_cost_usd",
                )
            },
            indent=2,
        )
    )
