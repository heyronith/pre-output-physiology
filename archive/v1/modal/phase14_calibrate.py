"""Modal GPU: Phase 14A same-prompt trajectory calibration (1536 continuations).

Stage 1: stochastic `Consideration:` (forced prefix), stop at `.` or newline, max 32.
Stage 2: append `\\n Response` (token 12107), then Phase-8/9/10 constrained greedy
choice between record and alternate. Candidate token ids are recomputed in the
post-consideration context. Calibration prompts only; no final/locked calls;
no activations.

Usage (clean tree required):

    uv run modal run modal/phase14_calibrate.py
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

APP_NAME = "pre-output-physiology-phase14a-trajectory-calibration"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
L40S_USD_PER_HOUR = 1.95
N_EXPECTED = 1536
STAGE1_MAX_NEW_TOKENS = 32

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_PATH = REPO_ROOT / "data/processed/phase14_design/calibration_prompts.jsonl"
SCHEDULE_PATH = (
    REPO_ROOT / "data/processed/phase14_design/calibration_sampling_schedule.jsonl"
)
CONFIG_PATH = REPO_ROOT / "configs/experiments/phase14_same_prompt_trajectory.yaml"
MATRIX_PATH = REPO_ROOT / "artifacts/phase14a_design/design_matrix.json"

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


def _local_setup() -> tuple[
    dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    import sys

    import yaml

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from prepare_phase8_design import sha_ids

    dirty = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise SystemExit(f"STOP: working tree dirty:\n{dirty}")
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if cfg.get("status") != "phase14a_same_prompt_trajectory_calibration_authorized":
        raise SystemExit(f"unexpected status {cfg.get('status')}")
    auth = cfg["authorizations"]
    if auth.get("calibration_trajectory_calls_authorized") is not True:
        raise SystemExit("calibration not authorized")
    for key in (
        "final_model_calls_authorized",
        "locked_family_model_calls_authorized",
        "activation_extraction_authorized",
        "probe_fitting_authorized",
        "phase5_probe_scoring_authorized",
        "causal_intervention_authorized",
        "sampling_grid_extension_authorized",
        "post_result_prompt_changes_authorized",
    ):
        if auth.get(key) is not False:
            raise SystemExit(f"{key} must remain false")
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    prompts = [
        json.loads(x)
        for x in PROMPTS_PATH.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    schedule = [
        json.loads(x)
        for x in SCHEDULE_PATH.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    ordered_prompts = sorted(prompts, key=lambda r: r["prompt_group_id"])
    digest = _sha_bytes(
        "\n".join(
            f"{r['example_id']}\t{r['prompt_text']}" for r in ordered_prompts
        ).encode()
    )
    if (
        digest != cfg["design"]["calibration_prompt_text_sha256"]
        or digest != matrix["calibration_prompt_text_sha256"]
    ):
        raise SystemExit(f"calibration prompt hash mismatch {digest}")
    if sha_ids([r["continuation_id"] for r in schedule]) != matrix[
        "sampling_schedule_sha256"
    ]:
        raise SystemExit("sampling schedule hash mismatch")
    if len(schedule) != N_EXPECTED:
        raise SystemExit("schedule size")
    by_pg = {p["prompt_group_id"]: p for p in prompts}
    label_free = []
    for i, s in enumerate(sorted(schedule, key=lambda r: r["continuation_id"])):
        p = by_pg[s["prompt_group_id"]]
        label_free.append(
            {
                "row_index": i,
                "continuation_id": s["continuation_id"],
                "prompt_group_id": s["prompt_group_id"],
                "prompt_text": p["prompt_text"],
                "temperature": s["temperature"],
                "top_p": s["top_p"],
                "sample_seed": s["sample_seed"],
                "sample_index": s["sample_index"],
            }
        )
    git_commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    prov = {
        "git_commit": git_commit,
        "runner_sha256": _sha_bytes(Path(__file__).read_bytes()),
        "calibration_prompt_text_sha256": digest,
        "sampling_schedule_sha256": matrix["sampling_schedule_sha256"],
    }
    return prov, prompts, label_free, schedule


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 90,
    volumes={MODEL_CACHE_DIR: model_volume},
    memory=65536,
)
def run_trajectories(payload_json: str) -> dict[str, Any]:
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
        extract_first_sentence,
    )

    rows = json.loads(payload_json)
    if len(rows) != N_EXPECTED:
        raise ValueError("payload size")
    allowed = {
        "row_index",
        "continuation_id",
        "prompt_group_id",
        "prompt_text",
        "temperature",
        "top_p",
        "sample_seed",
        "sample_index",
    }
    for r in rows:
        if set(r) != allowed:
            raise ValueError(f"payload fields not label-free: {set(r)}")

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

    def parse_candidates(prompt_text: str) -> list[str]:
        line = next(
            ln for ln in prompt_text.split("\n") if ln.startswith("Candidate states: ")
        )
        body = line[len("Candidate states: ") :]
        parts = [p.strip() for p in body.split("|")]
        if len(parts) != 2:
            raise RuntimeError("candidate line parse failed")
        return parts

    n_forwards = 0
    out = []
    with torch.inference_mode():
        for r in rows:
            fmt = format_chat(tok, r["prompt_text"])
            base_ids = tok.encode(fmt, add_special_tokens=False)
            stage1_prefix = tok.encode(
                fmt + CONSIDERATION_PREFIX, add_special_tokens=False
            )
            if stage1_prefix[: len(base_ids)] != base_ids:
                raise RuntimeError("consideration prefix broken")

            torch.manual_seed(int(r["sample_seed"]))
            torch.cuda.manual_seed_all(int(r["sample_seed"]))

            generated: list[int] = []
            hit_stop = False
            for _ in range(STAGE1_MAX_NEW_TOKENS):
                ids = torch.tensor(
                    [stage1_prefix + generated], dtype=torch.long, device="cuda:0"
                )
                n_forwards += 1
                logits = model(
                    input_ids=ids, attention_mask=torch.ones_like(ids)
                ).logits[0, -1].float()
                scaled = logits / float(r["temperature"])
                probs = torch.softmax(scaled, dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cum = torch.cumsum(sorted_probs, dim=-1)
                mask = cum > float(r["top_p"])
                mask[0] = False
                sorted_probs = sorted_probs.masked_fill(mask, 0.0)
                sorted_probs = sorted_probs / sorted_probs.sum()
                pick = int(torch.multinomial(sorted_probs, 1).item())
                token = int(sorted_idx[pick].item())
                generated.append(token)
                piece = tok.decode([token])
                if "." in piece or "\n" in piece or token == tok.eos_token_id:
                    hit_stop = True
                    break

            cont_text = tok.decode(generated, skip_special_tokens=True)
            consideration_decoded = extract_first_sentence(cont_text)
            # Stage-2 context uses the terminated first sentence (period/newline),
            # then `\n Response` whose final token must be 12107.
            full_before_response = (
                fmt + CONSIDERATION_PREFIX + consideration_decoded
            )
            without = tok.encode(full_before_response, add_special_tokens=False)
            with_response = tok.encode(
                full_before_response + RESPONSE_CONTINUATION, add_special_tokens=False
            )
            if with_response[: len(without)] != without:
                raise RuntimeError("Response continuation does not preserve stage1 text ids")
            appended = with_response[len(without) :]
            # In-context `\n Response` is typically [13, 12107]; bare encode may differ.
            # Protocol requires Response remain token 12107 as the final appended id.
            if not appended or appended[-1] != CONTROLLED_PREFIX_TOKEN_ID:
                raise RuntimeError(
                    f"Response not 12107 after consideration; got {appended}"
                )
            if CONTROLLED_PREFIX_TOKEN_ID not in appended:
                raise RuntimeError(f"Response token 12107 missing from {appended}")
            stage2_prefix_ids = list(with_response)

            cands_states = parse_candidates(r["prompt_text"])
            cands = []
            for state in cands_states:
                full = tok.encode(
                    full_before_response + RESPONSE_CONTINUATION + " " + state,
                    add_special_tokens=False,
                )
                if full[: len(stage2_prefix_ids)] != stage2_prefix_ids:
                    raise RuntimeError("candidate continuation broke prefix")
                cands.append(list(full[len(stage2_prefix_ids) :]))
            validate_candidates(cands)

            def next_logits(
                generated_c: list[int], prefix: list[int] = stage2_prefix_ids
            ):
                nonlocal n_forwards
                ids = torch.tensor(
                    [prefix + generated_c], dtype=torch.long, device="cuda:0"
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
            out.append(
                {
                    "row_index": r["row_index"],
                    "continuation_id": r["continuation_id"],
                    "consideration_raw": cont_text,
                    "consideration_text": consideration_decoded,
                    "stage1_generated_ids": generated,
                    "stage1_hit_stop": hit_stop,
                    "stage1_n_tokens": len(generated),
                    "response_token_id": CONTROLLED_PREFIX_TOKEN_ID,
                    "candidate_states_visible_order": cands_states,
                    "candidate_ids_visible_order": cands,
                    "constrained_chosen_visible_index": res["chosen_index"],
                    "constrained_generated_ids": res["generated"],
                    "constrained_steps": res["steps"],
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
    prov, prompts, payload, schedule = _local_setup()
    by_pg = {p["prompt_group_id"]: p for p in prompts}
    by_sched = {s["continuation_id"]: s for s in schedule}
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"phase14a_calibration_{ts}_{prov['git_commit'][:8]}"
    print(f"Launching Phase 14A trajectory calibration run_id={run_id}")
    result = run_trajectories.remote(json.dumps(payload))
    by_idx = {r["row_index"]: r for r in result["results"]}
    merged = []
    for i, row in enumerate(payload):
        res = by_idx[i]
        p = by_pg[row["prompt_group_id"]]
        s = by_sched[row["continuation_id"]]
        chosen = res["candidate_states_visible_order"][
            res["constrained_chosen_visible_index"]
        ]
        merged.append(
            {
                "continuation_id": row["continuation_id"],
                "prompt_group_id": row["prompt_group_id"],
                "family": p["family"],
                "k": p["k"],
                "temperature": s["temperature"],
                "top_p": s["top_p"],
                "sample_index": s["sample_index"],
                "sample_seed": s["sample_seed"],
                "record_state": p["record_state"],
                "alternate_state": p["alternate_state"],
                "record_listed_first": p["record_listed_first"],
                "consideration_raw": res["consideration_raw"],
                "consideration_text": res["consideration_text"],
                "stage1_generated_ids": res["stage1_generated_ids"],
                "stage1_n_tokens": res["stage1_n_tokens"],
                "stage1_hit_stop": res["stage1_hit_stop"],
                "response_token_id": res["response_token_id"],
                "candidate_states_visible_order": res["candidate_states_visible_order"],
                "candidate_ids_visible_order": res["candidate_ids_visible_order"],
                "chosen_state": chosen,
                "constrained_generated_ids": res["constrained_generated_ids"],
                "constrained_steps": res["constrained_steps"],
                "activation_extracted": False,
                "probe_scored": False,
            }
        )
    run_dir = REPO_ROOT / "artifacts/runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    raw = run_dir / "calibration_continuations.jsonl"
    raw.write_text(
        "".join(json.dumps(m, sort_keys=True) + "\n" for m in merged), "utf-8"
    )
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        **prov,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dtype": "bfloat16",
        "quantization": None,
        "batch_size": 1,
        "temperature_grid": [0.7, 0.9, 1.1],
        "top_p": 0.95,
        "n_continuations": len(merged),
        "n_forwards": result["n_forwards"],
        "raw_continuations_sha256": _sha_bytes(raw.read_bytes()),
        "wall_seconds": result["wall_seconds"],
        "estimated_cost_usd": result["estimated_cost_usd"],
        "final_model_calls": 0,
        "locked_model_calls": 0,
        "activations_collected": False,
        "probe_scored": False,
        "phase5_probe_scored": False,
        "causal_intervention": False,
    }
    (run_dir / "calibration_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", "utf-8"
    )
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in (
                    "run_id",
                    "n_continuations",
                    "n_forwards",
                    "wall_seconds",
                    "estimated_cost_usd",
                )
            },
            indent=2,
        )
    )
