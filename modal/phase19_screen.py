"""Modal GPU: Phase 19A unseen validation family screen (S1 / S2 / fresh).

Reuses Phase-14 two-stage assay. New families only; Phase-18 train untouched;
no activations.

Usage (clean tree):

    uv run modal run modal/phase19_screen.py --stage s1
    uv run modal run modal/phase19_screen.py --stage s2
    uv run modal run modal/phase19_screen.py --stage fresh
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

APP_NAME = "pre-output-physiology-phase19a-unseen-validation-screen"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
L40S_USD_PER_HOUR = 1.95
STAGE1_MAX_NEW_TOKENS = 32

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_PATH = REPO_ROOT / "data/processed/phase19_design/screen_prompts.jsonl"
SELECTED_PROMPTS_PATH = (
    REPO_ROOT / "data/processed/phase19_design/selected_validation_prompts.jsonl"
)
CONFIG_PATH = REPO_ROOT / "configs/experiments/phase19_unseen_validation.yaml"
MATRIX_PATH = REPO_ROOT / "artifacts/phase19a_design/design_matrix.json"
S1_SCHEDULE = REPO_ROOT / "data/processed/phase19_design/s1_sampling_schedule.jsonl"
S2_SCHEDULE = REPO_ROOT / "data/processed/phase19_design/s2_sampling_schedule.jsonl"
FRESH_SCHEDULE = (
    REPO_ROOT / "data/processed/phase19_design/fresh_sampling_schedule.jsonl"
)

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


def _local_setup(stage: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict]]:
    import sys

    import yaml

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from prepare_phase8_design import sha_ids

    from pre_output_physiology.phase19_screen import (  # noqa: E402
        N_FRESH_CONTINUATIONS,
        N_S1_CONTINUATIONS,
        NEW_FAMILIES,
        PHASE15_RULE_HASH,
        STATUS_AUTHORIZED,
        STATUS_S1_DONE,
        STATUS_S2_DONE_AWAITING_FRESH,
        TEMPERATURE,
        THRESHOLD_HASH,
    )

    dirty = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise SystemExit(f"STOP: working tree dirty:\n{dirty}")
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    status = cfg.get("status")
    auth = cfg["authorizations"]

    if stage == "s1":
        if status != STATUS_AUTHORIZED:
            raise SystemExit(f"S1 requires authorized status, got {status}")
        if auth.get("s1_model_calls_authorized") is not True:
            raise SystemExit("S1 not authorized")
        schedule_path = S1_SCHEDULE
        prompts_path = PROMPTS_PATH
        expected_n = N_S1_CONTINUATIONS
        schedule_key = "s1_schedule_sha256"
        auth_gpu = "modal_gpu_authorized"
    elif stage == "s2":
        if status != STATUS_S1_DONE:
            raise SystemExit(f"S2 unexpected status {status}")
        if auth.get("s2_model_calls_authorized") is not True:
            raise SystemExit("S2 not authorized")
        if not S2_SCHEDULE.is_file():
            raise SystemExit("S2 schedule missing — run S1 evaluate first")
        schedule_path = S2_SCHEDULE
        prompts_path = PROMPTS_PATH
        expected_n = None
        schedule_key = "s2_schedule_sha256"
        auth_gpu = "modal_gpu_authorized"
    elif stage == "fresh":
        if status != STATUS_S2_DONE_AWAITING_FRESH:
            raise SystemExit(f"fresh unexpected status {status}")
        if auth.get("fresh_confirmation_model_calls_authorized") is not True:
            raise SystemExit("fresh not authorized")
        if not FRESH_SCHEDULE.is_file():
            raise SystemExit("fresh schedule missing")
        schedule_path = FRESH_SCHEDULE
        prompts_path = SELECTED_PROMPTS_PATH
        expected_n = N_FRESH_CONTINUATIONS
        schedule_key = "fresh_schedule_sha256"
        auth_gpu = "modal_gpu_authorized"
    else:
        raise SystemExit(f"unknown stage {stage}")

    if auth.get(auth_gpu) is not True:
        raise SystemExit("modal GPU not authorized")

    for key in (
        "locked_family_model_calls_authorized",
        "activation_extraction_authorized",
        "probe_fitting_authorized",
        "phase5_probe_scoring_authorized",
        "temperature_search_authorized",
        "prompt_changes_authorized",
        "payoff_changes_authorized",
        "decoder_changes_authorized",
        "onset_rule_changes_authorized",
        "threshold_changes_authorized",
        "selection_rule_changes_authorized",
        "phase18_train_model_calls_authorized",
        "causal_intervention_authorized",
    ):
        if auth.get(key) is not False:
            raise SystemExit(f"{key} must remain false")

    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    if matrix["threshold_hash"] != THRESHOLD_HASH:
        raise SystemExit("threshold hash mismatch")
    if matrix["phase15_rule_hash"] != PHASE15_RULE_HASH:
        raise SystemExit("phase15 rule hash mismatch")
    if float(matrix["temperature"]) != float(TEMPERATURE):
        raise SystemExit("temperature mismatch")

    prompts = [
        json.loads(x)
        for x in prompts_path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    schedule = [
        json.loads(x)
        for x in schedule_path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    if expected_n is not None and len(schedule) != expected_n:
        raise SystemExit(f"schedule size {len(schedule)} != {expected_n}")
    if schedule_key not in matrix:
        raise SystemExit(f"{schedule_key} not frozen in matrix")
    if sha_ids([r["continuation_id"] for r in schedule]) != matrix[schedule_key]:
        raise SystemExit(f"{schedule_key} mismatch")

    allowed = set(NEW_FAMILIES)
    if any(p["family"] not in allowed for p in prompts):
        raise SystemExit("non-Phase19 family in prompts")
    if any(s.get("family") not in allowed for s in schedule):
        raise SystemExit("non-Phase19 family in schedule")

    by_pg = {p["prompt_group_id"]: p for p in prompts}
    payload = []
    for i, s in enumerate(sorted(schedule, key=lambda r: r["continuation_id"])):
        p = by_pg[s["prompt_group_id"]]
        payload.append(
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
    stage_label = {"s1": "S1", "s2": "S2", "fresh": "P19A"}[stage]
    prov = {
        "git_commit": git_commit,
        "runner_sha256": _sha_bytes(Path(__file__).read_bytes()),
        "stage": stage_label,
        "n_continuations": len(payload),
        "schedule_sha256": sha_ids([r["continuation_id"] for r in schedule]),
        "threshold_hash": THRESHOLD_HASH,
        "phase15_rule_hash": PHASE15_RULE_HASH,
    }
    return prov, prompts, payload


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 240,
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
            if not appended or appended[-1] != CONTROLLED_PREFIX_TOKEN_ID:
                raise RuntimeError(
                    f"Response not 12107 after consideration; got {appended}"
                )
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
def main(stage: str = "s1") -> None:
    stage = stage.lower().strip()
    prov, prompts, payload = _local_setup(stage)
    by_pg = {p["prompt_group_id"]: p for p in prompts}
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"phase19a_{stage}_{ts}_{prov['git_commit'][:8]}"
    print(f"Launching Phase 19A {prov['stage']} run_id={run_id} n={len(payload)}")
    result = run_trajectories.remote(json.dumps(payload))
    by_idx = {r["row_index"]: r for r in result["results"]}
    merged = []
    for i, row in enumerate(payload):
        res = by_idx[i]
        p = by_pg[row["prompt_group_id"]]
        chosen = res["candidate_states_visible_order"][
            res["constrained_chosen_visible_index"]
        ]
        merged.append(
            {
                "continuation_id": row["continuation_id"],
                "stage": prov["stage"],
                "prompt_group_id": row["prompt_group_id"],
                "family": p["family"],
                "k": p["k"],
                "temperature": row["temperature"],
                "top_p": row["top_p"],
                "sample_index": row["sample_index"],
                "sample_seed": row["sample_seed"],
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
    raw = run_dir / "continuations.jsonl"
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
        "temperature": 0.9,
        "top_p": 0.95,
        "n_continuations": len(merged),
        "n_forwards": result["n_forwards"],
        "raw_continuations_sha256": _sha_bytes(raw.read_bytes()),
        "wall_seconds": result["wall_seconds"],
        "estimated_cost_usd": result["estimated_cost_usd"],
        "locked_model_calls": 0,
        "phase18_train_model_calls": 0,
        "activations_collected": False,
        "probe_scored": False,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", "utf-8"
    )
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in (
                    "run_id",
                    "stage",
                    "n_continuations",
                    "n_forwards",
                    "wall_seconds",
                    "estimated_cost_usd",
                )
            },
            indent=2,
        )
    )
