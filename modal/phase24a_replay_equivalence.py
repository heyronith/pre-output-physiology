"""Modal: Phase-24A live cached decoding vs teacher-forced replay equivalence.

Usage (authorized; clean tree except phase24 artifacts/reports):

    uv run modal run modal/phase24a_replay_equivalence.py
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

APP_NAME = "pre-output-physiology-phase24a-replay"
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/experiments/phase24a_replay_equivalence.yaml"
PILOT_PATH = REPO_ROOT / "artifacts/phase24a_replay/pilot_manifest.json"
CONTRACT_PATH = REPO_ROOT / "artifacts/phase24a_replay/contract.json"
OUT_ROOT = REPO_ROOT / "artifacts/phase24a_replay"

L40S_USD_PER_HOUR = 1.95
A100_80GB_USD_PER_HOUR = 2.50
GPU_TYPE = "A100-80GB"
GPU_USD_PER_HOUR = A100_80GB_USD_PER_HOUR
# Live vs replay compared on the same GPU; A100 used because L40S was capacity-queued.
# K=20 generation historically used L40S; this pilot does not regenerate K=20.

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


def _dirty_ok() -> None:
    raw = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True
    ).strip()
    if not raw:
        return
    for line in raw.splitlines():
        path = line[3:].strip().strip('"')
        if path.startswith(
            (
                "artifacts/phase24a_replay/",
                "reports/phase24a_replay",
                "configs/experiments/phase24a_replay",
                "docs/decision_log.md",
            )
        ):
            continue
        if "phase24" in path.replace("\\", "/"):
            continue
        raise SystemExit(f"STOP: dirty tree:\n{raw}")


def _local_setup() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _dirty_ok()
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if cfg.get("status") != "phase24a_replay_equivalence_authorized":
        raise SystemExit(f"requires authorized status, got {cfg.get('status')}")
    auth = cfg.get("authorizations", {})
    if auth.get("modal_gpu_mistral_replay_pilot_authorized") is not True:
        raise SystemExit("modal_gpu_mistral_replay_pilot_authorized must be true")
    for key in (
        "k20_activation_extraction_authorized",
        "probe_fitting_authorized",
        "sae_analysis_authorized",
        "k_gt_20_generation_authorized",
        "openai_grading_api_authorized",
        "physiology_authorized",
        "causal_intervention_authorized",
    ):
        if auth.get(key) is not False:
            raise SystemExit(f"{key} must be false")

    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    pilot = json.loads(PILOT_PATH.read_text(encoding="utf-8"))
    if int(pilot["n"]) != 16 or len(pilot["rows"]) != 16:
        raise SystemExit("STOP: pilot must be exactly 16 rows")
    return {"contract": contract, "pilot": pilot, "cfg": cfg}, pilot["rows"]


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 3,
    volumes={MODEL_CACHE_DIR: model_volume},
    memory=65536,
)
def live_and_replay(payload_json: str) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    from pre_output_physiology.phase21_roleplay import format_prompt_for_generation
    from pre_output_physiology.phase24a_replay import (
        MAX_NEW_TOKENS,
        MODEL_ID,
        MODEL_REVISION,
        N_TRANSFORMER_BLOCKS,
        TEMPERATURE,
        align_live_replay_positions,
        cosine_similarity,
        post_block_hidden_index,
        relative_l2_error,
    )

    rows = json.loads(payload_json)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=MODEL_CACHE_DIR, use_fast=True
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        cache_dir=MODEL_CACHE_DIR,
        low_cpu_mem_usage=True,
    )
    model.to("cuda:0")
    model.eval()
    attn_impl = getattr(model.config, "_attn_implementation", None) or getattr(
        model.config, "attn_implementation", "unspecified"
    )
    max_pos = int(getattr(model.config, "max_position_embeddings", -1))

    per_response: list[dict[str, Any]] = []
    all_cos: list[float] = []
    all_rel: list[float] = []
    # Diagnostics aggregators
    logit_cos: list[float] = []
    logprob_abs_diff: list[float] = []
    top1_agree = 0
    top1_n = 0
    by_layer_cos: dict[int, list[float]] = {i: [] for i in range(N_TRANSFORMER_BLOCKS)}
    by_layer_rel: dict[int, list[float]] = {i: [] for i in range(N_TRANSFORMER_BLOCKS)}
    # Heatmap accumulators: mean rel L2 over responses for layer × gen_step (cap steps)
    MAX_HEAT_T = 64
    heat_sum = np.zeros((N_TRANSFORMER_BLOCKS, MAX_HEAT_T), dtype=np.float64)
    heat_cnt = np.zeros((N_TRANSFORMER_BLOCKS, MAX_HEAT_T), dtype=np.int64)

    live_store: dict[str, Any] = {}
    replay_store: dict[str, Any] = {}

    def _hidden_stack(hidden_states: tuple, seq_index: int) -> np.ndarray:
        # shape [n_layers, hidden]
        layers = []
        for layer in range(N_TRANSFORMER_BLOCKS):
            h = hidden_states[post_block_hidden_index(layer)][0, seq_index, :]
            layers.append(h.detach().float().cpu().numpy().astype(np.float16))
        return np.stack(layers, axis=0)

    with torch.inference_mode():
        for ri, r in enumerate(rows):
            prompt_text = format_prompt_for_generation(
                tok, r["scenario"], r["question"], r["answer_prefix"]
            )
            prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
            prompt_n = len(prompt_ids)
            seed = int(r["sample_seed"])
            set_seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            # ---- LIVE cached decoding ----
            live_hiddens: list[np.ndarray] = []  # each [32, H] at a recorded position
            live_logits: list[np.ndarray] = []
            generated: list[int] = []
            stopping_reason = "max_new_tokens"

            ids = torch.tensor([prompt_ids], dtype=torch.long, device="cuda:0")
            out = model(
                ids,
                use_cache=True,
                output_hidden_states=True,
            )
            past = out.past_key_values
            live_hiddens.append(_hidden_stack(out.hidden_states, prompt_n - 1))
            live_logits.append(out.logits[0, -1, :].detach().float().cpu().numpy())

            for _step in range(MAX_NEW_TOKENS):
                logits = out.logits[0, -1, :].float()
                # temperature sampling T=1.0
                probs = torch.softmax(logits / TEMPERATURE, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())
                generated.append(next_id)
                if next_id == tok.eos_token_id:
                    stopping_reason = "eos"
                    # Still record state after EOS token was processed? Live loop
                    # processes the EOS by feeding it; record that position.
                    next_t = torch.tensor([[next_id]], dtype=torch.long, device="cuda:0")
                    out = model(
                        next_t,
                        past_key_values=past,
                        use_cache=True,
                        output_hidden_states=True,
                    )
                    past = out.past_key_values
                    live_hiddens.append(_hidden_stack(out.hidden_states, 0))
                    live_logits.append(
                        out.logits[0, -1, :].detach().float().cpu().numpy()
                    )
                    break
                next_t = torch.tensor([[next_id]], dtype=torch.long, device="cuda:0")
                out = model(
                    next_t,
                    past_key_values=past,
                    use_cache=True,
                    output_hidden_states=True,
                )
                past = out.past_key_values
                live_hiddens.append(_hidden_stack(out.hidden_states, 0))
                live_logits.append(out.logits[0, -1, :].detach().float().cpu().numpy())

            n_gen = len(generated)
            # live_hiddens[0]=prompt_end; [1]=after gen_0; ...
            # If EOS break after recording, len(live_hiddens)=1+n_gen
            assert len(live_hiddens) == 1 + n_gen

            # ---- REPLAY teacher forcing (full sequence; no future for prefix) ----
            full_ids = prompt_ids + generated
            full_t = torch.tensor([full_ids], dtype=torch.long, device="cuda:0")
            rout = model(full_t, use_cache=False, output_hidden_states=True)
            positions = align_live_replay_positions(prompt_n, n_gen)

            replay_hiddens: list[np.ndarray] = []
            replay_logits: list[np.ndarray] = []
            for pos in positions:
                idx = int(pos["sequence_index"])
                replay_hiddens.append(_hidden_stack(rout.hidden_states, idx))
                replay_logits.append(
                    rout.logits[0, idx, :].detach().float().cpu().numpy()
                )

            # ---- Compare ----
            resp_cos: list[float] = []
            resp_rel: list[float] = []
            for p_i, pos in enumerate(positions):
                for layer in range(N_TRANSFORMER_BLOCKS):
                    lv = live_hiddens[p_i][layer].astype(np.float32)
                    rv = replay_hiddens[p_i][layer].astype(np.float32)
                    c = cosine_similarity(lv.tolist(), rv.tolist())
                    e = relative_l2_error(lv.tolist(), rv.tolist())
                    resp_cos.append(c)
                    resp_rel.append(e)
                    all_cos.append(c)
                    all_rel.append(e)
                    by_layer_cos[layer].append(c)
                    by_layer_rel[layer].append(e)
                    gen_t = pos["generated_token_index"]
                    if gen_t is not None and gen_t < MAX_HEAT_T:
                        heat_sum[layer, gen_t] += e
                        heat_cnt[layer, gen_t] += 1

                # Logit diagnostic at this position
                lc = live_logits[p_i]
                rc = replay_logits[p_i]
                logit_cos.append(cosine_similarity(lc.tolist(), rc.tolist()))
                # Absolute logprob diff for the *actual subsequent* token if any
                if p_i < len(generated):
                    nxt = generated[p_i]
                    # live distribution was used to sample nxt at this position
                    live_logp = float(lc[nxt] - np.logaddexp.reduce(lc))
                    replay_logp = float(rc[nxt] - np.logaddexp.reduce(rc))
                    logprob_abs_diff.append(abs(live_logp - replay_logp))
                    top1_n += 1
                    if int(np.argmax(lc)) == int(np.argmax(rc)):
                        top1_agree += 1

            pid = r["prompt_id"]
            live_store[pid] = {
                "prompt_token_ids": prompt_ids,
                "generated_token_ids": generated,
                "prompt_n_tokens": prompt_n,
                "n_generated": n_gen,
                "stopping_reason": stopping_reason,
                "sample_seed": seed,
                "hiddens_f16": np.stack(live_hiddens, axis=0),  # [P, 32, H]
            }
            replay_store[pid] = {
                "full_token_ids": full_ids,
                "hiddens_f16": np.stack(replay_hiddens, axis=0),
            }

            per_response.append(
                {
                    "prompt_id": pid,
                    "sample_seed": seed,
                    "prompt_n_tokens": prompt_n,
                    "n_generated_tokens": n_gen,
                    "stopping_reason": stopping_reason,
                    "n_comparisons": len(resp_cos),
                    "median_cosine": float(np.median(resp_cos)),
                    "min_cosine": float(np.min(resp_cos)),
                    "median_rel_l2": float(np.median(resp_rel)),
                    "max_rel_l2": float(np.max(resp_rel)),
                    "completion_preview": tok.decode(
                        generated, skip_special_tokens=True
                    )[:200],
                }
            )
            print(
                f"[{ri + 1}/{len(rows)}] {pid} n_gen={n_gen} "
                f"med_cos={np.median(resp_cos):.8f} med_rel={np.median(resp_rel):.6e}",
                flush=True,
            )

    heat = np.full_like(heat_sum, np.nan, dtype=np.float64)
    mask = heat_cnt > 0
    heat[mask] = heat_sum[mask] / heat_cnt[mask]

    # Serialize stores as lists for JSON transport (npz large) —
    # return compact summaries; raw arrays saved locally after return via bytes.
    # Modal return size limits: send metrics + base64-less structure; save npz in
    # local entrypoint from returned float16 arrays encoded as nested lists is too big.
    # Instead write npz inside the remote to volume then copy — simpler: return
    # metrics only and write npz files as temporary paths on volume.
    vol_dir = Path(MODEL_CACHE_DIR) / "phase24a_runs"
    vol_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    live_path = vol_dir / f"live_{ts}.npz"
    replay_path = vol_dir / f"replay_{ts}.npz"
    live_kw = {}
    replay_kw = {}
    meta_rows = []
    for pid, lv in live_store.items():
        live_kw[f"{pid}__hiddens"] = lv["hiddens_f16"]
        live_kw[f"{pid}__prompt_ids"] = np.asarray(lv["prompt_token_ids"], dtype=np.int32)
        live_kw[f"{pid}__gen_ids"] = np.asarray(lv["generated_token_ids"], dtype=np.int32)
        rp = replay_store[pid]
        replay_kw[f"{pid}__hiddens"] = rp["hiddens_f16"]
        replay_kw[f"{pid}__full_ids"] = np.asarray(rp["full_token_ids"], dtype=np.int32)
        meta_rows.append(
            {
                "prompt_id": pid,
                "prompt_n_tokens": lv["prompt_n_tokens"],
                "n_generated": lv["n_generated"],
                "stopping_reason": lv["stopping_reason"],
                "sample_seed": lv["sample_seed"],
            }
        )
    np.savez_compressed(live_path, **live_kw)
    np.savez_compressed(replay_path, **replay_kw)
    model_volume.commit()

    return {
        "wall_seconds": time.time() - t0,
        "attn_implementation": str(attn_impl),
        "max_position_embeddings": max_pos,
        "per_response": per_response,
        "all_cos": all_cos,
        "all_rel": all_rel,
        "logit_cos": logit_cos,
        "logprob_abs_diff": logprob_abs_diff,
        "top1_agree": top1_agree,
        "top1_n": top1_n,
        "by_layer_cos_median": {
            str(k): float(np.median(v)) if v else float("nan")
            for k, v in by_layer_cos.items()
        },
        "by_layer_rel_median": {
            str(k): float(np.median(v)) if v else float("nan")
            for k, v in by_layer_rel.items()
        },
        "heatmap_mean_rel_l2": heat.tolist(),
        "live_npz_path": str(live_path),
        "replay_npz_path": str(replay_path),
        "meta_rows": meta_rows,
        "n_comparisons": len(all_cos),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


@app.local_entrypoint()
def main() -> None:
    from pre_output_physiology.phase24a_replay import (
        GUARANTEE,
        STATUS_FAIL,
        STATUS_PASS,
        evaluate_activation_gates,
        summarize_metric_values,
    )
    from pre_output_physiology.provenance import utc_now_iso, write_json

    meta, rows = _local_setup()
    print(f"Phase-24A replay pilot: n={len(rows)}", flush=True)
    result = live_and_replay.remote(json.dumps(rows))

    # Copy npz from Modal volume via re-download is awkward; volume paths are remote.
    # Re-fetch by reading volume through a tiny helper — Modal volumes persist.
    # For local provenance, pull bytes with modal volume — use remote read function.
    live_bytes, replay_bytes = _fetch_npz.remote(
        result["live_npz_path"], result["replay_npz_path"]
    )

    run_ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    git_commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    short = hashlib.sha256(git_commit.encode()).hexdigest()[:8]
    run_id = f"phase24a_replay_{run_ts}_{short}"
    run_dir = OUT_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    live_path = run_dir / "live_activations.npz"
    replay_path = run_dir / "replay_activations.npz"
    live_path.write_bytes(live_bytes)
    replay_path.write_bytes(replay_bytes)
    live_sha = hashlib.sha256(live_bytes).hexdigest()
    replay_sha = hashlib.sha256(replay_bytes).hexdigest()

    cos_sum = summarize_metric_values(result["all_cos"])
    rel_sum = summarize_metric_values(result["all_rel"])
    gates = evaluate_activation_gates(cos_sum, rel_sum)
    status = STATUS_PASS if gates["passed"] else STATUS_FAIL

    logit_cos_sum = summarize_metric_values(result["logit_cos"])
    logprob_sum = summarize_metric_values(result["logprob_abs_diff"])
    top1_rate = (
        result["top1_agree"] / result["top1_n"] if result["top1_n"] else float("nan")
    )

    cost_usd = float(result["wall_seconds"]) / 3600.0 * GPU_USD_PER_HOUR
    summary = {
        "created_at": utc_now_iso(),
        "status": status,
        "run_id": run_id,
        "git_commit": git_commit,
        "model_id": result["model_id"],
        "model_revision": result["model_revision"],
        "tokenizer_revision": result["model_revision"],
        "dtype": "bfloat16",
        "attn_implementation": result["attn_implementation"],
        "max_position_embeddings": result.get("max_position_embeddings"),
        "n_prompts_requested": 16,
        "n_prompts_completed": len(result["per_response"]),
        "n_activation_comparisons": result["n_comparisons"],
        "cosine": cos_sum,
        "relative_l2": rel_sum,
        "gates": gates,
        "by_layer_cos_median": result["by_layer_cos_median"],
        "by_layer_rel_median": result["by_layer_rel_median"],
        "heatmap_mean_rel_l2_layer_by_gen_t": result["heatmap_mean_rel_l2"],
        "logit_diagnostics": {
            "logit_cosine": logit_cos_sum,
            "abs_logprob_diff_actual_next_token": logprob_sum,
            "top1_agreement_rate": top1_rate,
            "top1_n": result["top1_n"],
            "note": "Diagnostics only — not additional PASS/FAIL gates",
        },
        "per_response": result["per_response"],
        "live_activations_sha256": live_sha,
        "replay_activations_sha256": replay_sha,
        "pilot_prompt_ids_sha256": meta["pilot"]["prompt_ids_sha256"],
        "wall_seconds": result["wall_seconds"],
        "estimated_cost_usd": cost_usd,
        "gpu": GPU_TYPE,
        "gpu_note": (
            "A100-80GB used for live↔replay same-device comparison after L40S "
            "capacity queue; does not regenerate historical K=20 corpus"
        ),
        "guarantee": GUARANTEE,
        "authorizations_after": {
            "modal_gpu_mistral_replay_pilot_authorized": False,
            "k20_activation_extraction_authorized": False,
            "probe_fitting_authorized": False,
            "sae_analysis_authorized": False,
            "k_gt_20_generation_authorized": False,
            "physiology_authorized": False,
        },
    }
    write_json(run_dir / "summary.json", summary)
    write_json(OUT_ROOT / "summary.json", summary)
    write_json(
        OUT_ROOT / "freeze.json",
        {
            "status": status,
            "run_id": run_id,
            "git_commit": git_commit,
            "gates": gates,
            "cosine": cos_sum,
            "relative_l2": rel_sum,
            "live_activations_sha256": live_sha,
            "replay_activations_sha256": replay_sha,
            "pilot_prompt_ids_sha256": meta["pilot"]["prompt_ids_sha256"],
            "attn_implementation": result["attn_implementation"],
            "model_revision": result["model_revision"],
            "authorizations_after": summary["authorizations_after"],
        },
    )
    write_json(
        run_dir / "manifest.json",
        {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "git_commit": git_commit,
            "n": len(result["per_response"]),
            "live_activations_sha256": live_sha,
            "replay_activations_sha256": replay_sha,
            "attn_implementation": result["attn_implementation"],
            "cost_usd": cost_usd,
            "wall_seconds": result["wall_seconds"],
        },
    )

    # Update config status / consume auth
    cfg_path = CONFIG_PATH
    text = cfg_path.read_text(encoding="utf-8")
    text = text.replace(
        "status: phase24a_replay_equivalence_authorized",
        f"status: {status}",
        1,
    )
    text = text.replace(
        "modal_gpu_mistral_replay_pilot_authorized: true",
        "modal_gpu_mistral_replay_pilot_authorized: false",
        1,
    )
    cfg_path.write_text(text, encoding="utf-8")

    print(json.dumps({"status": status, "run_id": run_id, "gates": gates}, indent=2))


@app.function(
    image=image,
    timeout=60 * 30,
    volumes={MODEL_CACHE_DIR: model_volume},
)
def _fetch_npz(live_path: str, replay_path: str) -> tuple[bytes, bytes]:
    return Path(live_path).read_bytes(), Path(replay_path).read_bytes()
