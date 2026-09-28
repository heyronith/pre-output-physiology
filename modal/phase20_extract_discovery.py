"""Modal GPU: Phase 20A2 discovery activation extraction (TRAIN balanced subset only).

Truncated-prefix last-token residual hooks on blocks [0,4,...,31] at
k0 / end0 / end2 / end4. Validation prompts never extracted here.

Usage (clean tree):

    uv run modal run modal/phase20_extract_discovery.py
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

import modal

APP_NAME = "pre-output-physiology-phase20a2-discovery-extract"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
L40S_USD_PER_HOUR = 1.95
LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31]
POSITIONS = ["end0", "end2", "end4"]
POS_OFFSET = {"end0": 0, "end2": 2, "end4": 4}
COSINE_MIN = 0.9999

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/experiments/phase20_preanswer_physiology.yaml"
MATRIX_PATH = REPO_ROOT / "artifacts/phase20a_design/design_matrix.json"
PROMPTS_PATH = REPO_ROOT / "data/processed/phase20_design/physiology_prompts.jsonl"
BALANCED_PATH = REPO_ROOT / "data/processed/phase20_design/balanced_subset.jsonl"
POP_SUMMARY = REPO_ROOT / "artifacts/phase20a1_population/population_summary.json"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.44.2",
        "accelerate==0.33.0",
        "safetensors==0.4.4",
        "numpy==1.26.4",
        "pyyaml==6.0.2",
        "huggingface_hub==0.24.6",
    )
    .add_local_python_source("pre_output_physiology")
)
app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name("preoutput-mistral-cache", create_if_missing=True)
MODEL_CACHE_DIR = "/vol/hf_cache"


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def _local_setup() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict]]:
    import sys

    import yaml

    sys.path.insert(0, str(REPO_ROOT / "scripts"))

    from pre_output_physiology.phase20_physiology import (
        PHASE15_RULE_HASH,
        STATUS_POP_PASS,
        THRESHOLD_HASH,
        TRAIN_IDS_SHA256,
    )

    dirty = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise SystemExit(f"STOP: working tree dirty:\n{dirty}")
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if cfg.get("status") != STATUS_POP_PASS:
        raise SystemExit(f"requires population pass status, got {cfg.get('status')}")
    auth = cfg["authorizations"]
    if auth.get("activation_extraction_authorized") is not True:
        raise SystemExit("discovery activation not authorized")
    if auth.get("validation_activation_extraction_authorized") is not False:
        raise SystemExit("validation activation must remain false")
    if auth.get("phase5_probe_scoring_authorized") is not False:
        raise SystemExit("phase5 scoring must remain false")

    pop = json.loads(POP_SUMMARY.read_text(encoding="utf-8"))
    if not pop["gates"]["passed"]:
        raise SystemExit("population gates failed")
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    if matrix["threshold_hash"] != THRESHOLD_HASH:
        raise SystemExit("threshold hash")
    if matrix["train_prompt_group_ids_sha256"] != TRAIN_IDS_SHA256:
        raise SystemExit("train hash")

    prompts = {
        json.loads(x)["prompt_group_id"]: json.loads(x)
        for x in PROMPTS_PATH.read_text(encoding="utf-8").splitlines()
        if x.strip()
    }
    balanced = [
        json.loads(x)
        for x in BALANCED_PATH.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    train_bal = [r for r in balanced if r["split"] == "train"]
    if not train_bal:
        raise SystemExit("no train balanced rows")
    # Load full continuation records for consideration text / body ids.
    run_id = pop["run_id"]
    cont_path = REPO_ROOT / "artifacts/runs" / run_id / "continuations.jsonl"
    conts = {
        json.loads(x)["continuation_id"]: json.loads(x)
        for x in cont_path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    }
    payload = []
    for i, b in enumerate(sorted(train_bal, key=lambda r: r["continuation_id"])):
        c = conts[b["continuation_id"]]
        p = prompts[b["prompt_group_id"]]
        # Label-free extract fields only (policy label kept local-side for joining).
        payload.append(
            {
                "row_index": i,
                "continuation_id": b["continuation_id"],
                "prompt_group_id": b["prompt_group_id"],
                "prompt_text": p["prompt_text"],
                "consideration_text": c["consideration_text"],
                "stage1_body_ids": c["stage1_body_ids"],
                "record_state": p["record_state"],
                "alternate_state": p["alternate_state"],
            }
        )
    git_commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    # Attach labels only in local merge after extract (not sent to GPU as banned).
    label_map = {
        b["continuation_id"]: b["final_policy_label"] for b in train_bal
    }
    meta_local = {
        b["continuation_id"]: b for b in train_bal
    }
    prov = {
        "git_commit": git_commit,
        "runner_sha256": _sha_bytes(Path(__file__).read_bytes()),
        "stage": "P20A2",
        "n_trajectories": len(payload),
        "balanced_subset_sha256": pop["balanced_subset"]["balanced_subset_sha256"],
        "threshold_hash": THRESHOLD_HASH,
        "phase15_rule_hash": PHASE15_RULE_HASH,
        "layers": LAYERS,
        "positions": POSITIONS,
        "generation_run_id": run_id,
    }
    return prov, payload, {"labels": label_map, "meta": meta_local, "prompts": prompts}


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 240,
    volumes={MODEL_CACHE_DIR: model_volume},
    memory=65536,
)
def extract_discovery(payload_json: str) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from pre_output_physiology.phase8_design import format_chat
    from pre_output_physiology.phase14_design import CONSIDERATION_PREFIX

    rows = json.loads(payload_json)
    allowed = {
        "row_index",
        "continuation_id",
        "prompt_group_id",
        "prompt_text",
        "consideration_text",
        "stage1_body_ids",
        "record_state",
        "alternate_state",
    }
    for r in rows:
        if set(r) != allowed:
            raise ValueError(f"bad fields: {set(r)}")
        for banned in ("final_policy_label", "label", "split"):
            if banned in r:
                raise ValueError(f"label leaked: {banned}")

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

    captured: dict[int, Any] = {}
    handles = []
    for li in LAYERS:

        def make_hook(layer_i: int):
            def hook(_m, _inp, output):  # noqa: ANN001
                act = output[0] if isinstance(output, tuple) else output
                captured[layer_i] = act

            return hook

        handles.append(model.model.layers[li].register_forward_hook(make_hook(li)))

    def forward_vec(input_ids_list: list[int]) -> np.ndarray:
        ids = torch.tensor([input_ids_list], dtype=torch.long, device="cuda:0")
        captured.clear()
        model(input_ids=ids, attention_mask=torch.ones_like(ids))
        if set(captured) != set(LAYERS):
            raise RuntimeError("hook miss")
        acts = []
        for li in LAYERS:
            vec = captured[li][0, -1].float().detach().cpu().numpy()
            if not np.isfinite(vec).all():
                raise RuntimeError("NaN/Inf")
            acts.append(vec.astype(np.float32))
        return np.stack(acts, axis=0).astype(np.float32)  # [9, H]

    # Preflight repeatability
    sample = rows[0]
    fmt0 = format_chat(tok, sample["prompt_text"])
    ids0 = tok.encode(fmt0, add_special_tokens=False)
    a_rep1 = forward_vec(ids0)
    a_rep2 = forward_vec(ids0)
    cos = min(_cosine(a_rep1[j], a_rep2[j]) for j in range(len(LAYERS)))
    if cos < COSINE_MIN:
        raise RuntimeError(f"k0 repeatability cosine {cos} < {COSINE_MIN}")

    n = len(rows)
    h = int(a_rep1.shape[1])
    acts_k0 = np.zeros((n, len(LAYERS), h), dtype=np.float32)
    acts_pos = {pos: np.zeros((n, len(LAYERS), h), dtype=np.float32) for pos in POSITIONS}
    meta_rows = []

    with torch.inference_mode():
        for r in rows:
            i = int(r["row_index"])
            fmt = format_chat(tok, r["prompt_text"])
            prompt_ids = tok.encode(fmt, add_special_tokens=False)
            acts_k0[i] = forward_vec(prompt_ids)

            full = tok.encode(
                fmt + CONSIDERATION_PREFIX + r["consideration_text"],
                add_special_tokens=False,
            )
            body = list(r["stage1_body_ids"])
            if len(body) < 5:
                raise RuntimeError(f"body tokens <5 for {r['continuation_id']}")
            # Sanity: full should end with body
            if full[-len(body) :] != body:
                # Rebuild body from full after consideration prefix
                pref = tok.encode(fmt + CONSIDERATION_PREFIX, add_special_tokens=False)
                body = full[len(pref) :]
                if len(body) < 5:
                    raise RuntimeError("rebuilt body <5")

            for pos, off in POS_OFFSET.items():
                if off == 0:
                    ids = full
                else:
                    ids = full[:-off]
                if len(ids) < len(prompt_ids) + 1:
                    raise RuntimeError(f"truncated too short for {pos}")
                acts_pos[pos][i] = forward_vec(ids)

            meta_rows.append(
                {
                    "row_index": i,
                    "continuation_id": r["continuation_id"],
                    "prompt_group_id": r["prompt_group_id"],
                    "n_prompt_tokens": len(prompt_ids),
                    "n_full_tokens": len(full),
                    "n_stage1_body_tokens": len(body),
                }
            )

    for hnd in handles:
        hnd.remove()

    # k0 identity within prompt_group
    by_pg: dict[str, list[int]] = {}
    for m in meta_rows:
        by_pg.setdefault(m["prompt_group_id"], []).append(m["row_index"])
    k0_ok = True
    k0_details = []
    for pg, idxs in by_pg.items():
        ref = acts_k0[idxs[0]]
        mins = []
        for j in idxs[1:]:
            mins.append(min(_cosine(ref[li], acts_k0[j][li]) for li in range(len(LAYERS))))
        mn = min(mins) if mins else 1.0
        k0_details.append({"prompt_group_id": pg, "min_cosine": mn, "n": len(idxs)})
        if mn < COSINE_MIN:
            k0_ok = False

    wall = time.time() - t0
    # Return arrays as lists for JSON transport is too large — use safetensors bytes via
    # writing inside remote and returning path is hard; instead return base64? Better:
    # save to volume and download. Simpler: return numpy via modal pickle (default).
    return {
        "meta_rows": meta_rows,
        "activations_k0": acts_k0,
        "activations_end0": acts_pos["end0"],
        "activations_end2": acts_pos["end2"],
        "activations_end4": acts_pos["end4"],
        "layers": LAYERS,
        "hidden_size": h,
        "k0_identity_ok": k0_ok,
        "k0_details": k0_details,
        "preflight_cosine": cos,
        "wall_seconds": wall,
        "estimated_cost_usd": wall / 3600.0 * L40S_USD_PER_HOUR,
        "n_forwards": n * (1 + len(POSITIONS)) + 2,
    }


@app.local_entrypoint()
def main() -> None:
    from safetensors.numpy import save_file

    from pre_output_physiology.phase15_onset import (
        LABEL_ALTERNATE as LA,
    )
    from pre_output_physiology.phase15_onset import (
        LABEL_RECORD as LR,
    )

    prov, payload, local = _local_setup()
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"phase20a2_disc_extract_{ts}_{prov['git_commit'][:8]}"
    print(f"Launching discovery extract run_id={run_id} n={len(payload)}")
    result = extract_discovery.remote(json.dumps(payload))
    if not result["k0_identity_ok"]:
        raise SystemExit(f"k0 identity FAILED: {result['k0_details']}")

    n = len(payload)
    labels = np.zeros(n, dtype=np.int32)
    families = []
    prompt_ids = []
    cont_ids = []
    for r in sorted(payload, key=lambda x: x["row_index"]):
        i = r["row_index"]
        lab = local["labels"][r["continuation_id"]]
        labels[i] = 1 if lab == LA else 0  # 1 = alternate
        if lab not in (LR, LA):
            raise SystemExit(f"bad label {lab}")
        families.append(local["meta"][r["continuation_id"]]["family"])
        prompt_ids.append(r["prompt_group_id"])
        cont_ids.append(r["continuation_id"])

    run_dir = REPO_ROOT / "artifacts/runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    tensors = {
        "activations_k0": np.asarray(result["activations_k0"], dtype=np.float32),
        "activations_end0": np.asarray(result["activations_end0"], dtype=np.float32),
        "activations_end2": np.asarray(result["activations_end2"], dtype=np.float32),
        "activations_end4": np.asarray(result["activations_end4"], dtype=np.float32),
        "labels_alternate": labels,
        "layers": np.asarray(result["layers"], dtype=np.int32),
    }
    act_path = run_dir / "discovery_activations.safetensors"
    save_file(tensors, str(act_path))
    meta = {
        "continuation_ids": cont_ids,
        "prompt_group_ids": prompt_ids,
        "families": families,
        "meta_rows": result["meta_rows"],
        "k0_details": result["k0_details"],
        "k0_identity_ok": result["k0_identity_ok"],
        "preflight_cosine": result["preflight_cosine"],
    }
    (run_dir / "discovery_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", "utf-8"
    )
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        **prov,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dtype": "bfloat16",
        "activation_storage_dtype": "float32",
        "n_trajectories": n,
        "activations_sha256": _sha_bytes(act_path.read_bytes()),
        "wall_seconds": result["wall_seconds"],
        "estimated_cost_usd": result["estimated_cost_usd"],
        "k0_identity_ok": result["k0_identity_ok"],
        "locked_model_calls": 0,
        "validation_activations_collected": False,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", "utf-8"
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "n_trajectories": n,
                "k0_identity_ok": result["k0_identity_ok"],
                "wall_seconds": result["wall_seconds"],
                "estimated_cost_usd": result["estimated_cost_usd"],
                "activations_sha256": manifest["activations_sha256"],
            },
            indent=2,
        )
    )
