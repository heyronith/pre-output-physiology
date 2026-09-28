"""Modal GPU: Phase 20A3 validation activation extraction (frozen block/position only).

Extracts only the selected discovery block and Stage-1 position for the
balanced Phase-19 validation trajectories. No layer/time search.

Usage (clean tree):

    uv run modal run modal/phase20_extract_validation.py
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

APP_NAME = "pre-output-physiology-phase20a3-validation-extract"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
L40S_USD_PER_HOUR = 1.95
COSINE_MIN = 0.9999

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/experiments/phase20_preanswer_physiology.yaml"
PROMPTS_PATH = REPO_ROOT / "data/processed/phase20_design/physiology_prompts.jsonl"
BALANCED_PATH = REPO_ROOT / "data/processed/phase20_design/balanced_subset.jsonl"
POP_SUMMARY = REPO_ROOT / "artifacts/phase20a1_population/population_summary.json"
FROZEN_PATH = REPO_ROOT / "artifacts/phase20a2_discovery/frozen_candidate.json"

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


def _local_setup() -> tuple[dict[str, Any], list[dict], dict]:
    import yaml

    from pre_output_physiology.phase20_physiology import STATUS_DISC_PASS

    dirty = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise SystemExit(f"STOP: working tree dirty:\n{dirty}")
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if cfg.get("status") != STATUS_DISC_PASS:
        raise SystemExit(f"requires discovery pass, got {cfg.get('status')}")
    auth = cfg["authorizations"]
    if auth.get("validation_activation_extraction_authorized") is not True:
        raise SystemExit("validation extraction not authorized")
    if auth.get("phase5_probe_scoring_authorized") is not False:
        raise SystemExit("phase5 scoring must remain false")
    if not FROZEN_PATH.is_file():
        raise SystemExit("missing frozen candidate")
    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    pop = json.loads(POP_SUMMARY.read_text(encoding="utf-8"))
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
    val_bal = [r for r in balanced if r["split"] == "validation"]
    conts = {
        json.loads(x)["continuation_id"]: json.loads(x)
        for x in (
            REPO_ROOT / "artifacts/runs" / pop["run_id"] / "continuations.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()
        if x.strip()
    }
    payload = []
    for i, b in enumerate(sorted(val_bal, key=lambda r: r["continuation_id"])):
        c = conts[b["continuation_id"]]
        p = prompts[b["prompt_group_id"]]
        payload.append(
            {
                "row_index": i,
                "continuation_id": b["continuation_id"],
                "prompt_group_id": b["prompt_group_id"],
                "prompt_text": p["prompt_text"],
                "consideration_text": c["consideration_text"],
                "stage1_body_ids": c["stage1_body_ids"],
            }
        )
    git_commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    label_map = {b["continuation_id"]: b["final_policy_label"] for b in val_bal}
    meta_map = {b["continuation_id"]: b for b in val_bal}
    prov = {
        "git_commit": git_commit,
        "runner_sha256": _sha_bytes(Path(__file__).read_bytes()),
        "stage": "P20A3",
        "n_trajectories": len(payload),
        "frozen_block": frozen["block"],
        "frozen_position": frozen["position"],
        "frozen_activation_probe_sha256": frozen["activation_probe_sha256"],
        "generation_run_id": pop["run_id"],
    }
    return prov, payload, {
        "labels": label_map,
        "meta": meta_map,
        "frozen": frozen,
    }


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 120,
    volumes={MODEL_CACHE_DIR: model_volume},
    memory=65536,
)
def extract_validation(payload_json: str, block: int, position: str) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from pre_output_physiology.phase8_design import format_chat
    from pre_output_physiology.phase14_design import CONSIDERATION_PREFIX

    rows = json.loads(payload_json)
    off = {"end0": 0, "end2": 2, "end4": 4}[position]
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

    def hook(_m, _inp, output):  # noqa: ANN001
        act = output[0] if isinstance(output, tuple) else output
        captured[block] = act

    handle = model.model.layers[block].register_forward_hook(hook)

    def forward_vec(input_ids_list: list[int]) -> np.ndarray:
        ids = torch.tensor([input_ids_list], dtype=torch.long, device="cuda:0")
        captured.clear()
        model(input_ids=ids, attention_mask=torch.ones_like(ids))
        vec = captured[block][0, -1].float().detach().cpu().numpy()
        if not np.isfinite(vec).all():
            raise RuntimeError("NaN/Inf")
        return vec.astype(np.float32)

    n = len(rows)
    # probe shape from first forward
    sample_fmt = format_chat(tok, rows[0]["prompt_text"])
    h = forward_vec(tok.encode(sample_fmt, add_special_tokens=False)).shape[0]
    acts_k0 = np.zeros((n, h), dtype=np.float32)
    acts_pos = np.zeros((n, h), dtype=np.float32)
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
            ids = full if off == 0 else full[:-off]
            acts_pos[i] = forward_vec(ids)
            meta_rows.append(
                {
                    "row_index": i,
                    "continuation_id": r["continuation_id"],
                    "prompt_group_id": r["prompt_group_id"],
                }
            )

    handle.remove()

    # k0 identity
    by_pg: dict[str, list[int]] = {}
    for m in meta_rows:
        by_pg.setdefault(m["prompt_group_id"], []).append(m["row_index"])
    k0_ok = True
    details = []
    for pg, idxs in by_pg.items():
        ref = acts_k0[idxs[0]]
        mn = min((_cosine(ref, acts_k0[j]) for j in idxs[1:]), default=1.0)
        details.append({"prompt_group_id": pg, "min_cosine": mn})
        if mn < COSINE_MIN:
            k0_ok = False

    wall = time.time() - t0
    return {
        "meta_rows": meta_rows,
        "activations_k0": acts_k0,
        "activations_selected": acts_pos,
        "k0_identity_ok": k0_ok,
        "k0_details": details,
        "wall_seconds": wall,
        "estimated_cost_usd": wall / 3600.0 * L40S_USD_PER_HOUR,
        "hidden_size": h,
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
    frozen = local["frozen"]
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"phase20a3_val_extract_{ts}_{prov['git_commit'][:8]}"
    print(
        f"Launching validation extract run_id={run_id} n={len(payload)} "
        f"block={frozen['block']} pos={frozen['position']}"
    )
    result = extract_validation.remote(
        json.dumps(payload), int(frozen["block"]), str(frozen["position"])
    )
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
        labels[i] = 1 if lab == LA else 0
        if lab not in (LR, LA):
            raise SystemExit(f"bad label {lab}")
        families.append(local["meta"][r["continuation_id"]]["family"])
        prompt_ids.append(r["prompt_group_id"])
        cont_ids.append(r["continuation_id"])

    run_dir = REPO_ROOT / "artifacts/runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    act_path = run_dir / "validation_activations.safetensors"
    save_file(
        {
            "activations_k0": np.asarray(result["activations_k0"], dtype=np.float32),
            "activations_selected": np.asarray(
                result["activations_selected"], dtype=np.float32
            ),
            "labels_alternate": labels,
            "block": np.asarray([frozen["block"]], dtype=np.int32),
        },
        str(act_path),
    )
    (run_dir / "validation_meta.json").write_text(
        json.dumps(
            {
                "continuation_ids": cont_ids,
                "prompt_group_ids": prompt_ids,
                "families": families,
                "k0_details": result["k0_details"],
                "k0_identity_ok": result["k0_identity_ok"],
                "frozen_block": frozen["block"],
                "frozen_position": frozen["position"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        "utf-8",
    )
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        **prov,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "n_trajectories": n,
        "activations_sha256": _sha_bytes(act_path.read_bytes()),
        "wall_seconds": result["wall_seconds"],
        "estimated_cost_usd": result["estimated_cost_usd"],
        "k0_identity_ok": result["k0_identity_ok"],
        "locked_model_calls": 0,
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
            },
            indent=2,
        )
    )
