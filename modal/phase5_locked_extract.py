"""Modal GPU: Phase 5D locked-family L12 controlled-prefix k1 extraction only.

No other layers/endpoints. No labels in GPU payload. Clean tree required.

Usage:

  uv run modal run modal/phase5_locked_extract.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import modal

APP_NAME = "pre-output-physiology-phase5d-locked"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
LAYER = 12
CONTROLLED_PREFIX_TOKEN_ID = 12107
COSINE_MIN = 0.9999
L40S_USD_PER_HOUR = 1.95
N_EXPECTED = 640
BATCH_SIZE = 1
LOCKED = {"harbor_dock_slip", "trail_marker_post"}
DISCOVERY = {
    "package_routing_bay",
    "gallery_wall_panel",
    "archive_box_label",
    "greenhouse_row_crop",
    "maker_bench_compartment",
    "kiosk_display_tile",
}
EXPECTED_LOCKED_PROMPT_TEXT_SHA256 = (
    "3a0ac9929e0971fd333bcc3d5a17b8b5b9d8b81dd6a75a8d5fe437dcd85a5097"
)
EXPECTED_LOCKED_PAIR_SHA256 = (
    "ee845e48546d6b40cfb8f6eab37f0580967be776b442b498df90189c17be34a1"
)
EXPECTED_PROBE_SHA256 = (
    "fa725af194eb1ca227301e6519029c818e942dab4bc130de060aa0754f4709c8"
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/experiments/phase5_strategic_discovery.yaml"
FINAL_PROMPTS_PATH = REPO_ROOT / "data/processed/phase5_design/final_candidate_prompts.jsonl"
FREEZE_PATH = REPO_ROOT / "artifacts/phase5d_locked_freeze/locked_population.json"
PROBE_PATH = (
    REPO_ROOT / "artifacts/phase5c_discovery_physiology/selected_probe_k1.npz"
)

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
)

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name("preoutput-mistral-cache", create_if_missing=True)
artifact_volume = modal.Volume.from_name(
    "preoutput-phase5-artifacts", create_if_missing=True
)
MODEL_CACHE_DIR = "/vol/hf_cache"
ART_DIR = "/vol/phase5_artifacts"


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_text(text: str) -> str:
    return _sha_bytes(text.encode("utf-8"))


def _sha_prompt_texts(rows: list[dict[str, Any]]) -> str:
    ordered = sorted(rows, key=lambda r: r["example_id"])
    payload = "\n".join(f"{r['example_id']}\t{r['prompt_text']}" for r in ordered)
    return _sha_text(payload)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def _collect_local_provenance() -> dict[str, Any]:
    dirty = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise SystemExit(f"STOP: working tree dirty:\n{dirty}")
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if cfg.get("status") != "phase5d_locked_test_authorized":
        raise SystemExit(f"unexpected status {cfg.get('status')}")
    auth = cfg.get("authorizations", {})
    if auth.get("locked_final_generation_authorized") is not True:
        raise SystemExit("locked_final_generation_authorized must be true")
    if auth.get("activation_extraction_authorized") is not True:
        raise SystemExit("activation_extraction_authorized must be true")
    if auth.get("probe_fitting_authorized") is not False:
        raise SystemExit("probe fitting must remain false (frozen probe only)")
    if auth.get("causal_intervention_authorized") is not False:
        raise SystemExit("causal must remain false")
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if freeze["pair_ids_sha256"] != EXPECTED_LOCKED_PAIR_SHA256:
        raise SystemExit("locked pair hash drift")
    if freeze["locked_prompt_text_sha256"] != EXPECTED_LOCKED_PROMPT_TEXT_SHA256:
        raise SystemExit("locked prompt hash drift")
    if freeze.get("activation_extraction_performed") is not False:
        raise SystemExit("freeze must be pre-extraction")
    probe_sha = _sha_bytes(PROBE_PATH.read_bytes())
    if probe_sha != EXPECTED_PROBE_SHA256:
        raise SystemExit(f"probe sha mismatch: {probe_sha}")
    git_commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    return {
        "git_commit": git_commit,
        "working_tree_clean": True,
        "extractor_sha256": _sha_bytes(Path(__file__).read_bytes()),
        "probe_sha256": probe_sha,
        "locked_pair_ids_sha256": EXPECTED_LOCKED_PAIR_SHA256,
        "locked_prompt_text_sha256": EXPECTED_LOCKED_PROMPT_TEXT_SHA256,
        "layer": LAYER,
        "endpoint": "k1",
        "controlled_prefix_token_id": CONTROLLED_PREFIX_TOKEN_ID,
        "future_response_tokens_present": False,
        "condition_labels_in_extractor": False,
        "no_retraining": True,
        "no_layer_selection": True,
        "causal_intervention_authorized": False,
    }


def _load_locked_label_free_rows() -> list[dict[str, Any]]:
    rows = []
    with FINAL_PROMPTS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("pool") != "locked" or row.get("split") != "final":
                continue
            if row["family"] not in LOCKED:
                raise SystemExit(f"unexpected locked family {row['family']}")
            if row["family"] in DISCOVERY:
                raise SystemExit("discovery family leaked into locked load")
            rows.append(
                {
                    "example_id": row["example_id"],
                    "base_scenario_id": row["base_scenario_id"],
                    "family": row["family"],
                    "prompt_text": row["prompt_text"],
                }
            )
    if len(rows) != N_EXPECTED:
        raise SystemExit(f"expected {N_EXPECTED}, got {len(rows)}")
    if _sha_prompt_texts(
        [
            {**r, "prompt_text": r["prompt_text"]}
            for r in [
                json.loads(line)
                for line in FINAL_PROMPTS_PATH.read_text(encoding="utf-8").splitlines()
                if line.strip()
                and json.loads(line).get("pool") == "locked"
                and json.loads(line).get("split") == "final"
            ]
        ]
    ) != EXPECTED_LOCKED_PROMPT_TEXT_SHA256:
        # Recompute properly
        full = []
        with FINAL_PROMPTS_PATH.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    r = json.loads(line)
                    if r.get("pool") == "locked" and r.get("split") == "final":
                        full.append(r)
        if _sha_prompt_texts(full) != EXPECTED_LOCKED_PROMPT_TEXT_SHA256:
            raise SystemExit("locked prompt hash mismatch")
    for r in rows:
        for banned in ("condition_id", "objective_target", "label", "record_state"):
            if banned in r:
                raise SystemExit(f"banned field in extract payload: {banned}")
    return sorted(rows, key=lambda r: r["example_id"])


def _load_model(cache_dir: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=cache_dir, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
    )
    model.to("cuda:0")
    model.eval()
    return tokenizer, model


def _register_hook(model, layer: int):
    import torch

    captured: dict[str, torch.Tensor] = {}

    def hook(_m, _inp, output):  # noqa: ANN001
        act = output[0] if isinstance(output, tuple) else output
        captured["act"] = act

    handle = model.model.layers[layer].register_forward_hook(hook)
    return captured, handle


def _forward_k1(model, captured, prompt_ids: list[int]) -> np.ndarray:
    import torch

    ids_k1 = list(prompt_ids) + [CONTROLLED_PREFIX_TOKEN_ID]
    if ids_k1[-1] != CONTROLLED_PREFIX_TOKEN_ID:
        raise RuntimeError("prefix identity failure")
    ids = torch.tensor([ids_k1], dtype=torch.long, device="cuda:0")
    mask = torch.ones_like(ids)
    captured.clear()
    model(input_ids=ids, attention_mask=mask)
    vec = captured["act"][0, -1].float().detach().cpu().numpy().astype(np.float32)
    if not np.isfinite(vec).all():
        raise RuntimeError("NaN/Inf activation")
    return vec


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 40,
    volumes={MODEL_CACHE_DIR: model_volume, ART_DIR: artifact_volume},
    memory=65536,
)
def run_repeatability_preflight(
    rows_jsonl: str,
    *,
    provenance: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    import torch

    rows = [json.loads(line) for line in rows_jsonl.splitlines() if line.strip()]
    sample = rows[:8]
    t0 = time.time()
    tokenizer, model = _load_model(MODEL_CACHE_DIR)
    captured, handle = _register_hook(model, LAYER)
    cosines = []
    with torch.inference_mode():
        for row in sample:
            messages = [{"role": "user", "content": row["prompt_text"]}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = list(tokenizer.encode(formatted, add_special_tokens=False))
            a1 = _forward_k1(model, captured, prompt_ids)
            a2 = _forward_k1(model, captured, prompt_ids)
            cosines.append(_cosine(a1, a2))
    handle.remove()
    min_cos = float(min(cosines))
    wall_s = time.time() - t0
    model_volume.commit()
    return {
        "run_id": run_id,
        "passed": min_cos >= COSINE_MIN,
        "cosine_min_required": COSINE_MIN,
        "repeatability_min_cosine": min_cos,
        "n_samples": len(sample),
        "layer": LAYER,
        "endpoint": "k1",
        "wall_seconds": wall_s,
        "estimated_cost_usd": (wall_s / 3600.0) * L40S_USD_PER_HOUR,
        "provenance": provenance,
    }


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 120,
    volumes={MODEL_CACHE_DIR: model_volume, ART_DIR: artifact_volume},
    memory=65536,
)
def extract_locked_l12_k1(
    rows_jsonl: str,
    *,
    provenance: dict[str, Any],
    run_id: str,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from safetensors.numpy import save_file

    if not preflight.get("passed"):
        raise RuntimeError("STOP: preflight failed")
    rows = [json.loads(line) for line in rows_jsonl.splitlines() if line.strip()]
    if len(rows) != N_EXPECTED:
        raise ValueError(f"expected {N_EXPECTED}")
    if any(r["family"] not in LOCKED for r in rows):
        raise RuntimeError("non-locked family")
    if any(r["family"] in DISCOVERY for r in rows):
        raise RuntimeError("discovery family present")

    t0 = time.time()
    tokenizer, model = _load_model(MODEL_CACHE_DIR)
    model_load_s = time.time() - t0
    captured, handle = _register_hook(model, LAYER)
    hidden = int(model.config.hidden_size)
    acts = np.zeros((len(rows), hidden), dtype=np.float32)
    example_ids = []
    n_prompt_tokens = []
    gen_t0 = time.time()
    with torch.inference_mode():
        for i, row in enumerate(rows):
            messages = [{"role": "user", "content": row["prompt_text"]}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = list(tokenizer.encode(formatted, add_special_tokens=False))
            acts[i] = _forward_k1(model, captured, prompt_ids)
            example_ids.append(row["example_id"])
            n_prompt_tokens.append(len(prompt_ids))
            if (i + 1) % 100 == 0:
                print(f"extracted {i + 1}/{len(rows)}")
    handle.remove()
    extract_s = time.time() - gen_t0
    wall_s = time.time() - t0
    if not np.isfinite(acts).all():
        raise RuntimeError("non-finite activations")

    out_dir = Path(ART_DIR) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    act_path = out_dir / "activations_l12_k1.safetensors"
    save_file(
        {
            "activations_l12_k1": acts,
            "n_prompt_tokens": np.asarray(n_prompt_tokens, dtype=np.int32),
            "layer": np.asarray([LAYER], dtype=np.int32),
            "k1_prefix_token_id": np.asarray(
                [CONTROLLED_PREFIX_TOKEN_ID], dtype=np.int32
            ),
        },
        str(act_path),
    )
    meta = {
        "example_ids": example_ids,
        "base_scenario_ids": [r["base_scenario_id"] for r in rows],
        "families": [r["family"] for r in rows],
    }
    (out_dir / "extraction_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    act_sha = _sha_bytes(act_path.read_bytes())
    artifact_volume.commit()
    model_volume.commit()
    return {
        "run_id": run_id,
        "n_rows": len(rows),
        "layer": LAYER,
        "endpoint": "k1",
        "activations_sha256": act_sha,
        "wall_seconds": wall_s,
        "model_load_seconds": model_load_s,
        "extraction_seconds": extract_s,
        "estimated_cost_usd": (wall_s / 3600.0) * L40S_USD_PER_HOUR,
        "future_response_tokens_present": False,
        "condition_labels_present": False,
        "discovery_rows_present": False,
        "other_layers_extracted": False,
        "k0_extracted": False,
        "preflight": {
            "passed": True,
            "repeatability_min_cosine": preflight["repeatability_min_cosine"],
        },
        "provenance": provenance,
    }


@app.local_entrypoint()
def main() -> None:
    prov = _collect_local_provenance()
    rows = _load_locked_label_free_rows()
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"phase5d_locked_extract_{ts}_{prov['git_commit'][:8]}"
    rows_jsonl = "\n".join(json.dumps(r, sort_keys=True) for r in rows)

    print(f"Launching locked L12/k1 preflight run_id={run_id}")
    preflight = run_repeatability_preflight.remote(
        rows_jsonl, provenance=prov, run_id=f"{run_id}_preflight"
    )
    print(
        json.dumps(
            {
                "preflight_passed": preflight["passed"],
                "repeatability_min_cosine": preflight["repeatability_min_cosine"],
            },
            indent=2,
        )
    )
    if not preflight["passed"]:
        raise SystemExit(
            f"STOP: preflight min cosine {preflight['repeatability_min_cosine']} "
            f"< {COSINE_MIN}"
        )

    print(f"Launching locked L12/k1 extract n={len(rows)}")
    result = extract_locked_l12_k1.remote(
        rows_jsonl, provenance=prov, run_id=run_id, preflight=preflight
    )
    local_out = REPO_ROOT / "artifacts/runs" / run_id
    local_out.mkdir(parents=True, exist_ok=True)
    (local_out / "preflight_repeatability.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Downloading volume artifacts {run_id} ...")
    subprocess.run(
        [
            "modal",
            "volume",
            "get",
            "preoutput-phase5-artifacts",
            run_id,
            str(local_out),
            "--force",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )
    nested = local_out / run_id
    if nested.is_dir():
        for p in nested.iterdir():
            dest = local_out / p.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(p), str(dest))
        nested.rmdir()

    act_path = local_out / "activations_l12_k1.safetensors"
    if _sha_bytes(act_path.read_bytes()) != result["activations_sha256"]:
        raise SystemExit("activation sha mismatch")

    manifest = {
        **{k: v for k, v in result.items() if k != "provenance"},
        "git_commit": prov["git_commit"],
        "working_tree_clean": True,
        "extractor_sha256": prov["extractor_sha256"],
        "probe_sha256": EXPECTED_PROBE_SHA256,
        "locked_pair_ids_sha256": EXPECTED_LOCKED_PAIR_SHA256,
        "locked_prompt_text_sha256": EXPECTED_LOCKED_PROMPT_TEXT_SHA256,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "preflight_repeatability_min_cosine": preflight["repeatability_min_cosine"],
        "preflight_passed": True,
        "activations_path_gitignored": str(act_path.relative_to(REPO_ROOT)),
        "probe_retrained": False,
        "probe_recalibrated": False,
        "layer_reselected": False,
        "causal_interventions_performed": False,
    }
    (local_out / "extraction_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "n_rows": result["n_rows"],
                "wall_seconds": result["wall_seconds"],
                "estimated_cost_usd": result["estimated_cost_usd"],
                "activations_sha256": result["activations_sha256"],
                "preflight_min_cosine": preflight["repeatability_min_cosine"],
            },
            indent=2,
        )
    )
