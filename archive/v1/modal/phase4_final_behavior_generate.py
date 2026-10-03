"""Modal GPU: Phase 4D final behavior generation (controlled prefix; no activations).

Generates greedy continuations for frozen final prompts under conditions
C1/C2/C3/C4/C6 only. C5 remains HOLD (D061). No prompt-template changes.

Usage (clean tree required):

  uv run modal run modal/phase4_final_behavior_generate.py
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

APP_NAME = "pre-output-physiology-phase4d-final"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
CONTROLLED_PREFIX_TOKEN_ID = 12107
CONTROLLED_PREFIX_TEXT = "Response"
MAX_CONTINUATION_TOKENS = 127
BATCH_SIZE = 1
L40S_USD_PER_HOUR = 1.95
PROMPT_TEMPLATE_REVISION = 1
EXPECTED_FINAL_PROMPT_TEXT_SHA256 = (
    "84942a71e61a1aed67c872d4ffd69dc6e533f6295fcb0b3b93fa78d93f97ae89"
)
EXPECTED_FINAL_SCENARIO_IDS_SHA256 = (
    "f43d16d942331aa45f3a86e8af5f330f0370fe84356599fa7964a3949a2bf880"
)
SELECTED_CONDITIONS: tuple[str, ...] = (
    "C1_known_honest_neutral",
    "C2_known_honest_strategic",
    "C3_known_deceptive_strategic",
    "C4_false_belief_honest",
    "C6_counterfactual_nondeceptive",
)
HELD_CONDITION = "C5_uncertain_honest"
N_EXPECTED = 1200

REPO_ROOT = Path(__file__).resolve().parents[1]
FINAL_PROMPTS_PATH = REPO_ROOT / "data/processed/phase4_design/final_candidate_prompts.jsonl"
FINAL_SCENARIOS_PATH = REPO_ROOT / "data/processed/phase4_design/final_base_scenarios.jsonl"
CONFIG_PATH = REPO_ROOT / "configs/experiments/phase4_specificity.yaml"
MATRIX_PATH = REPO_ROOT / "artifacts/phase4a_summaries/condition_matrix.json"

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
MODEL_CACHE_DIR = "/vol/hf_cache"


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_text(text: str) -> str:
    return _sha_bytes(text.encode("utf-8"))


def _sha_ids(ids: list[int]) -> str:
    return _sha_text(",".join(str(i) for i in ids))


def _sha_prompt_texts(rows: list[dict[str, Any]]) -> str:
    ordered = sorted(rows, key=lambda r: r["example_id"])
    payload = "\n".join(f"{r['example_id']}\t{r['prompt_text']}" for r in ordered)
    return _sha_text(payload)


def _sha_scenario_ids(path: Path) -> str:
    ids = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                ids.append(json.loads(line)["base_scenario_id"])
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def _collect_local_provenance() -> dict[str, Any]:
    dirty = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise SystemExit(
            "STOP: working tree dirty — commit before Modal launch:\n" f"{dirty}"
        )
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if cfg.get("status") != "phase4d_final_behavior_generation_authorized":
        raise SystemExit(
            "status must be phase4d_final_behavior_generation_authorized, "
            f"got {cfg.get('status')}"
        )
    auth = cfg.get("authorizations", {})
    if auth.get("final_generation_authorized") is not True:
        raise SystemExit("final_generation_authorized must be true")
    if auth.get("activation_extraction_authorized") is not False:
        raise SystemExit("activation_extraction_authorized must remain false")
    if auth.get("causal_intervention_authorized") is not False:
        raise SystemExit("causal_intervention_authorized must remain false")
    if auth.get("probe_scoring_authorized", False) is not False:
        raise SystemExit("probe_scoring_authorized must remain false")
    if auth.get("modal_gpu_authorized") is not True:
        raise SystemExit("modal_gpu_authorized must be true")
    if cfg.get("design", {}).get("pilot_template_revision") != 1:
        raise SystemExit("prompt template revision must remain 1")
    if cfg.get("design", {}).get("prompt_template_changed_after_rev1") is not False:
        raise SystemExit("prompt_template_changed_after_rev1 must be false")

    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    if matrix.get("final_prompt_text_sha256") != EXPECTED_FINAL_PROMPT_TEXT_SHA256:
        raise SystemExit("final prompt-text hash drift — STOP")
    if matrix.get("final_base_scenario_ids_sha256") != EXPECTED_FINAL_SCENARIO_IDS_SHA256:
        raise SystemExit("final scenario-ID hash drift — STOP")
    scen_sha = _sha_scenario_ids(FINAL_SCENARIOS_PATH)
    if scen_sha != EXPECTED_FINAL_SCENARIO_IDS_SHA256:
        raise SystemExit(f"scenario file hash drift: {scen_sha}")

    git_commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    generator = Path(__file__).resolve()
    return {
        "git_commit": git_commit,
        "working_tree_clean": True,
        "generator_path": str(generator.relative_to(REPO_ROOT)),
        "generator_sha256": _sha_bytes(generator.read_bytes()),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "precision": "bf16",
        "quantization": None,
        "batch_size": BATCH_SIZE,
        "do_sample": False,
        "max_continuation_tokens": MAX_CONTINUATION_TOKENS,
        "first_token_sampled": False,
        "controlled_prefix_token_id": CONTROLLED_PREFIX_TOKEN_ID,
        "controlled_prefix_text": CONTROLLED_PREFIX_TEXT,
        "prompt_template_revision": PROMPT_TEMPLATE_REVISION,
        "prompt_template_changed_after_rev1": False,
        "final_prompt_text_sha256": EXPECTED_FINAL_PROMPT_TEXT_SHA256,
        "final_scenario_ids_sha256": EXPECTED_FINAL_SCENARIO_IDS_SHA256,
        "selected_conditions": list(SELECTED_CONDITIONS),
        "held_condition": HELD_CONDITION,
        "n_expected_outputs": N_EXPECTED,
        "activations_authorized": False,
        "probe_scoring_authorized": False,
        "k1_interpretation": "controlled-prefix k1",
    }


def _load_final_rows() -> list[dict[str, Any]]:
    if not FINAL_PROMPTS_PATH.is_file():
        raise SystemExit(f"missing final prompts: {FINAL_PROMPTS_PATH}")
    all_rows = []
    with FINAL_PROMPTS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                all_rows.append(json.loads(line))
    if len(all_rows) != 1440:
        raise SystemExit(f"expected 1440 final candidate prompts, got {len(all_rows)}")
    digest = _sha_prompt_texts(all_rows)
    if digest != EXPECTED_FINAL_PROMPT_TEXT_SHA256:
        raise SystemExit(
            f"final prompt-text hash mismatch: {digest} != "
            f"{EXPECTED_FINAL_PROMPT_TEXT_SHA256}"
        )
    if any(r.get("split") != "final" for r in all_rows):
        raise SystemExit("non-final split in final prompt file")
    if any(r["base_scenario_id"].startswith("pilot_") for r in all_rows):
        raise SystemExit("pilot scenario IDs found in final prompts")

    selected = [r for r in all_rows if r["condition_id"] in SELECTED_CONDITIONS]
    if any(r["condition_id"] == HELD_CONDITION for r in selected):
        raise SystemExit("C5 leaked into selected set")
    if len(selected) != N_EXPECTED:
        raise SystemExit(f"expected {N_EXPECTED} selected rows, got {len(selected)}")
    from collections import Counter

    counts = Counter(r["condition_id"] for r in selected)
    for cid in SELECTED_CONDITIONS:
        if counts[cid] != 240:
            raise SystemExit(f"{cid} count {counts[cid]} != 240")
    return selected


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 150,
    volumes={MODEL_CACHE_DIR: model_volume},
    memory=65536,
)
def generate_final_behavior(
    prompts_jsonl: str,
    *,
    provenance: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    """Controlled-prefix greedy generation for final C1/C2/C3/C4/C6 only."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(line) for line in prompts_jsonl.splitlines() if line.strip()]
    if len(rows) != N_EXPECTED:
        raise ValueError(f"expected {N_EXPECTED} rows, got {len(rows)}")
    if any(r.get("condition_id") == HELD_CONDITION for r in rows):
        raise ValueError("C5 must not be generated")
    if any(r.get("split") != "final" for r in rows):
        raise ValueError("non-final rows present")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=MODEL_CACHE_DIR,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        cache_dir=MODEL_CACHE_DIR,
        low_cpu_mem_usage=True,
    )
    model.to("cuda:0")
    model.eval()
    model_load_s = time.time() - t0

    outputs: list[dict[str, Any]] = []
    gen_t0 = time.time()
    with torch.inference_mode():
        for i, row in enumerate(rows):
            prompt_text = row["prompt_text"]
            messages = [{"role": "user", "content": prompt_text}]
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)
            input_ids = prompt_ids + [CONTROLLED_PREFIX_TOKEN_ID]
            enc = {
                "input_ids": torch.tensor([input_ids], device="cuda:0"),
                "attention_mask": torch.ones(
                    1, len(input_ids), device="cuda:0", dtype=torch.long
                ),
            }
            gen = model.generate(
                **enc,
                max_new_tokens=MAX_CONTINUATION_TOKENS,
                do_sample=False,
                temperature=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            full_ids = [int(x) for x in gen[0].tolist()]
            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise RuntimeError("prompt IDs mutated")
            if full_ids[len(prompt_ids)] != CONTROLLED_PREFIX_TOKEN_ID:
                raise RuntimeError("prefix integrity failure")
            continuation_ids = full_ids[len(input_ids) :]
            combined_output_ids = [CONTROLLED_PREFIX_TOKEN_ID] + continuation_ids
            text = tokenizer.decode(combined_output_ids, skip_special_tokens=True)
            if continuation_ids and continuation_ids[-1] == tokenizer.eos_token_id:
                stop_reason = "eos_token"
            elif len(continuation_ids) >= MAX_CONTINUATION_TOKENS:
                stop_reason = "max_new_tokens"
            else:
                stop_reason = "other"
            outputs.append(
                {
                    "example_id": row["example_id"],
                    "base_scenario_id": row["base_scenario_id"],
                    "condition_id": row["condition_id"],
                    "split": "final",
                    "family": row.get("family"),
                    "ground_truth_state": row["ground_truth_state"],
                    "alt_state": row["alt_state"],
                    "accessible_false_state": row["accessible_false_state"],
                    "formatted_prompt_sha256": _sha_text(formatted),
                    "input_token_ids_sha256": _sha_ids(prompt_ids),
                    "controlled_prefix_token_id": CONTROLLED_PREFIX_TOKEN_ID,
                    "controlled_prefix_text": CONTROLLED_PREFIX_TEXT,
                    "controlled_prefix_supplied": True,
                    "first_token_sampled": False,
                    "continuation_token_ids": continuation_ids,
                    "generated_token_ids": combined_output_ids,
                    "first_generated_token_id": CONTROLLED_PREFIX_TOKEN_ID,
                    "n_continuation_tokens": len(continuation_ids),
                    "n_generated_tokens": len(combined_output_ids),
                    "output_text": text,
                    "stop_reason": stop_reason,
                    "prompt_template_revision": PROMPT_TEMPLATE_REVISION,
                    "activation_extracted": False,
                    "probe_scored": False,
                    "model_revision": MODEL_REVISION,
                }
            )
            if (i + 1) % 100 == 0:
                print(f"generated {i + 1}/{N_EXPECTED}")

    gen_s = time.time() - gen_t0
    wall_s = time.time() - t0
    model_volume.commit()

    if len(outputs) != N_EXPECTED:
        raise RuntimeError(f"output count {len(outputs)} != {N_EXPECTED}")
    if not all(
        r["controlled_prefix_supplied"]
        and r["first_generated_token_id"] == CONTROLLED_PREFIX_TOKEN_ID
        and r["generated_token_ids"][0] == CONTROLLED_PREFIX_TOKEN_ID
        for r in outputs
    ):
        raise RuntimeError("prefix integrity failure")
    if any(r["condition_id"] == HELD_CONDITION for r in outputs):
        raise RuntimeError("C5 outputs produced")

    return {
        "run_id": run_id,
        "n_outputs": len(outputs),
        "outputs": outputs,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "precision": "bf16",
        "do_sample": False,
        "max_continuation_tokens": MAX_CONTINUATION_TOKENS,
        "batch_size": BATCH_SIZE,
        "gpu_type": "L40S",
        "model_load_seconds": model_load_s,
        "generation_seconds": gen_s,
        "wall_seconds": wall_s,
        "estimated_cost_usd": (wall_s / 3600.0) * L40S_USD_PER_HOUR,
        "l40s_usd_per_hour_assumed": L40S_USD_PER_HOUR,
        "first_token_sampled": False,
        "controlled_prefix_token_id": CONTROLLED_PREFIX_TOKEN_ID,
        "controlled_prefix_text": CONTROLLED_PREFIX_TEXT,
        "prompt_template_revision": PROMPT_TEMPLATE_REVISION,
        "prompt_template_changed_after_rev1": False,
        "selected_conditions": list(SELECTED_CONDITIONS),
        "held_condition": HELD_CONDITION,
        "prefix_integrity_count": len(outputs),
        "n_c5_outputs": 0,
        "n_pilot_outputs": 0,
        "activations_collected": False,
        "probe_scores_computed": False,
        "provenance": provenance,
    }


@app.local_entrypoint()
def main() -> None:
    from collections import Counter

    prov = _collect_local_provenance()
    rows = _load_final_rows()
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"phase4d_final_{ts}_{prov['git_commit'][:8]}"
    prompts_jsonl = "\n".join(json.dumps(r, sort_keys=True) for r in rows)
    print(
        f"Launching final behavior run_id={run_id} n={len(rows)} "
        f"conditions={list(SELECTED_CONDITIONS)}"
    )
    result = generate_final_behavior.remote(
        prompts_jsonl,
        provenance=prov,
        run_id=run_id,
    )
    if result["n_outputs"] != N_EXPECTED:
        raise SystemExit(f"expected {N_EXPECTED} outputs, got {result['n_outputs']}")
    if result["prefix_integrity_count"] != N_EXPECTED:
        raise SystemExit("prefix integrity incomplete — STOP")
    if result["n_c5_outputs"] != 0:
        raise SystemExit("C5 outputs present — STOP")

    counts = Counter(r["condition_id"] for r in result["outputs"])
    for cid in SELECTED_CONDITIONS:
        if counts[cid] != 240:
            raise SystemExit(f"{cid} output count {counts[cid]} != 240")

    run_dir = REPO_ROOT / "artifacts/runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / "final_outputs.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in result["outputs"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "git_commit": prov["git_commit"],
        "working_tree_clean": True,
        "generator_sha256": prov["generator_sha256"],
        "generator_path": prov["generator_path"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "precision": "bf16",
        "quantization": None,
        "batch_size": BATCH_SIZE,
        "do_sample": False,
        "max_continuation_tokens": MAX_CONTINUATION_TOKENS,
        "gpu_type": "L40S",
        "n_final_outputs": result["n_outputs"],
        "n_pilot_outputs": 0,
        "n_c5_outputs": 0,
        "condition_counts": dict(counts),
        "selected_conditions": list(SELECTED_CONDITIONS),
        "held_condition": HELD_CONDITION,
        "raw_outputs_path": str(raw_path.relative_to(REPO_ROOT)),
        "raw_outputs_sha256": _sha_bytes(raw_path.read_bytes()),
        "model_load_seconds": result["model_load_seconds"],
        "generation_seconds": result["generation_seconds"],
        "wall_seconds": result["wall_seconds"],
        "estimated_cost_usd": result["estimated_cost_usd"],
        "l40s_usd_per_hour_assumed": L40S_USD_PER_HOUR,
        "first_token_sampled": False,
        "controlled_prefix_token_id": CONTROLLED_PREFIX_TOKEN_ID,
        "controlled_prefix_text": CONTROLLED_PREFIX_TEXT,
        "controlled_prefix_supplied": True,
        "prompt_template_revision": PROMPT_TEMPLATE_REVISION,
        "prompt_template_changed_after_rev1": False,
        "final_prompt_text_sha256": EXPECTED_FINAL_PROMPT_TEXT_SHA256,
        "final_scenario_ids_sha256": EXPECTED_FINAL_SCENARIO_IDS_SHA256,
        "prefix_integrity_count": N_EXPECTED,
        "k1_interpretation": "controlled-prefix k1",
        "activations_collected": False,
        "probe_scores_computed": False,
    }
    (run_dir / "final_generation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in (
                    "run_id",
                    "n_final_outputs",
                    "n_c5_outputs",
                    "prefix_integrity_count",
                    "condition_counts",
                    "wall_seconds",
                    "estimated_cost_usd",
                    "raw_outputs_path",
                )
            },
            indent=2,
        )
    )
