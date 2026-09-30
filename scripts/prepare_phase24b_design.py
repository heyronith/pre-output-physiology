#!/usr/bin/env python3
"""Freeze Phase-24B live-capture design: 16 DEV mixed prompts × 6 seeds."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology import phase21_roleplay as p21  # noqa: E402
from pre_output_physiology.phase24b_live import (  # noqa: E402
    ATTN_IMPLEMENTATION,
    BATCH_SIZE,
    BEHAVIORAL_YIELD,
    CAPTURE_GATES,
    DO_SAMPLE,
    DTYPE,
    GEMMA_HF_ID,
    GEMMA_REVISION,
    GUARANTEE,
    LAYER_COORDINATE_CONVENTION,
    MAX_NEW_TOKENS,
    MAX_SEQUENCE_LENGTH,
    MODEL_ID,
    MODEL_REVISION,
    N_CANARY,
    N_PILOT_PROMPTS,
    N_REPLICATES,
    N_TRAJECTORIES,
    TEMPERATURE,
    TOKENIZER_REVISION,
    build_schedule,
    canary_schedule_rows,
    layer_hook_module_paths,
    select_pilot_prompt_ids,
    sha256_json,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

INV24A = REPO_ROOT / "artifacts/phase24a_replay/k20_label_only_inventory.json"
PROMPTS21 = REPO_ROOT / "data/processed/phase21_roleplay/prompts.jsonl"
OUT = REPO_ROOT / "artifacts/phase24b_live"
CFG = REPO_ROOT / "configs/experiments/phase24b_live_capture.yaml"


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(x)
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def main() -> int:
    if MODEL_ID != p21.MODEL_ID or MODEL_REVISION != p21.MODEL_REVISION:
        raise SystemExit("STOP: model pin != Phase-21")
    if TEMPERATURE != p21.TEMPERATURE or MAX_NEW_TOKENS != p21.MAX_NEW_TOKENS:
        raise SystemExit("STOP: generation settings != Phase-21")

    inv = json.loads(INV24A.read_text(encoding="utf-8"))
    eligible = list(inv["development"]["qualifying_prompt_ids"])
    pilot_ids = select_pilot_prompt_ids(eligible)
    prompts = {r["prompt_id"]: r for r in _load_jsonl(PROMPTS21)}
    for pid in pilot_ids:
        if pid not in prompts:
            raise SystemExit(f"STOP: missing prompt {pid}")

    schedule = build_schedule(pilot_ids)
    canary = canary_schedule_rows(schedule)
    git_commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()

    schedule_rows_full = []
    for row in schedule:
        p = prompts[row["prompt_id"]]
        schedule_rows_full.append(
            {
                **row,
                "scenario": p["scenario"],
                "question": p["question"],
                "answer_prefix": p["answer_prefix"],
            }
        )
    canary_full = [
        next(r for r in schedule_rows_full if r["trajectory_id"] == c["trajectory_id"])
        for c in canary
    ]

    contract = {
        "phase": "phase24b",
        "created_at": utc_now_iso(),
        "git_commit": git_commit,
        "starting_sha": "afb20fa87818fccd5201288f3dededc5860b896e",
        "phase24a_status": "phase24a_replay_equivalence_failed_live_recording_required",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "dtype": DTYPE,
        "quantization": None,
        "attention_implementation": ATTN_IMPLEMENTATION,
        "generation": {
            "temperature": TEMPERATURE,
            "do_sample": DO_SAMPLE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "batch_size": BATCH_SIZE,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
        },
        "prompt_formatting": "format_prompt_for_generation / fold_system (Phase-21)",
        "layer_coordinates": LAYER_COORDINATE_CONVENTION,
        "hook_paths": layer_hook_module_paths(),
        "capture_gates_frozen": CAPTURE_GATES,
        "behavioral_yield_frozen": BEHAVIORAL_YIELD,
        "grader": {
            "hf_id": GEMMA_HF_ID,
            "revision": GEMMA_REVISION,
            "note": "Phase-23C validated response-level only; no onset",
        },
        "enrichment_warning": (
            "16 prompts are historically switch-capable DEVELOPMENT prompts; "
            "do not use Phase-24B frequencies for population prevalence."
        ),
        "guarantee": GUARANTEE,
    }

    pilot = {
        "created_at": utc_now_iso(),
        "git_commit": git_commit,
        "selection_method": (
            "sha256_sort_28_dev_label_only_mixed_take_first_16"
        ),
        "eligible_source": str(INV24A),
        "n_eligible": len(eligible),
        "n_pilot_prompts": N_PILOT_PROMPTS,
        "n_replicates": N_REPLICATES,
        "n_trajectories": N_TRAJECTORIES,
        "n_canary": N_CANARY,
        "prompt_ids": pilot_ids,
        "prompt_ids_sha256": sha256_json(pilot_ids),
        "schedule_sha256": sha256_json(
            [
                {
                    "trajectory_id": r["trajectory_id"],
                    "prompt_id": r["prompt_id"],
                    "replicate_index": r["replicate_index"],
                    "sample_seed": r["sample_seed"],
                }
                for r in schedule
            ]
        ),
        "seed_rule": "sha256(phase24b|{prompt_id}|{replicate_index})[:8] as int",
        "canary_trajectory_ids": [c["trajectory_id"] for c in canary],
        "note": (
            "Enriched DEVELOPMENT-only. Held-out 11 mixed prompts unused. "
            "Not for prevalence estimation."
        ),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "contract.json", contract)
    write_json(OUT / "pilot_manifest.json", pilot)
    write_json(OUT / "schedule.json", {"rows": schedule_rows_full, **pilot})
    write_json(OUT / "canary_manifest.json", {"rows": canary_full, "n": N_CANARY})
    with (OUT / "schedule.jsonl").open("w", encoding="utf-8") as f:
        for row in schedule_rows_full:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    cfg = f"""# Phase 24B — live activation capture feasibility pilot

experiment_id: phase24b_live_capture
phase: phase24b
status: phase24b_live_capture_authorized
description: >
  Instrumentation + sampling feasibility: record live Mistral activations during
  AR generation on 16 enriched DEVELOPMENT mixed prompts × 6 rollouts.
  Phase 24A replay failed; no probes/physiology/onset/OpenAI/held-out.
starting_sha: afb20fa87818fccd5201288f3dededc5860b896e

model:
  model_id: {MODEL_ID}
  revision: {MODEL_REVISION}
  tokenizer_revision: {TOKENIZER_REVISION}
  dtype: {DTYPE}
  attention_implementation: {ATTN_IMPLEMENTATION}

generation:
  temperature: {TEMPERATURE}
  do_sample: true
  max_new_tokens: {MAX_NEW_TOKENS}
  batch_size: 1

pilot:
  n_prompts: {N_PILOT_PROMPTS}
  n_replicates: {N_REPLICATES}
  n_trajectories: {N_TRAJECTORIES}
  prompt_ids_sha256: {pilot['prompt_ids_sha256']}
  schedule_sha256: {pilot['schedule_sha256']}
  manifest: artifacts/phase24b_live/pilot_manifest.json

grader:
  hf_id: {GEMMA_HF_ID}
  revision: {GEMMA_REVISION}

gates_frozen:
  min_completed: {CAPTURE_GATES['min_completed']}
  behavioral_min_prompts_both: {BEHAVIORAL_YIELD['min_prompts_with_both_hd']}
  behavioral_min_honest: {BEHAVIORAL_YIELD['min_honest_total']}
  behavioral_min_deceptive: {BEHAVIORAL_YIELD['min_deceptive_total']}

authorizations:
  modal_gpu_mistral_canary_authorized: true
  modal_gpu_mistral_live_capture_authorized: true
  modal_gpu_gemma_grading_authorized: true
  probe_fitting_authorized: false
  physiology_authorized: false
  sae_analysis_authorized: false
  causal_intervention_authorized: false
  k_gt_20_generation_authorized: false
  k20_replay_extraction_authorized: false
  onset_grading_authorized: false
  openai_grading_api_authorized: false
  held_out_mixed_prompts_authorized: false
  threshold_changes_authorized: false

notes: >
  D159: Phase 24B authorized after 24A replay FAIL. Live capture only.
  Enriched DEV prompts; no prevalence claims.
"""
    CFG.write_text(cfg, encoding="utf-8")
    print(
        json.dumps(
            {
                "prompt_ids": pilot_ids,
                "prompt_ids_sha256": pilot["prompt_ids_sha256"],
                "schedule_sha256": pilot["schedule_sha256"],
                "n_trajectories": N_TRAJECTORIES,
                "canary_ids": pilot["canary_trajectory_ids"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
