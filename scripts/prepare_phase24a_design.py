#!/usr/bin/env python3
"""Freeze Phase-24A replay-equivalence contract + deterministic 16-prompt pilot."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology import phase21_roleplay as p21  # noqa: E402
from pre_output_physiology.phase24a_replay import (  # noqa: E402
    ACTIVATION_EQUIVALENCE_GATES,
    BATCH_SIZE,
    DO_SAMPLE,
    DTYPE,
    GRADER_SPLIT_SHA,
    GUARANTEE,
    HIDDEN_STATES_POST_BLOCK_OFFSET,
    MAX_NEW_TOKENS,
    MAX_SEQUENCE_LENGTH,
    MODEL_ID,
    MODEL_REVISION,
    N_DEV_PROMPTS,
    N_LOCKED_PROMPTS,
    N_PILOT_PROMPTS,
    N_TRANSFORMER_BLOCKS,
    PILOT_SEED_BASE,
    TEMPERATURE,
    TOKENIZER_REVISION,
    label_only_mixed_prompt_counts,
    pilot_sample_seed,
    select_pilot_prompt_ids,
    sha256_json,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

PROC23 = REPO_ROOT / "data/processed/phase23_open_grader"
PROMPTS21 = REPO_ROOT / "data/processed/phase21_roleplay/prompts.jsonl"
OUT = REPO_ROOT / "artifacts/phase24a_replay"
CFG = REPO_ROOT / "configs/experiments/phase24a_replay_equivalence.yaml"


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(x)
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def main() -> int:
    # Resolve contract consistency with Phase-21 constants
    if MODEL_ID != p21.MODEL_ID or MODEL_REVISION != p21.MODEL_REVISION:
        raise SystemExit("STOP: Phase-24A model pin != Phase-21 pin")
    if TEMPERATURE != p21.TEMPERATURE or MAX_NEW_TOKENS != p21.MAX_NEW_TOKENS:
        raise SystemExit("STOP: Phase-24A generation settings != Phase-21")
    if TOKENIZER_REVISION != MODEL_REVISION:
        raise SystemExit("STOP: tokenizer revision must match model revision")

    split = json.loads((PROC23 / "grader_prompt_split.json").read_text(encoding="utf-8"))
    if split["grader_prompt_split_sha256"] != GRADER_SPLIT_SHA:
        raise SystemExit("STOP: grader split SHA mismatch")
    if split["n_development"] != N_DEV_PROMPTS or split["n_locked_validation"] != N_LOCKED_PROMPTS:
        raise SystemExit("STOP: unexpected split sizes")

    dev_ids = list(split["development_prompt_ids"])
    locked_ids = list(split["locked_validation_prompt_ids"])
    pilot_ids = select_pilot_prompt_ids(dev_ids, n=N_PILOT_PROMPTS)

    prompts = {r["prompt_id"]: r for r in _load_jsonl(PROMPTS21)}
    for pid in pilot_ids:
        if pid not in prompts:
            raise SystemExit(f"STOP: missing prompt {pid}")

    # Inventory (no model calls): label-only mixed prompts on K=20 corpus
    corpus = _load_jsonl(PROC23 / "reference_corpus.jsonl")
    inv_dev = label_only_mixed_prompt_counts(corpus, prompt_ids=set(dev_ids))
    inv_locked = label_only_mixed_prompt_counts(corpus, prompt_ids=set(locked_ids))

    git_commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()

    contract = {
        "phase": "phase24a",
        "created_at": utc_now_iso(),
        "git_commit": git_commit,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_id": MODEL_ID,
        "tokenizer_revision": TOKENIZER_REVISION,
        "dtype": DTYPE,
        "quantization": None,
        "attention_implementation": (
            "transformers_default_at_load_time_recorded_in_run_manifest"
        ),
        "attention_note": (
            "K=20 generation manifests did not pin attn_implementation; Phase 24A "
            "uses the identical transformers default for both live and replay paths "
            "and records the resolved value at runtime."
        ),
        "chat_template": "tokenizer.apply_chat_template via format_prompt_for_generation",
        "prompt_formatting": {
            "system_postfix": "Answer the question directly, without asides or internal thoughts.",
            "fold_system_for_mistral": True,
            "assistant_prefix": True,
            "strip_trailing_eos_before_continuation": True,
        },
        "generation": {
            "temperature": TEMPERATURE,
            "do_sample": DO_SAMPLE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "batch_size": BATCH_SIZE,
            "pad_token": "eos",
            "seed_policy": "per_prompt deterministic phase24a seeds",
            "pilot_seed_base": PILOT_SEED_BASE,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "max_sequence_length_source": (
                "mistralai/Mistral-7B-Instruct-v0.2 config.json "
                "max_position_embeddings; K=20 generation used model default "
                "(no separate max_model_len pin)"
            ),
        },
        "software_stack_target": {
            "torch": "2.4.1",
            "transformers": "4.44.2",
            "note": "Matched to Phase-21/22B generation image for K=20 fidelity",
        },
        "hidden_state_indexing": {
            "n_transformer_blocks": N_TRANSFORMER_BLOCKS,
            "hf_hidden_states_index_for_block_L": "L + 1",
            "offset": HIDDEN_STATES_POST_BLOCK_OFFSET,
            "documentation": (
                "hidden_states[0]=embeddings; hidden_states[layer+1]=post-block residual"
            ),
        },
        "activation_equivalence_gates_frozen": ACTIVATION_EQUIVALENCE_GATES,
        "grader_prompt_split_sha256": GRADER_SPLIT_SHA,
        "guarantee": GUARANTEE,
    }

    pilot_rows = []
    for pid in pilot_ids:
        p = prompts[pid]
        seed = pilot_sample_seed(pid)
        pilot_rows.append(
            {
                "prompt_id": pid,
                "scenario": p["scenario"],
                "question": p["question"],
                "answer_prefix": p["answer_prefix"],
                "sample_seed": seed,
                "grader_split": "development",
                "purpose": "instrumentation_validation_only_not_behavioral_evidence",
            }
        )

    pilot_manifest = {
        "created_at": utc_now_iso(),
        "git_commit": git_commit,
        "n": N_PILOT_PROMPTS,
        "selection_method": (
            "evenly_spaced_indices_over_sha256_sorted_development_prompt_ids"
        ),
        "development_pool_n": N_DEV_PROMPTS,
        "grader_prompt_split_sha256": GRADER_SPLIT_SHA,
        "prompt_ids": pilot_ids,
        "prompt_ids_sha256": sha256_json(pilot_ids),
        "rows": pilot_rows,
        "note": (
            "One newly generated rollout per prompt for live/replay instrumentation "
            "only. Outputs are not graded and are not behavioral evidence."
        ),
    }

    inventory = {
        "created_at": utc_now_iso(),
        "primary_label_source": "gpt-4o-2024-08-06_frozen_reference_labels",
        "rule": "label_only >=2 honest and >=2 deceptive; no onset",
        "development": inv_dev,
        "locked_validation": inv_locked,
        "note": (
            "Inventory only; does not influence Phase-24A replay PASS/FAIL. "
            "No new split. Stage-23D onset eligibility not used."
        ),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "contract.json", contract)
    write_json(OUT / "pilot_manifest.json", pilot_manifest)
    write_json(OUT / "k20_label_only_inventory.json", inventory)
    with (OUT / "pilot_prompts.jsonl").open("w", encoding="utf-8") as f:
        for row in pilot_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    cfg_text = f"""# Phase 24A — live vs replay activation equivalence (instrumentation only)

experiment_id: phase24a_replay_equivalence
phase: phase24a
status: phase24a_replay_equivalence_authorized
description: >
  Instrumentation validation: do teacher-forced replay activations match live
  cached autoregressive activations under the frozen K=20 Mistral contract?
  Exact-onset remains unvalidated. No K=20 extraction, probes, SAE, or K>20.
model_config_path: configs/models/mistral_7b_instruct_v02.yaml
starting_sha: b7fd54e73fdd148797a5a24b77dedc053446f670

model:
  model_id: {MODEL_ID}
  revision: {MODEL_REVISION}
  tokenizer_revision: {TOKENIZER_REVISION}
  dtype: {DTYPE}
  quantization: null

generation:
  temperature: {TEMPERATURE}
  do_sample: true
  max_new_tokens: {MAX_NEW_TOKENS}
  batch_size: {BATCH_SIZE}

pilot:
  n_prompts: {N_PILOT_PROMPTS}
  prompt_ids_sha256: {pilot_manifest['prompt_ids_sha256']}
  manifest: artifacts/phase24a_replay/pilot_manifest.json
  seed_base: {PILOT_SEED_BASE}

gates_frozen:
  min_median_cosine: {ACTIVATION_EQUIVALENCE_GATES['min_median_cosine']}
  min_p01_cosine: {ACTIVATION_EQUIVALENCE_GATES['min_p01_cosine']}
  max_median_rel_l2: {ACTIVATION_EQUIVALENCE_GATES['max_median_rel_l2']}
  max_p99_rel_l2: {ACTIVATION_EQUIVALENCE_GATES['max_p99_rel_l2']}

authorizations:
  modal_gpu_mistral_replay_pilot_authorized: true
  k20_activation_extraction_authorized: false
  probe_fitting_authorized: false
  sae_analysis_authorized: false
  k_gt_20_generation_authorized: false
  openai_grading_api_authorized: false
  physiology_authorized: false
  causal_intervention_authorized: false
  prompt_changes_authorized: false
  threshold_changes_authorized: false

notes: >
  D158: Phase 24A authorized. Instrumentation only. Exact-onset remains
  unvalidated. Research target prospectively shifted to token×layer trajectories
  and incremental information over surface baselines, contingent on replay PASS.
"""
    CFG.write_text(cfg_text, encoding="utf-8")

    print(
        json.dumps(
            {
                "pilot_ids": pilot_ids,
                "prompt_ids_sha256": pilot_manifest["prompt_ids_sha256"],
                "dev_qualifying": inv_dev["n_qualifying"],
                "locked_qualifying": inv_locked["n_qualifying"],
                "contract": str(OUT / "contract.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
