#!/usr/bin/env python3
"""Freeze Phase-21 RoleplayDeception design: dataset pins, prompt split, generation schedule."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase21_roleplay import (  # noqa: E402
    APOLLO_COMMIT,
    APOLLO_REPO,
    DATASET_BLOB_SHA,
    DATASET_CONTENT_SHA256,
    GRADING_BLOB_SHA,
    GRADING_CONTENT_SHA256,
    GUARANTEE,
    MAX_NEW_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    N_PROMPTS,
    N_ROLLOUTS,
    N_TOTAL,
    SEED_BASE,
    STATUS_AUTHORIZED,
    TEMPERATURE,
    _sha_json,
    build_generation_schedule,
    deterministic_prompt_split,
    load_apollo_dataset,
    schedule_sha,
    sha256_file,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

RAW_DIR = REPO_ROOT / "data/raw/apollo_roleplaying"
PROC_DIR = REPO_ROOT / "data/processed/phase21_roleplay"
ART_DIR = REPO_ROOT / "artifacts/phase21a_design"
DATASET_PATH = RAW_DIR / "dataset.yaml"
GRADING_PATH = RAW_DIR / "grading_template.txt"


def main() -> int:
    dirty = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True
    ).strip()
    # Allow only expected new files during first freeze preparation itself.
    # After freeze commits, subsequent scripts enforce clean tree.

    for path, expect_content, expect_blob in [
        (DATASET_PATH, DATASET_CONTENT_SHA256, DATASET_BLOB_SHA),
        (GRADING_PATH, GRADING_CONTENT_SHA256, GRADING_BLOB_SHA),
    ]:
        got = sha256_file(path)
        if got != expect_content:
            raise SystemExit(f"content SHA mismatch for {path}: {got}")
        blob = subprocess.check_output(
            ["git", "hash-object", str(path)], text=True
        ).strip()
        if blob != expect_blob:
            raise SystemExit(f"blob SHA mismatch for {path}: {blob}")

    prompts = load_apollo_dataset(DATASET_PATH)
    prompt_ids = [p["prompt_id"] for p in prompts]
    split = deterministic_prompt_split(prompt_ids)
    schedule = build_generation_schedule(prompts, split["split_by_prompt_id"])
    sched_hash = schedule_sha(schedule)

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    ART_DIR.mkdir(parents=True, exist_ok=True)

    # Write prompts jsonl (source fields only — no completions from yaml)
    prompts_path = PROC_DIR / "prompts.jsonl"
    with prompts_path.open("w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, sort_keys=True) + "\n")

    write_json(PROC_DIR / "prompt_split.json", split)
    with (PROC_DIR / "generation_schedule.jsonl").open("w", encoding="utf-8") as f:
        for row in schedule:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    threshold = {
        "apollo_repo": APOLLO_REPO,
        "apollo_commit": APOLLO_COMMIT,
        "dataset_blob_sha": DATASET_BLOB_SHA,
        "dataset_content_sha256": DATASET_CONTENT_SHA256,
        "grading_blob_sha": GRADING_BLOB_SHA,
        "grading_content_sha256": GRADING_CONTENT_SHA256,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "n_prompts": N_PROMPTS,
        "n_rollouts": N_ROLLOUTS,
        "n_total": N_TOTAL,
        "temperature": TEMPERATURE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed_base": SEED_BASE,
        "prompt_split_sha256": split["prompt_split_sha256"],
        "generation_schedule_sha256": sched_hash,
        "n_train_prompts": split["n_train"],
        "n_test_prompts": split["n_test"],
        "block": 12,
        "primary_offset": "k1",
        "probe_C": 0.01,
        "min_train_qualifying": 25,
        "min_test_qualifying": 8,
        "grader_model": "gpt-4o-2024-08-06",
        "do_sample": True,
        "top_p_override": None,
    }
    threshold_hash = _sha_json(threshold)
    threshold["threshold_hash"] = threshold_hash
    write_json(ART_DIR / "threshold_manifest.json", threshold)
    write_json(PROC_DIR / "threshold_manifest.json", threshold)

    design_matrix = {
        "created_at": utc_now_iso(),
        "phase": "phase21",
        "status": STATUS_AUTHORIZED,
        "apollo_repo": APOLLO_REPO,
        "apollo_commit": APOLLO_COMMIT,
        "dataset_blob_sha": DATASET_BLOB_SHA,
        "dataset_content_sha256": DATASET_CONTENT_SHA256,
        "grading_blob_sha": GRADING_BLOB_SHA,
        "grading_content_sha256": GRADING_CONTENT_SHA256,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "n_prompts": N_PROMPTS,
        "n_rollouts": N_ROLLOUTS,
        "n_continuations": N_TOTAL,
        "temperature": TEMPERATURE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed_base": SEED_BASE,
        "prompt_split_sha256": split["prompt_split_sha256"],
        "generation_schedule_sha256": sched_hash,
        "threshold_hash": threshold_hash,
        "n_train_prompts": split["n_train"],
        "n_test_prompts": split["n_test"],
        "train_prompt_ids": split["train_prompt_ids"],
        "test_prompt_ids": split["test_prompt_ids"],
        "prompts_path": str(prompts_path.relative_to(REPO_ROOT)),
        "schedule_path": "data/processed/phase21_roleplay/generation_schedule.jsonl",
        "guarantee": GUARANTEE,
        "working_tree_note": dirty[:200] if dirty else "clean_at_prepare",
    }
    write_json(ART_DIR / "design_matrix.json", design_matrix)
    write_json(PROC_DIR / "design_matrix.json", design_matrix)

    print(
        json.dumps(
            {
                "status": STATUS_AUTHORIZED,
                "n_prompts": N_PROMPTS,
                "n_train": split["n_train"],
                "n_test": split["n_test"],
                "n_schedule": len(schedule),
                "prompt_split_sha256": split["prompt_split_sha256"],
                "generation_schedule_sha256": sched_hash,
                "threshold_hash": threshold_hash,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
