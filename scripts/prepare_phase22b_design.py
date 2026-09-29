#!/usr/bin/env python3
"""Freeze Phase-22B additional-rollout design (r10–r19) before any new model calls."""

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
    GRADER_MODEL,
    MAX_NEW_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    N_PROMPTS,
    N_ROLLOUTS,
    TEMPERATURE,
    load_apollo_dataset,
    sha256_file,
)
from pre_output_physiology.phase22b_sampling import (  # noqa: E402
    GUARANTEE,
    N_ADDITIONAL,
    PHASE21_GEN_RUN,
    PHASE21_GRADE_RUN,
    PHASE21_ONSET_RUN,
    PHASE21_SCHEDULE_SHA,
    PHASE21_SPLIT_SHA,
    ROLLOUT_OFFSET,
    SEED_BASE,
    STATUS_AUTHORIZED,
    build_additional_schedule,
    schedule_sha,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

RAW_DIR = REPO_ROOT / "data/raw/apollo_roleplaying"
PHASE21_PROC = REPO_ROOT / "data/processed/phase21_roleplay"
PROC_DIR = REPO_ROOT / "data/processed/phase22b_k20"
ART_DIR = REPO_ROOT / "artifacts/phase22b_design"
DATASET_PATH = RAW_DIR / "dataset.yaml"
GRADING_PATH = RAW_DIR / "grading_template.txt"


def main() -> int:
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

    # Reuse frozen Phase-21 prompts + split (do not recompute)
    prompts = [
        json.loads(x)
        for x in (PHASE21_PROC / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    if len(prompts) != N_PROMPTS:
        raise SystemExit(f"prompts {len(prompts)} != {N_PROMPTS}")
    # Sanity: reload from Apollo matches
    apollo = load_apollo_dataset(DATASET_PATH)
    if [p["prompt_id"] for p in apollo] != [p["prompt_id"] for p in prompts]:
        raise SystemExit("Phase-21 prompts.jsonl diverged from Apollo dataset")

    split = json.loads((PHASE21_PROC / "prompt_split.json").read_text(encoding="utf-8"))
    if split["prompt_split_sha256"] != PHASE21_SPLIT_SHA:
        raise SystemExit(
            f"split SHA mismatch: {split['prompt_split_sha256']} != {PHASE21_SPLIT_SHA}"
        )

    # Verify Phase-21 schedule SHA unchanged
    p21_sched = [
        json.loads(x)
        for x in (PHASE21_PROC / "generation_schedule.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if x.strip()
    ]
    if schedule_sha(p21_sched) != PHASE21_SCHEDULE_SHA:
        raise SystemExit("Phase-21 schedule SHA drift")

    schedule = build_additional_schedule(prompts, split["split_by_prompt_id"])
    sched_hash = schedule_sha(schedule)

    # Disjointness vs Phase-21
    p21_seeds = {r["sample_seed"] for r in p21_sched}
    p21_ids = {r["continuation_id"] for r in p21_sched}
    new_seeds = {r["sample_seed"] for r in schedule}
    new_ids = {r["continuation_id"] for r in schedule}
    if p21_seeds & new_seeds:
        raise SystemExit(f"seed collision: {list(p21_seeds & new_seeds)[:5]}")
    if p21_ids & new_ids:
        raise SystemExit(f"continuation_id collision: {list(p21_ids & new_ids)[:5]}")

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    ART_DIR.mkdir(parents=True, exist_ok=True)

    with (PROC_DIR / "generation_schedule.jsonl").open("w", encoding="utf-8") as f:
        for row in schedule:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    # Symlink/copy pointers to Phase-21 prompts + split (write copies for self-contained path)
    with (PROC_DIR / "prompts.jsonl").open("w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, sort_keys=True) + "\n")
    write_json(PROC_DIR / "prompt_split.json", split)

    design = {
        "phase": "phase22b",
        "status": STATUS_AUTHORIZED,
        "created_at": utc_now_iso(),
        "seed_base": SEED_BASE,
        "rollout_offset": ROLLOUT_OFFSET,
        "n_prompts": N_PROMPTS,
        "n_additional_rollouts_per_prompt": N_ROLLOUTS,
        "n_continuations": N_ADDITIONAL,
        "rollout_indices": list(range(ROLLOUT_OFFSET, ROLLOUT_OFFSET + N_ROLLOUTS)),
        "temperature": TEMPERATURE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": True,
        "top_p_override": None,
        "dtype": "bfloat16",
        "batch_size": 1,
        "grader_model": GRADER_MODEL,
        "n_train_prompts": split["n_train"],
        "n_test_prompts": split["n_test"],
        "prompt_split_sha256": PHASE21_SPLIT_SHA,
        "generation_schedule_sha256": sched_hash,
        "phase21_generation_schedule_sha256": PHASE21_SCHEDULE_SHA,
        "phase21_gen_run_id": PHASE21_GEN_RUN,
        "phase21_grade_run_id": PHASE21_GRADE_RUN,
        "phase21_onset_run_id": PHASE21_ONSET_RUN,
        "train_prompt_ids": split["train_prompt_ids"],
        "test_prompt_ids": split["test_prompt_ids"],
        "prompts_path": str((PROC_DIR / "prompts.jsonl").relative_to(REPO_ROOT)),
        "schedule_path": str(
            (PROC_DIR / "generation_schedule.jsonl").relative_to(REPO_ROOT)
        ),
        "apollo_repo": APOLLO_REPO,
        "apollo_commit": APOLLO_COMMIT,
        "dataset_blob_sha": DATASET_BLOB_SHA,
        "dataset_content_sha256": DATASET_CONTENT_SHA256,
        "grading_blob_sha": GRADING_BLOB_SHA,
        "grading_content_sha256": GRADING_CONTENT_SHA256,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "guarantee": GUARANTEE,
    }
    write_json(PROC_DIR / "design_matrix.json", design)
    write_json(ART_DIR / "design_matrix.json", design)

    print(
        json.dumps(
            {
                "status": STATUS_AUTHORIZED,
                "n_continuations": N_ADDITIONAL,
                "seed_base": SEED_BASE,
                "seed_range": [SEED_BASE, SEED_BASE + N_ADDITIONAL - 1],
                "generation_schedule_sha256": sched_hash,
                "prompt_split_sha256": PHASE21_SPLIT_SHA,
                "rollout_indices": design["rollout_indices"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
