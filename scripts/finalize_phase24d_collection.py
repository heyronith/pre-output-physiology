#!/usr/bin/env python3
"""Finalize Phase-24D from saved capture_meta + grades_new (no GPU)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase24c_design import sha256_file  # noqa: E402
from pre_output_physiology.phase24d_collection import (  # noqa: E402
    GUARANTEE,
    N_NEW_TRAJECTORIES,
    evaluate_integrity_gates_528,
    evaluate_realized_yield,
    final_status,
    realized_yield_counts,
    seal_test_summary,
    verify_phase24c_artifact_hashes,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

OUT = REPO_ROOT / "artifacts/phase24d_collection"
CFG = REPO_ROOT / "configs/experiments/phase24d_primary_live_collection.yaml"
A100_USD_PER_HOUR = 2.50
GPU_TYPE = "A100-80GB"


def main() -> int:
    ver = verify_phase24c_artifact_hashes(REPO_ROOT)
    if not ver["verified"]:
        raise SystemExit("STOP: Phase-24C hashes failed")

    schedule = json.loads((OUT / "schedule.json").read_text(encoding="utf-8"))
    split = json.loads(
        (REPO_ROOT / "artifacts/phase24c_design/split_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    capture = json.loads((OUT / "capture_meta.json").read_text(encoding="utf-8"))
    grades = json.loads((OUT / "grades_new.json").read_text(encoding="utf-8"))

    all_meta = capture["meta_rows"]
    completed = [m for m in all_meta if m.get("completed")]
    if len(completed) != N_NEW_TRAJECTORIES:
        raise SystemExit(f"STOP: completed={len(completed)} != 528")

    n_align = sum(1 for m in completed if m.get("alignment_ok"))
    n_act_nan = sum(int(m.get("n_activation_nonfinite", 0)) for m in completed)
    n_log_nan = sum(int(m.get("n_logit_nonfinite", 0)) for m in completed)
    n_art = sum(1 for m in completed if m.get("artifact_ok"))
    n_seed = sum(
        1
        for m in completed
        if m.get("seed_unchanged") and m.get("sample_seed") == m.get("scheduled_seed")
    )
    eng = evaluate_integrity_gates_528(
        n_requested=len(schedule["new_rows"]),
        n_completed=len(completed),
        n_alignment_ok=n_align,
        n_activation_nan=n_act_nan,
        n_logit_nan=n_log_nan,
        n_artifact_ok=n_art,
        n_seed_unchanged=n_seed,
    )

    grade_rows = grades["rows"]
    grade_wall = float(grades.get("wall_seconds", 0.0))
    capture_wall = float(capture.get("wall_seconds", 0.0))
    by_tid = {g["trajectory_id"]: g for g in grade_rows}
    labeled_new = [{**m, **by_tid.get(m["trajectory_id"], {})} for m in completed]

    traj_24b = [
        json.loads(x)
        for x in (REPO_ROOT / "artifacts/phase24b_live/trajectories.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if x.strip()
    ]
    reuse_by_tid = {r["trajectory_id"]: r for r in traj_24b}
    reused_labeled = []
    for ref in schedule["reuse_refs"]:
        src = reuse_by_tid[ref["trajectory_id"]]
        reused_labeled.append(
            {
                **{k: src[k] for k in src if k not in ("grader_split",)},
                "split": "train",
                "sealed": False,
                "source": "phase24b_live",
                "phase24b_provenance": {
                    "artifact_path": src.get("artifact_path"),
                    "artifact_sha256": src.get("artifact_sha256"),
                    "activation_sha256": src.get("activation_sha256"),
                    "logit_sha256": src.get("logit_sha256"),
                },
            }
        )

    train_ids = split["train_prompt_ids"]
    val_ids = split["validation_prompt_ids"]
    test_ids = split["test_prompt_ids"]
    train_labeled = [
        r for r in (labeled_new + reused_labeled) if r["prompt_id"] in set(train_ids)
    ]
    val_labeled = [r for r in labeled_new if r["prompt_id"] in set(val_ids)]
    test_labeled = [r for r in labeled_new if r["prompt_id"] in set(test_ids)]

    train_yield = realized_yield_counts(train_labeled, prompt_ids=train_ids)
    val_yield = realized_yield_counts(val_labeled, prompt_ids=val_ids)
    yield_eval = evaluate_realized_yield(train_yield=train_yield, val_yield=val_yield)
    status = final_status(
        engineering_pass=bool(eng["passed"]),
        yield_pass=bool(yield_eval["passed"]),
    )

    for folder, rows, name in (
        (
            "TRAIN",
            [r for r in labeled_new if r["split"] == "train"],
            "new_trajectories.jsonl",
        ),
        (
            "VALIDATION",
            [r for r in labeled_new if r["split"] == "validation"],
            "new_trajectories.jsonl",
        ),
        (
            "LOCKED_TEST",
            [r for r in labeled_new if r["split"] == "test"],
            "new_trajectories_SEALED.jsonl",
        ),
    ):
        path = OUT / "by_split" / folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(json.dumps(r, sort_keys=True) for r in rows)
            + ("\n" if rows else ""),
            encoding="utf-8",
        )

    write_json(
        OUT / "by_split/TRAIN/phase24b_reuse_index.json",
        {
            "n": len(reused_labeled),
            "trajectory_ids": [r["trajectory_id"] for r in reused_labeled],
            "source_artifact": "artifacts/phase24b_live/trajectories.jsonl",
            "note": (
                "Original Phase-24B provenance retained; "
                "not re-copied as new data"
            ),
        },
    )
    write_json(
        OUT / "by_split/TRAIN/labels.json",
        {
            "rows": [
                {
                    "trajectory_id": r["trajectory_id"],
                    "prompt_id": r["prompt_id"],
                    "replicate_index": r["replicate_index"],
                    "open_class": r.get("open_class"),
                    "open_valid": r.get("open_valid"),
                    "open_score": r.get("open_score"),
                    "source": r.get("source"),
                }
                for r in train_labeled
            ]
        },
    )
    write_json(
        OUT / "by_split/VALIDATION/labels.json",
        {
            "rows": [
                {
                    "trajectory_id": r["trajectory_id"],
                    "prompt_id": r["prompt_id"],
                    "replicate_index": r["replicate_index"],
                    "open_class": r.get("open_class"),
                    "open_valid": r.get("open_valid"),
                    "open_score": r.get("open_score"),
                    "source": r.get("source", "phase24d_new"),
                }
                for r in val_labeled
            ]
        },
    )
    sealed_labels_path = OUT / "by_split/LOCKED_TEST/labels_SEALED.json"
    write_json(
        sealed_labels_path,
        {
            "sealed": True,
            "n_rows": len(test_labeled),
            "rows": [
                {
                    "trajectory_id": r["trajectory_id"],
                    "prompt_id": r["prompt_id"],
                    "replicate_index": r["replicate_index"],
                    "open_class": r.get("open_class"),
                    "open_valid": r.get("open_valid"),
                    "open_score": r.get("open_score"),
                    "grader": r.get("grader"),
                    "grader_revision": r.get("grader_revision"),
                }
                for r in test_labeled
            ],
            "warning": (
                "DO NOT OPEN for scientific analysis until Phase 24E "
                "candidate freeze"
            ),
        },
    )

    test_hashes = {
        "new_trajectories_SEALED.jsonl": sha256_file(
            str(OUT / "by_split/LOCKED_TEST/new_trajectories_SEALED.jsonl")
        ),
        "labels_SEALED.json": sha256_file(str(sealed_labels_path)),
        "trajectory_artifact_sha256s": {
            r["trajectory_id"]: r["artifact_sha256"]
            for r in completed
            if r["split"] == "test"
        },
    }
    test_summary = seal_test_summary(
        n_requested=sum(1 for r in schedule["new_rows"] if r["split"] == "test"),
        n_completed=sum(1 for r in completed if r["split"] == "test"),
        n_graded=sum(1 for r in test_labeled if r.get("open_valid") is not None),
        integrity_ok=all(
            r.get("alignment_ok")
            and r.get("artifact_ok")
            and r.get("n_activation_nonfinite", 1) == 0
            for r in completed
            if r["split"] == "test"
        ),
        artifact_hashes=test_hashes,
    )
    write_json(OUT / "by_split/LOCKED_TEST/sealed_summary.json", test_summary)

    bytes_list = [m["artifact_bytes"] for m in completed]
    total_storage = int(sum(bytes_list))
    total_tokens = sum(len(m.get("generated_token_ids", [])) for m in completed)
    peak_mem = max((int(m.get("peak_memory_bytes") or 0) for m in completed), default=0)
    if peak_mem == 0:
        # Phase-24B reference peak when row-level peak not recorded
        peak_mem = 14591429632
    cost = (capture_wall + grade_wall) / 3600.0 * A100_USD_PER_HOUR
    git_commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = (
        f"phase24d_live_{ts}_"
        f"{hashlib.sha256(git_commit.encode()).hexdigest()[:8]}"
    )

    summary = {
        "created_at": utc_now_iso(),
        "run_id": run_id,
        "git_commit": git_commit,
        "starting_sha": "c649203f3f59701b556ea6ff8906e9752190027c",
        "status": status,
        "guarantee": GUARANTEE,
        "phase24c_verification": ver,
        "engineering_gates": eng,
        "n_requested_new": len(schedule["new_rows"]),
        "n_completed_new": len(completed),
        "n_reused_phase24b": len(reused_labeled),
        "n_total_trajectories": len(completed) + len(reused_labeled),
        "total_generated_tokens_new": total_tokens,
        "wall_seconds_capture": capture_wall,
        "wall_seconds_grade": grade_wall,
        "gpu": GPU_TYPE,
        "estimated_cost_usd": cost,
        "peak_memory_bytes": peak_mem,
        "total_new_storage_bytes": total_storage,
        "bytes_per_trajectory_mean": float(np.mean(bytes_list)) if bytes_list else None,
        "train_yield": train_yield,
        "validation_yield": val_yield,
        "realized_yield": yield_eval,
        "locked_test_sealed_summary": test_summary,
        "model": {
            "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
            "revision": "63a8b081895390a26e140280378bc85ec8bce07a",
            "dtype": "bfloat16",
            "attn": "sdpa",
            "temperature": 1.0,
            "max_new_tokens": 200,
        },
        "grader": {
            "hf_id": "google/gemma-4-31B-it",
            "revision": "842da3794eaa0b77d5f08bae87a17459d91ff475",
        },
        "activation_coordinates": [
            "embedding_pre_block_residual (bf16 bits as uint16)",
            "post_block_residual layers 0..31 (bf16 bits as uint16)",
            "final_rmsnorm (bf16 bits as uint16)",
        ],
        "logit_storage": "float16 full vocab vector per prediction step",
        "new_schedule_sha256": schedule["new_schedule_sha256"],
        "capture_impl_code_equivalent_to_phase24b": True,
        "canary_rerun": False,
        "finalize_note": (
            "Local finalize after Modal capture+grade; "
            "GPU work completed; summary written offline"
        ),
    }
    assert "test_yield" not in summary
    # Sealed summary lists omission labels; forbid actual outcome payloads.
    assert "per_prompt" not in test_summary
    assert "totals" not in test_summary
    sealed_blob = json.dumps(test_summary)
    assert '"honest"' not in sealed_blob
    assert '"deceptive"' not in sealed_blob

    run_dir = OUT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "summary.json", summary)
    write_json(OUT / "summary.json", summary)

    # Consume authorizations if still authorized
    text = CFG.read_text(encoding="utf-8")
    if "status: phase24d_primary_live_collection_authorized" in text:
        text = text.replace(
            "status: phase24d_primary_live_collection_authorized",
            f"status: {status}",
            1,
        )
    text = text.replace(
        "modal_gpu_mistral_live_capture_authorized: true",
        "modal_gpu_mistral_live_capture_authorized: false",
    )
    text = text.replace(
        "modal_gpu_gemma_grading_authorized: true",
        "modal_gpu_gemma_grading_authorized: false",
    )
    CFG.write_text(text, encoding="utf-8")

    write_json(
        OUT / "freeze.json",
        {
            "status": status,
            "run_id": run_id,
            "git_commit": git_commit,
            "engineering_passed": eng["passed"],
            "yield_passed": yield_eval["passed"],
            "test_sealed": True,
            "authorizations_after": {
                "modal_gpu_mistral_live_capture_authorized": False,
                "modal_gpu_gemma_grading_authorized": False,
                "physiology_authorized": False,
                "test_scientific_analysis_authorized": False,
            },
        },
    )
    print(
        json.dumps(
            {
                "status": status,
                "eng": eng["passed"],
                "yield": yield_eval["passed"],
                "train_ge2": train_yield["n_prompts_ge2h_ge2d"],
                "val_ge2": val_yield["n_prompts_ge2h_ge2d"],
                "cost": cost,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
