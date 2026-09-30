#!/usr/bin/env python3
"""Prepare Phase-24D collection schedule from frozen Phase-24C design (no GPU)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase24c_design import sha256_file  # noqa: E402
from pre_output_physiology.phase24d_collection import (  # noqa: E402
    GUARANTEE,
    N_NEW_TRAJECTORIES,
    N_REUSED_24B,
    N_TOTAL_TRAJECTORIES,
    SELECTED_K,
    STARTING_SHA,
    build_collection_schedule,
    capture_impl_source_sha,
    verify_phase24c_artifact_hashes,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

OUT = REPO_ROOT / "artifacts/phase24d_collection"
CFG = REPO_ROOT / "configs/experiments/phase24d_primary_live_collection.yaml"
TEMPLATE = REPO_ROOT / "artifacts/phase24c_design/generation_schedule_template.json"
SPLIT = REPO_ROOT / "artifacts/phase24c_design/split_manifest.json"
PROMPTS = REPO_ROOT / "data/processed/phase21_roleplay/prompts.jsonl"
PILOT = REPO_ROOT / "artifacts/phase24b_live/pilot_manifest.json"
MODAL_24B = REPO_ROOT / "modal/phase24b_live_capture.py"
MODAL_24D = REPO_ROOT / "modal/phase24d_live_collection.py"


def main() -> int:
    ver = verify_phase24c_artifact_hashes(REPO_ROOT)
    if not ver["verified"]:
        raise SystemExit(f"STOP: Phase-24C artifact hash mismatch:\n{json.dumps(ver, indent=2)}")

    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    pilot = json.loads(PILOT.read_text(encoding="utf-8"))
    prompts = [
        json.loads(x)
        for x in PROMPTS.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    by_id = {r["prompt_id"]: r for r in prompts}

    schedule = build_collection_schedule(
        schedule_template=template,
        prompts_by_id=by_id,
        pilot_24b_ids=pilot["prompt_ids"],
    )

    # Capture-impl equivalence vs Phase 24B (canary skip if identical)
    sha_24b = capture_impl_source_sha(MODAL_24B)
    capture_equiv = {
        "phase24b__generate_one_sha256": sha_24b,
        "note": (
            "Phase-24D modal embeds a byte-identical _generate_one; "
            "verified after modal file exists / updated by analyze step."
        ),
    }
    if MODAL_24D.exists():
        sha_24d = capture_impl_source_sha(MODAL_24D)
        capture_equiv["phase24d__generate_one_sha256"] = sha_24d
        capture_equiv["code_equivalent"] = sha_24b == sha_24d
        capture_equiv["canary_required"] = sha_24b != sha_24d
        if sha_24b != sha_24d:
            raise SystemExit(
                "STOP: Phase-24D _generate_one differs from Phase-24B; "
                "rerun 4-pair canary before collection."
            )

    git_commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()

    OUT.mkdir(parents=True, exist_ok=True)
    # Split directories for immutable artifact separation
    for sub in ("TRAIN", "VALIDATION", "LOCKED_TEST"):
        (OUT / "by_split" / sub).mkdir(parents=True, exist_ok=True)

    write_json(OUT / "phase24c_hash_verification.json", ver)
    write_json(OUT / "schedule.json", schedule)
    write_json(
        OUT / "contract.json",
        {
            "created_at": utc_now_iso(),
            "git_commit": git_commit,
            "starting_sha": STARTING_SHA,
            "selected_k": SELECTED_K,
            "n_new": N_NEW_TRAJECTORIES,
            "n_reuse": N_REUSED_24B,
            "n_total": N_TOTAL_TRAJECTORIES,
            "train_sha256": split["train_sha256"],
            "validation_sha256": split["validation_sha256"],
            "test_sha256": split["test_sha256"],
            "new_schedule_sha256": schedule["new_schedule_sha256"],
            "reuse_refs_sha256": schedule["reuse_refs_sha256"],
            "capture_equivalence": capture_equiv,
            "guarantee": GUARANTEE,
            "authorizations_intent": {
                "modal_gpu_mistral_live_capture_authorized": True,
                "modal_gpu_gemma_grading_authorized": True,
                "physiology_authorized": False,
                "probe_fitting_authorized": False,
                "test_scientific_analysis_authorized": False,
            },
        },
    )

    # Write new schedule jsonl by split (no TEST label fields)
    for split_name, folder in (
        ("train", "TRAIN"),
        ("validation", "VALIDATION"),
        ("test", "LOCKED_TEST"),
    ):
        rows = [r for r in schedule["new_rows"] if r["split"] == split_name]
        path = OUT / "by_split" / folder / "new_schedule.jsonl"
        path.write_text(
            "\n".join(json.dumps(r, sort_keys=True) for r in rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )

    design_sha = sha256_file(
        str(REPO_ROOT / "artifacts/phase24c_design/design_matrix.json")
    )
    split_sha = sha256_file(
        str(REPO_ROOT / "artifacts/phase24c_design/split_manifest.json")
    )
    template_sha = sha256_file(str(TEMPLATE))
    cfg = f"""# Phase 24D — primary live K=16 collection (capture + grade only)

experiment_id: phase24d_primary_live_collection
phase: phase24d
status: phase24d_primary_live_collection_authorized
description: >
  Execute frozen Phase-24C design at K=16. Collect 528 new live trajectories,
  reuse 96 Phase-24B TRAIN pilot trajectories, grade new responses with Gemma.
  No physiology / probes / TEST scientific analysis.
starting_sha: {STARTING_SHA}
selected_k: {SELECTED_K}
n_new_trajectories: {N_NEW_TRAJECTORIES}
n_reused_phase24b: {N_REUSED_24B}
n_total_trajectories: {N_TOTAL_TRAJECTORIES}

phase24c:
  design_matrix_sha256: {design_sha}
  split_manifest_sha256: {split_sha}
  schedule_template_sha256: {template_sha}
  new_schedule_sha256: {schedule['new_schedule_sha256']}

model:
  model_id: mistralai/Mistral-7B-Instruct-v0.2
  revision: 63a8b081895390a26e140280378bc85ec8bce07a
  tokenizer_revision: 63a8b081895390a26e140280378bc85ec8bce07a
  dtype: bfloat16
  attention_implementation: sdpa

generation:
  temperature: 1.0
  do_sample: true
  max_new_tokens: 200
  batch_size: 1

grader:
  hf_id: google/gemma-4-31B-it
  revision: 842da3794eaa0b77d5f08bae87a17459d91ff475

authorizations:
  modal_gpu_mistral_live_capture_authorized: true
  modal_gpu_gemma_grading_authorized: true
  modal_gpu_mistral_canary_authorized: false
  probe_fitting_authorized: false
  physiology_authorized: false
  sae_analysis_authorized: false
  causal_intervention_authorized: false
  onset_grading_authorized: false
  openai_grading_api_authorized: false
  test_scientific_analysis_authorized: false
  auroc_analysis_authorized: false
  layer_token_search_authorized: false

notes: >
  D161: Phase 24D authorized after 24C design freeze. Capture + grade only.
  LOCKED TEST outcomes remain sealed. Capture loop code-equivalent to Phase 24B.
"""
    CFG.write_text(cfg, encoding="utf-8")
    write_json(
        OUT / "prepare_hashes.json",
        {
            "schedule.json": sha256_file(str(OUT / "schedule.json")),
            "contract.json": sha256_file(str(OUT / "contract.json")),
            "config": sha256_file(str(CFG)),
            "phase24c_hash_verification.json": sha256_file(
                str(OUT / "phase24c_hash_verification.json")
            ),
            "new_schedule_sha256": schedule["new_schedule_sha256"],
        },
    )
    print(
        json.dumps(
            {
                "verified_phase24c": True,
                "n_new": schedule["n_new"],
                "n_reuse": schedule["n_reuse"],
                "new_schedule_sha256": schedule["new_schedule_sha256"],
                "capture_equivalence": capture_equiv,
                "config": str(CFG),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
