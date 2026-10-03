#!/usr/bin/env python3
"""Phase 27 Gemma grading preflight (CPU only; no Gemma weights / no labels)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from pre_output_physiology.phase27_gemma_grading import (
    CHECKPOINT_BATCH_SIZE,
    EXPECTED_GRADER_PROMPT_SHA256,
    EXPECTED_GRADING_SPEC_SHA256,
    EXPECTED_INFERENCE_MANIFEST_SHA256,
    EXPECTED_RAW_SHA256,
    GRADER_MODEL,
    GRADER_REVISION,
    GRADER_SYSTEM,
    GRADING_PROTOCOL_VERSION,
    GRADING_STATUS_PREFLIGHT,
    N_PLANNED_GENERATIONS,
    RESULTS_VOLUME_NAME,
    assert_raw_source,
    build_grading_manifest,
    find_grader_input_leaks,
    load_gemma_tokenizer,
    load_grader_prompt_template,
    read_jsonl,
    render_gemma_model_input,
    build_grader_messages_from_template,
    run_key_from_grading_manifest_sha,
    sha256_file,
    verify_frozen_grader_model_input,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()
    root = Path(args.repo_root)
    errors: list[str] = []

    cfg = yaml.safe_load(
        (root / "configs/phase27_extreme_roleplay_feasibility.yaml").read_text(
            encoding="utf-8"
        )
    )
    if cfg.get("grading_status") != GRADING_STATUS_PREFLIGHT:
        errors.append(f"unexpected grading_status {cfg.get('grading_status')}")
    if cfg.get("authorizations", {}).get("grader_execution_authorized") is not False:
        errors.append("grader_execution_authorized must be false")
    for key in (
        "activation_capture_authorized",
        "output_hidden_states_authorized",
        "logit_save_authorized",
        "physiology_collection_authorized",
    ):
        if cfg.get("authorizations", {}).get(key) is not False:
            errors.append(f"{key} must be false")

    try:
        src = assert_raw_source(root)
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
        src = None

    if sha256_file(root / "docs/phase27_extreme_roleplay_grader_prompt.txt") != (
        EXPECTED_GRADER_PROMPT_SHA256
    ):
        errors.append("grader prompt SHA mismatch")
    if sha256_file(root / "docs/phase27_extreme_roleplay_grading_spec.md") != (
        EXPECTED_GRADING_SPEC_SHA256
    ):
        errors.append("grading spec SHA mismatch")
    if sha256_file(root / "artifacts/phase27/inference_manifest.jsonl") != (
        EXPECTED_INFERENCE_MANIFEST_SHA256
    ):
        errors.append("inference manifest SHA mismatch")

    man_path = root / "artifacts/phase27/grading_manifest.jsonl"
    if not man_path.exists():
        errors.append("missing grading_manifest.jsonl")
        jobs = []
    else:
        jobs = read_jsonl(man_path)
        if len(jobs) != N_PLANNED_GENERATIONS:
            errors.append(f"grading jobs {len(jobs)} != {N_PLANNED_GENERATIONS}")
        if len({j["job_id"] for j in jobs}) != N_PLANNED_GENERATIONS:
            errors.append("grading job_id uniqueness failed")
        if src is not None and {j["job_id"] for j in jobs} != {
            r["job_id"] for r in src["raw"]
        }:
            errors.append("grading job set != raw job set")

    # Rebuild with tokenizer and compare frozen model-input hashes
    tok = load_gemma_tokenizer()
    rebuilt = build_grading_manifest(repo_root=root, tokenizer=tok)
    if [j["grader_model_input_sha256"] for j in jobs] != [
        j["grader_model_input_sha256"] for j in rebuilt
    ]:
        errors.append("grading model-input hashes not reproducible")

    template = load_grader_prompt_template(root)
    for j in jobs[:3] + jobs[-3:]:
        leaks = find_grader_input_leaks(j["grader_user_text"].split("[Response]")[0])
        if leaks:
            errors.append(f"{j['grading_job_id']} leak: {leaks}")
        if "ground_truth" in j["grader_user_text"] or "advantageous_false" in j[
            "grader_user_text"
        ]:
            errors.append(f"{j['grading_job_id']}: metadata key in grader user text")
        msgs = build_grader_messages_from_template(
            scenario=j["scenario_text"],
            question=j["question_text"],
            response=j["response_text"],
            template=template,
        )
        text, ids = render_gemma_model_input(tok, msgs)
        try:
            verify_frozen_grader_model_input(
                tok,
                model_input_text=j["grader_model_input_text"],
                model_input_sha256=j["grader_model_input_sha256"],
                input_ids_sha256=j["grader_input_ids_sha256"],
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{j['grading_job_id']}: {exc}")
        if text != j["grader_model_input_text"]:
            errors.append(f"{j['grading_job_id']}: model input re-render drift")
        if j["grader"]["model_id"] != GRADER_MODEL:
            errors.append("grader model id drift")
        if j["grader"]["model_revision"] != GRADER_REVISION:
            errors.append("grader revision drift")
        if j["grader_system"] != GRADER_SYSTEM:
            errors.append("grader system drift")

    gcfg = cfg.get("grader") or {}
    if gcfg.get("model") != GRADER_MODEL or gcfg.get("revision") != GRADER_REVISION:
        errors.append("config grader model/revision mismatch")

    manifest_sha = sha256_file(man_path) if man_path.exists() else None
    report = {
        "ok": not errors,
        "errors": errors,
        "grading_protocol_version": GRADING_PROTOCOL_VERSION,
        "grading_status": cfg.get("grading_status"),
        "grader_execution_authorized": False,
        "raw_sha256": EXPECTED_RAW_SHA256,
        "inference_manifest_sha256": EXPECTED_INFERENCE_MANIFEST_SHA256,
        "grader_prompt_sha256": EXPECTED_GRADER_PROMPT_SHA256,
        "grading_spec_sha256": EXPECTED_GRADING_SPEC_SHA256,
        "grading_manifest_sha256": manifest_sha,
        "run_key": run_key_from_grading_manifest_sha(manifest_sha)
        if manifest_sha
        else None,
        "n_grading_jobs": len(jobs),
        "checkpoint_batch_size": CHECKPOINT_BATCH_SIZE,
        "results_volume": RESULTS_VOLUME_NAME,
        "gemma_model": GRADER_MODEL,
        "gemma_revision": GRADER_REVISION,
        "gemma_weights_loaded": False,
        "labels_produced": False,
        "mixed_eligible_computed": False,
        "pass_hold_computed": False,
        "physiology_disabled": True,
    }
    out = root / "artifacts/phase27/grading_preflight_validation.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        print("FAIL")
        for e in errors[:40]:
            print(f"  - {e}")
        return 1
    print("PASS — Phase 27 Gemma grading preflight ready (no labels).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
