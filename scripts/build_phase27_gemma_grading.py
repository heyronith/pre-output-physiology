#!/usr/bin/env python3
"""Build Phase 27 Gemma grading manifest (CPU tokenizer only; no Gemma weights)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from pre_output_physiology.phase27_gemma_grading import (
    EXPECTED_GRADER_PROMPT_SHA256,
    EXPECTED_GRADING_SPEC_SHA256,
    EXPECTED_RAW_SHA256,
    GRADING_PROTOCOL_VERSION,
    GRADING_STATUS_PREFLIGHT,
    build_grading_manifest,
    load_gemma_tokenizer,
    run_key_from_grading_manifest_sha,
    sha256_file,
    write_jsonl,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()
    root = Path(args.repo_root)
    cfg_path = root / "configs/phase27_extreme_roleplay_feasibility.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # Tokenizer only.
    tok = load_gemma_tokenizer()
    jobs = build_grading_manifest(repo_root=root, tokenizer=tok)

    out_manifest = root / "artifacts/phase27/grading_manifest.jsonl"
    write_jsonl(out_manifest, jobs)
    manifest_sha = sha256_file(out_manifest)
    run_key = run_key_from_grading_manifest_sha(manifest_sha)

    summary = {
        "grading_protocol_version": GRADING_PROTOCOL_VERSION,
        "grading_status": cfg.get("grading_status", GRADING_STATUS_PREFLIGHT),
        "grader_execution_authorized": bool(
            cfg.get("authorizations", {}).get("grader_execution_authorized")
        ),
        "n_grading_jobs": len(jobs),
        "grading_manifest_sha256": manifest_sha,
        "run_key": run_key,
        "raw_generations_sha256": EXPECTED_RAW_SHA256,
        "grader_prompt_sha256": EXPECTED_GRADER_PROMPT_SHA256,
        "grading_spec_sha256": sha256_file(
            root / "docs/phase27_extreme_roleplay_grading_spec.md"
        ),
        "gemma_weights_loaded": False,
        "labels_produced": False,
        "mixed_eligible_computed": False,
        "pass_hold_computed": False,
    }
    if summary["grading_spec_sha256"] != EXPECTED_GRADING_SPEC_SHA256:
        raise SystemExit(
            f"grading spec SHA mismatch: {summary['grading_spec_sha256']}"
        )
    (root / "artifacts/phase27/grading_build_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
