#!/usr/bin/env python3
"""Build Phase 27 scenario/prompt/manifest/seed artifacts (CPU only; no inference)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from pre_output_physiology.phase27_extreme_roleplay import (
    GRADING_SPEC_ID,
    PROTOCOL_VERSION,
    build_inference_manifest,
    build_prompt_bank,
    build_scenario_rows,
    build_seed_manifest,
    load_phase27_tokenizer,
    sha256_file,
    write_jsonl,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()
    root = Path(args.repo_root)
    cfg = yaml.safe_load(
        (root / "configs/phase27_extreme_roleplay_feasibility.yaml").read_text()
    )

    # Tokenizer only — never model weights.
    tokenizer = load_phase27_tokenizer()
    scenarios = build_scenario_rows(tokenizer=tokenizer)
    prompts = build_prompt_bank(scenarios)
    jobs = build_inference_manifest(prompts)
    seeds = build_seed_manifest(jobs)

    write_jsonl(root / cfg["paths"]["scenarios"], scenarios)
    write_jsonl(root / cfg["paths"]["prompts"], prompts)
    write_jsonl(root / cfg["paths"]["inference_manifest"], jobs)
    write_jsonl(root / cfg["paths"]["seed_manifest"], seeds)

    grading_sha = sha256_file(root / cfg["paths"]["grading_spec"])
    grader_prompt_sha = sha256_file(root / cfg["paths"]["grader_prompt"])
    print(
        json.dumps(
            {
                "protocol_version": PROTOCOL_VERSION,
                "grading_spec_id": GRADING_SPEC_ID,
                "n_scenarios": len(scenarios),
                "n_prompts": len(prompts),
                "n_jobs": len(jobs),
                "n_seeds": len(seeds),
                "scenarios_sha256": sha256_file(root / cfg["paths"]["scenarios"]),
                "prompts_sha256": sha256_file(root / cfg["paths"]["prompts"]),
                "inference_manifest_sha256": sha256_file(
                    root / cfg["paths"]["inference_manifest"]
                ),
                "seed_manifest_sha256": sha256_file(root / cfg["paths"]["seed_manifest"]),
                "grading_spec_sha256": grading_sha,
                "grader_prompt_sha256": grader_prompt_sha,
                "model_weights_loaded": False,
                "inference_executed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
