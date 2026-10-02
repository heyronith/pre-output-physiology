#!/usr/bin/env python3
"""Write SHA256 provenance manifest for Phase 26A-v3 (non-self-referential)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FILES = [
    "docs/phase26_research_protocol.md",
    "docs/phase26_construct_definitions.md",
    "docs/phase26_scenario_spec.md",
    "configs/phase26_consequences.yaml",
    "configs/phase26_behavior_labels.yaml",
    "configs/phase26_prompt_schema.yaml",
    "data/phase26/family_definitions_v3.json",
    "data/phase26/scenario_blueprints.jsonl",
    "src/pre_output_physiology/phase26_protocol.py",
    "scripts/generate_phase26a_v3_blueprints.py",
    "scripts/validate_phase26a.py",
    "scripts/finalize_phase26a_manifest.py",
    "tests/test_phase26a_protocol.py",
    "artifacts/phase26a/validation_report.json",
    "reports/phase26a_v3_final_freeze_report.md",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument(
        "--prior-v2-sha",
        default="b4bc3339435218691f250ba5f1ded8d5ccd331e5",
    )
    args = ap.parse_args()
    root = Path(args.repo_root)
    hashes = {rel: sha256_file(root / rel) for rel in FILES}
    out = {
        "branch": "phase26/protocol-freeze",
        "protocol_version": "phase26a_protocol_freeze_v3",
        "provenance_scheme": (
            "file_sha256_manifest_without_self_commit; "
            "v3_commit_sha_reported_externally_via_git"
        ),
        "prior_v2_freeze_commit": args.prior_v2_sha,
        "parent_endpoint_branch": "phase25/behavioral-screening",
        "parent_endpoint_sha": "43798403fc52c7b31106948774a8a0d983beece9",
        "gpu_used": False,
        "model_inference_used": False,
        "production_prompts_authored": False,
        "phase25_sealed_cohort_accessed": False,
        "n_families": 24,
        "n_state_instances": 48,
        "files": hashes,
    }
    path = root / "artifacts/phase26a/manifest_sha256.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
