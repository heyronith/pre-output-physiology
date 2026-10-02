#!/usr/bin/env python3
"""Write SHA256 provenance manifest for Phase 26A-v2 (non-self-referential)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

# Manifest hashes these files only. It does NOT embed the current commit SHA,
# avoiding a self-referential commit problem. Freeze/repair commit SHAs are
# recorded in reports/phase26a_v2_repair_report.md after commit creation.
FILES = [
    "docs/phase26_research_protocol.md",
    "docs/phase26_construct_definitions.md",
    "docs/phase26_scenario_spec.md",
    "configs/phase26_consequences.yaml",
    "configs/phase26_behavior_labels.yaml",
    "configs/phase26_prompt_schema.yaml",
    "data/phase26/scenario_blueprints.jsonl",
    "src/pre_output_physiology/phase26_protocol.py",
    "scripts/generate_phase26a_v2_blueprints.py",
    "scripts/validate_phase26a.py",
    "scripts/finalize_phase26a_manifest.py",
    "tests/test_phase26a_protocol.py",
    "artifacts/phase26a/validation_report.json",
    "reports/phase26a_v2_repair_report.md",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument(
        "--prior-freeze-sha",
        default="b6538fd3fadcbb23ae2229aa13c15a87d2b71395",
    )
    ap.add_argument(
        "--parent-endpoint-sha",
        default="43798403fc52c7b31106948774a8a0d983beece9",
    )
    args = ap.parse_args()
    root = Path(args.repo_root)

    hashes = {}
    for rel in FILES:
        p = root / rel
        if not p.exists():
            raise FileNotFoundError(rel)
        hashes[rel] = sha256_file(p)

    out = {
        "branch": "phase26/protocol-freeze",
        "protocol_version": "phase26a_protocol_freeze_v2",
        "provenance_scheme": (
            "file_sha256_manifest_without_self_commit; "
            "repair_commit_sha_recorded_in_freeze_report_after_commit"
        ),
        "prior_v1_freeze_commit": args.prior_freeze_sha,
        "parent_endpoint_branch": "phase25/behavioral-screening",
        "parent_endpoint_sha": args.parent_endpoint_sha,
        "gpu_used": False,
        "model_inference_used": False,
        "production_prompts_authored": False,
        "phase25_sealed_cohort_modified": False,
        "phase25_sealed_cohort_accessed": False,
        "files": hashes,
    }
    path = root / "artifacts/phase26a/manifest_sha256.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
