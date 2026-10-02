#!/usr/bin/env python3
"""Write SHA256 provenance manifest for Phase 26A freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

FILES = [
    "docs/phase26_research_protocol.md",
    "docs/phase26_construct_definitions.md",
    "docs/phase26_scenario_spec.md",
    "configs/phase26_consequences.yaml",
    "configs/phase26_behavior_labels.yaml",
    "configs/phase26_prompt_schema.yaml",
    "data/phase26/scenario_blueprints.jsonl",
    "src/pre_output_physiology/phase26_protocol.py",
    "scripts/validate_phase26a.py",
    "scripts/finalize_phase26a_manifest.py",
    "tests/test_phase26a_protocol.py",
    "artifacts/phase26a/validation_report.json",
    "reports/phase26a_protocol_freeze.md",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_rev(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--parent-endpoint-sha", default="")
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
        "protocol_version": "phase26a_protocol_freeze_v1",
        "parent_endpoint_branch": "phase25/behavioral-screening",
        "parent_endpoint_sha": args.parent_endpoint_sha or None,
        "freeze_commit_placeholder": "filled_after_commit",
        "gpu_used": False,
        "model_inference_used": False,
        "production_prompts_authored": False,
        "phase25_sealed_cohort_modified": False,
        "files": hashes,
    }
    path = root / "artifacts/phase26a/manifest_sha256.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
