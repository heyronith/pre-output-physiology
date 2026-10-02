#!/usr/bin/env python3
"""Finalize Phase 26C primary SHA256 manifest (prep artifacts only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from pre_output_physiology.phase26c_primary import (
    PARENT_PHASE26A_COMMIT,
    PARENT_PHASE26B_COMMIT,
    PHASE26A_SOURCE_FILES,
    PHASE26B_FROZEN_FILES,
    PROTOCOL_VERSION,
    sha256_bytes,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()
    root = Path(args.repo_root)
    cfg = yaml.safe_load(
        (root / "configs/phase26c_primary_behavioral_feasibility.yaml").read_text()
    )
    artifacts = [
        "configs/phase26c_primary_behavioral_feasibility.yaml",
        "src/pre_output_physiology/phase26c_primary.py",
        "scripts/preflight_phase26c_primary.py",
        "scripts/summarize_phase26c_primary.py",
        "scripts/finalize_phase26c_primary_manifest.py",
        "modal/phase26c_primary_behavioral_feasibility.py",
        "tests/test_phase26c_primary.py",
        "docs/phase26c_primary_behavioral_feasibility.md",
        "reports/phase26c_primary_readiness.md",
        cfg["paths"]["prompt_bank"],
        cfg["paths"]["template_assignment"],
        cfg["paths"]["selection_manifest"],
        cfg["paths"]["inference_manifest"],
        cfg["paths"]["seed_manifest"],
        cfg["paths"]["preflight"],
        cfg["paths"]["config_hash_manifest"],
    ]
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "parent_phase26a_commit": PARENT_PHASE26A_COMMIT,
        "parent_phase26b_commit": PARENT_PHASE26B_COMMIT,
        "inference_executed": False,
        "gpu_used": False,
        "model_weights_loaded": False,
        "phase26a_source_sha256": {
            rel: sha256_bytes((root / rel).read_bytes()) for rel in PHASE26A_SOURCE_FILES
        },
        "phase26b_frozen_sha256": {
            rel: sha256_bytes((root / rel).read_bytes()) for rel in PHASE26B_FROZEN_FILES
        },
        "artifact_sha256": {
            rel: sha256_bytes((root / rel).read_bytes())
            for rel in artifacts
            if (root / rel).exists()
        },
    }
    out = root / cfg["paths"]["manifest_sha256"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
