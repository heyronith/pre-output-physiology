#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

FILES = [
    "docs/phase25_step1_v2_protocol.md",
    "configs/phase25_step1_v2_families.json",
    "scripts/generate_phase25_step1_v2.py",
    "scripts/validate_phase25_step1_v2.py",
    "tests/test_phase25_step1_v2.py",
    "data/phase25/scenario_bank_v2/base_scenarios.jsonl",
    "data/phase25/scenario_bank_v2/prompts.jsonl",
    "data/phase25/scenario_bank_v2/gold_answers.jsonl",
    "data/phase25/scenario_bank_v2/family_metadata.jsonl",
    "data/phase25/scenario_bank_v2/schema.json",
    "artifacts/phase25_step1_v2/validation_report.json",
    "reports/phase25_step1_v2_scenario_bank.md",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()
    root = Path(args.repo_root)
    hashes = {}
    for rel in FILES:
        b = (root / rel).read_bytes()
        hashes[rel] = hashlib.sha256(b).hexdigest()
    out = {
        "branch": "phase25/scenario-bank-v2",
        "v1_frozen_parent": "380a9eec889ad520abceda2f69852fa22ab93a6e",
        "v2_protocol_freeze_sha": "2a692d5067028268a9b68f03308c7a4d43690334",
        "gpu_used": False,
        "model_inference_used": False,
        "files": hashes,
    }
    p = root / "artifacts/phase25_step1_v2/manifest_sha256.json"
    p.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
