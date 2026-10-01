#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

FILES = [
    "data/phase25/scenario_bank/base_scenarios.jsonl",
    "data/phase25/scenario_bank/prompts.jsonl",
    "data/phase25/scenario_bank/gold_answers.jsonl",
    "data/phase25/scenario_bank/content_audit.jsonl",
    "data/phase25/scenario_bank/schema.json",
    "artifacts/phase25_step1/validation_report.json",
    "artifacts/phase25_step1/near_duplicate_report.json",
    "artifacts/phase25_step1/scientific_content_audit.json",
    "reports/phase25_step1_scenario_bank.md",
    "scripts/generate_phase25_step1_scenario_bank.py",
    "scripts/validate_phase25_step1_scenario_bank.py",
    "scripts/finalize_phase25_step1_manifest.py",
    "tests/test_phase25_step1_scenario_bank.py",
]

def h(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    a=ap.parse_args()
    missing=[x for x in FILES if not (a.repo_root/x).exists()]
    if missing:
        raise SystemExit(f"Missing manifest targets: {missing}")
    payload={
        "phase24_start_sha":"2af0c27fe034b3419dd9de8ccaafbb2780d6f899",
        "protocol_base_freeze_sha":"813776c6dbdeb96aa6f6cb87db78edd3c76b735a",
        "protocol_freeze_sha":"654750f713daf686a7f711de8c836de478bb5e93",
        "branch":"phase25/scenario-bank",
        "gpu_used":False,
        "model_inference_used":False,
        "files":{rel:h(a.repo_root/rel) for rel in FILES},
    }
    out=a.repo_root/"artifacts/phase25_step1/manifest_sha256.json"
    out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")

if __name__=="__main__":
    main()
