#!/usr/bin/env python3
"""Validate Phase 26A protocol freeze artifacts (no model inference)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pre_output_physiology.phase26_protocol import (
    PROTOCOL_VERSION,
    load_yaml,
    read_jsonl,
    validate_bank,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument(
        "--write-report",
        default="artifacts/phase26a/validation_report.json",
    )
    args = ap.parse_args()
    root = Path(args.repo_root)

    blueprints = read_jsonl(root / "data/phase26/scenario_blueprints.jsonl")
    consequences = load_yaml(root / "configs/phase26_consequences.yaml")
    labels = load_yaml(root / "configs/phase26_behavior_labels.yaml")
    schema = load_yaml(root / "configs/phase26_prompt_schema.yaml")

    report = validate_bank(blueprints, consequences, labels, schema)
    report["validator"] = "scripts/validate_phase26a.py"
    report["protocol_version"] = PROTOCOL_VERSION

    out = root / args.write_report
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if report["pass"]:
        print(f"PASS — {report['n_scenarios']} scenarios; report → {out}")
        return 0
    print("FAIL")
    for e in report["errors"]:
        print(f"  - {e}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
