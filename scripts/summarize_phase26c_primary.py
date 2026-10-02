#!/usr/bin/env python3
"""Parse, label, summarize Phase 26C primary raw generations (post-run; CPU only)."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from pre_output_physiology.phase26_protocol import parse_report
from pre_output_physiology.phase26c_primary import (
    PROTOCOL_VERSION,
    aggregate_summaries,
    label_row,
    read_jsonl,
    summarize_family,
    validate_raw_against_manifest,
    write_jsonl,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()
    root = Path(args.repo_root)
    cfg = yaml.safe_load(
        (root / "configs/phase26c_primary_behavioral_feasibility.yaml").read_text()
    )
    raw_path = root / cfg["paths"]["raw_generations"]
    if not raw_path.exists():
        raise SystemExit(
            f"raw generations not found: {raw_path} "
            "(expected only after authorized GPU execution)"
        )
    rows = read_jsonl(raw_path)
    manifest_jobs = read_jsonl(root / cfg["paths"]["inference_manifest"])
    try:
        validate_raw_against_manifest(rows, manifest_jobs)
    except ValueError as exc:
        raise SystemExit(f"STOP summarization: {exc}") from exc

    for r in rows:
        if r.get("technical_failure"):
            r["parse_valid"] = False
            r["parsed_decision"] = None
            r["malformed_reason"] = "technical_failure"
        else:
            parsed = parse_report(r.get("raw_response_text") or "")
            r["parsed_decision"] = parsed["parsed_decision"]
            r["parse_valid"] = parsed["parse_valid"]
            r["malformed_reason"] = parsed.get("malformed_reason")

    by_fam: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_fam[r["scenario_id"]].append(r)

    family_summaries = []
    labeled: list[dict] = []
    for sid in sorted(by_fam):
        fam_rows = by_fam[sid]
        summary = summarize_family(fam_rows)
        k_verified = summary["k_verified"]
        gold = summary["gold_decision"]
        for r in fam_rows:
            lr = dict(r)
            lab = label_row(r, gold=gold, k_verified=k_verified)
            lr["behavioral_primary"] = lab["primary"]
            lr["secondary_false_direction"] = lab["secondary_false_direction"]
            lr["k_verified_family"] = k_verified
            labeled.append(lr)
        family_summaries.append(summary)

    cell_summaries = [c for s in family_summaries for c in s["cells"]]
    paired_xe = [p for s in family_summaries for p in s["paired_xe"]]
    agg = aggregate_summaries(family_summaries)

    write_jsonl(root / cfg["paths"]["labeled_generations"], labeled)
    write_jsonl(root / cfg["paths"]["per_cell_summary"], cell_summaries)
    write_jsonl(root / cfg["paths"]["per_family_summary"], family_summaries)
    write_jsonl(root / cfg["paths"]["paired_xe_summary"], paired_xe)
    (root / cfg["paths"]["aggregate_summary"]).write_text(
        json.dumps(agg, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    malformed = Counter()
    for r in labeled:
        if not r.get("parse_valid") or r.get("technical_failure"):
            malformed[r["condition"]] += 1

    print(
        json.dumps(
            {
                "protocol_version": PROTOCOL_VERSION,
                "n_generations": len(labeled),
                "n_technical_failures": sum(
                    1 for r in labeled if r.get("technical_failure")
                ),
                "n_k_verified_families": agg["n_k_verified_families"],
                "gate_verdict": agg["go_no_go"]["phase26c_gate_verdict"],
                "malformed_by_condition": dict(malformed),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
