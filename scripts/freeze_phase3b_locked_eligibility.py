#!/usr/bin/env python3
"""Freeze locked-test eligibility (lengths/IDs only; no predictive metrics).

Must run before Modal locked-test extraction. Does not use labels for decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.provenance import write_json  # noqa: E402
from pre_output_physiology.trajectory import (  # noqa: E402
    PREFIX_LENGTHS_K,
    analyze_prompt_response_boundary,
    eligible_for_k,
)

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
SURFACE_FREEZE_SHA = (
    "b25c5095c1cba440103343f90408c4caac0b59785bdc28b57a46abea405bcd00"
)


def _sha_ids(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=str(REPO_ROOT / "data/processed/phase3_roleplay"),
    )
    parser.add_argument(
        "--out",
        default=str(
            REPO_ROOT / "artifacts/phase3b_locked/locked_eligibility_freeze.json"
        ),
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    freeze_path = REPO_ROOT / "artifacts/phase3b_dev/surface_baseline_freeze.json"
    freeze_sha = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
    if freeze_sha != SURFACE_FREEZE_SHA:
        raise SystemExit(f"surface freeze hash drift: {freeze_sha}")

    rows = []
    with (data_dir / "locked_test_metadata.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != 500:
        raise SystemExit(f"expected 500 locked-test rows, got {len(rows)}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, use_fast=True
    )
    prepared = []
    for row in rows:
        tok = analyze_prompt_response_boundary(
            row["input_formatted"], row["model_outputs"], tokenizer
        )
        if not tok.prompt_prefix_exact or tok.boundary_token_straddle:
            raise SystemExit(f"boundary fail {row['example_id']}")
        prepared.append(
            {
                "example_id": row["example_id"],
                "prompt_sha256": row["prompt_sha256"],
                "n_resp": len(tok.response_suffix_ids),
            }
        )
    groups = {p["prompt_sha256"] for p in prepared}
    if len(groups) != 53:
        raise SystemExit(f"expected 53 prompt groups, got {len(groups)}")

    eligibility: dict[str, dict] = {}
    for k in PREFIX_LENGTHS_K:
        elig = [p for p in prepared if eligible_for_k(p["n_resp"], k)]
        eligibility[str(k)] = {
            "n_eligible": len(elig),
            "n_excluded_short": len(prepared) - len(elig),
            "eligible_ids_sha256": _sha_ids([p["example_id"] for p in elig]),
            "n_prompt_groups": len({p["prompt_sha256"] for p in elig}),
        }

    k0 = len(groups)
    kgt0 = sum(eligibility[str(k)]["n_eligible"] for k in PREFIX_LENGTHS_K if k > 0)
    artifact = {
        "created_at": "2026-09-25T16:03:30+00:00",
        "split": "locked_test",
        "n_rows": 500,
        "n_prompt_groups": 53,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "surface_baseline_freeze_sha256": SURFACE_FREEZE_SHA,
        "eligibility": eligibility,
        "projected_k0_forwards": k0,
        "projected_kgt0_forwards": kgt0,
        "projected_n_scientific_forwards": k0 + kgt0,
        "labels_used": False,
        "predictive_metrics_computed": False,
    }
    write_json(out_path, artifact)
    print(
        json.dumps(
            {
                "out": str(out_path),
                "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
                "projected_n_scientific_forwards": k0 + kgt0,
                "eligibility_n": {
                    k: eligibility[k]["n_eligible"] for k in eligibility
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
