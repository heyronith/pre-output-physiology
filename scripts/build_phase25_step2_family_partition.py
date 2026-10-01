#!/usr/bin/env python3
"""Build prospective Phase-25 Step-2 family partition (no model calls)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pre_output_physiology.phase25_step2 import build_family_partition, read_jsonl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()
    root = Path(args.repo_root)
    meta = read_jsonl(root / "data/phase25/scenario_bank_v2/family_metadata.jsonl")
    part = build_family_partition(meta)
    out = root / "splits/phase25_step2_family_partition.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(part, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "n_screen": part["n_screen_families"],
                "n_sealed": part["n_sealed_families"],
                "structure_counts_screen": part["structure_counts_screen"],
                "structure_counts_sealed": part["structure_counts_sealed"],
                "path": str(out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
