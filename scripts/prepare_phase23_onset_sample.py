#!/usr/bin/env python3
"""Freeze Phase-23D onset-validation sample (before Stage 4)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase23_open_grader import (  # noqa: E402
    N_ONSET_VALIDATION_SAMPLE,
    ONSET_ACCEPTANCE,
    _sha_json,
    select_onset_validation_sample,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

PROC = REPO_ROOT / "data/processed/phase23_open_grader"
ART = REPO_ROOT / "artifacts/phase23_onset_sample"


def main() -> int:
    corpus = [
        json.loads(x)
        for x in (PROC / "reference_corpus.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    ids = select_onset_validation_sample(corpus, n=N_ONSET_VALIDATION_SAMPLE)
    by_id = {r["continuation_id"]: r for r in corpus}
    rows = [by_id[cid] for cid in ids]
    ART.mkdir(parents=True, exist_ok=True)
    with (ART / "onset_validation.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    manifest = {
        "created_at": utc_now_iso(),
        "git_commit": subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "n": len(ids),
        "onset_sample_sha256": _sha_json(ids),
        "onset_acceptance_thresholds_frozen": ONSET_ACCEPTANCE,
        "continuation_ids": ids,
    }
    write_json(ART / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
