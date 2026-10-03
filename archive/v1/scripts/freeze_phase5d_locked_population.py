#!/usr/bin/env python3
"""Freeze Phase 5D locked pair population before any locked model call."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase5_locked import (  # noqa: E402
    CONFIRMATION_CRITERIA,
    EXPECTED_FINAL_PROMPT_TEXT_SHA256,
    EXPECTED_LOCKED_BASE_IDS_SHA256,
    EXPECTED_LOCKED_PAIR_N,
    EXPECTED_LOCKED_PROMPT_N,
    EXPECTED_LOCKED_PROMPT_TEXT_SHA256,
    FROZEN_CANDIDATE,
    LOCKED_FAMILIES,
    PROMPT_TEMPLATE_REVISION,
)
from pre_output_physiology.phase5_split import sha_prompt_texts, sha_sorted_ids  # noqa: E402
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    out = REPO_ROOT / "artifacts/phase5d_locked_freeze"
    out.mkdir(parents=True, exist_ok=True)
    data_dir = REPO_ROOT / "data/processed/phase5_design"

    scenarios = _load_jsonl(data_dir / "locked_base_scenarios.jsonl")
    if len(scenarios) != EXPECTED_LOCKED_PAIR_N:
        raise SystemExit(f"expected {EXPECTED_LOCKED_PAIR_N} locked bases")
    if any(s["family"] not in LOCKED_FAMILIES for s in scenarios):
        raise SystemExit("non-locked family in locked scenarios")
    if any(s.get("pool") != "locked" for s in scenarios):
        raise SystemExit("pool != locked")

    pair_ids = sorted(s["base_scenario_id"] for s in scenarios)
    pair_hash = sha_sorted_ids(pair_ids)
    if pair_hash != EXPECTED_LOCKED_BASE_IDS_SHA256:
        raise SystemExit(f"locked pair hash drift: {pair_hash}")

    all_prompts = _load_jsonl(data_dir / "final_candidate_prompts.jsonl")
    if sha_prompt_texts(all_prompts) != EXPECTED_FINAL_PROMPT_TEXT_SHA256:
        raise SystemExit("final prompt corpus hash drift")
    locked_prompts = [
        r
        for r in all_prompts
        if r.get("pool") == "locked" and r.get("split") == "final"
    ]
    if len(locked_prompts) != EXPECTED_LOCKED_PROMPT_N:
        raise SystemExit(f"expected {EXPECTED_LOCKED_PROMPT_N} locked prompts")
    if any(r.get("prompt_template_revision") != PROMPT_TEMPLATE_REVISION for r in locked_prompts):
        raise SystemExit("prompt revision must remain 2")
    prompt_hash = sha_prompt_texts(locked_prompts)
    if prompt_hash != EXPECTED_LOCKED_PROMPT_TEXT_SHA256:
        raise SystemExit(f"locked prompt hash drift: {prompt_hash}")

    per_family = {}
    for fam in LOCKED_FAMILIES:
        n = sum(1 for s in scenarios if s["family"] == fam)
        per_family[fam] = {"n_pairs": n, "n_prompts": n * 2}

    probe_path = REPO_ROOT / FROZEN_CANDIDATE["probe_artifact"]
    import hashlib

    probe_sha = hashlib.sha256(probe_path.read_bytes()).hexdigest()
    if probe_sha != FROZEN_CANDIDATE["probe_sha256"]:
        raise SystemExit(f"frozen probe sha drift: {probe_sha}")

    payload = {
        "created_at": utc_now_iso(),
        "phase": "phase5d",
        "pre_model_call_freeze": True,
        "model_generation_performed": False,
        "activation_extraction_performed": False,
        "locked_families": list(LOCKED_FAMILIES),
        "n_pairs": EXPECTED_LOCKED_PAIR_N,
        "n_prompts": EXPECTED_LOCKED_PROMPT_N,
        "pair_ids": pair_ids,
        "pair_ids_sha256": pair_hash,
        "locked_prompt_text_sha256": prompt_hash,
        "final_prompt_text_sha256": EXPECTED_FINAL_PROMPT_TEXT_SHA256,
        "prompt_template_revision": PROMPT_TEMPLATE_REVISION,
        "per_family": per_family,
        "frozen_candidate": FROZEN_CANDIDATE,
        "confirmation_criteria": CONFIRMATION_CRITERIA,
        "primary_estimand": "all_320_designed_pairs",
        "behavior_valid_sensitivity_only": True,
        "no_retraining": True,
        "no_layer_selection": True,
        "causal_interventions": False,
    }
    write_json(out / "locked_population.json", payload)
    print(
        json.dumps(
            {
                "n_pairs": EXPECTED_LOCKED_PAIR_N,
                "pair_ids_sha256": pair_hash,
                "locked_prompt_text_sha256": prompt_hash,
                "probe_sha256": probe_sha,
                "out": str(out.relative_to(REPO_ROOT)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
