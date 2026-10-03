#!/usr/bin/env python3
"""Freeze Phase 17A S1 design (discovery prompts + schedule) before model calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from prepare_phase8_design import sha_ids, sha_prompt_texts, sha_scenarios  # noqa: E402

from pre_output_physiology.phase15_onset import (  # noqa: E402
    RULE_HASH,
    RULE_MANIFEST,
    RULE_VERSION,
)
from pre_output_physiology.phase17_screen import (  # noqa: E402
    FINAL_PROMPT_GROUP_IDS_SHA256,
    FINAL_PROMPT_TEXT_SHA256,
    FINAL_SCENARIO_TEXT_SHA256,
    N_DISCOVERY_PROMPTS,
    N_S1_CONTINUATIONS,
    T_RATIONALE,
    TEMPERATURE,
    THRESHOLD_HASH,
    THRESHOLD_MANIFEST,
    TOP_P,
    build_s1_schedule,
    discovery_prompts,
    family_split,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        default=str(REPO_ROOT / "data/processed/phase17_design"),
    )
    ap.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "artifacts/phase17a_design"),
    )
    args = ap.parse_args()
    data = Path(args.data_dir)
    data.mkdir(parents=True, exist_ok=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    final = [
        json.loads(x)
        for x in (REPO_ROOT / "data/processed/phase14_design/final_prompts.jsonl")
        .read_text("utf-8")
        .splitlines()
        if x.strip()
    ]
    scen = [
        json.loads(x)
        for x in (
            REPO_ROOT / "data/processed/phase14_design/final_base_scenarios.jsonl"
        )
        .read_text("utf-8")
        .splitlines()
        if x.strip()
    ]
    if sha_scenarios(scen) != FINAL_SCENARIO_TEXT_SHA256:
        raise SystemExit("final scenario hash mismatch")
    if sha_ids([p["prompt_group_id"] for p in final]) != FINAL_PROMPT_GROUP_IDS_SHA256:
        raise SystemExit("final prompt-group hash mismatch")
    if sha_prompt_texts(final) != FINAL_PROMPT_TEXT_SHA256:
        raise SystemExit("final prompt-text hash mismatch")
    if RULE_VERSION != "phase15_onset_v1" or RULE_HASH != RULE_MANIFEST["rule_hash"]:
        raise SystemExit("Phase-15 rule drift")
    if RULE_HASH != "6bb40fa50eaf07b52e4709355bdb4652060658c2e73d3e41775b8bccac4bc24f":
        raise SystemExit("Phase-15 rule hash mismatch vs Phase-17 spec")

    split = family_split()
    frozen_split = json.loads(
        (REPO_ROOT / "artifacts/phase14a_design/family_split.json").read_text("utf-8")
    )
    for k in ("discovery_train", "discovery_validation", "locked_generalization"):
        if split[k] != frozen_split[k]:
            raise SystemExit(f"family split drift on {k}")

    disc = discovery_prompts(final)
    if len(disc) != N_DISCOVERY_PROMPTS:
        raise SystemExit("discovery count")
    locked_ids = [
        p["prompt_group_id"]
        for p in final
        if p["family"] in set(split["locked_generalization"])
    ]

    # Write discovery prompt slice (pointers into Phase-14 finals; text unchanged).
    disc_path = data / "discovery_prompts.jsonl"
    disc_path.write_text(
        "".join(json.dumps(p, sort_keys=True) + "\n" for p in disc), "utf-8"
    )
    s1 = build_s1_schedule(disc)
    s1_path = data / "s1_sampling_schedule.jsonl"
    s1_path.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in s1), "utf-8"
    )
    if len(s1) != N_S1_CONTINUATIONS:
        raise SystemExit("S1 size")

    # Ensure S1 seeds never overlap S2 base range.
    seeds = [r["sample_seed"] for r in s1]
    if max(seeds) >= 17_100_000:
        raise SystemExit("S1 seeds collide with S2 base")

    matrix = {
        "created_at": utc_now_iso(),
        "phase": "phase17a",
        "temperature": TEMPERATURE,
        "temperature_rationale": T_RATIONALE,
        "top_p": TOP_P,
        "n_discovery_prompts": N_DISCOVERY_PROMPTS,
        "n_s1_continuations": N_S1_CONTINUATIONS,
        "n_locked_prompts_untouched": len(locked_ids),
        "phase15_rule_version": RULE_VERSION,
        "phase15_rule_hash": RULE_HASH,
        "threshold_hash": THRESHOLD_HASH,
        "final_scenario_text_sha256": FINAL_SCENARIO_TEXT_SHA256,
        "final_prompt_group_ids_sha256": FINAL_PROMPT_GROUP_IDS_SHA256,
        "final_prompt_text_sha256": FINAL_PROMPT_TEXT_SHA256,
        "discovery_prompt_group_ids_sha256": sha_ids(
            [p["prompt_group_id"] for p in disc]
        ),
        "discovery_prompt_text_sha256": sha_prompt_texts(disc),
        "s1_schedule_sha256": sha_ids([r["continuation_id"] for r in s1]),
        "locked_prompt_group_ids_sha256": sha_ids(sorted(locked_ids)),
        "family_split": {
            "discovery_train": split["discovery_train"],
            "discovery_validation": split["discovery_validation"],
            "locked_generalization": split["locked_generalization"],
        },
        "model_calls_performed": False,
        "s2_schedule_frozen": False,
    }
    write_json(out / "design_matrix.json", matrix)
    write_json(out / "threshold_manifest.json", THRESHOLD_MANIFEST)
    write_json(out / "family_split.json", frozen_split)
    print(json.dumps({k: matrix[k] for k in (
        "discovery_prompt_group_ids_sha256",
        "s1_schedule_sha256",
        "threshold_hash",
        "n_s1_continuations",
        "temperature",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
