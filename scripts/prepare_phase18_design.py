#!/usr/bin/env python3
"""Freeze Phase 18A enriched cohort + fresh seed schedule before model calls."""

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
)
from pre_output_physiology.phase18_cohort import (  # noqa: E402
    EXCLUDED_FAMILIES,
    LOCKED_FAMILIES,
    N_COHORT,
    N_CONTINUATIONS,
    N_SAMPLES,
    PHASE15_RULE_HASH,
    PHASE17_CONFIRMED_IDS_SHA256,
    PHASE17_STATUS,
    QUALIFYING_FAMILIES,
    QUALIFYING_TRAIN_FAMILIES,
    QUALIFYING_VAL_FAMILIES,
    SEED_BASE,
    TEMPERATURE,
    THRESHOLD_HASH,
    THRESHOLD_MANIFEST,
    TOP_P,
    build_fresh_schedule,
    load_phase17_confirmed,
    select_enriched_cohort,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        default=str(REPO_ROOT / "data/processed/phase18_design"),
    )
    ap.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "artifacts/phase18a_design"),
    )
    ap.add_argument(
        "--phase17-screen",
        default=str(REPO_ROOT / "artifacts/phase17a_screen/screen_summary.json"),
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
    if RULE_HASH != PHASE15_RULE_HASH:
        raise SystemExit("Phase-15 rule hash mismatch")
    if RULE_HASH != "6bb40fa50eaf07b52e4709355bdb4652060658c2e73d3e41775b8bccac4bc24f":
        raise SystemExit("Phase-15 rule hash mismatch vs Phase-18 spec")

    screen = json.loads(Path(args.phase17_screen).read_text("utf-8"))
    if screen.get("status") != PHASE17_STATUS:
        raise SystemExit(f"Phase 17 must remain HOLD; got {screen.get('status')}")
    confirmed = load_phase17_confirmed(screen)
    cohort = select_enriched_cohort(confirmed)

    by_id = {p["prompt_group_id"]: p for p in final}
    selected_prompts = []
    for sid in cohort["selected_prompt_group_ids"]:
        p = by_id[sid]
        if p["family"] not in QUALIFYING_FAMILIES:
            raise SystemExit(f"non-qualifying family in cohort: {p['family']}")
        if p["family"] in LOCKED_FAMILIES or p["family"] in EXCLUDED_FAMILIES:
            raise SystemExit(f"excluded/locked family in cohort: {p['family']}")
        meta = next(s for s in cohort["selected"] if s["prompt_group_id"] == sid)
        row = dict(p)
        row["split_role"] = meta["split_role"]
        row["selection_digest"] = meta["selection_digest"]
        row["phase17_s2_n_record"] = meta["phase17_s2_n_record"]
        row["phase17_s2_n_alternate"] = meta["phase17_s2_n_alternate"]
        row["phase17_s2_n_valid"] = meta["phase17_s2_n_valid"]
        row["phase17_s2_alternate_fraction"] = meta["phase17_s2_alternate_fraction"]
        selected_prompts.append(row)

    if len(selected_prompts) != N_COHORT:
        raise SystemExit("cohort size")
    for fam in QUALIFYING_FAMILIES:
        n = sum(1 for p in selected_prompts if p["family"] == fam)
        if n != 6:
            raise SystemExit(f"{fam} has {n} != 6")

    prompts_path = data / "cohort_prompts.jsonl"
    prompts_path.write_text(
        "".join(json.dumps(p, sort_keys=True) + "\n" for p in selected_prompts),
        "utf-8",
    )

    schedule = build_fresh_schedule(cohort)
    sched_path = data / "fresh_sampling_schedule.jsonl"
    sched_path.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in schedule),
        "utf-8",
    )
    if len(schedule) != N_CONTINUATIONS:
        raise SystemExit("schedule size")
    seeds = [r["sample_seed"] for r in schedule]
    if min(seeds) < SEED_BASE:
        raise SystemExit("seed base collision")
    if max(seeds) >= 19_000_000:
        raise SystemExit("seed range unexpected")
    # Disjoint from Phase-14 (14M) and Phase-17 (17M / 17.1M).
    if any(s < 18_000_000 for s in seeds):
        raise SystemExit("seeds overlap prior phases")

    locked_ids = [
        p["prompt_group_id"]
        for p in final
        if p["family"] in set(LOCKED_FAMILIES)
    ]

    matrix = {
        "created_at": utc_now_iso(),
        "phase": "phase18a",
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "n_cohort": N_COHORT,
        "n_samples_per_prompt": N_SAMPLES,
        "n_continuations": N_CONTINUATIONS,
        "seed_base": SEED_BASE,
        "phase15_rule_version": RULE_VERSION,
        "phase15_rule_hash": RULE_HASH,
        "threshold_hash": THRESHOLD_HASH,
        "phase17_status": PHASE17_STATUS,
        "phase17_confirmed_ids_sha256": PHASE17_CONFIRMED_IDS_SHA256,
        "qualifying_families": list(QUALIFYING_FAMILIES),
        "qualifying_train_families": list(QUALIFYING_TRAIN_FAMILIES),
        "qualifying_validation_families": list(QUALIFYING_VAL_FAMILIES),
        "excluded_families": list(EXCLUDED_FAMILIES),
        "locked_families_untouched": list(LOCKED_FAMILIES),
        "cohort_prompt_group_ids": cohort["selected_prompt_group_ids"],
        "cohort_sha256": cohort["cohort_sha256"],
        "train_prompt_group_ids": cohort["train_prompt_group_ids"],
        "validation_prompt_group_ids": cohort["validation_prompt_group_ids"],
        "n_train": cohort["n_train"],
        "n_validation": cohort["n_validation"],
        "cohort_prompt_text_sha256": sha_prompt_texts(selected_prompts),
        "fresh_schedule_sha256": sha_ids([r["continuation_id"] for r in schedule]),
        "final_scenario_text_sha256": FINAL_SCENARIO_TEXT_SHA256,
        "final_prompt_group_ids_sha256": FINAL_PROMPT_GROUP_IDS_SHA256,
        "final_prompt_text_sha256": FINAL_PROMPT_TEXT_SHA256,
        "locked_prompt_group_ids_sha256": sha_ids(sorted(locked_ids)),
        "selection": {
            "salt": cohort["selection_salt"],
            "n_per_family": cohort["n_per_family"],
            "rule": "SHA256(salt+prompt_group_id) ascending; first 6 per qualifying family",
        },
        "model_calls_performed": False,
        "activations_authorized": False,
    }
    write_json(out / "design_matrix.json", matrix)
    write_json(out / "threshold_manifest.json", THRESHOLD_MANIFEST)
    write_json(out / "cohort.json", cohort)

    print(
        json.dumps(
            {
                "cohort_sha256": matrix["cohort_sha256"],
                "fresh_schedule_sha256": matrix["fresh_schedule_sha256"],
                "threshold_hash": matrix["threshold_hash"],
                "n_continuations": matrix["n_continuations"],
                "n_cohort": matrix["n_cohort"],
                "n_train": matrix["n_train"],
                "n_validation": matrix["n_validation"],
                "temperature": matrix["temperature"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
