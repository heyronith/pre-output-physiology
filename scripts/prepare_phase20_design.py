#!/usr/bin/env python3
"""Freeze Phase 20A physiology design before model calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from prepare_phase8_design import sha_ids, sha_prompt_texts  # noqa: E402
from prepare_phase19_design import build_prompts as p19_build_prompts  # noqa: E402
from prepare_phase19_design import build_scenarios as p19_build_scenarios  # noqa: E402

from pre_output_physiology.phase15_onset import RULE_HASH, RULE_MANIFEST, RULE_VERSION  # noqa: E402
from pre_output_physiology.phase20_physiology import (  # noqa: E402
    N_CONTINUATIONS,
    N_PROMPTS,
    N_SAMPLES,
    N_TRAIN,
    N_VAL,
    PHASE15_RULE_HASH,
    PHASE18_STATUS_REQUIRED,
    PHASE19_STATUS_REQUIRED,
    PHASE19_USABLE_IDS_SHA256,
    SEED_BASE,
    TEMPERATURE,
    THRESHOLD_HASH,
    THRESHOLD_MANIFEST,
    TOP_P,
    TRAIN_FAMILIES,
    TRAIN_IDS_SHA256,
    TRAIN_PROMPT_GROUP_IDS,
    VAL_FAMILIES,
    VAL_IDS_SHA256,
    VAL_PROMPT_GROUP_IDS,
    build_generation_schedule,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402


def load_train_prompts() -> list[dict]:
    final = [
        json.loads(x)
        for x in (REPO_ROOT / "data/processed/phase14_design/final_prompts.jsonl")
        .read_text("utf-8")
        .splitlines()
        if x.strip()
    ]
    by = {p["prompt_group_id"]: p for p in final}
    out = []
    for pid in TRAIN_PROMPT_GROUP_IDS:
        p = dict(by[pid])
        p["split"] = "train"
        p["physiology_role"] = "train"
        out.append(p)
    return out


def load_val_prompts() -> list[dict]:
    """Regenerate Phase-19 screen prompts and take the frozen usable IDs."""
    scenarios = p19_build_scenarios()
    prompts = p19_build_prompts(scenarios)
    by = {p["prompt_group_id"]: p for p in prompts}
    out = []
    for pid in VAL_PROMPT_GROUP_IDS:
        if pid not in by:
            raise SystemExit(f"missing val prompt {pid}")
        p = dict(by[pid])
        p["split"] = "validation"
        p["physiology_role"] = "validation"
        out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        default=str(REPO_ROOT / "data/processed/phase20_design"),
    )
    ap.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "artifacts/phase20a_design"),
    )
    args = ap.parse_args()
    data = Path(args.data_dir)
    data.mkdir(parents=True, exist_ok=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if RULE_VERSION != "phase15_onset_v1" or RULE_HASH != RULE_MANIFEST["rule_hash"]:
        raise SystemExit("Phase-15 rule drift")
    if RULE_HASH != PHASE15_RULE_HASH:
        raise SystemExit("rule hash mismatch")
    if RULE_HASH != "6bb40fa50eaf07b52e4709355bdb4652060658c2e73d3e41775b8bccac4bc24f":
        raise SystemExit("rule hash vs spec")

    p18 = json.loads(
        (REPO_ROOT / "artifacts/phase18a_confirmation/confirmation_summary.json").read_text()
    )
    if p18["status"] != PHASE18_STATUS_REQUIRED:
        raise SystemExit(f"Phase 18 status: {p18['status']}")
    p19 = json.loads(
        (REPO_ROOT / "artifacts/phase19a_confirmation/confirmation_summary.json").read_text()
    )
    if p19["status"] != PHASE19_STATUS_REQUIRED:
        raise SystemExit(f"Phase 19 status: {p19['status']}")
    if p19["fresh_usable_prompt_group_ids_sha256"] != PHASE19_USABLE_IDS_SHA256:
        raise SystemExit("Phase-19 usable hash mismatch")
    if sorted(p19["fresh_usable_prompt_group_ids"]) != list(VAL_PROMPT_GROUP_IDS):
        raise SystemExit("Phase-19 usable ID list mismatch")

    train = load_train_prompts()
    val = load_val_prompts()
    if len(train) != N_TRAIN or len(val) != N_VAL:
        raise SystemExit("population size")
    for p in train:
        if p["family"] not in TRAIN_FAMILIES:
            raise SystemExit(f"bad train family {p['family']}")
        if "daycare" in p["family"]:
            raise SystemExit("daycare forbidden")
    for p in val:
        if p["family"] not in VAL_FAMILIES:
            raise SystemExit(f"bad val family {p['family']}")

    all_prompts = sorted(train + val, key=lambda p: p["prompt_group_id"])
    if len(all_prompts) != N_PROMPTS:
        raise SystemExit("28 prompts")
    if sha_ids([p["prompt_group_id"] for p in train]) != TRAIN_IDS_SHA256:
        raise SystemExit("train ID hash")
    if sha_ids([p["prompt_group_id"] for p in val]) != VAL_IDS_SHA256:
        raise SystemExit("val ID hash")

    schedule = build_generation_schedule(all_prompts)
    if len(schedule) != N_CONTINUATIONS:
        raise SystemExit("schedule size")

    (data / "physiology_prompts.jsonl").write_text(
        "".join(json.dumps(p, sort_keys=True) + "\n" for p in all_prompts), "utf-8"
    )
    (data / "generation_schedule.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in schedule), "utf-8"
    )

    matrix = {
        "created_at": utc_now_iso(),
        "phase": "phase20a",
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "n_prompts": N_PROMPTS,
        "n_train": N_TRAIN,
        "n_validation": N_VAL,
        "n_samples": N_SAMPLES,
        "n_continuations": N_CONTINUATIONS,
        "seed_base": SEED_BASE,
        "phase15_rule_version": RULE_VERSION,
        "phase15_rule_hash": RULE_HASH,
        "threshold_hash": THRESHOLD_HASH,
        "train_prompt_group_ids": list(TRAIN_PROMPT_GROUP_IDS),
        "train_prompt_group_ids_sha256": TRAIN_IDS_SHA256,
        "validation_prompt_group_ids": list(VAL_PROMPT_GROUP_IDS),
        "validation_prompt_group_ids_sha256": VAL_IDS_SHA256,
        "prompt_text_sha256": sha_prompt_texts(all_prompts),
        "generation_schedule_sha256": sha_ids([r["continuation_id"] for r in schedule]),
        "phase18_status": PHASE18_STATUS_REQUIRED,
        "phase19_status": PHASE19_STATUS_REQUIRED,
        "daycare_excluded": True,
        "locked_families_untouched": True,
        "model_calls_performed": False,
        "activations_authorized_discovery": False,
        "activations_authorized_validation": False,
    }
    write_json(out / "design_matrix.json", matrix)
    write_json(out / "threshold_manifest.json", THRESHOLD_MANIFEST)
    write_json(
        out / "populations.json",
        {
            "train_prompt_group_ids": list(TRAIN_PROMPT_GROUP_IDS),
            "train_sha256": TRAIN_IDS_SHA256,
            "validation_prompt_group_ids": list(VAL_PROMPT_GROUP_IDS),
            "validation_sha256": VAL_IDS_SHA256,
            "train_families": list(TRAIN_FAMILIES),
            "validation_families": list(VAL_FAMILIES),
        },
    )
    print(
        json.dumps(
            {
                "train_sha256": TRAIN_IDS_SHA256,
                "validation_sha256": VAL_IDS_SHA256,
                "generation_schedule_sha256": matrix["generation_schedule_sha256"],
                "threshold_hash": THRESHOLD_HASH,
                "n_continuations": N_CONTINUATIONS,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
