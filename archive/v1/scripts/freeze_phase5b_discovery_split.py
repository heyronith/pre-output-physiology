#!/usr/bin/env python3
"""Freeze Phase 5B discovery train/validation split + primary all-pair populations.

Must run and be committed BEFORE discovery model generation.
No model calls. Locked families excluded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase5_conditions import (  # noqa: E402
    DISCOVERY_FAMILIES,
    LOCKED_FAMILIES,
    PREREGISTERED_BASELINES,
)
from pre_output_physiology.phase5_split import (  # noqa: E402
    FUTURE_PROBE_PROCEDURE,
    compute_discovery_family_split,
    sha_prompt_texts,
    sha_sorted_ids,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

EXPECTED_FINAL_PROMPT_TEXT_SHA256 = (
    "cdf2003ac0d568eee28a0feffa7b07e0ef208e637de40665aa9a5b1e8bc2920e"
)
EXPECTED_DISCOVERY_BASE_IDS_SHA256 = (
    "42a1d85d4c42045ec4f633c7ca7462f5aa80b58318daee300b0ed1edec8e7407"
)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=str(REPO_ROOT / "data/processed/phase5_design"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "artifacts/phase5b_discovery_split"),
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split = compute_discovery_family_split(DISCOVERY_FAMILIES)
    train_set = set(split["train_families"])
    val_set = set(split["validation_families"])
    if train_set & val_set:
        raise SystemExit("train/validation family overlap")
    if train_set | val_set != set(DISCOVERY_FAMILIES):
        raise SystemExit("train+val must equal discovery families")
    if set(LOCKED_FAMILIES) & (train_set | val_set):
        raise SystemExit("locked family leaked into split")

    discovery_scenarios = _load_jsonl(data_dir / "discovery_base_scenarios.jsonl")
    if len(discovery_scenarios) != 960:
        raise SystemExit(f"expected 960 discovery bases, got {len(discovery_scenarios)}")
    if any(s.get("pool") != "discovery" for s in discovery_scenarios):
        raise SystemExit("non-discovery row in discovery_base_scenarios")
    if any(s["family"] in LOCKED_FAMILIES for s in discovery_scenarios):
        raise SystemExit("locked family in discovery scenarios")

    discovery_ids = [s["base_scenario_id"] for s in discovery_scenarios]
    if sha_sorted_ids(discovery_ids) != EXPECTED_DISCOVERY_BASE_IDS_SHA256:
        raise SystemExit("discovery base ID hash drift vs Phase 5A freeze")

    train_scenarios = [s for s in discovery_scenarios if s["family"] in train_set]
    val_scenarios = [s for s in discovery_scenarios if s["family"] in val_set]
    if len(train_scenarios) != 640 or len(val_scenarios) != 320:
        raise SystemExit(
            f"unexpected split sizes {len(train_scenarios)}/{len(val_scenarios)}"
        )

    # No scenario-level mixing: every scenario in a family stays in that family's split.
    for s in train_scenarios:
        if s["family"] not in train_set:
            raise SystemExit("train scenario family mismatch")
    for s in val_scenarios:
        if s["family"] not in val_set:
            raise SystemExit("validation scenario family mismatch")

    train_pair_ids = sorted(s["base_scenario_id"] for s in train_scenarios)
    val_pair_ids = sorted(s["base_scenario_id"] for s in val_scenarios)
    if set(train_pair_ids) & set(val_pair_ids):
        raise SystemExit("train/validation pair ID overlap")

    all_prompts = _load_jsonl(data_dir / "final_candidate_prompts.jsonl")
    if sha_prompt_texts(all_prompts) != EXPECTED_FINAL_PROMPT_TEXT_SHA256:
        raise SystemExit("final prompt text hash drift — prompts must remain unchanged")
    discovery_prompts = [
        r
        for r in all_prompts
        if r.get("pool") == "discovery" and r.get("split") == "final"
    ]
    if len(discovery_prompts) != 1920:
        raise SystemExit(f"expected 1920 discovery prompts, got {len(discovery_prompts)}")
    if any(r["family"] in LOCKED_FAMILIES for r in discovery_prompts):
        raise SystemExit("locked family in discovery prompts")
    if any(r.get("prompt_template_revision") != 2 for r in discovery_prompts):
        raise SystemExit("discovery prompts must be revision 2")

    train_prompts = [r for r in discovery_prompts if r["family"] in train_set]
    val_prompts = [r for r in discovery_prompts if r["family"] in val_set]
    if len(train_prompts) != 1280 or len(val_prompts) != 640:
        raise SystemExit(
            f"unexpected prompt split sizes {len(train_prompts)}/{len(val_prompts)}"
        )

    per_family_counts = {}
    for fam in DISCOVERY_FAMILIES:
        n_base = sum(1 for s in discovery_scenarios if s["family"] == fam)
        n_prompts = sum(1 for r in discovery_prompts if r["family"] == fam)
        role = "train" if fam in train_set else "validation"
        per_family_counts[fam] = {
            "role": role,
            "n_base_scenarios": n_base,
            "n_prompts": n_prompts,
            "n_designed_pairs": n_base,
            "family_digest_sha256": next(
                row["digest_sha256"]
                for row in split["ranked_by_digest"]
                if row["family"] == fam
            ),
        }

    family_split_path = out_dir / "family_split.json"
    primary_path = out_dir / "primary_all_pair_populations.json"
    procedure_path = out_dir / "future_probe_procedure.json"

    family_payload = {
        "created_at": utc_now_iso(),
        "phase": "phase5b",
        "pre_generation_freeze": True,
        "model_generation_performed": False,
        **split,
        "per_family": per_family_counts,
        "discovery_base_scenario_ids_sha256": EXPECTED_DISCOVERY_BASE_IDS_SHA256,
        "final_prompt_text_sha256": EXPECTED_FINAL_PROMPT_TEXT_SHA256,
        "discovery_prompt_text_sha256": sha_prompt_texts(discovery_prompts),
        "train_prompt_text_sha256": sha_prompt_texts(train_prompts),
        "validation_prompt_text_sha256": sha_prompt_texts(val_prompts),
        "activations_collected": False,
        "probes_fit_or_scored": False,
        "locked_families_run": False,
    }
    write_json(family_split_path, family_payload)

    primary_payload = {
        "created_at": utc_now_iso(),
        "phase": "phase5b",
        "estimand_note": (
            "Primary physiology analysis uses all designed S2/S3 pairs independent of "
            "later behavioral compliance (post-treatment selection bias avoidance)."
        ),
        "pair_id_definition": "base_scenario_id (matched S2+S3 prompts)",
        "train": {
            "families": split["train_families"],
            "n_pairs": len(train_pair_ids),
            "pair_ids": train_pair_ids,
            "pair_ids_sha256": sha_sorted_ids(train_pair_ids),
            "n_prompts": len(train_prompts),
        },
        "validation": {
            "families": split["validation_families"],
            "n_pairs": len(val_pair_ids),
            "pair_ids": val_pair_ids,
            "pair_ids_sha256": sha_sorted_ids(val_pair_ids),
            "n_prompts": len(val_prompts),
        },
        "per_family_pair_counts": {
            fam: per_family_counts[fam]["n_designed_pairs"] for fam in DISCOVERY_FAMILIES
        },
        "sensitivity_behavior_valid_pairs": "frozen_after_generation_only",
        "locked_family_ids_present": False,
        "activations_collected": False,
        "probes_fit_or_scored": False,
    }
    if primary_payload["train"]["n_pairs"] != 640:
        raise SystemExit("train all-pair N != 640")
    if primary_payload["validation"]["n_pairs"] != 320:
        raise SystemExit("validation all-pair N != 320")
    write_json(primary_path, primary_payload)

    procedure_payload = {
        "created_at": utc_now_iso(),
        "phase": "phase5b",
        "executed": False,
        "procedure": FUTURE_PROBE_PROCEDURE,
        "preregistered_baselines": list(PREREGISTERED_BASELINES),
        "baselines_fit": False,
        "information_theoretic_note": (
            "Because record and objective target are present in privileged context, "
            "Phase 5 does not claim the condition is absent from text. The scientific "
            "target is a cross-scenario internal representation of objective-record "
            "conflict / strategic deception."
        ),
        "prohibitions": [
            "no_activation_extraction",
            "no_probe_fitting",
            "no_probe_scoring",
            "no_layer_selection",
            "no_locked_family_model_calls",
            "no_causal_intervention",
            "no_prompt_revision",
        ],
    }
    write_json(procedure_path, procedure_payload)

    # Lightweight discovery prompt index for Modal (IDs only; texts loaded from design).
    index = {
        "created_at": utc_now_iso(),
        "n_discovery_prompts": 1920,
        "train_families": split["train_families"],
        "validation_families": split["validation_families"],
        "discovery_prompt_text_sha256": family_payload["discovery_prompt_text_sha256"],
        "example_ids": sorted(r["example_id"] for r in discovery_prompts),
    }
    write_json(out_dir / "discovery_generation_index.json", index)

    print(
        json.dumps(
            {
                "train_families": split["train_families"],
                "validation_families": split["validation_families"],
                "train_all_pair_n": 640,
                "train_all_pair_sha256": primary_payload["train"]["pair_ids_sha256"],
                "validation_all_pair_n": 320,
                "validation_all_pair_sha256": primary_payload["validation"][
                    "pair_ids_sha256"
                ],
                "discovery_prompt_text_sha256": family_payload[
                    "discovery_prompt_text_sha256"
                ],
                "out_dir": str(out_dir.relative_to(REPO_ROOT)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
