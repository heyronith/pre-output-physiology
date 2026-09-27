#!/usr/bin/env python3
"""Freeze Phase 19A unseen validation family screen design before model calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from prepare_phase8_design import (  # noqa: E402
    counterbalance_report,
    make_family_scenarios,
    sha_ids,
    sha_prompt_texts,
    sha_scenarios,
)
from prepare_phase14_design import (  # noqa: E402
    prior_families,
    prior_topic_overlap,
    tokenization_audit,
)

from pre_output_physiology.phase14_design import (  # noqa: E402
    FIXED_K,
    build_trajectory_prompt,
)
from pre_output_physiology.phase15_onset import (  # noqa: E402
    RULE_HASH,
    RULE_MANIFEST,
    RULE_VERSION,
)
from pre_output_physiology.phase19_screen import (  # noqa: E402
    FAMILY_VOCAB,
    ID_PREFIX,
    N_PER_FAMILY,
    N_S1_CONTINUATIONS,
    N_SCREEN_PROMPTS,
    NEW_FAMILIES,
    PHASE15_RULE_HASH,
    PHASE18_STATUS_REQUIRED,
    SCENARIO_SEED,
    TEMPERATURE,
    THRESHOLD_HASH,
    THRESHOLD_MANIFEST,
    TOP_P,
    build_s1_schedule,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"


def build_scenarios() -> list[dict]:
    rows = []
    for fi, fam in enumerate(NEW_FAMILIES):
        rows += make_family_scenarios(
            FAMILY_VOCAB[fam],
            fam,
            n=N_PER_FAMILY,
            seed=SCENARIO_SEED + 1000 * fi,
            id_prefix=ID_PREFIX,
        )
    return rows


def build_prompts(scenarios: list[dict]) -> list[dict]:
    return [
        {
            "prompt_group_id": sc["base_scenario_id"],
            "example_id": sc["base_scenario_id"],
            "base_scenario_id": sc["base_scenario_id"],
            "family": sc["family"],
            "k": FIXED_K,
            "record_state": sc["record_state"],
            "alternate_state": sc["alternate_state"],
            "record_listed_first": sc["record_listed_first"],
            "prompt_text": build_trajectory_prompt(scenario=sc, k=FIXED_K),
            "behavior_label": None,
            "model_called": False,
            "phase": "phase19a",
        }
        for sc in scenarios
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        default=str(REPO_ROOT / "data/processed/phase19_design"),
    )
    ap.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "artifacts/phase19a_design"),
    )
    args = ap.parse_args()
    data = Path(args.data_dir)
    data.mkdir(parents=True, exist_ok=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if RULE_VERSION != "phase15_onset_v1" or RULE_HASH != RULE_MANIFEST["rule_hash"]:
        raise SystemExit("Phase-15 rule drift")
    if RULE_HASH != PHASE15_RULE_HASH:
        raise SystemExit("Phase-15 rule hash mismatch")
    if RULE_HASH != "6bb40fa50eaf07b52e4709355bdb4652060658c2e73d3e41775b8bccac4bc24f":
        raise SystemExit("Phase-15 rule hash mismatch vs Phase-19 spec")

    # Phase 18 remains validation-gate HOLD.
    p18 = json.loads(
        (
            REPO_ROOT / "artifacts/phase18a_confirmation/confirmation_summary.json"
        ).read_text("utf-8")
    )
    if p18.get("status") != PHASE18_STATUS_REQUIRED:
        raise SystemExit(f"Phase 18 must remain HOLD; got {p18.get('status')}")

    fams = set(NEW_FAMILIES)
    if len(fams) != 6:
        raise SystemExit("expected 6 new families")
    overlap_f = fams & prior_families()
    if overlap_f:
        raise SystemExit(f"families overlap prior phases: {sorted(overlap_f)}")

    scenarios = build_scenarios()
    if len(scenarios) != N_SCREEN_PROMPTS:
        raise SystemExit("scenario count")
    topics = {s["topic_sentence"] for s in scenarios}
    topic_overlap = prior_topic_overlap(topics)
    if topic_overlap:
        raise SystemExit(f"topic overlap prior: {topic_overlap[:5]}")
    cb = counterbalance_report(scenarios)
    if not all(v["balanced"] for v in cb.values()):
        raise SystemExit(f"counterbalance failed: {cb}")

    prompts = build_prompts(scenarios)
    if len(prompts) != N_SCREEN_PROMPTS:
        raise SystemExit("prompt count")
    for fam in NEW_FAMILIES:
        n = sum(1 for p in prompts if p["family"] == fam)
        if n != N_PER_FAMILY:
            raise SystemExit(f"{fam} has {n} != {N_PER_FAMILY}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True)
    audit = tokenization_audit(tok, prompts)

    s1 = build_s1_schedule(prompts)
    if len(s1) != N_S1_CONTINUATIONS:
        raise SystemExit("S1 schedule size")
    seeds = [r["sample_seed"] for r in s1]
    if min(seeds) < 19_000_000 or max(seeds) >= 19_100_000:
        raise SystemExit("S1 seed range")

    (data / "screen_base_scenarios.jsonl").write_text(
        "".join(json.dumps(s, sort_keys=True) + "\n" for s in scenarios), "utf-8"
    )
    (data / "screen_prompts.jsonl").write_text(
        "".join(json.dumps(p, sort_keys=True) + "\n" for p in prompts), "utf-8"
    )
    (data / "s1_sampling_schedule.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in s1), "utf-8"
    )

    matrix = {
        "created_at": utc_now_iso(),
        "phase": "phase19a",
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "n_screen_prompts": N_SCREEN_PROMPTS,
        "n_per_family": N_PER_FAMILY,
        "n_s1_continuations": N_S1_CONTINUATIONS,
        "new_families": list(NEW_FAMILIES),
        "scenario_seed": SCENARIO_SEED,
        "phase15_rule_version": RULE_VERSION,
        "phase15_rule_hash": RULE_HASH,
        "threshold_hash": THRESHOLD_HASH,
        "phase18_status": PHASE18_STATUS_REQUIRED,
        "screen_scenario_text_sha256": sha_scenarios(scenarios),
        "screen_prompt_group_ids_sha256": sha_ids(
            [p["prompt_group_id"] for p in prompts]
        ),
        "screen_prompt_text_sha256": sha_prompt_texts(prompts),
        "s1_schedule_sha256": sha_ids([r["continuation_id"] for r in s1]),
        "tokenization_audit": audit,
        "counterbalance": cb,
        "model_calls_performed": False,
        "s2_schedule_frozen": False,
        "phase18_train_prompts_untouched": True,
        "activations_authorized": False,
    }
    write_json(out / "design_matrix.json", matrix)
    write_json(out / "threshold_manifest.json", THRESHOLD_MANIFEST)
    write_json(
        out / "new_families.json",
        {"families": list(NEW_FAMILIES), "vocab": FAMILY_VOCAB},
    )
    print(
        json.dumps(
            {
                "screen_prompt_group_ids_sha256": matrix[
                    "screen_prompt_group_ids_sha256"
                ],
                "screen_prompt_text_sha256": matrix["screen_prompt_text_sha256"],
                "s1_schedule_sha256": matrix["s1_schedule_sha256"],
                "threshold_hash": matrix["threshold_hash"],
                "n_s1_continuations": matrix["n_s1_continuations"],
                "new_families": matrix["new_families"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
