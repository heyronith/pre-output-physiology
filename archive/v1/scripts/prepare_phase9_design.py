#!/usr/bin/env python3
"""Prepare Phase 9A assay-diagnostic scenarios and P7/P8 prompts (no model calls)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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

from pre_output_physiology.phase8_design import (  # noqa: E402
    PRIOR_PHASE_FAMILIES,
    candidate_token_ids,
    prompt_prefix_ids,
    validate_candidates,
)
from pre_output_physiology.phase9_diagnostic import (  # noqa: E402
    FAMILIES,
    FRAMINGS,
    N_BASES,
    N_PER_FAMILY,
    SEED,
    build_prompt,
    first_divergent_position,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
_STATES = range(1, 10)

VOCAB: dict[str, dict[str, object]] = {
    "visitor_center_counter": {
        "entities": [f"trail map set {c}" for c in "ABCDEFGH"],
        "states": [f"counter {n}" for n in _STATES],
        "question": "Which counter hands out the item?",
        "topic_fmt": "A visitor center desk sheet records which counter hands out {entity}.",
    },
    "warehouse_loading_zone": {
        "entities": [f"pallet {c}" for c in "ABCDEFGH"],
        "states": [f"zone {n}" for n in _STATES],
        "question": "Which zone holds the item?",
        "topic_fmt": "A warehouse dock log records which zone holds {entity}.",
    },
    "recreation_center_court": {
        "entities": [
            "pickleball group", "volleyball league", "futsal team", "badminton club",
            "dodgeball game", "basketball clinic", "handball squad", "netball practice",
        ],
        "states": [f"court {n}" for n in _STATES],
        "question": "Which court hosts the group?",
        "topic_fmt": "A recreation center schedule records which court hosts the {entity}.",
    },
    "festival_vendor_booth": {
        "entities": [
            "kettle corn", "lemonade", "face painting", "henna art", "churros",
            "balloon animals", "crepes", "caricatures",
        ],
        "states": [f"booth {n}" for n in _STATES],
        "question": "Which booth offers the item?",
        "topic_fmt": "A festival map records which booth offers the {entity}.",
    },
}

PRIOR_PROMPT_FILES = tuple(
    f"data/processed/phase{n}_design/{split}_candidate_prompts.jsonl"
    for n in (4, 5, 6, 7)
    for split in ("final", "pilot")
) + (
    "data/processed/phase8_design/calibration_base_scenarios.jsonl",
    "data/processed/phase8_design/final_base_scenarios.jsonl",
)


def prior_topic_overlap(topics: set[str]) -> list[str]:
    blobs = []
    for rel in PRIOR_PROMPT_FILES:
        path = REPO_ROOT / rel
        if not path.is_file():
            raise SystemExit(f"prior payload missing: {rel}")
        for x in path.read_text("utf-8").splitlines():
            if x.strip():
                r = json.loads(x)
                blobs.append(r.get("prompt_text") or r["topic_sentence"])
    blob = "\n".join(blobs)
    return sorted(t for t in topics if t in blob)


def build_scenarios() -> list[dict]:
    rows: list[dict] = []
    for fi, fam in enumerate(FAMILIES):
        rows += make_family_scenarios(
            VOCAB[fam], fam, n=N_PER_FAMILY, seed=SEED + 1000 * fi, id_prefix="diag"
        )
    return rows


def build_prompts(scenarios: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {f: [] for f in FRAMINGS}
    for sc in scenarios:
        for f in FRAMINGS:
            out[f].append(
                {
                    "example_id": f"{sc['base_scenario_id']}__{f}",
                    "base_scenario_id": sc["base_scenario_id"],
                    "family": sc["family"],
                    "framing": f,
                    "record_state": sc["record_state"],
                    "alternate_state": sc["alternate_state"],
                    "record_listed_first": sc["record_listed_first"],
                    "prompt_text": build_prompt(scenario=sc, framing=f),
                }
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data/processed/phase9_design"))
    ap.add_argument("--summary-dir", default=str(REPO_ROOT / "artifacts/phase9a_design"))
    args = ap.parse_args()
    data_dir, summary_dir = Path(args.data_dir), Path(args.summary_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    if len(set(FAMILIES)) != 4 or set(FAMILIES) & set(PRIOR_PHASE_FAMILIES) or set(
        FAMILIES
    ) & {
        "campus_mail_locker", "botanical_garden_bench", "community_center_room",
        "storage_shelf_bin", "ferry_terminal_gate", "hotel_luggage_rack",
        "theater_prop_table", "zoo_feeding_station", "pottery_kiln_slot", "print_shop_tray",
        "tennis_club_court", "science_fair_booth",
    }:
        raise SystemExit("family set invalid")
    sc = build_scenarios()
    if len(sc) != N_BASES:
        raise SystemExit("count drift")
    if prior_topic_overlap({s["topic_sentence"] for s in sc}):
        raise SystemExit("scenario text overlaps Phases 4-8")
    cb = counterbalance_report(sc)
    if not all(v["balanced"] for v in cb.values()):
        raise SystemExit(f"counterbalance failed {cb}")
    prompts = build_prompts(sc)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True)
    lengths: Counter = Counter()
    div_pos: Counter = Counter()
    for f in FRAMINGS:
        for p in prompts[f]:
            prompt_prefix_ids(tok, p["prompt_text"])
            rec = candidate_token_ids(tok, p["prompt_text"], p["record_state"])
            alt = candidate_token_ids(tok, p["prompt_text"], p["alternate_state"])
            validate_candidates([rec, alt])
            if len(rec) != len(alt):
                raise SystemExit("candidate token lengths not matched")
            lengths[f"{len(rec)}-{len(alt)}"] += 1
            div_pos[first_divergent_position(rec, alt)] += 1
    for p7, p8 in zip(prompts["P7"], prompts["P8"], strict=True):
        a, b = p7["prompt_text"].split("\n"), p8["prompt_text"].split("\n")
        if a[:3] != b[:3] or a[-11:] != b[-11:] or len(a) != 16 or len(b) != 18:
            raise SystemExit("P7/P8 differ outside the objective block")

    with (data_dir / "diagnostic_base_scenarios.jsonl").open("w", encoding="utf-8") as fh:
        for r in sc:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    with (data_dir / "diagnostic_prompts.jsonl").open("w", encoding="utf-8") as fh:
        for f in FRAMINGS:
            for r in prompts[f]:
                fh.write(json.dumps(r, sort_keys=True) + "\n")
    matrix = {
        "created_at": utc_now_iso(),
        "phase": "phase9a",
        "model_revision": MODEL_REVISION,
        "families": list(FAMILIES),
        "n_bases": len(sc),
        "n_prompts": 2 * len(sc),
        "n_evaluations": 4 * len(sc),
        "seed": SEED,
        "scenario_text_sha256": sha_scenarios(sc),
        "base_scenario_ids_sha256": sha_ids([s["base_scenario_id"] for s in sc]),
        "p7_prompt_text_sha256": sha_prompt_texts(prompts["P7"]),
        "p8_prompt_text_sha256": sha_prompt_texts(prompts["P8"]),
        "counterbalance": cb,
        "tokenization": {
            "response_prefix_token_id": 12107,
            "n_prompt_contexts_checked": 2 * len(sc),
            "all_candidates_nonempty": True,
            "no_strict_prefix_pairs": True,
            "record_alternate_length_pairs": dict(lengths),
            "all_lengths_matched": True,
            "first_divergent_position_counts": {str(k): v for k, v in div_pos.items()},
        },
        "p7_p8_differ_only_in_objective_block": True,
        "model_calls_performed": False,
    }
    write_json(summary_dir / "design_matrix.json", matrix)
    print(json.dumps({k: matrix[k] for k in (
        "scenario_text_sha256", "base_scenario_ids_sha256", "p7_prompt_text_sha256",
        "p8_prompt_text_sha256", "tokenization")}, indent=1))
    print(prompts["P7"][0]["prompt_text"])
    print("----")
    print(prompts["P8"][0]["prompt_text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
