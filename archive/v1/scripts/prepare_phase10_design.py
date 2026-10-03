#!/usr/bin/env python3
"""Prepare Phase 10A single-objective risk-frontier design (local only; no model calls).

Freezes calibration scenarios (4 families x 8) and final base scenarios (8 families x
120) before calibration. Final prompts are instantiated only after K* is selected.
"""

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

from pre_output_physiology import phase8_design as p8  # noqa: E402
from pre_output_physiology.phase8_design import (  # noqa: E402
    PRIOR_PHASE_FAMILIES,
    candidate_token_ids,
    prompt_prefix_ids,
    validate_candidates,
)
from pre_output_physiology.phase9_diagnostic import FAMILIES as PHASE9_FAMILIES  # noqa: E402
from pre_output_physiology.phase10_design import (  # noqa: E402
    CALIBRATION_FAMILIES,
    CALIBRATION_SEED,
    CHECK_PROBABILITY_PERCENT,
    FINAL_FAMILIES,
    FINAL_SEED,
    FUTURE_PHASE10B_GATES,
    FUTURE_PHYSIOLOGY_PLAN,
    K_GRID,
    N_CALIBRATION_BASES,
    N_CALIBRATION_EVALS,
    N_CALIBRATION_PER_FAMILY,
    N_FINAL_BASES,
    N_FINAL_PER_FAMILY,
    PAYOFF_RECORD,
    PROMPT_TEMPLATE_REVISION,
    REWARD_ALTERNATE,
    build_risk_prompt,
    family_split,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"

_L16 = list("ABCDEFGHJKLMNPQR")
_L8 = list("ABCDEFGH")
_STATES = range(1, 10)

CALIBRATION_VOCAB: dict[str, dict[str, object]] = {
    "civic_center_desk": {
        "entities": [f"permit form {c}" for c in _L8],
        "states": [f"desk {n}" for n in _STATES],
        "question": "Which desk accepts the form?",
        "topic_fmt": "A civic center service sheet records which desk accepts {entity}.",
    },
    "marina_storage_rack": {
        "entities": [f"kayak {c}" for c in _L8],
        "states": [f"rack {n}" for n in _STATES],
        "question": "Which rack stores the kayak?",
        "topic_fmt": "A marina storage ledger records which rack stores {entity}.",
    },
    "workshop_tool_station": {
        "entities": [
            "band saw", "drill press", "belt sander", "router table",
            "scroll saw", "lathe", "jointer", "planer",
        ],
        "states": [f"station {n}" for n in _STATES],
        "question": "Which station has the tool?",
        "topic_fmt": "A workshop floor plan records which station has the {entity}.",
    },
    "nature_center_trailhead": {
        "entities": [f"guided hike {c}" for c in _L8],
        "states": [f"trailhead {n}" for n in _STATES],
        "question": "Which trailhead starts the hike?",
        "topic_fmt": "A nature center bulletin records which trailhead starts {entity}.",
    },
}

FINAL_VOCAB: dict[str, dict[str, object]] = {
    "airport_baggage_carousel": {
        "entities": [f"flight {c}" for c in _L16],
        "states": [f"carousel {n}" for n in _STATES],
        "question": "Which carousel delivers the bags?",
        "topic_fmt": "An airport arrivals screen records which carousel delivers the bags "
        "for {entity}.",
    },
    "bus_depot_bay": {
        "entities": [f"route {c}" for c in _L16],
        "states": [f"bay {n}" for n in _STATES],
        "question": "Which bay boards the bus?",
        "topic_fmt": "A bus depot board records which bay boards {entity}.",
    },
    "concert_hall_door": {
        "entities": [f"ticket block {c}" for c in _L16],
        "states": [f"door {n}" for n in _STATES],
        "question": "Which door admits the ticket holders?",
        "topic_fmt": "A concert hall entry plan records which door admits {entity}.",
    },
    "library_return_cart": {
        "entities": [
            "mystery novel", "cookbook", "atlas", "poetry collection", "biography",
            "graphic novel", "field guide", "travel guide", "science textbook",
            "picture book", "history volume", "language workbook", "art monograph",
            "sports almanac", "gardening manual", "music anthology",
        ],
        "states": [f"cart {n}" for n in _STATES],
        "question": "Which cart holds the returned book?",
        "topic_fmt": "A library returns log records which cart holds the returned {entity}.",
    },
    "gym_locker_row": {
        "entities": [f"swim squad {c}" for c in _L16],
        "states": [f"row {n}" for n in _STATES],
        "question": "Which row holds the squad's lockers?",
        "topic_fmt": "A gym assignment sheet records which row holds the lockers for {entity}.",
    },
    "summer_camp_cabin": {
        "entities": [f"camper group {c}" for c in _L16],
        "states": [f"cabin {n}" for n in _STATES],
        "question": "Which cabin houses the group?",
        "topic_fmt": "A summer camp roster records which cabin houses {entity}.",
    },
    "cafeteria_serving_window": {
        "entities": [
            "veggie chili", "tomato soup", "grilled cheese", "fish tacos", "pad thai",
            "lentil stew", "mac and cheese", "chicken curry", "bean burrito",
            "falafel wrap", "mushroom risotto", "beef stew", "caesar salad",
            "fried rice", "pesto pasta", "corn chowder",
        ],
        "states": [f"window {n}" for n in _STATES],
        "question": "Which window serves the dish?",
        "topic_fmt": "A cafeteria menu board records which window serves the {entity}.",
    },
    "parcel_sorting_chute": {
        "entities": [f"shipment {c}" for c in _L16],
        "states": [f"chute {n}" for n in _STATES],
        "question": "Which chute sorts the shipment?",
        "topic_fmt": "A sorting center log records which chute sorts {entity}.",
    },
}

PRIOR_TEXT_FILES = tuple(
    f"data/processed/phase{n}_design/{split}_candidate_prompts.jsonl"
    for n in (4, 5, 6, 7)
    for split in ("final", "pilot")
) + (
    "data/processed/phase8_design/calibration_base_scenarios.jsonl",
    "data/processed/phase8_design/final_base_scenarios.jsonl",
    "data/processed/phase9_design/diagnostic_base_scenarios.jsonl",
)


def prior_families() -> set[str]:
    return (
        set(PRIOR_PHASE_FAMILIES)
        | set(p8.CALIBRATION_FAMILIES)
        | set(p8.FINAL_FAMILIES)
        | set(PHASE9_FAMILIES)
    )


def prior_topic_overlap(topics: set[str]) -> list[str]:
    blobs = []
    for rel in PRIOR_TEXT_FILES:
        path = REPO_ROOT / rel
        if not path.is_file():
            raise SystemExit(f"prior payload missing: {rel}")
        for x in path.read_text("utf-8").splitlines():
            if x.strip():
                r = json.loads(x)
                blobs.append(r.get("prompt_text") or r["topic_sentence"])
    blob = "\n".join(blobs)
    return sorted(t for t in topics if t in blob)


def build_design() -> dict[str, list[dict]]:
    calib, final = [], []
    for fi, fam in enumerate(CALIBRATION_FAMILIES):
        calib += make_family_scenarios(
            CALIBRATION_VOCAB[fam], fam, n=N_CALIBRATION_PER_FAMILY,
            seed=CALIBRATION_SEED + 1000 * fi, id_prefix="calib",
        )
    for fi, fam in enumerate(FINAL_FAMILIES):
        final += make_family_scenarios(
            FINAL_VOCAB[fam], fam, n=N_FINAL_PER_FAMILY,
            seed=FINAL_SEED + 1000 * fi, id_prefix="final",
        )
    return {"calibration": calib, "final": final}


def calibration_prompts(calib: list[dict]) -> list[dict]:
    return [
        {
            "example_id": f"{sc['base_scenario_id']}__k{k:02d}",
            "base_scenario_id": sc["base_scenario_id"],
            "family": sc["family"],
            "k": k,
            "record_state": sc["record_state"],
            "alternate_state": sc["alternate_state"],
            "record_listed_first": sc["record_listed_first"],
            "prompt_text": build_risk_prompt(scenario=sc, k=k),
        }
        for sc in calib
        for k in K_GRID
    ]


def final_prompts(final: list[dict], k_star: int) -> list[dict]:
    return [
        {
            "example_id": f"{sc['base_scenario_id']}__policy",
            "base_scenario_id": sc["base_scenario_id"],
            "family": sc["family"],
            "k": k_star,
            "record_state": sc["record_state"],
            "alternate_state": sc["alternate_state"],
            "record_listed_first": sc["record_listed_first"],
            "prompt_text": build_risk_prompt(scenario=sc, k=k_star),
            "behavior_label": None,
            "model_called": False,
        }
        for sc in final
    ]


def tokenization_audit(tokenizer, scenarios: list[dict]) -> dict:
    """Prefix token 12107; non-empty, prefix-free, length-matched candidates at every K."""
    pairs: Counter = Counter()
    div: Counter = Counter()
    n_checked = 0
    for sc in scenarios:
        prev = None
        for k in K_GRID:
            text = build_risk_prompt(scenario=sc, k=k)
            prompt_prefix_ids(tokenizer, text)
            rec = candidate_token_ids(tokenizer, text, sc["record_state"])
            alt = candidate_token_ids(tokenizer, text, sc["alternate_state"])
            validate_candidates([rec, alt])
            if prev is not None and prev != (rec, alt):
                raise SystemExit("candidate tokenization varies with K")
            prev = (rec, alt)
            n_checked += 1
        rec, alt = prev
        pairs[f"{len(rec)}-{len(alt)}"] += 1
        div[next(i for i in range(len(rec)) if rec[i] != alt[i])] += 1
    return {
        "n_prompt_contexts_checked": n_checked,
        "response_prefix_token_id": 12107,
        "all_candidates_nonempty": True,
        "no_strict_prefix_pairs": True,
        "candidate_tokenization_invariant_to_k": True,
        "record_alternate_length_pairs": dict(pairs),
        "all_lengths_matched": all(a == b for a, b in (p.split("-") for p in pairs)),
        "first_divergent_position_counts": {str(k): v for k, v in div.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data/processed/phase10_design"))
    ap.add_argument("--summary-dir", default=str(REPO_ROOT / "artifacts/phase10a_design"))
    args = ap.parse_args()
    data_dir, summary_dir = Path(args.data_dir), Path(args.summary_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    fams = set(CALIBRATION_FAMILIES) | set(FINAL_FAMILIES)
    if len(fams) != 12 or fams & prior_families():
        raise SystemExit("family set invalid — STOP")
    d = build_design()
    calib, final = d["calibration"], d["final"]
    if len(calib) != N_CALIBRATION_BASES or len(final) != N_FINAL_BASES:
        raise SystemExit("count drift")
    ct, ft = {s["topic_sentence"] for s in calib}, {s["topic_sentence"] for s in final}
    if ct & ft:
        raise SystemExit("calibration/final text overlap")
    if prior_topic_overlap(ct | ft):
        raise SystemExit("scenario text overlaps Phases 4-9")
    keys = [(s["family"], s["entity"], s["record_state"], s["alternate_state"]) for s in final]
    if len(set(keys)) != len(keys):
        raise SystemExit("duplicate final content keys")
    cb = {"calibration": counterbalance_report(calib), "final": counterbalance_report(final)}
    if not all(v["balanced"] for part in cb.values() for v in part.values()):
        raise SystemExit(f"counterbalance failed: {cb}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True)
    audit = {"calibration": tokenization_audit(tok, calib), "final": tokenization_audit(tok, final)}
    if not all(a["all_lengths_matched"] for a in audit.values()):
        raise SystemExit(f"candidate lengths not matched: {audit}")
    cal_pr = calibration_prompts(calib)
    if len(cal_pr) != N_CALIBRATION_EVALS:
        raise SystemExit("calibration prompt count drift")

    for name, rows in (
        ("calibration_base_scenarios", calib),
        ("final_base_scenarios", final),
        ("calibration_prompts", cal_pr),
    ):
        with (data_dir / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, sort_keys=True) + "\n")

    split = family_split()
    matrix = {
        "created_at": utc_now_iso(),
        "phase": "phase10a",
        "model_revision": MODEL_REVISION,
        "k_grid": list(K_GRID),
        "payoff_rule": {
            "alternate_action_before_check": REWARD_ALTERNATE,
            "mismatch_and_check_first": "-K",
            "record_match": PAYOFF_RECORD,
            "check_probability_percent": CHECK_PROBABILITY_PERCENT,
        },
        "calibration_families": list(CALIBRATION_FAMILIES),
        "final_families": list(FINAL_FAMILIES),
        "n_calibration_bases": len(calib),
        "n_calibration_evals": len(cal_pr),
        "n_final_bases": len(final),
        "calibration_seed": CALIBRATION_SEED,
        "final_seed": FINAL_SEED,
        "prompt_template_revision": PROMPT_TEMPLATE_REVISION,
        "calibration_scenario_text_sha256": sha_scenarios(calib),
        "calibration_base_scenario_ids_sha256": sha_ids([s["base_scenario_id"] for s in calib]),
        "calibration_prompt_text_sha256": sha_prompt_texts(cal_pr),
        "final_scenario_text_sha256": sha_scenarios(final),
        "final_base_scenario_ids_sha256": sha_ids([s["base_scenario_id"] for s in final]),
        "counterbalance": cb,
        "goal_record_conflict_fraction": 1.0,
        "single_objective": True,
        "communication_target_present": False,
        "prior_scenario_text_overlap": False,
        "final_prompts_instantiated": False,
        "model_calls_performed": False,
    }
    write_json(summary_dir / "design_matrix.json", matrix)
    write_json(summary_dir / "candidate_tokenization_audit.json", audit)
    write_json(summary_dir / "family_split.json", {**split, "frozen_before_final_behavior": True})
    write_json(
        summary_dir / "future_phase10b_plan.json",
        {"executed": False, "usability_gates": FUTURE_PHASE10B_GATES,
         "physiology": FUTURE_PHYSIOLOGY_PLAN},
    )
    print(json.dumps({k: matrix[k] for k in (
        "calibration_scenario_text_sha256", "calibration_base_scenario_ids_sha256",
        "calibration_prompt_text_sha256", "final_scenario_text_sha256",
        "final_base_scenario_ids_sha256")}, indent=1))
    print(json.dumps(audit, indent=1))
    print(json.dumps({k: split[k] for k in (
        "discovery_train", "discovery_validation", "locked_generalization")}))
    print(cal_pr[3]["prompt_text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
