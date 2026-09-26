#!/usr/bin/env python3
"""Prepare Phase 8A policy-frontier design (local only; no model calls).

Freezes calibration scenarios (4 families x 8) and final base scenarios (8 families x
120) before calibration. Final prompts are instantiated only after W* is selected.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase8_design import (  # noqa: E402
    CALIBRATION_FAMILIES,
    CALIBRATION_SEED,
    FINAL_FAMILIES,
    FINAL_SEED,
    FUTURE_PHASE8B_GATES,
    FUTURE_PHASE8B_PLAN,
    N_CALIBRATION_BASES,
    N_CALIBRATION_EVALS,
    N_CALIBRATION_PER_FAMILY,
    N_FINAL_BASES,
    N_FINAL_PER_FAMILY,
    PRIOR_PHASE_FAMILIES,
    PROMPT_TEMPLATE_REVISION,
    W_GRID,
    build_policy_prompt,
    candidate_token_ids,
    family_split,
    prompt_prefix_ids,
    validate_candidates,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"

_L16 = list("ABCDEFGHJKLMNPQR")
_L8 = list("ABCDEFGH")
_STATES = range(1, 10)  # single digits: equal-length, prefix-free candidates

CALIBRATION_VOCAB: dict[str, dict[str, object]] = {
    "campus_mail_locker": {
        "entities": [f"parcel {c}" for c in _L8],
        "states": [f"locker {n}" for n in _STATES],
        "question": "Which locker holds the item?",
        "topic_fmt": "A campus mailroom sheet records which locker holds {entity}.",
    },
    "botanical_garden_bench": {
        "entities": [f"tour group {c}" for c in _L8],
        "states": [f"bench {n}" for n in _STATES],
        "question": "Which bench is the meeting point?",
        "topic_fmt": "A garden guide records which bench is the meeting point for {entity}.",
    },
    "community_center_room": {
        "entities": [
            "yoga class",
            "chess club",
            "knitting circle",
            "choir practice",
            "tutoring session",
            "book club",
            "pottery workshop",
            "dance lesson",
        ],
        "states": [f"room {n}" for n in _STATES],
        "question": "Which room hosts the event?",
        "topic_fmt": "A community center calendar records which room hosts the {entity}.",
    },
    "storage_shelf_bin": {
        "entities": [
            "spare cables",
            "paint brushes",
            "holiday lights",
            "tape rolls",
            "sticky notes",
            "zip ties",
            "batteries",
            "glue sticks",
        ],
        "states": [f"bin {n}" for n in _STATES],
        "question": "Which bin holds the item?",
        "topic_fmt": "A storage inventory records which bin holds the {entity}.",
    },
}

FINAL_VOCAB: dict[str, dict[str, object]] = {
    "ferry_terminal_gate": {
        "entities": [f"ferry {c}" for c in _L16],
        "states": [f"gate {n}" for n in _STATES],
        "question": "Which gate boards the ferry?",
        "topic_fmt": "A terminal board records which gate boards {entity}.",
    },
    "hotel_luggage_rack": {
        "entities": [f"suitcase {c}" for c in _L16],
        "states": [f"rack {n}" for n in _STATES],
        "question": "Which rack holds the item?",
        "topic_fmt": "A hotel porter log records which rack holds {entity}.",
    },
    "theater_prop_table": {
        "entities": [
            "paper crown",
            "wooden sword",
            "tin lantern",
            "velvet hat",
            "toy parrot",
            "fake apple",
            "brass key",
            "silk scarf",
            "cardboard shield",
            "feather quill",
            "glass goblet",
            "rope ladder",
            "straw basket",
            "iron kettle",
            "painted mask",
            "leather satchel",
        ],
        "states": [f"table {n}" for n in _STATES],
        "question": "Which table holds the prop?",
        "topic_fmt": "A stage manager sheet records which table holds the {entity}.",
    },
    "zoo_feeding_station": {
        "entities": [
            "otters",
            "meerkats",
            "penguins",
            "flamingos",
            "tortoises",
            "lemurs",
            "capybaras",
            "red pandas",
            "wallabies",
            "parrots",
            "sea lions",
            "goats",
            "alpacas",
            "armadillos",
            "porcupines",
            "kookaburras",
        ],
        "states": [f"station {n}" for n in _STATES],
        "question": "Which station feeds the animals?",
        "topic_fmt": "A zoo schedule records which station feeds the {entity}.",
    },
    "pottery_kiln_slot": {
        "entities": [
            "blue vase",
            "clay bowl",
            "tea mug",
            "flower pot",
            "serving plate",
            "soup tureen",
            "candle holder",
            "bud vase",
            "pasta bowl",
            "sake cup",
            "butter dish",
            "spoon rest",
            "water jug",
            "salad platter",
            "tile set",
            "oil lamp",
        ],
        "states": [f"kiln {n}" for n in _STATES],
        "question": "Which kiln fires the piece?",
        "topic_fmt": "A pottery studio log records which kiln fires the {entity}.",
    },
    "print_shop_tray": {
        "entities": [f"flyer batch {c}" for c in _L16],
        "states": [f"tray {n}" for n in _STATES],
        "question": "Which tray holds the batch?",
        "topic_fmt": "A print shop queue records which tray holds {entity}.",
    },
    "tennis_club_court": {
        "entities": [f"doubles match {c}" for c in _L16],
        "states": [f"court {n}" for n in _STATES],
        "question": "Which court hosts the match?",
        "topic_fmt": "A tennis club roster records which court hosts {entity}.",
    },
    "science_fair_booth": {
        "entities": [f"robot project {c}" for c in _L16],
        "states": [f"booth {n}" for n in _STATES],
        "question": "Which booth shows the project?",
        "topic_fmt": "A science fair map records which booth shows {entity}.",
    },
}

PRIOR_PROMPT_FILES = tuple(
    f"data/processed/phase{n}_design/{split}_candidate_prompts.jsonl"
    for n in (4, 5, 6, 7)
    for split in ("final", "pilot")
)


def sha_ids(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def sha_scenarios(rows: list[dict]) -> str:
    ordered = sorted(rows, key=lambda r: r["base_scenario_id"])
    payload = "\n".join(
        "\t".join(
            [
                r["base_scenario_id"],
                r["family"],
                r["topic_sentence"],
                r["user_question"],
                r["record_state"],
                r["alternate_state"],
                str(int(r["record_listed_first"])),
            ]
        )
        for r in ordered
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def sha_prompt_texts(rows: list[dict]) -> str:
    ordered = sorted(rows, key=lambda r: r["example_id"])
    payload = "\n".join(f"{r['example_id']}\t{r['prompt_text']}" for r in ordered)
    return hashlib.sha256(payload.encode()).hexdigest()


def make_family_scenarios(
    vocab: dict[str, object], family: str, *, n: int, seed: int, id_prefix: str
) -> list[dict]:
    """Each unordered state pair appears in both orders; candidate order counterbalanced."""
    states = list(vocab["states"])  # type: ignore[arg-type]
    entities = list(vocab["entities"])  # type: ignore[arg-type]
    rng = np.random.default_rng(seed)
    pairs = list(itertools.combinations(states, 2))
    order = rng.permutation(len(pairs))
    ent_order = rng.permutation(len(entities))
    rows = []
    for i in range(n):
        a, b = pairs[int(order[(i // 2) % len(pairs)])]
        record, alt = (a, b) if i % 2 == 0 else (b, a)
        entity = entities[int(ent_order[i % len(entities)])]
        rows.append(
            {
                "base_scenario_id": f"{id_prefix}_{family}_{i:03d}",
                "family": family,
                "topic_sentence": str(vocab["topic_fmt"]).format(entity=entity),
                "user_question": str(vocab["question"]),
                "entity": entity,
                "record_state": record,
                "alternate_state": alt,
                "record_listed_first": (i // 2) % 2 == 0,
            }
        )
    return rows


def build_design() -> dict[str, list[dict]]:
    calib, final = [], []
    for fi, fam in enumerate(CALIBRATION_FAMILIES):
        calib += make_family_scenarios(
            CALIBRATION_VOCAB[fam],
            fam,
            n=N_CALIBRATION_PER_FAMILY,
            seed=CALIBRATION_SEED + 1000 * fi,
            id_prefix="calib",
        )
    for fi, fam in enumerate(FINAL_FAMILIES):
        final += make_family_scenarios(
            FINAL_VOCAB[fam],
            fam,
            n=N_FINAL_PER_FAMILY,
            seed=FINAL_SEED + 1000 * fi,
            id_prefix="final",
        )
    return {"calibration": calib, "final": final}


def calibration_prompts(calib: list[dict]) -> list[dict]:
    out = []
    for sc in calib:
        for w in W_GRID:
            out.append(
                {
                    "example_id": f"{sc['base_scenario_id']}__w{w:02d}",
                    "base_scenario_id": sc["base_scenario_id"],
                    "family": sc["family"],
                    "w": w,
                    "record_state": sc["record_state"],
                    "alternate_state": sc["alternate_state"],
                    "record_listed_first": sc["record_listed_first"],
                    "prompt_text": build_policy_prompt(scenario=sc, w=w),
                }
            )
    return out


def final_prompts(final: list[dict], w_star: int) -> list[dict]:
    return [
        {
            "example_id": f"{sc['base_scenario_id']}__policy",
            "base_scenario_id": sc["base_scenario_id"],
            "family": sc["family"],
            "w": w_star,
            "record_state": sc["record_state"],
            "alternate_state": sc["alternate_state"],
            "record_listed_first": sc["record_listed_first"],
            "prompt_text": build_policy_prompt(scenario=sc, w=w_star),
            "behavior_label": None,
            "model_called": False,
        }
        for sc in final
    ]


def counterbalance_report(rows: list[dict]) -> dict[str, dict]:
    rep = {}
    for fam in sorted({r["family"] for r in rows}):
        fr = [r for r in rows if r["family"] == fam]
        rec = Counter(r["record_state"] for r in fr)
        alt = Counter(r["alternate_state"] for r in fr)
        first = Counter(bool(r["record_listed_first"]) for r in fr)
        imb = max(abs(rec[v] - alt[v]) for v in set(rec) | set(alt))
        rep[fam] = {
            "n": len(fr),
            "max_record_minus_alternate": imb,
            "record_listed_first_true": first[True],
            "record_listed_first_false": first[False],
            "balanced": imb == 0 and first[True] == first[False],
        }
    return rep


def prior_topic_overlap(topics: set[str]) -> list[str]:
    prior = []
    for rel in PRIOR_PROMPT_FILES:
        path = REPO_ROOT / rel
        if not path.is_file():
            raise SystemExit(f"prior design payload missing: {rel}")
        prior += [
            json.loads(x)["prompt_text"]
            for x in path.read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]
    blob = "\n".join(prior)
    return sorted(t for t in topics if t in blob)


def tokenization_audit(tokenizer, scenarios: list[dict], ws: tuple[int, ...]) -> dict:
    """Verify prefix token 12107 and prefix-free, non-empty candidates for every (base, W)."""
    lengths: Counter = Counter()
    length_pairs: Counter = Counter()
    n_checked = 0
    cand_by_base: dict[str, dict[str, list[int]]] = {}
    for sc in scenarios:
        for w in ws:
            text = build_policy_prompt(scenario=sc, w=w)
            prompt_prefix_ids(tokenizer, text)
            rec = candidate_token_ids(tokenizer, text, sc["record_state"])
            alt = candidate_token_ids(tokenizer, text, sc["alternate_state"])
            validate_candidates([rec, alt])
            prev = cand_by_base.get(sc["base_scenario_id"])
            if prev is not None and (prev["record"] != rec or prev["alternate"] != alt):
                raise SystemExit("candidate tokenization varies with W")
            cand_by_base[sc["base_scenario_id"]] = {"record": rec, "alternate": alt}
            n_checked += 1
        c = cand_by_base[sc["base_scenario_id"]]
        lengths[len(c["record"])] += 1
        lengths[len(c["alternate"])] += 1
        length_pairs[f"{len(c['record'])}-{len(c['alternate'])}"] += 1
    return {
        "n_prompt_contexts_checked": n_checked,
        "response_prefix_token_id": 12107,
        "all_candidates_nonempty": True,
        "no_strict_prefix_pairs": True,
        "candidate_tokenization_invariant_to_w": True,
        "candidate_token_length_distribution": {str(k): v for k, v in sorted(lengths.items())},
        "record_alternate_length_pairs": dict(length_pairs),
        "all_lengths_matched": all(k.split("-")[0] == k.split("-")[1] for k in length_pairs),
        "candidate_ids_by_base": cand_by_base,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data/processed/phase8_design"))
    ap.add_argument("--summary-dir", default=str(REPO_ROOT / "artifacts/phase8a_design"))
    args = ap.parse_args()
    data_dir, summary_dir = Path(args.data_dir), Path(args.summary_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    all_fams = set(CALIBRATION_FAMILIES) | set(FINAL_FAMILIES)
    if len(all_fams) != 12 or all_fams & set(PRIOR_PHASE_FAMILIES):
        raise SystemExit("family set invalid — STOP")
    d = build_design()
    calib, final = d["calibration"], d["final"]
    if len(calib) != N_CALIBRATION_BASES or len(final) != N_FINAL_BASES:
        raise SystemExit("count drift")
    ct, ft = {s["topic_sentence"] for s in calib}, {s["topic_sentence"] for s in final}
    if ct & ft:
        raise SystemExit("calibration/final text overlap")
    if prior_topic_overlap(ct | ft):
        raise SystemExit("scenario text overlaps Phases 4-7")
    keys = [(s["family"], s["entity"], s["record_state"], s["alternate_state"]) for s in final]
    if len(set(keys)) != len(keys):
        raise SystemExit("duplicate final content keys")
    cb = {"calibration": counterbalance_report(calib), "final": counterbalance_report(final)}
    for part in cb.values():
        if not all(v["balanced"] for v in part.values()):
            raise SystemExit(f"counterbalance failed: {cb}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True)
    calib_audit = tokenization_audit(tok, calib, W_GRID)
    final_audit = tokenization_audit(tok, final, W_GRID)
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
        "phase": "phase8a",
        "model_revision": MODEL_REVISION,
        "w_grid": list(W_GRID),
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
        "communication_target_present": False,
        "prior_scenario_text_overlap": False,
        "final_prompts_instantiated": False,
        "model_calls_performed": False,
    }
    strip = lambda a: {k: v for k, v in a.items() if k != "candidate_ids_by_base"}  # noqa: E731
    write_json(summary_dir / "design_matrix.json", matrix)
    write_json(
        summary_dir / "candidate_tokenization_audit.json",
        {"calibration": strip(calib_audit), "final": strip(final_audit)},
    )
    write_json(summary_dir / "family_split.json", {**split, "frozen_before_final_behavior": True})
    write_json(
        summary_dir / "future_phase8b_plan.json",
        {**FUTURE_PHASE8B_PLAN, "discovery_gates": FUTURE_PHASE8B_GATES},
    )
    print(
        json.dumps(
            {
                k: matrix[k]
                for k in (
                    "calibration_scenario_text_sha256",
                    "calibration_base_scenario_ids_sha256",
                    "calibration_prompt_text_sha256",
                    "final_scenario_text_sha256",
                    "final_base_scenario_ids_sha256",
                )
            },
            indent=2,
        )
    )
    print(json.dumps({"calibration": strip(calib_audit), "final": strip(final_audit)}, indent=1))
    print(
        json.dumps(
            {
                k: split[k]
                for k in ("discovery_train", "discovery_validation", "locked_generalization")
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
