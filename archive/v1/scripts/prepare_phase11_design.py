#!/usr/bin/env python3
"""Prepare Phase 11A order-robust behavior design (local only; no model calls).

Reuses the frozen Phase-10 final bases; builds RF and AF prompts at K=10 for the 720
discovery bases only. Locked-family prompts are not built.
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

from prepare_phase8_design import sha_ids, sha_prompt_texts, sha_scenarios  # noqa: E402

from pre_output_physiology.phase8_design import (  # noqa: E402
    candidate_order,
    candidate_token_ids,
    prompt_prefix_ids,
    validate_candidates,
)
from pre_output_physiology.phase10_design import family_split  # noqa: E402
from pre_output_physiology.phase11_design import (  # noqa: E402
    DISCOVERY_TRAIN,
    DISCOVERY_VALIDATION,
    FIXED_K,
    FUTURE_PHASE11B_PLAN,
    GATES,
    K_RATIONALE,
    LOCKED,
    N_DISCOVERY_BASES,
    N_EVALUATIONS,
    ORDERS,
    PHASE10_FINAL_SCENARIO_IDS_SHA256,
    PHASE10_FINAL_SCENARIO_TEXT_SHA256,
    build_order_prompt,
    mask_candidate_line,
    order_variant,
    split_of,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
PHASE10_FINAL = REPO_ROOT / "data/processed/phase10_design/final_base_scenarios.jsonl"


def load_phase10_final() -> list[dict]:
    rows = [json.loads(x) for x in PHASE10_FINAL.read_text("utf-8").splitlines() if x.strip()]
    if (
        sha_scenarios(rows) != PHASE10_FINAL_SCENARIO_TEXT_SHA256
        or sha_ids([r["base_scenario_id"] for r in rows]) != PHASE10_FINAL_SCENARIO_IDS_SHA256
    ):
        raise SystemExit("Phase-10 final base hashes do not match — STOP")
    return rows


def discovery_prompts(final: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {o: [] for o in ORDERS}
    for sc in final:
        if sc["family"] in LOCKED:
            continue
        for o in ORDERS:
            v = order_variant(sc, o)
            out[o].append(
                {
                    "example_id": f"{sc['base_scenario_id']}__{o}",
                    "base_scenario_id": sc["base_scenario_id"],
                    "family": sc["family"],
                    "split": split_of(sc["family"]),
                    "order": o,
                    "k": FIXED_K,
                    "record_state": sc["record_state"],
                    "alternate_state": sc["alternate_state"],
                    "record_listed_first": v["record_listed_first"],
                    "prompt_text": build_order_prompt(sc, o),
                }
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data/processed/phase11_design"))
    ap.add_argument("--summary-dir", default=str(REPO_ROOT / "artifacts/phase11a_design"))
    args = ap.parse_args()
    data_dir, summary_dir = Path(args.data_dir), Path(args.summary_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    split = family_split()
    if (
        tuple(split["discovery_train"]) != DISCOVERY_TRAIN
        or tuple(split["discovery_validation"]) != DISCOVERY_VALIDATION
        or tuple(split["locked_generalization"]) != LOCKED
    ):
        raise SystemExit("family split drift — STOP")
    final = load_phase10_final()
    pr = discovery_prompts(final)
    if any(len(pr[o]) != N_DISCOVERY_BASES for o in ORDERS):
        raise SystemExit("discovery count drift")
    if any(p["family"] in LOCKED for o in ORDERS for p in pr[o]):
        raise SystemExit("locked family in payload")
    n_masked_identical = 0
    for rf, af in zip(pr["RF"], pr["AF"], strict=True):
        if rf["base_scenario_id"] != af["base_scenario_id"]:
            raise SystemExit("pairing drift")
        if mask_candidate_line(rf["prompt_text"]) != mask_candidate_line(af["prompt_text"]):
            raise SystemExit("RF/AF differ outside the candidate-list line")
        if rf["prompt_text"] == af["prompt_text"]:
            raise SystemExit("RF/AF identical")
        c_rf, c_af = candidate_order(rf), candidate_order(af)
        if c_rf != (rf["record_state"], rf["alternate_state"]) or c_af != c_rf[::-1]:
            raise SystemExit("candidate order wrong")
        n_masked_identical += 1

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True)
    pairs: Counter = Counter()
    div: Counter = Counter()
    order_invariant = True
    for rf, af in zip(pr["RF"], pr["AF"], strict=True):
        got = []
        for p in (rf, af):
            prefix = prompt_prefix_ids(tok, p["prompt_text"])
            if prefix[-1] != 12107:
                raise SystemExit("prefix does not end in 12107")
            rec = candidate_token_ids(tok, p["prompt_text"], p["record_state"])
            alt = candidate_token_ids(tok, p["prompt_text"], p["alternate_state"])
            validate_candidates([rec, alt])
            if len(rec) != len(alt):
                raise SystemExit("candidate lengths not matched")
            got.append((rec, alt))
        order_invariant &= got[0] == got[1]
        rec, alt = got[0]
        pairs[f"{len(rec)}-{len(alt)}"] += 1
        div[next(i for i in range(len(rec)) if rec[i] != alt[i])] += 1
    if not order_invariant:
        raise SystemExit("candidate tokenization differs between RF and AF")

    with (data_dir / "discovery_order_prompts.jsonl").open("w", encoding="utf-8") as fh:
        for o in ORDERS:
            for r in pr[o]:
                fh.write(json.dumps(r, sort_keys=True) + "\n")
    matrix = {
        "created_at": utc_now_iso(),
        "phase": "phase11a",
        "model_revision": MODEL_REVISION,
        "phase10_status": "phase10a_risk_frontier_sanity_hold",
        "fixed_k": FIXED_K,
        "k_rationale": K_RATIONALE,
        "reused_final_scenario_text_sha256": sha_scenarios(final),
        "reused_final_base_scenario_ids_sha256": sha_ids([r["base_scenario_id"] for r in final]),
        "family_split": {k: split[k] for k in (
            "discovery_train", "discovery_validation", "locked_generalization")},
        "n_discovery_bases": N_DISCOVERY_BASES,
        "n_evaluations": N_EVALUATIONS,
        "n_locked_bases_excluded": sum(r["family"] in LOCKED for r in final),
        "rf_prompt_text_sha256": sha_prompt_texts(pr["RF"]),
        "af_prompt_text_sha256": sha_prompt_texts(pr["AF"]),
        "discovery_base_ids_sha256": sha_ids([p["base_scenario_id"] for p in pr["RF"]]),
        "rf_af_byte_identical_after_masking_candidate_line": n_masked_identical,
        "tokenization": {
            "response_prefix_token_id": 12107,
            "n_prompt_contexts_checked": 2 * N_DISCOVERY_BASES,
            "all_candidates_nonempty": True,
            "no_strict_prefix_pairs": True,
            "all_lengths_matched": True,
            "candidate_tokenization_invariant_to_order": True,
            "record_alternate_length_pairs": dict(pairs),
            "first_divergent_position_counts": {str(k): v for k, v in div.items()},
        },
        "gates": GATES,
        "model_calls_performed": False,
    }
    write_json(summary_dir / "design_matrix.json", matrix)
    write_json(summary_dir / "future_phase11b_plan.json", FUTURE_PHASE11B_PLAN)
    print(json.dumps({k: matrix[k] for k in (
        "reused_final_scenario_text_sha256", "reused_final_base_scenario_ids_sha256",
        "rf_prompt_text_sha256", "af_prompt_text_sha256", "discovery_base_ids_sha256",
        "rf_af_byte_identical_after_masking_candidate_line", "n_locked_bases_excluded",
        "tokenization")}, indent=1))
    print(pr["RF"][0]["prompt_text"])
    print("----")
    print(pr["AF"][0]["prompt_text"].split("\n")[2])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
