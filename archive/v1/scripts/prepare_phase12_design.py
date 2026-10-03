#!/usr/bin/env python3
"""Prepare Phase 12A family-bias source diagnostic (local only; no model calls).

Selects 24 Phase-11 discovery bases per family by salted SHA256 of the base ID (no labels
read), builds A/B/C/D x RF/AF prompts, runs tokenization and neutralization gates, and
verifies provenance of the reused Phase-11 cell-A results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from prepare_phase8_design import sha_ids, sha_prompt_texts  # noqa: E402
from prepare_phase11_design import load_phase10_final  # noqa: E402

from pre_output_physiology.phase8_design import (  # noqa: E402
    candidate_order,
    candidate_token_ids,
    prompt_prefix_ids,
    validate_candidates,
)
from pre_output_physiology.phase11_design import (  # noqa: E402
    build_order_prompt,
    mask_candidate_line,
)
from pre_output_physiology.phase12_diagnostic import (  # noqa: E402
    CELL_NAMES,
    CELLS,
    FAMILIES,
    N_BASES,
    N_NEW_EVALUATIONS,
    N_PER_FAMILY,
    NEW_CELLS,
    ORDERS,
    SELECTION_SALT,
    assert_not_locked,
    build_cell_prompt,
    cell_scenario,
    neutralization_violations,
    select_bases,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
P11_PROMPTS = REPO_ROOT / "data/processed/phase11_design/discovery_order_prompts.jsonl"
P11_ART = REPO_ROOT / "artifacts/phase11a_behavior"
P11_RUN_ID = "phase11a_behavior_20260926T223153Z_aa9a743b"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text("utf-8").splitlines() if x.strip()]


def build_prompts(sel: list[dict]) -> list[dict]:
    rows = []
    for b in sel:
        for cell in CELLS:
            sc = cell_scenario(b, cell)
            for o in ORDERS:
                rows.append({
                    "example_id": f"{b['base_scenario_id']}__{cell}__{o}",
                    "base_scenario_id": b["base_scenario_id"],
                    "family": b["family"],
                    "cell": cell,
                    "order": o,
                    "record_state": sc["record_state"],
                    "alternate_state": sc["alternate_state"],
                    "record_listed_first": o == "RF",
                    "prompt_text": build_cell_prompt(b, cell, o),
                })
    return rows


def cell_hashes(rows: list[dict]) -> dict[str, str]:
    return {f"{c.lower()}_prompt_text_sha256": sha_prompt_texts([r for r in rows
                                                               if r["cell"] == c])
            for c in CELLS}


def verify_cell_a_provenance(sel: list[dict], rows: list[dict]) -> dict:
    manifest = json.loads((P11_ART / "behavior_manifest.json").read_text("utf-8"))
    if manifest["run_id"] != P11_RUN_ID:
        raise SystemExit("Phase-11 run id mismatch")
    raw_path = REPO_ROOT / "artifacts/runs" / P11_RUN_ID / "behavior_choices.jsonl"
    raw_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    if raw_sha != manifest["raw_choices_sha256"]:
        raise SystemExit("Phase-11 raw choices hash mismatch")
    raw = {r["example_id"]: r for r in _jsonl(raw_path)}
    p11_prompts = {r["example_id"]: r for r in _jsonl(P11_PROMPTS)}
    variants = {v["example_id"]: v for v in json.loads(
        (P11_ART / "variant_choices.json").read_text("utf-8"))}
    n = 0
    for b in sel:
        for o in ORDERS:
            eid = f"{b['base_scenario_id']}__{o}"
            a_row = next(r for r in rows if r["example_id"] == f"{b['base_scenario_id']}__A__{o}")
            if not (a_row["prompt_text"] == p11_prompts[eid]["prompt_text"]
                    == build_order_prompt(b, o)):
                raise SystemExit(f"cell A prompt differs from Phase-11 prompt: {eid}")
            ch = raw[eid]["chosen_state"]
            choice = "record" if ch == raw[eid]["record_state"] else "alternate"
            if variants[eid]["choice"] != choice:
                raise SystemExit("Phase-11 variant choice disagrees with raw run")
            n += 1
    return {
        "phase11_run_id": P11_RUN_ID,
        "phase11_git_commit": manifest["git_commit"],
        "phase11_raw_choices_sha256_verified": raw_sha,
        "n_cell_a_variants_reused": n,
        "cell_a_prompts_identical_to_phase11": True,
        "new_model_calls_for_cell_a": 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data/processed/phase12_design"))
    ap.add_argument("--summary-dir", default=str(REPO_ROOT / "artifacts/phase12a_design"))
    args = ap.parse_args()
    data_dir, summary_dir = Path(args.data_dir), Path(args.summary_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    final = load_phase10_final()
    sel = select_bases(final)
    assert_not_locked([b["family"] for b in sel])
    if len(sel) != N_BASES or Counter(b["family"] for b in sel) != Counter(
            {f: N_PER_FAMILY for f in FAMILIES}):
        raise SystemExit("selection count drift")
    rows = build_prompts(sel)
    by_base = {b["base_scenario_id"]: b for b in sel}

    viol = {}
    for r in rows:
        v = neutralization_violations(r["prompt_text"], by_base[r["base_scenario_id"]],
                                      r["cell"])
        if v:
            viol[r["example_id"]] = v
    if viol:
        raise SystemExit(f"neutralization failed: {list(viol.items())[:5]}")
    idx = {(r["base_scenario_id"], r["cell"], r["order"]): r for r in rows}
    for b in sel:
        for c in CELLS:
            rf, af = idx[(b["base_scenario_id"], c, "RF")], idx[(b["base_scenario_id"], c, "AF")]
            if mask_candidate_line(rf["prompt_text"]) != mask_candidate_line(af["prompt_text"]):
                raise SystemExit("RF/AF differ outside candidate line")
            if candidate_order(af) != candidate_order(rf)[::-1]:
                raise SystemExit("candidate order wrong")
        digits = {(idx[(b["base_scenario_id"], c, "RF")]["record_state"].split()[-1],
                   idx[(b["base_scenario_id"], c, "RF")]["alternate_state"].split()[-1])
                  for c in CELLS}
        if len(digits) != 1:
            raise SystemExit("record/alternate numbers differ across cells")
    d_texts = Counter(r["prompt_text"] for r in rows if r["cell"] == "D")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True)
    pairs: dict[str, Counter] = {c: Counter() for c in CELLS}
    n_checked = 0
    for r in rows:
        prefix = prompt_prefix_ids(tok, r["prompt_text"])
        if prefix[-1] != 12107:
            raise SystemExit("prefix does not end in 12107")
        rec = candidate_token_ids(tok, r["prompt_text"], r["record_state"])
        alt = candidate_token_ids(tok, r["prompt_text"], r["alternate_state"])
        validate_candidates([rec, alt])
        if len(rec) != len(alt):
            raise SystemExit(f"candidate lengths not matched: {r['example_id']}")
        pairs[r["cell"]][f"{len(rec)}-{len(alt)}"] += 1
        n_checked += 1

    prov = verify_cell_a_provenance(sel, rows)
    new_rows = [r for r in rows if r["cell"] in NEW_CELLS]
    if len(new_rows) != N_NEW_EVALUATIONS:
        raise SystemExit("new evaluation count drift")
    with (data_dir / "selected_bases.jsonl").open("w", encoding="utf-8") as fh:
        for b in sel:
            fh.write(json.dumps(b, sort_keys=True) + "\n")
    with (data_dir / "all_cell_prompts.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    with (data_dir / "new_cell_prompts.jsonl").open("w", encoding="utf-8") as fh:
        for r in new_rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    matrix = {
        "created_at": utc_now_iso(),
        "phase": "phase12a",
        "model_revision": MODEL_REVISION,
        "phase11_status": "phase11a_order_robust_behavior_hold",
        "families": list(FAMILIES),
        "n_per_family": N_PER_FAMILY,
        "n_bases": len(sel),
        "selection_rule": f'first {N_PER_FAMILY}/family by SHA256("{SELECTION_SALT}" + '
        "base_scenario_id); labels not read",
        "selected_base_ids_sha256": sha_ids([b["base_scenario_id"] for b in sel]),
        "cells": CELL_NAMES,
        **cell_hashes(rows),
        "new_prompt_text_sha256": sha_prompt_texts(new_rows),
        "n_new_evaluations": len(new_rows),
        "tokenization": {
            "response_prefix_token_id": 12107,
            "n_prompt_contexts_checked": n_checked,
            "all_candidates_nonempty": True,
            "no_strict_prefix_pairs": True,
            "all_lengths_matched": True,
            "length_pairs_by_cell": {c: dict(v) for c, v in pairs.items()},
        },
        "neutralization": {
            "rf_af_differ_only_in_candidate_line": True,
            "record_alternate_numbers_identical_across_cells": True,
            "B_no_state_noun": True,
            "C_no_family_semantic_or_entity_outside_states": True,
            "D_no_family_vocabulary": True,
            "n_violations": 0,
            "n_distinct_D_prompts": len(d_texts),
        },
        "cell_a_reuse": prov,
        "locked_families_included": False,
        "model_calls_performed": False,
    }
    write_json(summary_dir / "design_matrix.json", matrix)
    print(json.dumps({k: matrix[k] for k in (
        "selected_base_ids_sha256", "a_prompt_text_sha256", "b_prompt_text_sha256",
        "c_prompt_text_sha256", "d_prompt_text_sha256", "new_prompt_text_sha256",
        "tokenization", "neutralization", "cell_a_reuse")}, indent=1))
    b0 = sel[0]
    for c in CELLS:
        print(f"--- {c}")
        print("\n".join(idx[(b0["base_scenario_id"], c, "RF")]["prompt_text"].split("\n")[:3]))
        print(idx[(b0["base_scenario_id"], c, "RF")]["prompt_text"].split("\n")[10])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
