#!/usr/bin/env python3
"""Prepare Phase 13A semantic-component diagnostic (local only; no model calls).

Reuses the exact 144 Phase-12 selected bases. Builds the 2×2×2 T×Q×E factorial with
`slot` states everywhere. Cells `000` and `111` must be byte-identical to Phase-12 D
and B (respectively) so those RF/AF results can be reused without new model calls.
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

from pre_output_physiology.phase8_design import (  # noqa: E402
    candidate_order,
    candidate_token_ids,
    prompt_prefix_ids,
    validate_candidates,
)
from pre_output_physiology.phase11_design import mask_candidate_line  # noqa: E402
from pre_output_physiology.phase12_diagnostic import build_cell_prompt as p12_prompt  # noqa: E402
from pre_output_physiology.phase13_diagnostic import (  # noqa: E402
    CELLS,
    FAMILIES,
    GENERIC_QUESTION,
    N_BASES,
    N_NEW_EVALUATIONS,
    N_PER_FAMILY,
    NEW_CELLS,
    ORDERS,
    REUSE_CELLS,
    SELECTED_BASE_IDS_SHA256,
    assert_not_locked,
    build_cell_prompt,
    cell_scenario,
    neutralization_violations,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
P12_SELECTED = REPO_ROOT / "data/processed/phase12_design/selected_bases.jsonl"
P12_ALL = REPO_ROOT / "data/processed/phase12_design/all_cell_prompts.jsonl"
P12_ART = REPO_ROOT / "artifacts/phase12a_diagnostic"
P12_RUN_ID = "phase12a_diagnostic_20260926T224325Z_860e81be"
P12_RUN = REPO_ROOT / "artifacts/runs" / P12_RUN_ID


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text("utf-8").splitlines() if x.strip()]


def load_selected() -> list[dict]:
    rows = _jsonl(P12_SELECTED)
    if sha_ids([r["base_scenario_id"] for r in rows]) != SELECTED_BASE_IDS_SHA256:
        raise SystemExit("selected-base hash mismatch — STOP")
    if len(rows) != N_BASES or Counter(r["family"] for r in rows) != Counter(
        {f: N_PER_FAMILY for f in FAMILIES}
    ):
        raise SystemExit("selection count drift")
    assert_not_locked([r["family"] for r in rows])
    return rows


def build_prompts(sel: list[dict]) -> list[dict]:
    rows = []
    for b in sel:
        for cell in CELLS:
            sc = cell_scenario(b, cell)
            for o in ORDERS:
                rows.append(
                    {
                        "example_id": f"{b['base_scenario_id']}__{cell}__{o}",
                        "base_scenario_id": b["base_scenario_id"],
                        "family": b["family"],
                        "cell": cell,
                        "order": o,
                        "record_state": sc["record_state"],
                        "alternate_state": sc["alternate_state"],
                        "record_listed_first": o == "RF",
                        "prompt_text": build_cell_prompt(b, cell, o),
                    }
                )
    return rows


def cell_hashes(rows: list[dict]) -> dict[str, str]:
    return {
        f"cell_{c}_prompt_text_sha256": sha_prompt_texts([r for r in rows if r["cell"] == c])
        for c in CELLS
    }


def verify_reuse(sel: list[dict], rows: list[dict]) -> dict:
    """Require byte-identical prompts vs Phase-12 B/D before allowing reuse."""
    p12_prompts = {
        (r["base_scenario_id"], r["cell"], r["order"]): r
        for r in _jsonl(P12_ALL)
        if r["cell"] in ("B", "D")
    }
    # Also verify against Phase-12 builder directly.
    n_ok = 0
    for b in sel:
        for cell13, cell12 in REUSE_CELLS.items():
            for o in ORDERS:
                r13 = next(
                    r
                    for r in rows
                    if r["base_scenario_id"] == b["base_scenario_id"]
                    and r["cell"] == cell13
                    and r["order"] == o
                )
                built12 = p12_prompt(b, cell12, o)
                stored = p12_prompts[(b["base_scenario_id"], cell12, o)]["prompt_text"]
                if r13["prompt_text"] != built12 or r13["prompt_text"] != stored:
                    raise SystemExit(
                        f"reuse prompt mismatch {b['base_scenario_id']} {cell13}/{cell12} {o}"
                    )
                n_ok += 1
    manifest = json.loads((P12_ART / "diagnostic_manifest.json").read_text("utf-8"))
    if manifest["run_id"] != P12_RUN_ID:
        raise SystemExit("Phase-12 run id mismatch")
    raw_path = P12_RUN / "diagnostic_choices.jsonl"
    raw_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    if raw_sha != manifest["raw_choices_sha256"]:
        raise SystemExit("Phase-12 raw choices hash mismatch")
    # Map Phase-12 B/D choices onto selected bases.
    labels = json.loads((P12_ART / "base_labels_by_cell.json").read_text("utf-8"))
    reused_bases = {c13: 0 for c13 in REUSE_CELLS}
    for b in sel:
        for c13, c12 in REUSE_CELLS.items():
            hit = next(
                x for x in labels[c12] if x["base_scenario_id"] == b["base_scenario_id"]
            )
            if hit["family"] != b["family"]:
                raise SystemExit("family drift in reused labels")
            reused_bases[c13] += 1
    return {
        "phase12_run_id": P12_RUN_ID,
        "phase12_git_commit": manifest["git_commit"],
        "phase12_raw_choices_sha256_verified": raw_sha,
        "n_prompt_variants_byte_identical": n_ok,
        "reused_cells": REUSE_CELLS,
        "n_bases_per_reused_cell": reused_bases,
        "new_model_calls_for_reused_cells": 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data/processed/phase13_design"))
    ap.add_argument("--summary-dir", default=str(REPO_ROOT / "artifacts/phase13a_design"))
    args = ap.parse_args()
    data_dir, summary_dir = Path(args.data_dir), Path(args.summary_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    sel = load_selected()
    rows = build_prompts(sel)
    by_base = {b["base_scenario_id"]: b for b in sel}

    viol = {}
    for r in rows:
        v = neutralization_violations(
            r["prompt_text"], by_base[r["base_scenario_id"]], r["cell"]
        )
        if v:
            viol[r["example_id"]] = v
    if viol:
        raise SystemExit(f"neutralization failed: {list(viol.items())[:8]}")

    # Q=0 questions byte-identical across families.
    q0 = [r for r in rows if r["cell"][1] == "0"]
    q_lines = {
        next(
            ln
            for ln in r["prompt_text"].split("\n")
            if ln.startswith("User-visible question: ")
        )
        for r in q0
    }
    if q_lines != {f"User-visible question: {GENERIC_QUESTION}"}:
        raise SystemExit(f"Q=0 question drift: {q_lines}")

    idx = {(r["base_scenario_id"], r["cell"], r["order"]): r for r in rows}
    for b in sel:
        for c in CELLS:
            rf, af = idx[(b["base_scenario_id"], c, "RF")], idx[(b["base_scenario_id"], c, "AF")]
            if mask_candidate_line(rf["prompt_text"]) != mask_candidate_line(af["prompt_text"]):
                raise SystemExit("RF/AF differ outside candidate line")
            if candidate_order(af) != candidate_order(rf)[::-1]:
                raise SystemExit("candidate order wrong")
            if not rf["record_state"].startswith("slot ") or not rf["alternate_state"].startswith(
                "slot "
            ):
                raise SystemExit("non-slot state")
        digits = {
            (
                idx[(b["base_scenario_id"], c, "RF")]["record_state"].split()[-1],
                idx[(b["base_scenario_id"], c, "RF")]["alternate_state"].split()[-1],
            )
            for c in CELLS
        }
        if len(digits) != 1:
            raise SystemExit("record/alternate numbers differ across cells")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True)
    pairs: Counter = Counter()
    n_checked = 0
    for r in rows:
        prefix = prompt_prefix_ids(tok, r["prompt_text"])
        if prefix[-1] != 12107:
            raise SystemExit("prefix does not end in 12107")
        rec = candidate_token_ids(tok, r["prompt_text"], r["record_state"])
        alt = candidate_token_ids(tok, r["prompt_text"], r["alternate_state"])
        validate_candidates([rec, alt])
        if len(rec) != len(alt):
            raise SystemExit(f"lengths not matched: {r['example_id']}")
        pairs[f"{len(rec)}-{len(alt)}"] += 1
        n_checked += 1

    reuse = verify_reuse(sel, rows)
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
        "phase": "phase13a",
        "model_revision": MODEL_REVISION,
        "phase12_status": "phase12a_family_bias_source_diagnostic_complete_awaiting_audit",
        "families": list(FAMILIES),
        "n_per_family": N_PER_FAMILY,
        "n_bases": len(sel),
        "selected_base_ids_sha256": SELECTED_BASE_IDS_SHA256,
        "cells": list(CELLS),
        "factors": "T=topic, Q=question, E=entity; state namespace always slot",
        **cell_hashes(rows),
        "new_prompt_text_sha256": sha_prompt_texts(new_rows),
        "n_new_evaluations": len(new_rows),
        "tokenization": {
            "response_prefix_token_id": 12107,
            "n_prompt_contexts_checked": n_checked,
            "all_candidates_nonempty": True,
            "no_strict_prefix_pairs": True,
            "all_lengths_matched": True,
            "all_states_are_slot": True,
            "record_alternate_length_pairs": dict(pairs),
        },
        "neutralization": {
            "rf_af_differ_only_in_candidate_line": True,
            "record_alternate_numbers_identical_across_cells": True,
            "no_original_state_nouns": True,
            "Q0_question_byte_identical": True,
            "n_violations": 0,
        },
        "cell_reuse": reuse,
        "locked_families_included": False,
        "model_calls_performed": False,
    }
    write_json(summary_dir / "design_matrix.json", matrix)
    print(
        json.dumps(
            {
                k: matrix[k]
                for k in (
                    "selected_base_ids_sha256",
                    *[f"cell_{c}_prompt_text_sha256" for c in CELLS],
                    "new_prompt_text_sha256",
                    "tokenization",
                    "neutralization",
                    "cell_reuse",
                )
            },
            indent=1,
        )
    )
    b0 = sel[0]
    for c in CELLS:
        p = idx[(b0["base_scenario_id"], c, "RF")]["prompt_text"]
        print(f"--- {c}")
        print("\n".join(p.split("\n")[:3]))
        print(next(ln for ln in p.split("\n") if ln.startswith("User-visible")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
