"""Phase 17A — policy-unstable exact-prompt enrichment screen.

Behavior-only S1/S2 screen of frozen Phase-14 discovery finals at T=0.9
using the Phase-14 two-stage assay and frozen Phase-15 onset validity rule.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

from pre_output_physiology.phase14_design import (
    FINAL_FAMILIES,
    family_split,
)
from pre_output_physiology.phase15_onset import (
    CAT_ASSIGN,
    LABEL_ALTERNATE,
    LABEL_RECORD,
    PHASE14_RAW_CONTINUATIONS_SHA256,
    RULE_HASH,
    RULE_VERSION,
    annotate_continuation,
)

PHASE15_RULE_HASH = RULE_HASH
PHASE15_RULE_VERSION = RULE_VERSION

# --- Frozen Phase-14 final pool hashes (must match design_matrix) ---
FINAL_SCENARIO_TEXT_SHA256 = (
    "caeef7d1a1f941c87e44aff070ddbbf21102bc15e506cd15eb52aed03b8b2407"
)
FINAL_PROMPT_GROUP_IDS_SHA256 = (
    "d470586b1e8055d1a7b07af4551eeffdcdbb569c0d56bb30810d41e1df514dbd"
)
FINAL_PROMPT_TEXT_SHA256 = (
    "e35ece844c15a0393b218d67d67fe1bcd3000bf2770f41bd4613715dfbafe736"
)

TEMPERATURE = 0.9
TOP_P = 0.95
N_SAMPLES = 16
N_DISCOVERY_PROMPTS = 240
N_S1_CONTINUATIONS = N_DISCOVERY_PROMPTS * N_SAMPLES  # 3840

S1_SEED_BASE = 17_000_000
S2_SEED_BASE = 17_100_000  # disjoint from S1

# Frozen S1 candidate threshold (before any Phase-17 model calls).
S1_MIN_VALID = 14
S1_MIN_RECORD = 2
S1_MIN_ALTERNATE = 2

# Frozen S2 confirmation threshold (S2 alone; before any Phase-17 model calls).
S2_MIN_VALID = 14
S2_MIN_RECORD = 3
S2_MIN_ALTERNATE = 3

# Discovery gates (frozen before calls).
TRAIN_MIN_CONFIRMED = 20
TRAIN_MIN_PER_FAMILY = 4
VAL_MIN_CONFIRMED = 10
VAL_MIN_PER_FAMILY = 4

SELECTED_SALT = "phase17_selected_v1|"
MAX_SELECTED_PER_FAMILY = 8

STATUS_AUTHORIZED = "phase17a_policy_unstable_screen_authorized"
STATUS_PASS = "phase17a_policy_unstable_screen_pass_awaiting_audit"
STATUS_HOLD = "phase17a_policy_unstable_screen_hold"

T_RATIONALE = (
    "From Phase-15 calibration-only data: T=0.9 had 13/32 prompt groups with "
    "≥1 example of each policy vs 12/32 at T=1.1; both had 6/32 with ≥3/class; "
    "T=0.9 had slightly higher proposition-valid fraction. Selected "
    "prospectively before any Phase-17 prompt is run. No other T in Phase 17."
)

GUARANTEE = (
    "PHASE 17A WAS A BEHAVIOR-ONLY PROSPECTIVE ENRICHMENT SCREEN FOR EXACT "
    "PROMPTS WITH REPRODUCIBLE STOCHASTIC POLICY DIVERSITY. PHASES 14A AND 15A "
    "REMAIN HOLDS, AND PHASE 16A FOUND SAMPLING-ONLY RESCUE OF RANDOM PROMPTS "
    "UNSUPPORTED. NO ACTIVATIONS OR PHYSIOLOGY PROBES WERE USED FOR SCREENING. "
    "NO LOCKED PROMPTS WERE RUN. TEMPERATURE, PROMPTS, PAYOFFS, DECODER, ONSET "
    "RULE, AND SCREENING THRESHOLDS WERE NOT TUNED AFTER MODEL CALLS."
)


def _sha_json(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha_ids(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def discovery_prompts(final_prompts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    split = family_split()
    disc = set(split["discovery_train"] + split["discovery_validation"])
    locked = set(split["locked_generalization"])
    out = [p for p in final_prompts if p["family"] in disc]
    bad = [p for p in final_prompts if p["family"] in locked]
    if len(out) != N_DISCOVERY_PROMPTS:
        raise ValueError(f"expected {N_DISCOVERY_PROMPTS} discovery prompts, got {len(out)}")
    if len(bad) != 80:
        raise ValueError(f"expected 80 locked prompts, got {len(bad)}")
    return sorted(out, key=lambda p: p["prompt_group_id"])


def s1_sample_seed(prompt_index: int, sample_index: int) -> int:
    return S1_SEED_BASE + prompt_index * N_SAMPLES + sample_index


def s2_sample_seed(candidate_index: int, sample_index: int) -> int:
    return S2_SEED_BASE + candidate_index * N_SAMPLES + sample_index


def build_s1_schedule(discovery: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(discovery) != N_DISCOVERY_PROMPTS:
        raise ValueError("discovery size")
    rows = []
    for pi, p in enumerate(discovery):
        for si in range(N_SAMPLES):
            seed = s1_sample_seed(pi, si)
            cid = f"s1|{p['prompt_group_id']}|t{TEMPERATURE}|s{si}|seed{seed}"
            rows.append(
                {
                    "continuation_id": cid,
                    "stage": "S1",
                    "prompt_group_id": p["prompt_group_id"],
                    "family": p["family"],
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "sample_index": si,
                    "sample_seed": seed,
                    "prompt_index": pi,
                }
            )
    if len(rows) != N_S1_CONTINUATIONS:
        raise ValueError("S1 schedule size")
    return rows


def build_s2_schedule(
    candidate_prompt_group_ids: Sequence[str],
    discovery_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    ordered = sorted(candidate_prompt_group_ids)
    rows = []
    for ci, pg in enumerate(ordered):
        p = discovery_by_id[pg]
        for si in range(N_SAMPLES):
            seed = s2_sample_seed(ci, si)
            cid = f"s2|{pg}|t{TEMPERATURE}|s{si}|seed{seed}"
            rows.append(
                {
                    "continuation_id": cid,
                    "stage": "S2",
                    "prompt_group_id": pg,
                    "family": p["family"],
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "sample_index": si,
                    "sample_seed": seed,
                    "candidate_index": ci,
                }
            )
    return rows


def threshold_manifest() -> dict[str, Any]:
    body = {
        "temperature": TEMPERATURE,
        "temperature_rationale": T_RATIONALE,
        "top_p": TOP_P,
        "n_samples_per_prompt_per_stage": N_SAMPLES,
        "phase15_rule_version": PHASE15_RULE_VERSION,
        "phase15_rule_hash": PHASE15_RULE_HASH,
        "s1_candidate_rule": {
            "min_valid": S1_MIN_VALID,
            "min_record": S1_MIN_RECORD,
            "min_alternate": S1_MIN_ALTERNATE,
        },
        "s2_confirmation_rule": {
            "min_valid": S2_MIN_VALID,
            "min_record": S2_MIN_RECORD,
            "min_alternate": S2_MIN_ALTERNATE,
            "independent_of_s1": True,
        },
        "discovery_gates": {
            "train_min_confirmed": TRAIN_MIN_CONFIRMED,
            "train_min_per_family": TRAIN_MIN_PER_FAMILY,
            "validation_min_confirmed": VAL_MIN_CONFIRMED,
            "validation_min_per_family": VAL_MIN_PER_FAMILY,
        },
        "selection": {
            "salt": SELECTED_SALT,
            "max_per_family": MAX_SELECTED_PER_FAMILY,
            "sort_key": 'SHA256("phase17_selected_v1|" + prompt_group_id)',
        },
        "s1_seed_base": S1_SEED_BASE,
        "s2_seed_base": S2_SEED_BASE,
        "phase14_final_hashes": {
            "final_scenario_text_sha256": FINAL_SCENARIO_TEXT_SHA256,
            "final_prompt_group_ids_sha256": FINAL_PROMPT_GROUP_IDS_SHA256,
            "final_prompt_text_sha256": FINAL_PROMPT_TEXT_SHA256,
        },
    }
    return {**body, "threshold_hash": _sha_json(body)}


THRESHOLD_MANIFEST = threshold_manifest()
THRESHOLD_HASH = THRESHOLD_MANIFEST["threshold_hash"]


def summarize_prompt_group(annotated: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts for one prompt group at one stage."""
    n_rec = sum(
        1
        for r in annotated
        if r["phase15_valid_pre_answer"] and r["final_policy_label"] == LABEL_RECORD
    )
    n_alt = sum(
        1
        for r in annotated
        if r["phase15_valid_pre_answer"]
        and r["final_policy_label"] == LABEL_ALTERNATE
    )
    n_assign = sum(1 for r in annotated if r["onset_category"] == CAT_ASSIGN)
    n_other_inv = sum(
        1
        for r in annotated
        if (not r["phase15_valid_pre_answer"]) and r["onset_category"] != CAT_ASSIGN
    )
    n_valid = n_rec + n_alt
    return {
        "n_continuations": len(annotated),
        "n_valid": n_valid,
        "n_record": n_rec,
        "n_alternate": n_alt,
        "n_assignment_invalid": n_assign,
        "n_other_invalid": n_other_inv,
        "valid_fraction": n_valid / len(annotated) if annotated else 0.0,
        "alternate_fraction_among_valid": (n_alt / n_valid) if n_valid else None,
        "record_fraction_among_valid": (n_rec / n_valid) if n_valid else None,
    }


def is_s1_candidate(summary: dict[str, Any]) -> bool:
    return (
        summary["n_valid"] >= S1_MIN_VALID
        and summary["n_record"] >= S1_MIN_RECORD
        and summary["n_alternate"] >= S1_MIN_ALTERNATE
    )


def is_s2_confirmed(summary: dict[str, Any]) -> bool:
    return (
        summary["n_valid"] >= S2_MIN_VALID
        and summary["n_record"] >= S2_MIN_RECORD
        and summary["n_alternate"] >= S2_MIN_ALTERNATE
    )


def per_prompt_summaries(
    annotated: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in annotated:
        by[r["prompt_group_id"]].append(r)
    out = {}
    for pg, rows in by.items():
        s = summarize_prompt_group(rows)
        s["prompt_group_id"] = pg
        s["family"] = rows[0]["family"]
        out[pg] = s
    return out


def evaluate_gates(
    confirmed: list[dict[str, Any]],
) -> dict[str, Any]:
    split = family_split()
    train_fams = list(split["discovery_train"])
    val_fams = list(split["discovery_validation"])
    by_fam = Counter(c["family"] for c in confirmed)
    train_confirmed = [c for c in confirmed if c["family"] in train_fams]
    val_confirmed = [c for c in confirmed if c["family"] in val_fams]
    train_per = {f: by_fam[f] for f in train_fams}
    val_per = {f: by_fam[f] for f in val_fams}
    train_ok = (
        len(train_confirmed) >= TRAIN_MIN_CONFIRMED
        and all(v >= TRAIN_MIN_PER_FAMILY for v in train_per.values())
    )
    val_ok = (
        len(val_confirmed) >= VAL_MIN_CONFIRMED
        and all(v >= VAL_MIN_PER_FAMILY for v in val_per.values())
    )
    return {
        "train": {
            "n_confirmed": len(train_confirmed),
            "per_family": train_per,
            "min_overall": TRAIN_MIN_CONFIRMED,
            "min_per_family": TRAIN_MIN_PER_FAMILY,
            "passed": train_ok,
        },
        "validation": {
            "n_confirmed": len(val_confirmed),
            "per_family": val_per,
            "min_overall": VAL_MIN_CONFIRMED,
            "min_per_family": VAL_MIN_PER_FAMILY,
            "passed": val_ok,
        },
        "passed": train_ok and val_ok,
    }


def select_prompts(confirmed: list[dict[str, Any]]) -> dict[str, Any]:
    """SHA-ordered cap at most MAX_SELECTED_PER_FAMILY per family."""
    by_fam: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in confirmed:
        by_fam[c["family"]].append(c)
    selected: list[dict[str, Any]] = []
    digests: dict[str, str] = {}
    for _fam, rows in sorted(by_fam.items()):
        scored = []
        for r in rows:
            d = hashlib.sha256(
                f"{SELECTED_SALT}{r['prompt_group_id']}".encode()
            ).hexdigest()
            digests[r["prompt_group_id"]] = d
            scored.append((d, r))
        scored.sort(key=lambda x: x[0])
        for d, r in scored[:MAX_SELECTED_PER_FAMILY]:
            selected.append(
                {
                    "prompt_group_id": r["prompt_group_id"],
                    "family": r["family"],
                    "selection_digest": d,
                    "s2_n_record": r["n_record"],
                    "s2_n_alternate": r["n_alternate"],
                    "s2_n_valid": r["n_valid"],
                }
            )
    selected.sort(key=lambda r: r["prompt_group_id"])
    ids = [r["prompt_group_id"] for r in selected]
    return {
        "selected": selected,
        "selected_prompt_group_ids": ids,
        "selected_prompt_group_ids_sha256": _sha_ids(ids),
        "n_selected": len(selected),
        "n_per_family": dict(Counter(r["family"] for r in selected)),
        "selection_digests": digests,
        "salt": SELECTED_SALT,
        "max_per_family": MAX_SELECTED_PER_FAMILY,
    }


def policy_entropy(n_rec: int, n_alt: int) -> float | None:
    n = n_rec + n_alt
    if n <= 0:
        return None
    h = 0.0
    for c in (n_rec, n_alt):
        if c <= 0:
            continue
        p = c / n
        h -= p * math.log(p, 2)
    return h


def reproducibility_diagnostics(
    s1_by_pg: dict[str, dict[str, Any]],
    s2_by_pg: dict[str, dict[str, Any]],
    candidate_ids: Sequence[str],
) -> dict[str, Any]:
    """S1 vs S2 alternate-rate diagnostics on S1 candidates that received S2."""
    pairs = []
    for pg in candidate_ids:
        a1 = s1_by_pg[pg].get("alternate_fraction_among_valid")
        a2 = s2_by_pg[pg].get("alternate_fraction_among_valid")
        if a1 is None or a2 is None:
            continue
        pairs.append((pg, a1, a2, s1_by_pg[pg], s2_by_pg[pg]))
    if not pairs:
        return {
            "n_paired": 0,
            "pearson_alt_fraction": None,
            "median_abs_alt_diff": None,
            "n_switched_record_heavy_to_alternate_heavy": 0,
            "n_switched_alternate_heavy_to_record_heavy": 0,
            "s2_entropy_median": None,
            "s2_entropy_iqr": None,
        }
    xs = [p[1] for p in pairs]
    ys = [p[2] for p in pairs]
    # Pearson
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    pearson = (num / (denx * deny)) if denx > 0 and deny > 0 else None
    abs_diffs = sorted(abs(a - b) for _, a, b, _, _ in pairs)
    mid = len(abs_diffs) // 2
    med_abs = (
        abs_diffs[mid]
        if len(abs_diffs) % 2
        else (abs_diffs[mid - 1] + abs_diffs[mid]) / 2
    )
    switch_r_to_a = 0
    switch_a_to_r = 0
    for _, a1, a2, _, _ in pairs:
        if a1 < 0.5 <= a2:
            switch_r_to_a += 1
        elif a1 >= 0.5 > a2:
            switch_a_to_r += 1
    ents = [
        policy_entropy(p[4]["n_record"], p[4]["n_alternate"])
        for p in pairs
    ]
    ents_f = sorted(e for e in ents if e is not None)
    def _pct(xs: list[float], p: float) -> float | None:
        if not xs:
            return None
        idx = min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))
        return xs[idx]

    return {
        "n_paired": n,
        "pearson_alt_fraction": pearson,
        "median_abs_alt_diff": med_abs,
        "n_switched_record_heavy_to_alternate_heavy": switch_r_to_a,
        "n_switched_alternate_heavy_to_record_heavy": switch_a_to_r,
        "s2_entropy_median": _pct(ents_f, 0.5),
        "s2_entropy_iqr": (
            [_pct(ents_f, 0.25), _pct(ents_f, 0.75)] if ents_f else None
        ),
    }


def family_stage_table(
    summaries: dict[str, dict[str, Any]],
    *,
    flag_key: str | None = None,
) -> dict[str, Any]:
    split = family_split()
    fams = list(split["discovery_train"] + split["discovery_validation"])
    out: dict[str, Any] = {}
    for fam in fams:
        rows = [s for s in summaries.values() if s["family"] == fam]
        flagged = (
            [s for s in rows if s.get(flag_key)] if flag_key else rows
        )
        out[fam] = {
            "n_prompts": len(rows),
            "n_flagged": len(flagged) if flag_key else None,
            "total_record": sum(s["n_record"] for s in rows),
            "total_alternate": sum(s["n_alternate"] for s in rows),
            "total_valid": sum(s["n_valid"] for s in rows),
            "total_assignment_invalid": sum(s["n_assignment_invalid"] for s in rows),
            "total_other_invalid": sum(s["n_other_invalid"] for s in rows),
        }
    return out


# Re-exports for scripts
__all__ = [
    "TEMPERATURE",
    "TOP_P",
    "N_SAMPLES",
    "N_DISCOVERY_PROMPTS",
    "N_S1_CONTINUATIONS",
    "S1_SEED_BASE",
    "S2_SEED_BASE",
    "THRESHOLD_MANIFEST",
    "THRESHOLD_HASH",
    "STATUS_AUTHORIZED",
    "STATUS_PASS",
    "STATUS_HOLD",
    "T_RATIONALE",
    "GUARANTEE",
    "FINAL_SCENARIO_TEXT_SHA256",
    "FINAL_PROMPT_GROUP_IDS_SHA256",
    "FINAL_PROMPT_TEXT_SHA256",
    "PHASE15_RULE_HASH",
    "PHASE15_RULE_VERSION",
    "PHASE14_RAW_CONTINUATIONS_SHA256",
    "discovery_prompts",
    "build_s1_schedule",
    "build_s2_schedule",
    "annotate_continuation",
    "per_prompt_summaries",
    "is_s1_candidate",
    "is_s2_confirmed",
    "evaluate_gates",
    "select_prompts",
    "reproducibility_diagnostics",
    "family_stage_table",
    "family_split",
    "FINAL_FAMILIES",
    "_sha_ids",
]
