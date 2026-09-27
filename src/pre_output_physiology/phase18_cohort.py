"""Phase 18A — enriched near-boundary cohort fresh confirmation.

Behavior-only. Selects 24 Phase-17 S2-confirmed prompts via deterministic SHA,
then confirms phenotype on an independent 48-continuation seed batch.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

from pre_output_physiology.phase15_onset import (
    LABEL_ALTERNATE,
    LABEL_RECORD,
    RULE_HASH,
    RULE_VERSION,
    annotate_continuation,
)
from pre_output_physiology.phase17_screen import TEMPERATURE as _P17_T
from pre_output_physiology.phase17_screen import TOP_P as _P17_TOP_P
from pre_output_physiology.phase17_screen import policy_entropy

PHASE17_TEMPERATURE = _P17_T
PHASE17_TOP_P = _P17_TOP_P

PHASE15_RULE_HASH = RULE_HASH
PHASE15_RULE_VERSION = RULE_VERSION

PHASE17_STATUS = "phase17a_policy_unstable_screen_hold"
PHASE17_CONFIRMED_IDS_SHA256 = (
    "78068558226919eac00d21e6df68b86b2b4f0e584cf067786c1b28450d9ed2c5"
)

QUALIFYING_TRAIN_FAMILIES: tuple[str, ...] = (
    "bowling_alley_lane",
    "radio_studio_booth",
    "veterinary_kennel_run",
)
QUALIFYING_VAL_FAMILIES: tuple[str, ...] = ("daycare_cubby_shelf",)
QUALIFYING_FAMILIES: tuple[str, ...] = (
    QUALIFYING_TRAIN_FAMILIES + QUALIFYING_VAL_FAMILIES
)
EXCLUDED_FAMILIES: tuple[str, ...] = (
    "climbing_gym_route",
    "subway_turnstile_bank",
)
LOCKED_FAMILIES: tuple[str, ...] = (
    "museum_cloakroom_hook",
    "pier_mooring_slip",
)

MIN_CONFIRMED_PER_QUALIFYING_FAMILY = 6
N_PER_FAMILY = 6
N_COHORT = N_PER_FAMILY * len(QUALIFYING_FAMILIES)  # 24
N_SAMPLES = 48
N_CONTINUATIONS = N_COHORT * N_SAMPLES  # 1152

SELECTION_SALT = "phase18_enriched_v1|"
SEED_BASE = 18_000_000  # disjoint from Phase-14 (14M) and Phase-17 (17M/17.1M)

TEMPERATURE = 0.9
TOP_P = 0.95

# Fresh usability (frozen before Phase-18 model calls).
FRESH_MIN_VALID = 42
FRESH_MIN_RECORD = 6
FRESH_MIN_ALTERNATE = 6

# Gates (frozen before Phase-18 model calls).
TRAIN_MIN_USABLE = 15
TRAIN_MIN_PER_FAMILY = 5
VAL_MIN_USABLE = 5  # of 6 daycare

STATUS_AUTHORIZED = "phase18a_enriched_cohort_fresh_confirmation_authorized"
STATUS_PASS = "phase18a_enriched_cohort_fresh_confirmation_pass_awaiting_audit"
STATUS_HOLD = "phase18a_enriched_cohort_fresh_confirmation_hold"

GUARANTEE = (
    "PHASE 17A REMAINS A FAMILY-COVERAGE HOLD. PHASE 18A DEFINED A NEW "
    "BEHAVIORALLY ENRICHED NEAR-BOUNDARY PROMPT POPULATION USING ONLY PRIOR "
    "PHASE-17 S2 BEHAVIOR AND DETERMINISTIC SHA SELECTION. PHASE 18A USED AN "
    "INDEPENDENT FRESH SEED BATCH TO CONFIRM THAT POPULATION BEFORE ANY "
    "PHYSIOLOGY. NO ACTIVATIONS OR PHYSIOLOGY PROBES WERE COLLECTED OR SCORED. "
    "NO LOCKED PROMPTS WERE RUN. PROMPTS, TEMPERATURE, PAYOFFS, DECODER, ONSET "
    "RULE, SELECTION RULE, AND FRESH-CONFIRMATION THRESHOLDS WERE NOT TUNED "
    "AFTER PHASE-18 MODEL CALLS."
)

assert TEMPERATURE == PHASE17_TEMPERATURE
assert TOP_P == PHASE17_TOP_P


def _sha_ids(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def _sha_json(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_phase17_confirmed(
    screen_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    if screen_summary.get("status") != PHASE17_STATUS:
        raise ValueError(
            f"Phase 17 must remain HOLD; got {screen_summary.get('status')}"
        )
    confirmed = [
        s
        for s in screen_summary["s2_per_prompt"].values()
        if s.get("s2_confirmed")
    ]
    ids = sorted(c["prompt_group_id"] for c in confirmed)
    if ids != sorted(screen_summary.get("confirmed_prompt_group_ids", [])):
        raise ValueError("confirmed ID list mismatch vs s2_per_prompt")
    if _sha_ids(ids) != PHASE17_CONFIRMED_IDS_SHA256:
        raise ValueError(
            f"confirmed IDs hash mismatch: {_sha_ids(ids)} != {PHASE17_CONFIRMED_IDS_SHA256}"
        )
    return confirmed


def select_enriched_cohort(
    confirmed: list[dict[str, Any]],
) -> dict[str, Any]:
    """SHA-order first N_PER_FAMILY confirmed prompts in each qualifying family."""
    by_fam: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in confirmed:
        by_fam[c["family"]].append(c)

    for fam in QUALIFYING_FAMILIES:
        n = len(by_fam[fam])
        if n < MIN_CONFIRMED_PER_QUALIFYING_FAMILY:
            raise ValueError(
                f"family {fam} has {n} confirmed < {MIN_CONFIRMED_PER_QUALIFYING_FAMILY}"
            )
    # Excluded/locked families may have confirmed prompts (e.g. climbing) but
    # never enter the Phase-18 cohort (QUALIFYING_FAMILIES only).

    selected: list[dict[str, Any]] = []
    digests: dict[str, str] = {}
    for fam in QUALIFYING_FAMILIES:
        scored = []
        for c in by_fam[fam]:
            d = hashlib.sha256(
                f"{SELECTION_SALT}{c['prompt_group_id']}".encode()
            ).hexdigest()
            digests[c["prompt_group_id"]] = d
            scored.append((d, c))
        scored.sort(key=lambda x: x[0])
        for d, c in scored[:N_PER_FAMILY]:
            role = (
                "physiology_development_validation"
                if fam in QUALIFYING_VAL_FAMILIES
                else "physiology_development_train"
            )
            selected.append(
                {
                    "prompt_group_id": c["prompt_group_id"],
                    "family": fam,
                    "split_role": role,
                    "selection_digest": d,
                    "phase17_s2_n_record": c["n_record"],
                    "phase17_s2_n_alternate": c["n_alternate"],
                    "phase17_s2_n_valid": c["n_valid"],
                    "phase17_s2_alternate_fraction": c.get(
                        "alternate_fraction_among_valid"
                    ),
                }
            )

    selected.sort(key=lambda r: r["prompt_group_id"])
    ids = [r["prompt_group_id"] for r in selected]
    if len(ids) != N_COHORT:
        raise ValueError(f"expected {N_COHORT} selected, got {len(ids)}")
    per_fam = Counter(r["family"] for r in selected)
    if any(per_fam[f] != N_PER_FAMILY for f in QUALIFYING_FAMILIES):
        raise ValueError(f"per-family imbalance: {dict(per_fam)}")

    train_ids = [
        r["prompt_group_id"]
        for r in selected
        if r["split_role"] == "physiology_development_train"
    ]
    val_ids = [
        r["prompt_group_id"]
        for r in selected
        if r["split_role"] == "physiology_development_validation"
    ]
    return {
        "selection_salt": SELECTION_SALT,
        "n_per_family": N_PER_FAMILY,
        "qualifying_families": list(QUALIFYING_FAMILIES),
        "excluded_families": list(EXCLUDED_FAMILIES),
        "locked_families_untouched": list(LOCKED_FAMILIES),
        "selected": selected,
        "selected_prompt_group_ids": ids,
        "cohort_sha256": _sha_ids(ids),
        "n_selected": len(ids),
        "n_per_family_counts": dict(per_fam),
        "train_prompt_group_ids": sorted(train_ids),
        "validation_prompt_group_ids": sorted(val_ids),
        "n_train": len(train_ids),
        "n_validation": len(val_ids),
        "selection_digests": digests,
        "phase17_confirmed_ids_sha256": PHASE17_CONFIRMED_IDS_SHA256,
    }


def sample_seed(prompt_index: int, sample_index: int) -> int:
    return SEED_BASE + prompt_index * N_SAMPLES + sample_index


def build_fresh_schedule(cohort: dict[str, Any]) -> list[dict[str, Any]]:
    ordered = sorted(
        cohort["selected"], key=lambda r: r["prompt_group_id"]
    )
    rows = []
    for pi, p in enumerate(ordered):
        for si in range(N_SAMPLES):
            seed = sample_seed(pi, si)
            cid = (
                f"p18|{p['prompt_group_id']}|t{TEMPERATURE}|s{si}|seed{seed}"
            )
            rows.append(
                {
                    "continuation_id": cid,
                    "stage": "P18A",
                    "prompt_group_id": p["prompt_group_id"],
                    "family": p["family"],
                    "split_role": p["split_role"],
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "sample_index": si,
                    "sample_seed": seed,
                    "prompt_index": pi,
                }
            )
    if len(rows) != N_CONTINUATIONS:
        raise ValueError("schedule size")
    seeds = [r["sample_seed"] for r in rows]
    if min(seeds) < SEED_BASE or max(seeds) >= 19_000_000:
        raise ValueError("seed range unexpected")
    if min(seeds) < 18_000_000:
        raise ValueError("seeds collide with prior phases")
    return rows


def threshold_manifest() -> dict[str, Any]:
    body = {
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "n_samples": N_SAMPLES,
        "n_cohort": N_COHORT,
        "n_continuations": N_CONTINUATIONS,
        "phase15_rule_version": PHASE15_RULE_VERSION,
        "phase15_rule_hash": PHASE15_RULE_HASH,
        "selection_salt": SELECTION_SALT,
        "n_per_family": N_PER_FAMILY,
        "qualifying_families": list(QUALIFYING_FAMILIES),
        "seed_base": SEED_BASE,
        "fresh_usability": {
            "min_valid": FRESH_MIN_VALID,
            "min_record": FRESH_MIN_RECORD,
            "min_alternate": FRESH_MIN_ALTERNATE,
        },
        "gates": {
            "train_min_usable": TRAIN_MIN_USABLE,
            "train_min_per_family": TRAIN_MIN_PER_FAMILY,
            "validation_min_usable": VAL_MIN_USABLE,
        },
        "phase17_status_unchanged": PHASE17_STATUS,
        "phase17_confirmed_ids_sha256": PHASE17_CONFIRMED_IDS_SHA256,
    }
    return {**body, "threshold_hash": _sha_json(body)}


THRESHOLD_MANIFEST = threshold_manifest()
THRESHOLD_HASH = THRESHOLD_MANIFEST["threshold_hash"]


def summarize_prompt_group(annotated: list[dict[str, Any]]) -> dict[str, Any]:
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
    n_valid = n_rec + n_alt
    n_assign = sum(
        1
        for r in annotated
        if r.get("onset_category") == "candidate_assignment_proposition"
    )
    return {
        "n_continuations": len(annotated),
        "n_valid": n_valid,
        "n_record": n_rec,
        "n_alternate": n_alt,
        "n_assignment_invalid": n_assign,
        "n_other_invalid": len(annotated) - n_valid - n_assign,
        "valid_fraction": n_valid / len(annotated) if annotated else 0.0,
        "alternate_fraction_among_valid": (n_alt / n_valid) if n_valid else None,
        "record_fraction_among_valid": (n_rec / n_valid) if n_valid else None,
        "policy_entropy": policy_entropy(n_rec, n_alt),
    }


def is_fresh_usable(summary: dict[str, Any]) -> bool:
    return (
        summary["n_valid"] >= FRESH_MIN_VALID
        and summary["n_record"] >= FRESH_MIN_RECORD
        and summary["n_alternate"] >= FRESH_MIN_ALTERNATE
    )


def evaluate_gates(usable: list[dict[str, Any]]) -> dict[str, Any]:
    train = [
        u
        for u in usable
        if u["family"] in QUALIFYING_TRAIN_FAMILIES
    ]
    val = [u for u in usable if u["family"] in QUALIFYING_VAL_FAMILIES]
    train_per = {f: 0 for f in QUALIFYING_TRAIN_FAMILIES}
    for u in train:
        train_per[u["family"]] += 1
    train_ok = (
        len(train) >= TRAIN_MIN_USABLE
        and all(v >= TRAIN_MIN_PER_FAMILY for v in train_per.values())
    )
    val_ok = len(val) >= VAL_MIN_USABLE
    return {
        "train": {
            "n_usable": len(train),
            "per_family": train_per,
            "min_overall": TRAIN_MIN_USABLE,
            "min_per_family": TRAIN_MIN_PER_FAMILY,
            "passed": train_ok,
        },
        "validation": {
            "n_usable": len(val),
            "per_family": {QUALIFYING_VAL_FAMILIES[0]: len(val)},
            "min_usable": VAL_MIN_USABLE,
            "passed": val_ok,
        },
        "passed": train_ok and val_ok,
    }


def spearman(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None

    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    denx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    deny = math.sqrt(sum((b - my) ** 2 for b in ry))
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)


def reproducibility_diagnostics(
    per_prompt: list[dict[str, Any]],
) -> dict[str, Any]:
    pairs = [
        p
        for p in per_prompt
        if p.get("phase17_s2_alternate_fraction") is not None
        and p.get("phase18_alternate_fraction") is not None
    ]
    if not pairs:
        return {
            "n_paired": 0,
            "pearson_alt_fraction": None,
            "spearman_alt_fraction": None,
            "median_abs_alt_diff": None,
            "n_switched_record_heavy_to_alternate_heavy": 0,
            "n_switched_alternate_heavy_to_record_heavy": 0,
        }
    xs = [p["phase17_s2_alternate_fraction"] for p in pairs]
    ys = [p["phase18_alternate_fraction"] for p in pairs]
    abs_diffs = sorted(abs(a - b) for a, b in zip(xs, ys, strict=True))
    mid = len(abs_diffs) // 2
    med = (
        abs_diffs[mid]
        if len(abs_diffs) % 2
        else (abs_diffs[mid - 1] + abs_diffs[mid]) / 2
    )
    sw_r_a = sum(1 for a, b in zip(xs, ys, strict=True) if a < 0.5 <= b)
    sw_a_r = sum(1 for a, b in zip(xs, ys, strict=True) if a >= 0.5 > b)
    return {
        "n_paired": len(pairs),
        "pearson_alt_fraction": pearson(xs, ys),
        "spearman_alt_fraction": spearman(xs, ys),
        "median_abs_alt_diff": med,
        "n_switched_record_heavy_to_alternate_heavy": sw_r_a,
        "n_switched_alternate_heavy_to_record_heavy": sw_a_r,
    }


__all__ = [
    "PHASE17_STATUS",
    "PHASE17_CONFIRMED_IDS_SHA256",
    "QUALIFYING_FAMILIES",
    "QUALIFYING_TRAIN_FAMILIES",
    "QUALIFYING_VAL_FAMILIES",
    "N_COHORT",
    "N_SAMPLES",
    "N_CONTINUATIONS",
    "TEMPERATURE",
    "TOP_P",
    "SEED_BASE",
    "THRESHOLD_MANIFEST",
    "THRESHOLD_HASH",
    "STATUS_AUTHORIZED",
    "STATUS_PASS",
    "STATUS_HOLD",
    "GUARANTEE",
    "PHASE15_RULE_HASH",
    "PHASE15_RULE_VERSION",
    "load_phase17_confirmed",
    "select_enriched_cohort",
    "build_fresh_schedule",
    "summarize_prompt_group",
    "is_fresh_usable",
    "evaluate_gates",
    "reproducibility_diagnostics",
    "annotate_continuation",
    "_sha_ids",
]
