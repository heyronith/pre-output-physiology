"""Phase 19A — unseen multi-family validation cohort screen.

Behavior-only. Builds 6 never-before-used low-stakes families (24 prompts each),
S1/S2 screens them for policy-unstable prompts, then fresh-confirms a
SHA-selected 12-prompt validation cohort. Phase-18 train prompts untouched.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

from pre_output_physiology.phase15_onset import (
    CAT_ASSIGN,
    LABEL_ALTERNATE,
    LABEL_RECORD,
    RULE_HASH,
    RULE_VERSION,
    annotate_continuation,
)
from pre_output_physiology.phase17_screen import policy_entropy
from pre_output_physiology.phase18_cohort import (
    FRESH_MIN_ALTERNATE,
    FRESH_MIN_RECORD,
    FRESH_MIN_VALID,
)
from pre_output_physiology.phase18_cohort import (
    STATUS_HOLD as PHASE18_STATUS,
)

PHASE15_RULE_HASH = RULE_HASH
PHASE15_RULE_VERSION = RULE_VERSION

PHASE18_STATUS_REQUIRED = PHASE18_STATUS  # validation-gate HOLD

# --- Six completely new families (never used in prior phases) ---
NEW_FAMILIES: tuple[str, ...] = (
    "florist_cooler_shelf",
    "hardware_pegboard_hook",
    "community_pool_lane",
    "music_school_practice_room",
    "toy_library_checkout_bin",
    "mini_golf_tee_pad",
)

_L16 = list("ABCDEFGHJKLMNPQR")
_STATES = range(1, 10)

FAMILY_VOCAB: dict[str, dict[str, object]] = {
    "florist_cooler_shelf": {
        "entities": [f"bouquet order {c}" for c in _L16],
        "states": [f"shelf {n}" for n in _STATES],
        "question": "Which shelf holds the bouquet?",
        "topic_fmt": "A florist cooler chart records which shelf holds {entity}.",
    },
    "hardware_pegboard_hook": {
        "entities": [f"tool tag {c}" for c in _L16],
        "states": [f"hook {n}" for n in _STATES],
        "question": "Which hook holds the tool?",
        "topic_fmt": "A hardware shop pegboard log records which hook holds {entity}.",
    },
    "community_pool_lane": {
        "entities": [f"swim class {c}" for c in _L16],
        "states": [f"lane {n}" for n in _STATES],
        "question": "Which lane hosts the class?",
        "topic_fmt": "A community pool schedule records which lane hosts {entity}.",
    },
    "music_school_practice_room": {
        "entities": [f"student block {c}" for c in _L16],
        "states": [f"room {n}" for n in _STATES],
        "question": "Which room hosts the block?",
        "topic_fmt": (
            "A music school booking board records which room hosts {entity}."
        ),
    },
    "toy_library_checkout_bin": {
        "entities": [f"toy kit {c}" for c in _L16],
        "states": [f"bin {n}" for n in _STATES],
        "question": "Which bin holds the kit?",
        "topic_fmt": "A toy library checkout log records which bin holds {entity}.",
    },
    "mini_golf_tee_pad": {
        "entities": [f"party group {c}" for c in _L16],
        "states": [f"pad {n}" for n in _STATES],
        "question": "Which pad starts the group?",
        "topic_fmt": "A mini-golf starter sheet records which pad starts {entity}.",
    },
}

N_PER_FAMILY = 24
N_SCREEN_PROMPTS = N_PER_FAMILY * len(NEW_FAMILIES)  # 144
SCENARIO_SEED = 19_140_001
ID_PREFIX = "p19"

TEMPERATURE = 0.9
TOP_P = 0.95
N_SAMPLES_SCREEN = 16
N_S1_CONTINUATIONS = N_SCREEN_PROMPTS * N_SAMPLES_SCREEN  # 2304

S1_SEED_BASE = 19_000_000
S2_SEED_BASE = 19_100_000
FRESH_SEED_BASE = 19_200_000

# S1 / S2 thresholds (same as Phase 17; frozen before Phase-19 calls).
S1_MIN_VALID = 14
S1_MIN_RECORD = 2
S1_MIN_ALTERNATE = 2
S2_MIN_VALID = 14
S2_MIN_RECORD = 3
S2_MIN_ALTERNATE = 3

MIN_CONFIRMED_PER_QUALIFYING_FAMILY = 6
MIN_QUALIFYING_FAMILIES = 2
N_SELECTED_FAMILIES = 2
N_PROMPTS_PER_SELECTED_FAMILY = 6
N_SELECTED_PROMPTS = N_SELECTED_FAMILIES * N_PROMPTS_PER_SELECTED_FAMILY  # 12
N_SAMPLES_FRESH = 48
N_FRESH_CONTINUATIONS = N_SELECTED_PROMPTS * N_SAMPLES_FRESH  # 576

FAMILY_SELECT_SALT = "phase19_validation_family_v1|"
PROMPT_SELECT_SALT = "phase19_validation_prompt_v1|"

# Fresh usability (unchanged from Phase 18).
assert FRESH_MIN_VALID == 42
assert FRESH_MIN_RECORD == 6
assert FRESH_MIN_ALTERNATE == 6
FRESH_MIN_PER_FAMILY = 5  # of 6

STATUS_AUTHORIZED = "phase19a_unseen_validation_family_screen_authorized"
STATUS_S1_DONE = "phase19a_unseen_validation_family_screen_s1_complete_awaiting_s2"
STATUS_S2_DONE_AWAITING_FRESH = (
    "phase19a_unseen_validation_family_screen_s2_complete_awaiting_fresh"
)
STATUS_FAMILY_HOLD = "phase19a_unseen_validation_family_screen_hold"
STATUS_COHORT_HOLD = "phase19a_unseen_validation_cohort_hold"
STATUS_PASS = "phase19a_unseen_validation_cohort_pass_awaiting_audit"

GUARANTEE = (
    "PHASE 18A REMAINS A VALIDATION-GATE HOLD. PHASE 19A IS A NEW PROSPECTIVE "
    "ATTEMPT TO BUILD A MULTI-FAMILY UNSEEN VALIDATION COHORT. FAILURE OF THE "
    "FROZEN PHASE-19 FAMILY OR FRESH-CONFIRMATION GATES RETIRES THIS SAME-PROMPT "
    "SYNTHETIC ASSAY RATHER THAN TRIGGERING FURTHER TUNING."
)


def _sha_json(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha_ids(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def s1_sample_seed(prompt_index: int, sample_index: int) -> int:
    return S1_SEED_BASE + prompt_index * N_SAMPLES_SCREEN + sample_index


def s2_sample_seed(candidate_index: int, sample_index: int) -> int:
    return S2_SEED_BASE + candidate_index * N_SAMPLES_SCREEN + sample_index


def fresh_sample_seed(prompt_index: int, sample_index: int) -> int:
    return FRESH_SEED_BASE + prompt_index * N_SAMPLES_FRESH + sample_index


def build_s1_schedule(prompts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(prompts, key=lambda p: p["prompt_group_id"])
    if len(ordered) != N_SCREEN_PROMPTS:
        raise ValueError(f"expected {N_SCREEN_PROMPTS} prompts")
    rows = []
    for pi, p in enumerate(ordered):
        for si in range(N_SAMPLES_SCREEN):
            seed = s1_sample_seed(pi, si)
            cid = f"p19s1|{p['prompt_group_id']}|t{TEMPERATURE}|s{si}|seed{seed}"
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
    candidate_ids: Sequence[str],
    by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    ordered = sorted(candidate_ids)
    rows = []
    for ci, pg in enumerate(ordered):
        p = by_id[pg]
        for si in range(N_SAMPLES_SCREEN):
            seed = s2_sample_seed(ci, si)
            cid = f"p19s2|{pg}|t{TEMPERATURE}|s{si}|seed{seed}"
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


def build_fresh_schedule(
    selected_prompts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ordered = sorted(selected_prompts, key=lambda p: p["prompt_group_id"])
    if len(ordered) != N_SELECTED_PROMPTS:
        raise ValueError(f"expected {N_SELECTED_PROMPTS} selected prompts")
    rows = []
    for pi, p in enumerate(ordered):
        for si in range(N_SAMPLES_FRESH):
            seed = fresh_sample_seed(pi, si)
            cid = f"p19f|{p['prompt_group_id']}|t{TEMPERATURE}|s{si}|seed{seed}"
            rows.append(
                {
                    "continuation_id": cid,
                    "stage": "P19A",
                    "prompt_group_id": p["prompt_group_id"],
                    "family": p["family"],
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "sample_index": si,
                    "sample_seed": seed,
                    "prompt_index": pi,
                }
            )
    if len(rows) != N_FRESH_CONTINUATIONS:
        raise ValueError("fresh schedule size")
    return rows


def threshold_manifest() -> dict[str, Any]:
    body = {
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "n_screen_prompts": N_SCREEN_PROMPTS,
        "n_per_family": N_PER_FAMILY,
        "new_families": list(NEW_FAMILIES),
        "n_samples_screen": N_SAMPLES_SCREEN,
        "n_s1_continuations": N_S1_CONTINUATIONS,
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
        "family_qualify_rule": {
            "min_confirmed": MIN_CONFIRMED_PER_QUALIFYING_FAMILY,
            "min_qualifying_families": MIN_QUALIFYING_FAMILIES,
        },
        "selection": {
            "family_salt": FAMILY_SELECT_SALT,
            "prompt_salt": PROMPT_SELECT_SALT,
            "n_families": N_SELECTED_FAMILIES,
            "n_prompts_per_family": N_PROMPTS_PER_SELECTED_FAMILY,
        },
        "fresh_usability": {
            "min_valid": FRESH_MIN_VALID,
            "min_record": FRESH_MIN_RECORD,
            "min_alternate": FRESH_MIN_ALTERNATE,
            "min_per_family": FRESH_MIN_PER_FAMILY,
        },
        "s1_seed_base": S1_SEED_BASE,
        "s2_seed_base": S2_SEED_BASE,
        "fresh_seed_base": FRESH_SEED_BASE,
        "phase18_status_unchanged": PHASE18_STATUS_REQUIRED,
        "scenario_seed": SCENARIO_SEED,
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
    n_assign = sum(1 for r in annotated if r.get("onset_category") == CAT_ASSIGN)
    n_valid = n_rec + n_alt
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


def per_prompt_summaries(annotated: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by: dict[str, list] = defaultdict(list)
    for r in annotated:
        by[r["prompt_group_id"]].append(r)
    out = {}
    for pg, rows in by.items():
        s = summarize_prompt_group(rows)
        s["prompt_group_id"] = pg
        s["family"] = rows[0]["family"]
        out[pg] = s
    return out


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


def is_fresh_usable(summary: dict[str, Any]) -> bool:
    return (
        summary["n_valid"] >= FRESH_MIN_VALID
        and summary["n_record"] >= FRESH_MIN_RECORD
        and summary["n_alternate"] >= FRESH_MIN_ALTERNATE
    )


def family_confirmed_counts(
    confirmed: list[dict[str, Any]],
) -> dict[str, int]:
    c = Counter(x["family"] for x in confirmed)
    return {f: c.get(f, 0) for f in NEW_FAMILIES}


def qualifying_families(confirmed: list[dict[str, Any]]) -> list[str]:
    counts = family_confirmed_counts(confirmed)
    return sorted(
        f for f, n in counts.items() if n >= MIN_CONFIRMED_PER_QUALIFYING_FAMILY
    )


def select_validation_cohort(
    confirmed: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return SHA-selected 2 families × 6 prompts, or None if <2 qualify."""
    qual = qualifying_families(confirmed)
    if len(qual) < MIN_QUALIFYING_FAMILIES:
        return None
    scored = [
        (hashlib.sha256(f"{FAMILY_SELECT_SALT}{f}".encode()).hexdigest(), f)
        for f in qual
    ]
    scored.sort(key=lambda x: x[0])
    selected_fams = [f for _, f in scored[:N_SELECTED_FAMILIES]]
    by_fam: dict[str, list] = defaultdict(list)
    for c in confirmed:
        if c["family"] in selected_fams:
            by_fam[c["family"]].append(c)
    selected_prompts: list[dict[str, Any]] = []
    for fam in selected_fams:
        scored_p = [
            (
                hashlib.sha256(
                    f"{PROMPT_SELECT_SALT}{c['prompt_group_id']}".encode()
                ).hexdigest(),
                c,
            )
            for c in by_fam[fam]
        ]
        scored_p.sort(key=lambda x: x[0])
        for dig, c in scored_p[:N_PROMPTS_PER_SELECTED_FAMILY]:
            selected_prompts.append(
                {
                    "prompt_group_id": c["prompt_group_id"],
                    "family": fam,
                    "selection_digest": dig,
                    "phase19_s2_n_record": c["n_record"],
                    "phase19_s2_n_alternate": c["n_alternate"],
                    "phase19_s2_n_valid": c["n_valid"],
                    "phase19_s2_alternate_fraction": c.get(
                        "alternate_fraction_among_valid"
                    ),
                }
            )
    ids = sorted(p["prompt_group_id"] for p in selected_prompts)
    return {
        "qualifying_families": qual,
        "qualifying_family_digests": {f: d for d, f in scored},
        "selected_families": selected_fams,
        "selected_family_digests": {
            f: hashlib.sha256(f"{FAMILY_SELECT_SALT}{f}".encode()).hexdigest()
            for f in selected_fams
        },
        "n_confirmed_by_family": family_confirmed_counts(confirmed),
        "selected_prompts": sorted(selected_prompts, key=lambda p: p["prompt_group_id"]),
        "selected_prompt_group_ids": ids,
        "cohort_sha256": _sha_ids(ids),
        "n_selected": len(ids),
    }


def evaluate_fresh_gates(usable: list[dict[str, Any]]) -> dict[str, Any]:
    per = {f: 0 for f in sorted({u["family"] for u in usable})}
    # Ensure both expected families appear even if zero.
    for u in usable:
        per[u["family"]] = per.get(u["family"], 0) + 1
    passed = (
        len(per) >= N_SELECTED_FAMILIES
        and all(v >= FRESH_MIN_PER_FAMILY for v in per.values())
        and len(usable) >= FRESH_MIN_PER_FAMILY * N_SELECTED_FAMILIES
    )
    # Stricter: every selected family must have ≥5.
    # Caller should pass expected_families.
    return {
        "per_family": per,
        "min_per_family": FRESH_MIN_PER_FAMILY,
        "n_usable": len(usable),
        "passed": passed,
    }


def evaluate_fresh_gates_for_families(
    usable: list[dict[str, Any]],
    expected_families: Sequence[str],
) -> dict[str, Any]:
    per = {f: 0 for f in expected_families}
    for u in usable:
        if u["family"] in per:
            per[u["family"]] += 1
    passed = all(per[f] >= FRESH_MIN_PER_FAMILY for f in expected_families)
    return {
        "per_family": per,
        "min_per_family": FRESH_MIN_PER_FAMILY,
        "n_usable": len(usable),
        "passed": passed,
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
    mx, my = sum(rx) / n, sum(ry) / n
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
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)


__all__ = [
    "NEW_FAMILIES",
    "FAMILY_VOCAB",
    "N_SCREEN_PROMPTS",
    "N_S1_CONTINUATIONS",
    "N_SELECTED_PROMPTS",
    "N_FRESH_CONTINUATIONS",
    "N_SAMPLES_SCREEN",
    "N_SAMPLES_FRESH",
    "TEMPERATURE",
    "TOP_P",
    "THRESHOLD_MANIFEST",
    "THRESHOLD_HASH",
    "STATUS_AUTHORIZED",
    "STATUS_S1_DONE",
    "STATUS_S2_DONE_AWAITING_FRESH",
    "STATUS_FAMILY_HOLD",
    "STATUS_COHORT_HOLD",
    "STATUS_PASS",
    "GUARANTEE",
    "PHASE15_RULE_HASH",
    "PHASE18_STATUS_REQUIRED",
    "build_s1_schedule",
    "build_s2_schedule",
    "build_fresh_schedule",
    "summarize_prompt_group",
    "per_prompt_summaries",
    "is_s1_candidate",
    "is_s2_confirmed",
    "is_fresh_usable",
    "qualifying_families",
    "select_validation_cohort",
    "evaluate_fresh_gates_for_families",
    "annotate_continuation",
    "_sha_ids",
]
