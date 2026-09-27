"""Phase 15A — proposition-onset Stage-1 reanalysis of Phase-14 calibration.

Text-only annotation. No model calls, activations, or learned classifiers.
Rules are frozen before recomputing behavior statistics.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from typing import Any

from pre_output_physiology.phase14_design import (
    LABEL_ALTERNATE,
    LABEL_INVALID,
    LABEL_LEAK,
    LABEL_RECORD,
    TEMPERATURE_GRID,
    _assignment_leak,
    classify_stage1,
    label_continuation,
)

PHASE14_RUN_ID = "phase14a_calibration_20260926T231646Z_e6206543"
PHASE14_RAW_CONTINUATIONS_SHA256 = (
    "b1c003b3282387ef43e0e3d8b31a14978fbfe32105eac8cdffd740ff19aa44b8"
)
N_CONTINUATIONS = 1536
N_PROMPT_GROUPS = 32

RULE_VERSION = "phase15_onset_v1"

# Categories (activation-independent; Stage-1 text + candidate strings only).
CAT_CLEAN = "clean_no_candidate_mention"
CAT_MENTION = "candidate_mention_only"
CAT_ASSIGN = "candidate_assignment_proposition"
CAT_INVALID = "invalid_other"

# Feasibility (diagnostic; does NOT select T*).
MIN_RECORD_FRACTION = 0.25
MIN_ALTERNATE_FRACTION = 0.25
MIN_PROMPT_GROUPS_GE3 = 16
MIN_VALID_FRACTION = 0.90
MAX_ASSIGNMENT_LEAK_RATE = 0.10

STATUS_AUTHORIZED = "phase15a_same_prompt_onset_reanalysis_authorized"
STATUS_PROMISING = "phase15a_same_prompt_onset_reanalysis_promising_awaiting_audit"
STATUS_HOLD = "phase15a_same_prompt_onset_reanalysis_hold"

GUARANTEE = (
    "PHASE 14A REMAINS A HOLD UNDER ITS ORIGINAL FROZEN RULES. PHASE 15A ONLY "
    "REANALYZED THE EXISTING CALIBRATION TRAJECTORIES USING A PROPOSITION-ONSET "
    "VALIDITY DEFINITION FROZEN BEFORE RECOMPUTING BEHAVIOR STATISTICS. NO MODEL "
    "CALLS, ACTIVATIONS, PROBES, FINAL/LOCKED PROMPTS, RESAMPLING, PROMPT TUNING, "
    "TEMPERATURE EXTENSION, OR CAUSAL INTERVENTIONS WERE PERFORMED."
)


def _sha_json(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


# Linguistic assignment patterns (form only; audited against Stage-1 vocabulary).
# Each pattern uses {esc} for the escaped candidate state.
_ASSIGNMENT_PATTERN_SPECS: tuple[tuple[str, str], ...] = (
    # Phase-14 base patterns (retained).
    ("prep_state", r"\b(?:is|at|in|on|to|for)\s+{esc}\b"),
    ("state_verb", r"\b{esc}\s+(?:is|holds|boards|serves|houses|admits)\b"),
    ("choose_state", r"\b(?:choose|select|pick|answer|response)\s+(?:is\s+)?{esc}\b"),
    ("goes_to_state", r"\bgoes\s+(?:to|at)\s+{esc}\b"),
    # Phase-15 coverage additions from Stage-1 vocabulary audit (linguistic form).
    ("head_to_state", r"\bhead(?:ing)?\s+(?:to|towards)\s+{esc}\b"),
    ("go_to_state", r"\bgo(?:ing)?\s+to\s+{esc}\b"),
    ("checking_state", r"\bcheck(?:ing)?\s+{esc}\b"),
    ("state_last_used", r"\b{esc}\s+was\s+last\s+used\b"),
    ("last_used_state", r"\blast\s+used\b[^.]*\b{esc}\b"),
    ("responsible_state", r"\b{esc}\s+is\s+responsible\b"),
    ("preference_for_state", r"\bpreference\s+for\s+{esc}\b"),
    ("near_state", r"\b(?:near|located\s+near)\s+{esc}\b"),
    ("redirect_to_state", r"\bredirect(?:\s+\w+){0,3}\s+to\s+{esc}\b"),
    ("next_action_at_state", r"\bnext\s+action\b[^.]*\b(?:at|to)\s+{esc}\b"),
    ("take_place_at_state",
     r"\btake\s+(?:their\s+)?(?:next\s+)?(?:action\s+)?(?:place\s+)?at\s+{esc}\b"),
    ("action_at_state",
     r"\b(?:perform|prepare\s+for)\s+(?:an\s+)?"
     r"(?:action|their\s+next\s+action)\s+at\s+{esc}\b"),
    ("headed_to_state", r"\bheaded\s+to\s+{esc}\b"),
    ("use_state", r"\b(?:use|using|used)\s+{esc}\b"),
)


def assignment_pattern_catalog() -> list[dict[str, str]]:
    return [{"name": n, "pattern": p} for n, p in _ASSIGNMENT_PATTERN_SPECS]


def rule_manifest() -> dict[str, Any]:
    """Frozen before recomputing any behavior statistics."""
    catalog = assignment_pattern_catalog()
    body = {
        "rule_version": RULE_VERSION,
        "categories": [CAT_CLEAN, CAT_MENTION, CAT_ASSIGN, CAT_INVALID],
        "valid_pre_answer": [CAT_CLEAN, CAT_MENTION],
        "invalid_pre_answer": [CAT_ASSIGN, CAT_INVALID],
        "assignment_patterns": catalog,
        "notes": (
            "Exact candidate substring without a matched assignment pattern is "
            "candidate_mention_only. Disjunctive either/or comparisons without a "
            "singular assignment pattern are mention-only. Annotation uses only "
            "Stage-1 text plus record/alternate strings."
        ),
    }
    return {**body, "rule_hash": _sha_json(body)}


RULE_MANIFEST = rule_manifest()
RULE_HASH = RULE_MANIFEST["rule_hash"]


def find_assignment_propositions(
    text: str, record: str, alternate: str
) -> list[dict[str, str]]:
    """Return matched assignment hits: pattern name + implicated candidate."""
    low = (text or "").lower()
    hits: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for name, tmpl in _ASSIGNMENT_PATTERN_SPECS:
        for cand, label in ((record, "record"), (alternate, "alternate")):
            pat = tmpl.replace("{esc}", re.escape(cand.lower()))
            if re.search(pat, low):
                key = (name, label)
                if key not in seen:
                    seen.add(key)
                    hits.append(
                        {
                            "pattern_name": name,
                            "pattern": tmpl,
                            "candidate_implicated": cand,
                            "candidate_role": label,
                        }
                    )
    return hits


def classify_onset_stage1(
    consideration_text: str, record: str, alternate: str
) -> dict[str, Any]:
    """Phase-15 onset category from Stage-1 text only (no choice/activations)."""
    text = (consideration_text or "").strip()
    if not text:
        return {
            "onset_category": CAT_INVALID,
            "valid_pre_answer": False,
            "exact_candidate_mention": False,
            "assignment_hits": [],
            "matched_proposition_patterns": [],
            "candidates_implicated": [],
            "reasons": ["empty_consideration"],
        }
    hits = find_assignment_propositions(text, record, alternate)
    mention = (record in text) or (alternate in text)
    if hits:
        return {
            "onset_category": CAT_ASSIGN,
            "valid_pre_answer": False,
            "exact_candidate_mention": mention,
            "assignment_hits": hits,
            "matched_proposition_patterns": [h["pattern_name"] for h in hits],
            "candidates_implicated": sorted({h["candidate_implicated"] for h in hits}),
            "reasons": ["assignment_proposition"],
        }
    if mention:
        return {
            "onset_category": CAT_MENTION,
            "valid_pre_answer": True,
            "exact_candidate_mention": True,
            "assignment_hits": [],
            "matched_proposition_patterns": [],
            "candidates_implicated": [],
            "reasons": ["exact_candidate_substring_without_assignment"],
        }
    return {
        "onset_category": CAT_CLEAN,
        "valid_pre_answer": True,
        "exact_candidate_mention": False,
        "assignment_hits": [],
        "matched_proposition_patterns": [],
        "candidates_implicated": [],
        "reasons": [],
    }


def original_phase14_flags(
    consideration_text: str, record: str, alternate: str
) -> dict[str, Any]:
    """Preserve Phase-14 Stage-1 flags without overwriting them later."""
    s1 = classify_stage1(consideration_text, record, alternate)
    return {
        "phase14_exact_candidate_substring": "exact_candidate_substring" in s1["reasons"],
        "phase14_assignment_proposition": "assignment_proposition" in s1["reasons"],
        "phase14_stage1_reasons": list(s1["reasons"]),
        "phase14_stage1_leak": bool(s1["leak"]),
    }


def annotate_continuation(row: dict[str, Any]) -> dict[str, Any]:
    """Attach Phase-15 onset labels while retaining original Phase-14 labels."""
    consideration = row["consideration_text"]
    record = row["record_state"]
    alternate = row["alternate_state"]
    stage1_ok = bool(consideration) and bool(row.get("stage1_n_tokens"))
    p14 = label_continuation(
        consideration_text=consideration,
        chosen_state=row["chosen_state"],
        record=record,
        alternate=alternate,
        stage1_ok_format=stage1_ok,
    )
    flags = original_phase14_flags(consideration, record, alternate)
    onset = classify_onset_stage1(consideration, record, alternate)
    # Final policy from deterministic Stage-2 choice (not used in onset annotation).
    if row["chosen_state"] == record:
        policy = LABEL_RECORD
    elif row["chosen_state"] == alternate:
        policy = LABEL_ALTERNATE
    else:
        policy = LABEL_INVALID
    return {
        "continuation_id": row["continuation_id"],
        "prompt_group_id": row["prompt_group_id"],
        "family": row["family"],
        "temperature": float(row["temperature"]),
        "sample_index": row["sample_index"],
        "sample_seed": row["sample_seed"],
        "record_state": record,
        "alternate_state": alternate,
        "chosen_state": row["chosen_state"],
        "consideration_text": consideration,
        # Original Phase-14 (never overwritten).
        "phase14_label": p14["label"],
        "phase14_valid_nonleaking": p14["valid_nonleaking"],
        **flags,
        # Phase-15 onset.
        "onset_category": onset["onset_category"],
        "phase15_valid_pre_answer": onset["valid_pre_answer"],
        "matched_proposition_patterns": onset["matched_proposition_patterns"],
        "candidates_implicated": onset["candidates_implicated"],
        "assignment_hits": onset["assignment_hits"],
        "onset_reasons": onset["reasons"],
        "final_policy_label": policy,
        "rule_version": RULE_VERSION,
        "rule_hash": RULE_HASH,
    }


def _policy_fractions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rec = sum(1 for r in rows if r["final_policy_label"] == LABEL_RECORD)
    alt = sum(1 for r in rows if r["final_policy_label"] == LABEL_ALTERNATE)
    n = rec + alt
    return {
        "n": n,
        "n_record": rec,
        "n_alternate": alt,
        "record_fraction": (rec / n) if n else None,
        "alternate_fraction": (alt / n) if n else None,
    }


def rates_by_temperature(annotated: list[dict[str, Any]]) -> dict[float, dict[str, Any]]:
    out: dict[float, dict[str, Any]] = {}
    for t in TEMPERATURE_GRID:
        subset = [r for r in annotated if float(r["temperature"]) == float(t)]
        n = len(subset)
        valid = [r for r in subset if r["phase15_valid_pre_answer"]]
        assign = [r for r in subset if r["onset_category"] == CAT_ASSIGN]
        by_pg: dict[str, Counter[str]] = defaultdict(Counter)
        for r in valid:
            if r["final_policy_label"] in (LABEL_RECORD, LABEL_ALTERNATE):
                by_pg[r["prompt_group_id"]][r["final_policy_label"]] += 1
        ge1 = sum(
            1
            for c in by_pg.values()
            if c[LABEL_RECORD] >= 1 and c[LABEL_ALTERNATE] >= 1
        )
        ge3 = sum(
            1
            for c in by_pg.values()
            if c[LABEL_RECORD] >= 3 and c[LABEL_ALTERNATE] >= 3
        )
        valid_counts = [sum(c.values()) for c in by_pg.values()]
        # Include prompts with zero valid as 0 for median over all 32 groups.
        while len(valid_counts) < N_PROMPT_GROUPS:
            valid_counts.append(0)
        valid_counts_sorted = sorted(valid_counts)
        mid = len(valid_counts_sorted) // 2
        if len(valid_counts_sorted) % 2:
            median_valid = float(valid_counts_sorted[mid])
        else:
            median_valid = (
                valid_counts_sorted[mid - 1] + valid_counts_sorted[mid]
            ) / 2.0
        pol = _policy_fractions(valid)
        out[float(t)] = {
            "n": n,
            "n_valid": len(valid),
            "n_assignment_proposition": len(assign),
            "valid_fraction": len(valid) / n if n else 0.0,
            "assignment_proposition_leak_rate": len(assign) / n if n else 0.0,
            "record_fraction": pol["record_fraction"],
            "alternate_fraction": pol["alternate_fraction"],
            "n_record": pol["n_record"],
            "n_alternate": pol["n_alternate"],
            "n_prompt_groups_ge1_each_class": ge1,
            "n_prompt_groups_ge3_each_class": ge3,
            "median_valid_continuations_per_prompt": median_valid,
            "onset_category_counts": dict(Counter(r["onset_category"] for r in subset)),
        }
    return out


def replication_worthy(rates: dict[float, dict[str, Any]]) -> dict[str, Any]:
    eligible: list[float] = []
    details: dict[str, Any] = {}
    for t in TEMPERATURE_GRID:
        r = rates[float(t)]
        rec = r["record_fraction"]
        alt = r["alternate_fraction"]
        ok = (
            rec is not None
            and alt is not None
            and rec >= MIN_RECORD_FRACTION
            and alt >= MIN_ALTERNATE_FRACTION
            and r["n_prompt_groups_ge3_each_class"] >= MIN_PROMPT_GROUPS_GE3
            and r["valid_fraction"] >= MIN_VALID_FRACTION
            and r["assignment_proposition_leak_rate"] <= MAX_ASSIGNMENT_LEAK_RATE
        )
        details[str(t)] = {
            "replication_worthy": ok,
            "checks": {
                "record_ge_0.25": rec is not None and rec >= MIN_RECORD_FRACTION,
                "alternate_ge_0.25": alt is not None and alt >= MIN_ALTERNATE_FRACTION,
                "ge3_groups_ge_16": r["n_prompt_groups_ge3_each_class"]
                >= MIN_PROMPT_GROUPS_GE3,
                "valid_ge_0.90": r["valid_fraction"] >= MIN_VALID_FRACTION,
                "assign_leak_le_0.10": r["assignment_proposition_leak_rate"]
                <= MAX_ASSIGNMENT_LEAK_RATE,
            },
        }
        if ok:
            eligible.append(float(t))
    return {
        "replication_worthy_temperatures": eligible,
        "details": details,
        "selects_t_star": False,
    }


def policy_by_onset_category(
    annotated: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for t in TEMPERATURE_GRID:
        per_cat: dict[str, Any] = {}
        subset = [r for r in annotated if float(r["temperature"]) == float(t)]
        for cat in (CAT_CLEAN, CAT_MENTION, CAT_ASSIGN):
            rows = [r for r in subset if r["onset_category"] == cat]
            per_cat[cat] = _policy_fractions(rows)
        out[str(t)] = per_cat
    return out


def recovery_vs_phase14(annotated: list[dict[str, Any]]) -> dict[str, Any]:
    """How many alternate samples/groups become newly usable under Phase-15 validity."""
    by_t: dict[str, Any] = {}
    for t in TEMPERATURE_GRID:
        subset = [r for r in annotated if float(r["temperature"]) == float(t)]
        newly = [
            r
            for r in subset
            if r["phase15_valid_pre_answer"] and not r["phase14_valid_nonleaking"]
        ]
        newly_alt = [r for r in newly if r["final_policy_label"] == LABEL_ALTERNATE]
        newly_rec = [r for r in newly if r["final_policy_label"] == LABEL_RECORD]
        # Per prompt-group: alternate counts under P14 vs P15 validity
        p14_alt: dict[str, int] = defaultdict(int)
        p15_alt: dict[str, int] = defaultdict(int)
        for r in subset:
            pg = r["prompt_group_id"]
            if r["final_policy_label"] != LABEL_ALTERNATE:
                continue
            if r["phase14_valid_nonleaking"]:
                p14_alt[pg] += 1
            if r["phase15_valid_pre_answer"]:
                p15_alt[pg] += 1
        groups_gained_alt = sum(
            1
            for pg in set(p14_alt) | set(p15_alt)
            if p15_alt[pg] > p14_alt[pg]
        )
        groups_newly_with_alt = sum(
            1
            for pg in set(p14_alt) | set(p15_alt)
            if p14_alt[pg] == 0 and p15_alt[pg] > 0
        )
        by_t[str(t)] = {
            "newly_valid_continuations": len(newly),
            "newly_valid_record": len(newly_rec),
            "newly_valid_alternate": len(newly_alt),
            "prompt_groups_with_increased_alternate": groups_gained_alt,
            "prompt_groups_newly_gaining_any_alternate": groups_newly_with_alt,
        }
    return by_t


def cross_tab_validity(annotated: list[dict[str, Any]]) -> dict[str, int]:
    tab: Counter[str] = Counter()
    for r in annotated:
        key = (
            f"phase14_{'valid' if r['phase14_valid_nonleaking'] else 'invalid'}"
            f"|phase15_{'valid' if r['phase15_valid_pre_answer'] else 'invalid'}"
        )
        tab[key] += 1
    return dict(tab)


def coverage_audit_vs_phase14_assignment(annotated: list[dict[str, Any]]) -> dict[str, Any]:
    """Sanity: Phase-15 should catch every Phase-14 assignment flag (superset)."""
    p14_assign = [r for r in annotated if r["phase14_assignment_proposition"]]
    missed = [
        r["continuation_id"]
        for r in p14_assign
        if r["onset_category"] != CAT_ASSIGN
    ]
    return {
        "phase14_assignment_count": len(p14_assign),
        "phase15_covers_all_phase14_assignments": len(missed) == 0,
        "missed_continuation_ids": missed,
    }


# Re-export phase14 helpers used by scripts/tests
__all__ = [
    "PHASE14_RUN_ID",
    "PHASE14_RAW_CONTINUATIONS_SHA256",
    "N_CONTINUATIONS",
    "RULE_VERSION",
    "RULE_HASH",
    "RULE_MANIFEST",
    "CAT_CLEAN",
    "CAT_MENTION",
    "CAT_ASSIGN",
    "CAT_INVALID",
    "STATUS_AUTHORIZED",
    "STATUS_PROMISING",
    "STATUS_HOLD",
    "GUARANTEE",
    "TEMPERATURE_GRID",
    "LABEL_RECORD",
    "LABEL_ALTERNATE",
    "LABEL_LEAK",
    "LABEL_INVALID",
    "annotate_continuation",
    "classify_onset_stage1",
    "find_assignment_propositions",
    "rates_by_temperature",
    "replication_worthy",
    "policy_by_onset_category",
    "recovery_vs_phase14",
    "cross_tab_validity",
    "coverage_audit_vs_phase14_assignment",
    "rule_manifest",
    "_assignment_leak",
]
