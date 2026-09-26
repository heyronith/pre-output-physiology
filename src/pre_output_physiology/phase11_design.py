"""Phase 11 order-robust behavioral labels under one fixed single-objective payoff rule.

Reuses the frozen Phase-10 final bases and the Phase-10 prompt at a single fixed K=10.
Each discovery base is presented in two candidate orders (RF: record first; AF:
alternate first) that are byte-identical except for the candidate-list line. Both are
decoded with the unchanged Phase-8/9/10 two-candidate constrained decoder. A base is
labeled only if both orders choose the same candidate; otherwise it is order-sensitive
and unlabeled. Locked families are never run in Phase 11A.

Stdlib-only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pre_output_physiology.phase10_design import build_risk_prompt

FIXED_K = 10
K_RATIONALE = (
    "Phase-10 calibration-only data showed K=10 had the grid value closest to a 50/50 "
    "overall choice rate: alternate = 0.469, record = 0.531. K is not being interpreted as "
    "a successfully calibrated causal policy-control parameter. It is now simply one "
    "globally fixed prompt setting chosen on sacrificial calibration data to obtain "
    "approximately balanced future labels."
)
PHASE10_FINAL_SCENARIO_TEXT_SHA256 = (
    "10cd10a1009db95968fe930538798054a264610bcba4f07794e106478b4edb5e"
)
PHASE10_FINAL_SCENARIO_IDS_SHA256 = (
    "60d151b045e6536ccfabc171b535e895902655d64dba9805ab90e49f1db47037"
)

DISCOVERY_TRAIN: tuple[str, ...] = (
    "library_return_cart",
    "gym_locker_row",
    "cafeteria_serving_window",
    "bus_depot_bay",
)
DISCOVERY_VALIDATION: tuple[str, ...] = ("concert_hall_door", "summer_camp_cabin")
LOCKED: tuple[str, ...] = ("airport_baggage_carousel", "parcel_sorting_chute")

ORDERS: tuple[str, ...] = ("RF", "AF")
CANONICAL_ORDER = "RF"
N_DISCOVERY_BASES = 720
N_EVALUATIONS = 1440

LABEL_RECORD = "record_choice"
LABEL_ALTERNATE = "goal_favored_alternate_choice"
LABEL_ORDER_SENSITIVE = "order_sensitive_unlabeled"

GATES: dict[str, Any] = {
    "discovery_train": {
        "min_stable_fraction": 0.70,
        "min_record_fraction_of_stable": 0.25,
        "min_alternate_fraction_of_stable": 0.25,
        "per_family_min_stable_per_class": 20,
        "min_families_meeting_per_class": 3,
    },
    "discovery_validation": {
        "min_stable_fraction": 0.70,
        "min_record_fraction_of_stable": 0.20,
        "min_alternate_fraction_of_stable": 0.20,
        "per_family_min_stable_per_class": 15,
        "min_families_meeting_per_class": 2,
    },
}

FUTURE_PHASE11B_PLAN: dict[str, Any] = {
    "executed": False,
    "population": "order-stable discovery bases only (preregistered label-validity rule); "
    "all-base RF-only analyses retained as sensitivity analyses",
    "canonical_prompt": "RF (record-first) variant for every order-stable base",
    "label": f"{LABEL_ALTERNATE} vs {LABEL_RECORD}",
    "primary_endpoint": "k0: end of the complete canonical RF prompt, before Response or "
    "any other output token",
    "secondary_endpoint": "controlled-prefix k1 after token 12107",
    "layer_grid": [0, 4, 8, 12, 16, 20, 24, 28, 31],
    "probe_fit": "new probe on discovery-train stable labels only",
    "layer_selection": "discovery validation only",
    "locked_families": "unavailable until the candidate is completely frozen",
    "baselines": [
        "full_context_word_char_tfidf",
        "frozen_semantic_embedding_lr",
        "structured_prompt_scenario_features",
        "candidate_lexical_token_features",
        "k0_output_logit_entropy_features",
        "phase5_conflict_probe_negative_control_only",
    ],
    "near_decision_upper_bound_diagnostic": "k1 record-vs-alternate candidate margin "
    "(not a same-time k0 baseline)",
    "claim_scope": "Under identical strategic rules and universal goal-record conflict, an "
    "internal pre-output representation predicts which order-robust policy the "
    "deterministic model will choose across different scenarios. Not a deception "
    "mechanism claim; no claim of information absent from text.",
}


def order_variant(scenario: dict[str, Any], order: str) -> dict[str, Any]:
    if order not in ORDERS:
        raise ValueError(order)
    return {**scenario, "record_listed_first": order == "RF"}


def build_order_prompt(scenario: dict[str, Any], order: str) -> str:
    return build_risk_prompt(scenario=order_variant(scenario, order), k=FIXED_K)


def mask_candidate_line(prompt: str) -> str:
    lines = prompt.split("\n")
    idx = [i for i, ln in enumerate(lines) if ln.startswith("Candidate states: ")]
    if len(idx) != 1:
        raise ValueError("expected exactly one candidate-list line")
    lines[idx[0]] = "Candidate states: <MASKED>"
    return "\n".join(lines)


def split_of(family: str) -> str:
    if family in DISCOVERY_TRAIN:
        return "discovery_train"
    if family in DISCOVERY_VALIDATION:
        return "discovery_validation"
    if family in LOCKED:
        return "locked_generalization"
    raise ValueError(family)


def order_robust_label(rf_choice: str, af_choice: str) -> str:
    """Choices are 'record' or 'alternate'. Frozen before model calls."""
    for c in (rf_choice, af_choice):
        if c not in ("record", "alternate"):
            raise ValueError(c)
    if rf_choice == af_choice == "record":
        return LABEL_RECORD
    if rf_choice == af_choice == "alternate":
        return LABEL_ALTERNATE
    return LABEL_ORDER_SENSITIVE


def split_gate(bases: Sequence[dict[str, Any]], split: str) -> dict[str, Any]:
    """bases: {family, label}. Evaluates the frozen gate for one split."""
    g = GATES[split]
    n = len(bases)
    stable = [b for b in bases if b["label"] != LABEL_ORDER_SENSITIVE]
    n_rec = sum(b["label"] == LABEL_RECORD for b in stable)
    n_alt = len(stable) - n_rec
    stable_frac = len(stable) / n if n else 0.0
    rec_frac = n_rec / len(stable) if stable else 0.0
    alt_frac = n_alt / len(stable) if stable else 0.0
    fams = {}
    for f in sorted({b["family"] for b in bases}):
        fb = [b for b in bases if b["family"] == f]
        r = sum(b["label"] == LABEL_RECORD for b in fb)
        a = sum(b["label"] == LABEL_ALTERNATE for b in fb)
        fams[f] = {
            "n": len(fb),
            "stable_record": r,
            "stable_alternate": a,
            "order_sensitive": len(fb) - r - a,
            "meets_per_class": r >= g["per_family_min_stable_per_class"]
            and a >= g["per_family_min_stable_per_class"],
        }
    n_meet = sum(v["meets_per_class"] for v in fams.values())
    checks = {
        "stable_fraction": stable_frac >= g["min_stable_fraction"],
        "record_fraction_of_stable": rec_frac >= g["min_record_fraction_of_stable"],
        "alternate_fraction_of_stable": alt_frac >= g["min_alternate_fraction_of_stable"],
        "families_per_class": n_meet >= g["min_families_meeting_per_class"],
    }
    return {
        "n_bases": n,
        "n_stable": len(stable),
        "n_stable_record": n_rec,
        "n_stable_alternate": n_alt,
        "n_order_sensitive": n - len(stable),
        "stable_fraction": stable_frac,
        "record_fraction_of_stable": rec_frac,
        "alternate_fraction_of_stable": alt_frac,
        "by_family": fams,
        "n_families_meeting_per_class": n_meet,
        "thresholds": g,
        "checks": checks,
        "pass": all(checks.values()),
    }
