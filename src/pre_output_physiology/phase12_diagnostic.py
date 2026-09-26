"""Phase 12A family-bias source diagnostic (behavior only).

Phase 11A order-stable labels were almost fully determined by family (Cramér's V 0.905).
Phase 12A crosses family semantic shell (kept / generic) with family state namespace
(kept / common `slot`) on a deterministic, label-independent subset of 24 Phase-11
bases per discovery family:

    A FULL_FAMILY   = original Phase-10/11 prompt (reused Phase-11 RF/AF results)
    B SEMANTIC_ONLY = family shell and entity, state noun -> `slot`
    C STATE_ONLY    = generic shell and item ID, family state noun kept
    D FULL_NEUTRAL  = generic shell and item ID, `slot`

Record/alternate numbers, K=10, payoff rule, and decoder are unchanged. Stdlib-only.
"""

from __future__ import annotations

import hashlib
import math
import re
import string
from collections import Counter
from collections.abc import Sequence
from typing import Any

from pre_output_physiology.phase10_design import build_risk_prompt
from pre_output_physiology.phase11_design import (
    FIXED_K,
    LABEL_ALTERNATE,
    LABEL_ORDER_SENSITIVE,
    LABEL_RECORD,
    LOCKED,
)

FAMILIES: tuple[str, ...] = (
    "bus_depot_bay",
    "cafeteria_serving_window",
    "gym_locker_row",
    "library_return_cart",
    "concert_hall_door",
    "summer_camp_cabin",
)
STATE_NOUN: dict[str, str] = {
    "bus_depot_bay": "bay",
    "cafeteria_serving_window": "window",
    "gym_locker_row": "row",
    "library_return_cart": "cart",
    "concert_hall_door": "door",
    "summer_camp_cabin": "cabin",
}
NEUTRAL_NOUN = "slot"
# Model-visible family semantic/entity vocabulary (topic, question, entities).
FAMILY_SEMANTIC_TERMS: dict[str, tuple[str, ...]] = {
    "bus_depot_bay": ("bus", "depot", "route", "boards"),
    "cafeteria_serving_window": (
        "cafeteria", "menu", "serves", "dish", "chili", "soup", "cheese", "tacos",
        "thai", "stew", "curry", "burrito", "falafel", "risotto", "salad", "rice",
        "pasta", "chowder",
    ),
    "gym_locker_row": ("gym", "locker", "lockers", "swim", "squad", "squad's"),
    "library_return_cart": (
        "library", "returns", "returned", "book", "novel", "cookbook", "atlas",
        "poetry", "biography", "guide", "textbook", "volume", "workbook", "monograph",
        "almanac", "manual", "anthology",
    ),
    "concert_hall_door": ("concert", "hall", "ticket", "admits", "holders", "entry"),
    "summer_camp_cabin": ("summer", "camp", "camper", "houses", "roster", "group"),
}

CELLS: tuple[str, ...] = ("A", "B", "C", "D")
CELL_NAMES: dict[str, str] = {
    "A": "FULL_FAMILY",
    "B": "SEMANTIC_ONLY",
    "C": "STATE_ONLY",
    "D": "FULL_NEUTRAL",
}
NEW_CELLS: tuple[str, ...] = ("B", "C", "D")
ORDERS: tuple[str, ...] = ("RF", "AF")
N_PER_FAMILY = 24
N_BASES = N_PER_FAMILY * len(FAMILIES)  # 144
N_NEW_EVALUATIONS = N_BASES * len(NEW_CELLS) * len(ORDERS)  # 864
SELECTION_SALT = "phase12_family_source_v1|"
GENERIC_TOPIC = "A private assignment record gives a state for item {item_id}."
GENERIC_QUESTION = "Which listed state applies to the item?"
ITEM_IDS: tuple[str, ...] = tuple(string.ascii_uppercase[:N_PER_FAMILY])  # A..X

SUBSTANTIAL_V = 0.50
SUBSTANTIAL_RANGE = 0.50
REPLICATION_MIN_V = 0.70


def selection_digest(base_scenario_id: str) -> str:
    return hashlib.sha256(f"{SELECTION_SALT}{base_scenario_id}".encode()).hexdigest()


def select_bases(final_bases: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Label-independent: uses only base IDs. First 24 per family by salted SHA256."""
    out = []
    for fam in FAMILIES:
        fb = sorted((b for b in final_bases if b["family"] == fam),
                    key=lambda b: selection_digest(b["base_scenario_id"]))
        if len(fb) < N_PER_FAMILY:
            raise ValueError(f"too few bases for {fam}")
        for rank, b in enumerate(fb[:N_PER_FAMILY]):
            out.append({**b, "selection_rank": rank, "item_id": ITEM_IDS[rank]})
    return out


def _swap_noun(text: str, noun: str) -> str:
    return re.sub(rf"\b{re.escape(noun)}\b", NEUTRAL_NOUN, text)


def _digit(state: str) -> str:
    return state.rsplit(" ", 1)[-1]


def cell_scenario(base: dict[str, Any], cell: str) -> dict[str, Any]:
    fam = base["family"]
    noun = STATE_NOUN[fam]
    r, a = _digit(base["record_state"]), _digit(base["alternate_state"])
    if base["record_state"] != f"{noun} {r}" or base["alternate_state"] != f"{noun} {a}":
        raise ValueError("unexpected state namespace")
    if cell == "A":
        topic, question, sn = base["topic_sentence"], base["user_question"], noun
    elif cell == "B":
        topic = _swap_noun(base["topic_sentence"], noun)
        question = _swap_noun(base["user_question"], noun)
        sn = NEUTRAL_NOUN
    elif cell == "C":
        topic, question, sn = GENERIC_TOPIC.format(item_id=base["item_id"]), GENERIC_QUESTION, noun
    elif cell == "D":
        topic = GENERIC_TOPIC.format(item_id=base["item_id"])
        question, sn = GENERIC_QUESTION, NEUTRAL_NOUN
    else:
        raise ValueError(cell)
    return {
        "topic_sentence": topic,
        "user_question": question,
        "record_state": f"{sn} {r}",
        "alternate_state": f"{sn} {a}",
    }


def build_cell_prompt(base: dict[str, Any], cell: str, order: str) -> str:
    if order not in ORDERS:
        raise ValueError(order)
    sc = {**cell_scenario(base, cell), "record_listed_first": order == "RF"}
    return build_risk_prompt(scenario=sc, k=FIXED_K)


def _has_word(text: str, word: str) -> bool:
    return re.search(rf"(?<![A-Za-z']){re.escape(word)}(?![A-Za-z'])", text.lower()) is not None


def neutralization_violations(prompt: str, base: dict[str, Any], cell: str) -> list[str]:
    """Empty list means the cell's vocabulary rule holds for this prompt."""
    fam = base["family"]
    own = STATE_NOUN[fam]
    v: list[str] = []
    nouns = set(STATE_NOUN.values())
    if cell in ("B", "D"):
        v += [f"state noun {n}" for n in nouns if _has_word(prompt, n)]
    if cell == "C":
        stripped = re.sub(rf"\b{own} \d\b", "", prompt)
        v += [f"state noun {n} outside states" for n in nouns if _has_word(stripped, n)]
    if cell in ("C", "D"):
        terms = {t for ts in FAMILY_SEMANTIC_TERMS.values() for t in ts}
        v += [f"semantic term {t}" for t in sorted(terms) if _has_word(prompt, t)]
        if base["entity"].lower() in prompt.lower():
            v.append("entity string")
    if cell == "B":
        if base["entity"] not in prompt:
            v.append("entity missing")
        if not any(_has_word(prompt, t) for t in FAMILY_SEMANTIC_TERMS[fam]):
            v.append("family semantic shell missing")
    return v


def cramers_v(pairs: Sequence[tuple[str, str]]) -> float:
    n = len(pairs)
    if not n:
        return float("nan")
    rows = sorted({a for a, _ in pairs})
    cols = sorted({b for _, b in pairs})
    k = min(len(rows), len(cols)) - 1
    if k <= 0:
        return 0.0
    obs = Counter(pairs)
    ra = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    chi2 = sum((obs[(a, b)] - ra[a] * cb[b] / n) ** 2 / (ra[a] * cb[b] / n)
               for a in rows for b in cols)
    return math.sqrt(chi2 / (n * k))


def family_dependence(bases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """bases: {family, label} for one cell."""
    by_fam = {}
    for fam in FAMILIES:
        fb = [b for b in bases if b["family"] == fam]
        r = sum(b["label"] == LABEL_RECORD for b in fb)
        a = sum(b["label"] == LABEL_ALTERNATE for b in fb)
        by_fam[fam] = {
            "n": len(fb),
            "stable_record": r,
            "stable_alternate": a,
            "order_sensitive": len(fb) - r - a,
            "alternate_rate_of_stable": a / (r + a) if r + a else None,
        }
    stable = [b for b in bases if b["label"] != LABEL_ORDER_SENSITIVE]
    rates = [v["alternate_rate_of_stable"] for v in by_fam.values()
             if v["alternate_rate_of_stable"] is not None]
    v = cramers_v([(b["family"], b["label"]) for b in stable])
    rng = max(rates) - min(rates) if rates else float("nan")
    return {
        "n_bases": len(bases),
        "n_stable": len(stable),
        "stable_fraction": len(stable) / len(bases) if bases else float("nan"),
        "order_sensitive_rate": 1 - len(stable) / len(bases) if bases else float("nan"),
        "n_stable_record": sum(b["label"] == LABEL_RECORD for b in stable),
        "n_stable_alternate": sum(b["label"] == LABEL_ALTERNATE for b in stable),
        "by_family": by_fam,
        "cramers_v_family_label": v,
        "family_alternate_rate_range": rng,
        "substantial_family_dependence": bool(v >= SUBSTANTIAL_V and rng >= SUBSTANTIAL_RANGE),
    }


def interpretation(dep: dict[str, dict[str, Any]]) -> dict[str, Any]:
    a_v = dep["A"]["cramers_v_family_label"]
    replicated = a_v >= REPLICATION_MIN_V
    b, c, a = (dep[x]["substantial_family_dependence"] for x in ("B", "C", "A"))
    out: dict[str, Any] = {
        "thresholds": {
            "substantial": f"V >= {SUBSTANTIAL_V} and range >= {SUBSTANTIAL_RANGE}",
            "replication": f"A V >= {REPLICATION_MIN_V}",
        },
        "replication_passed": replicated,
        "A_cramers_v": a_v,
    }
    if not replicated:
        out["note"] = "subset nonrepresentative; B/C not interpreted"
        return out
    out.update({
        "state_vocabulary_contributor": c,
        "semantic_shell_contributor": b,
        "mixed_contributors": b and c,
        "interaction_or_other_family_structure": a and not b and not c,
    })
    return out


def matched_agreement(x: Sequence[dict[str, Any]], y: Sequence[dict[str, Any]]) -> dict:
    """Per-base agreement between two cells (label and RF choice)."""
    ym = {b["base_scenario_id"]: b for b in y}
    both = [(b, ym[b["base_scenario_id"]]) for b in x]
    st = [(p, q) for p, q in both
          if p["label"] != LABEL_ORDER_SENSITIVE and q["label"] != LABEL_ORDER_SENSITIVE]
    return {
        "n_bases": len(both),
        "label_exact_agreement": sum(p["label"] == q["label"] for p, q in both) / len(both),
        "n_both_stable": len(st),
        "stable_label_agreement": (sum(p["label"] == q["label"] for p, q in st) / len(st))
        if st else None,
        "rf_choice_agreement": sum(p["rf_choice"] == q["rf_choice"] for p, q in both)
        / len(both),
        "af_choice_agreement": sum(p["af_choice"] == q["af_choice"] for p, q in both)
        / len(both),
    }


def assert_not_locked(families: Sequence[str]) -> None:
    if set(families) & set(LOCKED):
        raise ValueError("locked family present")
