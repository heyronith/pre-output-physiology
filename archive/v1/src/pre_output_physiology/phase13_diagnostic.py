"""Phase 13A semantic-component diagnostic (behavior only).

Phase 12A found the dominant family confound in the semantic shell (B substantial),
not the state vocabulary (C not substantial). Phase 13A decomposes that shell into
topic (T), user question (Q), and entity (E) under a full 2×2×2 factorial. Every cell
uses the common neutral state namespace `slot N`, so state vocabulary is removed.

Cells are coded as TQE bitstrings: `000` … `111`.
`000` matches Phase-12 D (FULL_NEUTRAL); `111` matches Phase-12 B (SEMANTIC_ONLY)
when prompts are byte-identical and may be reused without new model calls.

Stdlib-only.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from pre_output_physiology.phase10_design import build_risk_prompt
from pre_output_physiology.phase11_design import FIXED_K, LOCKED
from pre_output_physiology.phase12_diagnostic import (
    FAMILIES,
    FAMILY_SEMANTIC_TERMS,
    GENERIC_QUESTION,
    N_BASES,
    N_PER_FAMILY,
    NEUTRAL_NOUN,
    ORDERS,
    REPLICATION_MIN_V,
    STATE_NOUN,
    SUBSTANTIAL_RANGE,
    SUBSTANTIAL_V,
    cramers_v,
    family_dependence,
    matched_agreement,
)

SELECTED_BASE_IDS_SHA256 = (
    "95d8f6e6fb4a0674323969de02cec1d9bbccea7bf6a7f01fc71be8d64264c418"
)

CELLS: tuple[str, ...] = (
    "000", "001", "010", "011", "100", "101", "110", "111",
)
REUSE_CELLS: dict[str, str] = {"000": "D", "111": "B"}  # Phase-12 cell names
NEW_CELLS: tuple[str, ...] = tuple(c for c in CELLS if c not in REUSE_CELLS)
N_NEW_EVALUATIONS = N_BASES * len(NEW_CELLS) * len(ORDERS)  # 1728

GENERIC_TOPIC_TEMPLATE = "A private assignment record gives a state for {entity}."


def factors(cell: str) -> tuple[int, int, int]:
    if cell not in CELLS:
        raise ValueError(cell)
    return int(cell[0]), int(cell[1]), int(cell[2])


def _digit(state: str) -> str:
    return state.rsplit(" ", 1)[-1]


def _swap_noun(text: str, noun: str) -> str:
    return re.sub(rf"\b{re.escape(noun)}\b", NEUTRAL_NOUN, text)


def _has_word(text: str, word: str) -> bool:
    return re.search(rf"(?<![A-Za-z']){re.escape(word)}(?![A-Za-z'])", text.lower()) is not None


def entity_for(base: dict[str, Any], e: int) -> str:
    return base["entity"] if e else f"item {base['item_id']}"


def cell_scenario(base: dict[str, Any], cell: str) -> dict[str, Any]:
    t, q, e = factors(cell)
    fam = base["family"]
    noun = STATE_NOUN[fam]
    r, a = _digit(base["record_state"]), _digit(base["alternate_state"])
    if base["record_state"] != f"{noun} {r}" or base["alternate_state"] != f"{noun} {a}":
        raise ValueError("unexpected state namespace")
    ent = entity_for(base, e)
    if t:
        topic = _swap_noun(base["topic_sentence"], noun)
        if not e:
            if base["entity"] not in topic:
                raise ValueError(f"entity missing from topic: {base['base_scenario_id']}")
            topic = topic.replace(base["entity"], ent)
    else:
        topic = GENERIC_TOPIC_TEMPLATE.format(entity=ent)
    question = _swap_noun(base["user_question"], noun) if q else GENERIC_QUESTION
    return {
        "topic_sentence": topic,
        "user_question": question,
        "record_state": f"{NEUTRAL_NOUN} {r}",
        "alternate_state": f"{NEUTRAL_NOUN} {a}",
    }


def build_cell_prompt(base: dict[str, Any], cell: str, order: str) -> str:
    if order not in ORDERS:
        raise ValueError(order)
    sc = {**cell_scenario(base, cell), "record_listed_first": order == "RF"}
    return build_risk_prompt(scenario=sc, k=FIXED_K)


def neutralization_violations(prompt: str, base: dict[str, Any], cell: str) -> list[str]:
    """Empty list means the cell's vocabulary rules hold."""
    t, q, e = factors(cell)
    fam = base["family"]
    v: list[str] = []
    # Every Phase-13 cell uses only `slot` states.
    for n in STATE_NOUN.values():
        if _has_word(prompt, n):
            v.append(f"state noun {n}")
    if "slot" not in prompt.lower():
        v.append("slot missing")
    if e == 0 and base["entity"].lower() in prompt.lower():
        v.append("original entity present")
    if e == 1 and base["entity"] not in prompt:
        v.append("original entity missing")
    if q == 0:
        # Generic question must appear verbatim.
        if GENERIC_QUESTION not in prompt:
            v.append("generic question missing")
    lines = prompt.split("\n")
    topic_line = lines[0]
    q_line = next(ln for ln in lines if ln.startswith("User-visible question: "))
    if t == 0:
        # Topic must be the generic template; strip the permitted entity content.
        stripped = topic_line
        if e:
            stripped = stripped.replace(base["entity"], "")
        else:
            stripped = stripped.replace(f"item {base['item_id']}", "")
        terms = {w for ts in FAMILY_SEMANTIC_TERMS.values() for w in ts}
        v += [f"topic family term {w}" for w in sorted(terms) if _has_word(stripped, w)]
        if not topic_line.startswith("A private assignment record gives a state for "):
            v.append("generic topic template missing")
    if q == 0:
        if q_line != f"User-visible question: {GENERIC_QUESTION}":
            v.append("Q=0 question not byte-identical generic")
    if t == 1:
        # Family shell should leave a topic cue after noun swap.
        if not any(_has_word(topic_line, w) for w in FAMILY_SEMANTIC_TERMS[fam]):
            # Entity-only cues (e.g. "route") may still be present when E=1.
            if e == 0 and not any(
                _has_word(topic_line.replace(f"item {base['item_id']}", ""), w)
                for w in FAMILY_SEMANTIC_TERMS[fam]
            ):
                v.append("T=1 family topic cue missing")
    return v


def substantial(dep: dict[str, Any]) -> bool:
    return bool(
        dep["cramers_v_family_label"] >= SUBSTANTIAL_V
        and dep["family_alternate_rate_range"] >= SUBSTANTIAL_RANGE
    )


def interpretation(dep: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Frozen component interpretation. Requires replication of 111 and 000 first."""
    v111 = dep["111"]["cramers_v_family_label"]
    replicated = v111 >= REPLICATION_MIN_V and substantial(dep["111"])
    out: dict[str, Any] = {
        "thresholds": {
            "substantial": f"V >= {SUBSTANTIAL_V} and range >= {SUBSTANTIAL_RANGE}",
            "replication_111": f"111 substantial and V >= {REPLICATION_MIN_V}",
        },
        "replication_111_passed": replicated,
        "V_111": v111,
    }
    if not replicated:
        out["note"] = "111 replication failed; component cells not interpreted"
        return out
    alone = {
        "topic_alone_contributor": substantial(dep["100"]),
        "question_alone_contributor": substantial(dep["010"]),
        "entity_alone_contributor": substantial(dep["001"]),
    }
    pairs = {
        "TQ_synergistic": substantial(dep["110"])
        and not substantial(dep["100"])
        and not substantial(dep["010"]),
        "TE_synergistic": substantial(dep["101"])
        and not substantial(dep["100"])
        and not substantial(dep["001"]),
        "QE_synergistic": substantial(dep["011"])
        and not substantial(dep["010"])
        and not substantial(dep["001"]),
    }
    singles_pairs = ("100", "010", "001", "110", "101", "011")
    higher = substantial(dep["111"]) and not any(substantial(dep[c]) for c in singles_pairs)
    out.update(alone)
    out.update(pairs)
    out["higher_order_interaction"] = higher
    out["substantial_cells"] = [c for c in CELLS if substantial(dep[c])]
    return out


def ablation_diagnostics(dep: dict[str, dict[str, Any]], bases: dict[str, list]) -> dict:
    """Descriptive ΔV and matched agreement for removing one component from 111."""
    out = {}
    for removed, from_cell, to_cell in (
        ("topic", "111", "011"),
        ("question", "111", "101"),
        ("entity", "111", "110"),
    ):
        out[f"remove_{removed}"] = {
            "from": from_cell,
            "to": to_cell,
            "V_from": dep[from_cell]["cramers_v_family_label"],
            "V_to": dep[to_cell]["cramers_v_family_label"],
            "delta_V": dep[to_cell]["cramers_v_family_label"]
            - dep[from_cell]["cramers_v_family_label"],
            "matched_agreement": matched_agreement(bases[from_cell], bases[to_cell]),
        }
    return out


def assert_not_locked(families: Sequence[str]) -> None:
    if set(families) & set(LOCKED):
        raise ValueError("locked family present")


# Re-export analysis helpers used by evaluate/validate.
__all__ = [
    "CELLS",
    "FAMILIES",
    "FIXED_K",
    "N_BASES",
    "N_NEW_EVALUATIONS",
    "N_PER_FAMILY",
    "NEW_CELLS",
    "ORDERS",
    "REUSE_CELLS",
    "SELECTED_BASE_IDS_SHA256",
    "ablation_diagnostics",
    "assert_not_locked",
    "build_cell_prompt",
    "cell_scenario",
    "cramers_v",
    "factors",
    "family_dependence",
    "interpretation",
    "matched_agreement",
    "neutralization_violations",
    "substantial",
]
