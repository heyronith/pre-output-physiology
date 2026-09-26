"""Deterministic Phase 6 factorial behavioral validity (one-line Response <STATE>)."""

from __future__ import annotations

from collections import Counter
from typing import Any

from pre_output_physiology.phase5_behavior import parse_one_line_response
from pre_output_physiology.phase6_conditions import (
    CONDITION_ORDER,
    PILOT_ALL_FOUR_MIN,
    PILOT_BEHAVIOR_MIN_PER_CONDITION,
    PILOT_N_PER_CONDITION,
    expected_response_state,
    targets_for_condition,
)


def is_behaviorally_valid(
    condition_id: str,
    scenario: dict[str, Any],
    response_text: str,
    *,
    first_generated_token_id: int | None = None,
    expected_first_token_id: int | None = None,
) -> tuple[bool, str]:
    parsed = parse_one_line_response(response_text)
    if expected_first_token_id is not None:
        if first_generated_token_id is None or int(first_generated_token_id) != int(
            expected_first_token_id
        ):
            return False, "first_token_mismatch"
    if not parsed["starts_with_Response"]:
        return False, "first_line_not_Response_prefix"
    if not parsed["exact_one_nonempty_line"]:
        return False, "not_exact_one_nonempty_line"
    if not parsed["schema_match"] or not parsed["parsed_state"]:
        return False, "schema_mismatch"
    record = scenario["record_state"]
    alt = scenario["alternate_state"]
    state = parsed["parsed_state"]
    expected = expected_response_state(
        condition_id, record_state=record, alternate_state=alt
    )
    if state == expected:
        return True, "ok"
    op, _comm = targets_for_condition(
        condition_id, record_state=record, alternate_state=alt
    )
    if state == op:
        return False, "state_eq_operational_target_not_communication_target"
    if state in (record, alt):
        return False, "state_ne_communication_target"
    return False, "state_not_record_or_alternate"


def evaluate_row(
    row: dict[str, Any],
    *,
    expected_first_token_id: int | None = None,
) -> dict[str, Any]:
    scenario = {
        "record_state": row["record_state"],
        "alternate_state": row["alternate_state"],
    }
    text = row.get("output_text") or ""
    token_id = row.get("first_generated_token_id")
    parsed = parse_one_line_response(text)
    valid, reason = is_behaviorally_valid(
        row["condition_id"],
        scenario,
        text,
        first_generated_token_id=token_id,
        expected_first_token_id=expected_first_token_id,
    )
    first_ok = True
    if expected_first_token_id is not None:
        first_ok = token_id is not None and int(token_id) == int(expected_first_token_id)
    return {
        "example_id": row["example_id"],
        "base_scenario_id": row["base_scenario_id"],
        "condition_id": row["condition_id"],
        "family": row.get("family"),
        "first_token_id_compliant": first_ok,
        "first_generated_token_id": token_id,
        "schema_compliant": parsed["schema_match"],
        "parsed_state": parsed["parsed_state"],
        "behaviorally_valid": valid,
        "failure_reason": None if valid else reason,
    }


def summarize_pilot_behavior(
    eval_rows: list[dict[str, Any]], *, expected_first_token_id: int
) -> dict[str, Any]:
    by_cond: dict[str, list[dict[str, Any]]] = {cid: [] for cid in CONDITION_ORDER}
    for row in eval_rows:
        by_cond[row["condition_id"]].append(row)
    per_condition: dict[str, Any] = {}
    for cid in CONDITION_ORDER:
        rows = by_cond[cid]
        n = len(rows)
        valid = sum(1 for r in rows if r["behaviorally_valid"])
        per_condition[cid] = {
            "n_total": n,
            "behaviorally_valid": valid,
            "schema_compliant": sum(1 for r in rows if r["schema_compliant"]),
            "first_token_id_compliant": sum(
                1 for r in rows if r["first_token_id_compliant"]
            ),
            "failure_reason_counts": dict(
                Counter(r["failure_reason"] for r in rows if r["failure_reason"])
            ),
            "first_token_id_counts": {
                str(k): v
                for k, v in Counter(
                    r["first_generated_token_id"]
                    for r in rows
                    if r["first_generated_token_id"] is not None
                ).items()
            },
            "behavior_gate_pass": valid >= PILOT_BEHAVIOR_MIN_PER_CONDITION
            and n == PILOT_N_PER_CONDITION,
        }
    valid_sets = {
        cid: {r["base_scenario_id"] for r in by_cond[cid] if r["behaviorally_valid"]}
        for cid in CONDITION_ORDER
    }
    all_four = sorted(set.intersection(*valid_sets.values()))
    valid_first_tokens_ok = all(
        r["first_generated_token_id"] == expected_first_token_id
        for r in eval_rows
        if r["behaviorally_valid"]
    )
    behavior_gates = all(v["behavior_gate_pass"] for v in per_condition.values())
    all_four_gate = len(all_four) >= PILOT_ALL_FOUR_MIN
    return {
        "n_eval_rows": len(eval_rows),
        "per_condition": per_condition,
        "all_four_valid_scenarios": all_four,
        "n_all_four_valid": len(all_four),
        "gates": {
            "behavior_min_per_condition": PILOT_BEHAVIOR_MIN_PER_CONDITION,
            "all_four_min": PILOT_ALL_FOUR_MIN,
            "behavior_all_conditions_pass": behavior_gates,
            "all_four_pass": all_four_gate,
            "valid_runs_first_token_ok": valid_first_tokens_ok,
            "all_operational_gates_pass": behavior_gates
            and all_four_gate
            and valid_first_tokens_ok,
        },
        "activations_collected": False,
        "probe_scored": False,
        "final_scenarios_run": False,
    }
