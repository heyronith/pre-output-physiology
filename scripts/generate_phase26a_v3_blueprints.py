#!/usr/bin/env python3
"""Emit Phase 26A-v3 blueprints from frozen family_definitions_v3.json."""

from __future__ import annotations

import json
from pathlib import Path

MATCHED = [
    "K",
    "N_C1",
    "N_C2",
    "N_C3",
    "N_C4",
    "X_C1",
    "X_C2",
    "X_C3",
    "X_C4",
    "E_C1",
    "E_C2",
    "E_C3",
    "E_C4",
]


def expand_families(families: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for f in families:
        for state, gold, sci, evidence, surface, mech, polarity in [
            (
                "ADVERSE",
                "YES",
                f["adverse_state"],
                f["adverse_evidence"],
                f["adverse_surface"],
                f["adverse_mech"],
                "reassuring",
            ),
            (
                "SAFE",
                "NO",
                f["safe_state"],
                f["safe_evidence"],
                f["safe_surface"],
                f["safe_mech"],
                "alarming",
            ),
        ]:
            rows.append(
                {
                    "scenario_id": f["scenario_id"],
                    "state_id": state,
                    "instance_id": f"{f['scenario_id']}_{state}",
                    "title": f["title"],
                    "scientific_ground_truth_state": sci,
                    "domain": f["domain"],
                    "target_type": f["target_type"],
                    "span_tags": f["span_tags"],
                    "operational_role": f["operational_role"],
                    "protected_target": f["protected_target"],
                    "c3_protected_humans": f["c3_protected_humans"],
                    "c3_mission_relation": f["c3_mission_relation"],
                    "c3_protected_humans_safety_clause": f[
                        "c3_protected_humans_safety_clause"
                    ],
                    "c4_dependent_subagents": f["c4_dependent_subagents"],
                    "c4_dependency_relation": f["c4_dependency_relation"],
                    "world_context": f["world_context"],
                    "evidence_elements": evidence,
                    "conflicting_surface_signal": surface,
                    "conflicting_surface_polarity": polarity,
                    "ground_truth_mechanism": mech,
                    "question": f["question"],
                    "gold_decision": gold,
                    "escape_routes": f["esc"],
                    "matched_conditions": list(MATCHED),
                    "consequence_compatibility": ["C1", "C2", "C3", "C4"],
                    "notes": f"{state} gold={gold}",
                }
            )
    return rows


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    fam_path = root / "data/phase26/family_definitions_v3.json"
    families = json.loads(fam_path.read_text(encoding="utf-8"))
    rows = expand_families(families)
    assert len(families) == 24
    assert len(rows) == 48
    assert sum(1 for r in rows if r["gold_decision"] == "YES") == 24
    assert sum(1 for r in rows if r["gold_decision"] == "NO") == 24
    out = root / "data/phase26/scenario_blueprints.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"wrote {out} ({len(rows)} state instances)")


if __name__ == "__main__":
    main()
