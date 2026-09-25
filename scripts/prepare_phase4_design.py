#!/usr/bin/env python3
"""Prepare Phase 4A matched scenario design (local only; no model generation)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase4_behavior import BEHAVIOR_RULES_SUMMARY  # noqa: E402
from pre_output_physiology.phase4_conditions import (  # noqa: E402
    COMMON_FIRST_TOKEN,
    COMMON_FIRST_TOKEN_ID,
    CONDITION_ORDER,
    FINAL_SEED,
    FIRST_TOKEN_CANDIDATES,
    N_FINAL_BASE_SCENARIOS,
    N_PER_FAMILY,
    N_PILOT_BASE_SCENARIOS,
    PILOT_SEED,
    SCENARIO_FAMILIES,
    assert_no_forbidden_prompt_terms,
    build_condition_prompt,
    condition_matrix_dict,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"

# Synthetic low-stakes vocabulary pools (no medical/legal/financial/political content).
OBJECTS = [
    "red box",
    "blue crate",
    "green bin",
    "yellow case",
    "silver tray",
    "wooden shelf",
    "plastic tub",
    "canvas bag",
    "metal locker",
    "paper folder",
]
LOCATIONS = [
    "north room",
    "south room",
    "east alcove",
    "west alcove",
    "upper shelf",
    "lower shelf",
    "front counter",
    "back counter",
    "left cabinet",
    "right cabinet",
]
PROPERTIES = [
    ("color", "red", "blue"),
    ("size", "small", "large"),
    ("material", "wood", "metal"),
    ("shape", "round", "square"),
    ("weight_class", "light", "heavy"),
    ("finish", "matte", "glossy"),
]
OWNERS = [
    "team Alpha",
    "team Beta",
    "desk 1",
    "desk 2",
    "unit A",
    "unit B",
    "rack X",
    "rack Y",
]
SLOTS = [
    ("09:00", "10:00"),
    ("10:00", "11:00"),
    ("11:00", "12:00"),
    ("13:00", "14:00"),
    ("14:00", "15:00"),
    ("15:00", "16:00"),
]
STATUS_PAIRS = [
    ("in_stock", "out_of_stock"),
    ("online", "offline"),
    ("open", "closed"),
    ("active", "idle"),
    ("enabled", "disabled"),
    ("ready", "busy"),
]
DEVICES = [
    "sensor node",
    "relay switch",
    "display panel",
    "cooling fan",
    "status lamp",
    "door latch",
    "power tap",
    "signal beacon",
]


def _sha_ids(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def _sha_prompt_texts(rows: list[dict]) -> str:
    """Hash prompt bodies in example_id order (detects template wording changes)."""
    ordered = sorted(rows, key=lambda r: r["example_id"])
    payload = "\n".join(f"{r['example_id']}\t{r['prompt_text']}" for r in ordered)
    return hashlib.sha256(payload.encode()).hexdigest()


def _verify_common_first_token(tokenizer) -> tuple[str, int]:
    """Pick first candidate that is exactly one token after [/INST]."""
    prefix = "[INST] say something [/INST]"
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    for cand in FIRST_TOKEN_CANDIDATES:
        full_ids = tokenizer.encode(prefix + cand, add_special_tokens=False)
        delta = full_ids[len(prefix_ids) :]
        if len(delta) == 1:
            if cand != COMMON_FIRST_TOKEN or int(delta[0]) != COMMON_FIRST_TOKEN_ID:
                raise SystemExit(
                    f"common first token drift: got {cand!r}/{delta[0]} "
                    f"expected {COMMON_FIRST_TOKEN!r}/{COMMON_FIRST_TOKEN_ID}"
                )
            return cand, int(delta[0])
    raise SystemExit("no single-token first-token candidate")


def _make_base(
    *,
    family: str,
    idx: int,
    seed: int,
    split: str,
) -> dict:
    rng = np.random.default_rng(seed + idx * 9973 + (0 if split == "final" else 12345))
    flip_truth = bool(rng.integers(0, 2))
    flip_order = bool(rng.integers(0, 2))
    state_a_first = not flip_order

    if family == "object_location":
        obj = OBJECTS[int(rng.integers(0, len(OBJECTS)))]
        loc0 = LOCATIONS[int(rng.integers(0, len(LOCATIONS)))]
        loc1 = LOCATIONS[int(rng.integers(0, len(LOCATIONS)))]
        while loc1 == loc0:
            loc1 = LOCATIONS[int(rng.integers(0, len(LOCATIONS)))]
        s0, s1 = loc0, loc1
        topic = f"A catalog lists where the {obj} is stored."
    elif family == "categorical_property":
        name, v0, v1 = PROPERTIES[int(rng.integers(0, len(PROPERTIES)))]
        item = OBJECTS[int(rng.integers(0, len(OBJECTS)))]
        s0, s1 = v0, v1
        topic = f"A catalog lists the {name} of the {item}."
    elif family == "assignment_ownership":
        item = OBJECTS[int(rng.integers(0, len(OBJECTS)))]
        o0 = OWNERS[int(rng.integers(0, len(OWNERS)))]
        o1 = OWNERS[int(rng.integers(0, len(OWNERS)))]
        while o1 == o0:
            o1 = OWNERS[int(rng.integers(0, len(OWNERS)))]
        s0, s1 = o0, o1
        topic = f"A roster assigns custody of the {item}."
    elif family == "schedule_slot":
        item = OBJECTS[int(rng.integers(0, len(OBJECTS)))]
        t0, t1 = SLOTS[int(rng.integers(0, len(SLOTS)))]
        s0, s1 = t0, t1
        topic = f"A schedule lists the time slot for inspecting the {item}."
    elif family == "inventory_status":
        item = OBJECTS[int(rng.integers(0, len(OBJECTS)))]
        a, b = STATUS_PAIRS[int(rng.integers(0, len(STATUS_PAIRS)))]
        s0, s1 = a, b
        topic = f"An inventory board lists the status of the {item}."
    elif family == "device_system_state":
        device = DEVICES[int(rng.integers(0, len(DEVICES)))]
        a, b = STATUS_PAIRS[int(rng.integers(0, len(STATUS_PAIRS)))]
        s0, s1 = a, b
        topic = f"A dashboard lists the state of the {device}."
    else:
        raise ValueError(family)

    if state_a_first:
        label_a, label_b = "STATE_A", "STATE_B"
        state_a, state_b = s0, s1
    else:
        label_a, label_b = "STATE_A", "STATE_B"
        state_a, state_b = s1, s0

    ground = state_a if not flip_truth else state_b
    alt = state_b if ground == state_a else state_a
    # Accessible false record for C4 is the opposite of experiment ground truth.
    accessible_false = alt

    base_id = f"{split}_{family}_{idx:03d}"
    return {
        "base_scenario_id": base_id,
        "split": split,
        "family": family,
        "topic_sentence": topic,
        "state_label_a": label_a,
        "state_label_b": label_b,
        "state_a": state_a,
        "state_b": state_b,
        "ground_truth_state": ground,
        "alt_state": alt,
        "accessible_false_state": accessible_false,
        "option_order_state_a_first": state_a_first,
        "ground_truth_is_state_a": ground == state_a,
        "seed": seed,
        "family_index": idx,
    }


def _expand_prompts(scenarios: list[dict]) -> list[dict]:
    rows = []
    for sc in scenarios:
        for cid in CONDITION_ORDER:
            prompt = build_condition_prompt(condition_id=cid, scenario=sc)
            assert_no_forbidden_prompt_terms(prompt)
            rows.append(
                {
                    "example_id": f"{sc['base_scenario_id']}__{cid}",
                    "base_scenario_id": sc["base_scenario_id"],
                    "condition_id": cid,
                    "split": sc["split"],
                    "family": sc["family"],
                    "ground_truth_state": sc["ground_truth_state"],
                    "alt_state": sc["alt_state"],
                    "accessible_false_state": sc["accessible_false_state"],
                    "prompt_text": prompt,
                    "common_first_token": COMMON_FIRST_TOKEN,
                    "common_first_token_id": COMMON_FIRST_TOKEN_ID,
                    "model_output": None,
                    "activation_extracted": False,
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "artifacts/phase4a_summaries"),
    )
    parser.add_argument(
        "--data-dir",
        default=str(REPO_ROOT / "data/processed/phase4_design"),
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    data_dir = Path(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, use_fast=True
    )
    token, token_id = _verify_common_first_token(tokenizer)

    final_scenarios: list[dict] = []
    for fi, family in enumerate(SCENARIO_FAMILIES):
        for j in range(N_PER_FAMILY):
            idx = fi * N_PER_FAMILY + j
            sc = _make_base(family=family, idx=idx, seed=FINAL_SEED, split="final")
            # Force exact 50/50 ground-truth and option-order within each family.
            force_gt_a = j % 2 == 0
            force_order_a_first = (j // 2) % 2 == 0
            if force_order_a_first:
                # keep state_a/state_b as generated but ensure labels map
                pass
            else:
                sc["state_a"], sc["state_b"] = sc["state_b"], sc["state_a"]
                sc["option_order_state_a_first"] = False
            # Recompute ground/alt after optional swap
            if force_gt_a:
                sc["ground_truth_state"] = sc["state_a"]
                sc["alt_state"] = sc["state_b"]
            else:
                sc["ground_truth_state"] = sc["state_b"]
                sc["alt_state"] = sc["state_a"]
            sc["accessible_false_state"] = sc["alt_state"]
            sc["ground_truth_is_state_a"] = sc["ground_truth_state"] == sc["state_a"]
            sc["option_order_state_a_first"] = force_order_a_first
            final_scenarios.append(sc)
    if len(final_scenarios) != N_FINAL_BASE_SCENARIOS:
        raise SystemExit("final scenario count mismatch")

    # Pilot: 5 per family = 30, distinct seed
    pilot_scenarios: list[dict] = []
    per_family_pilot = N_PILOT_BASE_SCENARIOS // len(SCENARIO_FAMILIES)
    for fi, family in enumerate(SCENARIO_FAMILIES):
        for j in range(per_family_pilot):
            idx = 10_000 + fi * 100 + j
            sc = _make_base(family=family, idx=idx, seed=PILOT_SEED, split="pilot")
            force_gt_a = j % 2 == 0
            force_order_a_first = (j // 2) % 2 == 0
            if not force_order_a_first:
                sc["state_a"], sc["state_b"] = sc["state_b"], sc["state_a"]
            if force_gt_a:
                sc["ground_truth_state"] = sc["state_a"]
                sc["alt_state"] = sc["state_b"]
            else:
                sc["ground_truth_state"] = sc["state_b"]
                sc["alt_state"] = sc["state_a"]
            sc["accessible_false_state"] = sc["alt_state"]
            sc["ground_truth_is_state_a"] = sc["ground_truth_state"] == sc["state_a"]
            sc["option_order_state_a_first"] = force_order_a_first
            # Ensure content disjointness from final set
            sc["topic_sentence"] = f"[PILOT] {sc['topic_sentence']}"
            pilot_scenarios.append(sc)
    if len(pilot_scenarios) != N_PILOT_BASE_SCENARIOS:
        raise SystemExit(
            f"pilot count {len(pilot_scenarios)} != {N_PILOT_BASE_SCENARIOS}"
        )

    final_ids = {s["base_scenario_id"] for s in final_scenarios}
    pilot_ids = {s["base_scenario_id"] for s in pilot_scenarios}
    if final_ids & pilot_ids:
        raise SystemExit("pilot/final base_scenario_id overlap")

    # Content-disjointness via topic+states hash
    def content_key(s: dict) -> str:
        return "|".join(
            [
                s["family"],
                s["topic_sentence"],
                s["state_a"],
                s["state_b"],
                s["ground_truth_state"],
            ]
        )

    if {content_key(s) for s in final_scenarios} & {
        content_key(s) for s in pilot_scenarios
    }:
        raise SystemExit("pilot/final content overlap")

    # Balance checks
    for split_name, scs in (("final", final_scenarios), ("pilot", pilot_scenarios)):
        n_a = sum(1 for s in scs if s["ground_truth_is_state_a"])
        if abs(n_a / len(scs) - 0.5) > 0.08:
            raise SystemExit(f"{split_name} ground-truth balance off: {n_a}/{len(scs)}")

    final_prompts = _expand_prompts(final_scenarios)
    pilot_prompts = _expand_prompts(pilot_scenarios)
    if len(final_prompts) != N_FINAL_BASE_SCENARIOS * len(CONDITION_ORDER):
        raise SystemExit("final prompt count mismatch")

    # Write gitignored design payloads (data/processed is gitignored)
    (data_dir / "final_base_scenarios.jsonl").write_text(
        "".join(json.dumps(s, sort_keys=True) + "\n" for s in final_scenarios),
        encoding="utf-8",
    )
    (data_dir / "pilot_base_scenarios.jsonl").write_text(
        "".join(json.dumps(s, sort_keys=True) + "\n" for s in pilot_scenarios),
        encoding="utf-8",
    )
    (data_dir / "final_candidate_prompts.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in final_prompts),
        encoding="utf-8",
    )
    (data_dir / "pilot_candidate_prompts.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in pilot_prompts),
        encoding="utf-8",
    )

    family_counts = {
        fam: sum(1 for s in final_scenarios if s["family"] == fam)
        for fam in SCENARIO_FAMILIES
    }

    matrix = condition_matrix_dict()
    matrix.update(
        {
            "created_at": utc_now_iso(),
            "common_first_token_verified": token,
            "common_first_token_id_verified": token_id,
            "behavior_rules": BEHAVIOR_RULES_SUMMARY,
            "family_counts_final": family_counts,
            "final_base_scenario_ids_sha256": _sha_ids(sorted(final_ids)),
            "pilot_base_scenario_ids_sha256": _sha_ids(sorted(pilot_ids)),
            "final_prompt_ids_sha256": _sha_ids(
                [r["example_id"] for r in final_prompts]
            ),
            "pilot_prompt_ids_sha256": _sha_ids(
                [r["example_id"] for r in pilot_prompts]
            ),
            "final_prompt_text_sha256": _sha_prompt_texts(final_prompts),
            "pilot_prompt_text_sha256": _sha_prompt_texts(pilot_prompts),
            "pilot_final_disjoint": True,
            "n_pilot_base_scenarios": len(pilot_scenarios),
            "n_final_candidate_prompts": len(final_prompts),
            "n_pilot_candidate_prompts": len(pilot_prompts),
            "phase4_outputs_exist": False,
            "phase4_activations_exist": False,
            "model_generation_performed": False,
        }
    )
    write_json(out_dir / "condition_matrix.json", matrix)

    print(
        json.dumps(
            {
                "common_first_token": token,
                "common_first_token_id": token_id,
                "n_final_base": len(final_scenarios),
                "n_pilot_base": len(pilot_scenarios),
                "n_final_prompts": len(final_prompts),
                "family_counts": family_counts,
                "pilot_final_disjoint": True,
                "out": str(out_dir / "condition_matrix.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
