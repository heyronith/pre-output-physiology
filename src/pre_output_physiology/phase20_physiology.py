"""Phase 20A — pre-answer physiology on same-prompt stochastic trajectories.

20A1: fresh 128-cont generation + balanced subset gates (behavior only).
20A2: discovery activations/probes/baselines on TRAIN only; freeze candidate.
20A3: validation extraction at frozen block/position only.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from pre_output_physiology.phase15_onset import (
    LABEL_ALTERNATE,
    LABEL_RECORD,
    RULE_HASH,
    RULE_VERSION,
    annotate_continuation,
)
from pre_output_physiology.phase18_cohort import STATUS_HOLD as PHASE18_STATUS
from pre_output_physiology.phase19_screen import STATUS_PASS as PHASE19_STATUS

PHASE15_RULE_HASH = RULE_HASH
PHASE15_RULE_VERSION = RULE_VERSION

PHASE18_STATUS_REQUIRED = PHASE18_STATUS
PHASE19_STATUS_REQUIRED = PHASE19_STATUS
PHASE19_USABLE_IDS_SHA256 = (
    "60099744abc25099558ca401b303f4bcff86736e107436260842054020e4f49f"
)

# --- Frozen populations ---
TRAIN_PROMPT_GROUP_IDS: tuple[str, ...] = (
    "final_bowling_alley_lane_003",
    "final_bowling_alley_lane_009",
    "final_bowling_alley_lane_012",
    "final_bowling_alley_lane_013",
    "final_bowling_alley_lane_016",
    "final_bowling_alley_lane_030",
    "final_radio_studio_booth_004",
    "final_radio_studio_booth_020",
    "final_radio_studio_booth_022",
    "final_radio_studio_booth_027",
    "final_radio_studio_booth_034",
    "final_veterinary_kennel_run_000",
    "final_veterinary_kennel_run_004",
    "final_veterinary_kennel_run_008",
    "final_veterinary_kennel_run_009",
    "final_veterinary_kennel_run_012",
    "final_veterinary_kennel_run_032",
)
TRAIN_FAMILIES: tuple[str, ...] = (
    "bowling_alley_lane",
    "radio_studio_booth",
    "veterinary_kennel_run",
)
VAL_PROMPT_GROUP_IDS: tuple[str, ...] = (
    "p19_mini_golf_tee_pad_000",
    "p19_mini_golf_tee_pad_008",
    "p19_mini_golf_tee_pad_009",
    "p19_mini_golf_tee_pad_011",
    "p19_mini_golf_tee_pad_015",
    "p19_mini_golf_tee_pad_017",
    "p19_music_school_practice_room_004",
    "p19_music_school_practice_room_005",
    "p19_music_school_practice_room_010",
    "p19_music_school_practice_room_014",
    "p19_music_school_practice_room_018",
)
VAL_FAMILIES: tuple[str, ...] = (
    "mini_golf_tee_pad",
    "music_school_practice_room",
)
LOCKED_FAMILIES: tuple[str, ...] = (
    "museum_cloakroom_hook",
    "pier_mooring_slip",
)

N_TRAIN = len(TRAIN_PROMPT_GROUP_IDS)  # 17
N_VAL = len(VAL_PROMPT_GROUP_IDS)  # 11
N_PROMPTS = N_TRAIN + N_VAL  # 28
N_SAMPLES = 128
N_CONTINUATIONS = N_PROMPTS * N_SAMPLES  # 3584

SEED_BASE = 20_000_000  # disjoint from 14M/17M/18M/19M

TEMPERATURE = 0.9
TOP_P = 0.95

MIN_STAGE1_TOKENS = 5
N_BALANCED_PER_CLASS = 8
N_BALANCED_PER_PROMPT = N_BALANCED_PER_CLASS * 2  # 16

# Population gates
TRAIN_MIN_ELIGIBLE = 15
TRAIN_MIN_PER_FAMILY = 4
VAL_MIN_ELIGIBLE = 9
VAL_MIN_PER_FAMILY = 4

# Physiology
BLOCKS: tuple[int, ...] = (0, 4, 8, 12, 16, 20, 24, 28, 31)
POSITIONS: tuple[str, ...] = ("end0", "end2", "end4")
POSITION_OFFSET: dict[str, int] = {"end0": 0, "end2": 2, "end4": 4}
POSITION_TIE_RANK: dict[str, int] = {"end4": 0, "end2": 1, "end0": 2}  # earlier better

PROBE_C = 0.01
PROBE_MAX_ITER = 500
PROBE_SEED = 42
K0_AUROC_TOL = 0.02
K0_COSINE_MIN = 0.9999

DISCOVERY_MIN_AUROC = 0.60
DISCOVERY_MIN_DELTA = 0.03

VAL_MIN_AUROC = 0.65
VAL_MIN_AUROC_CI_LOW = 0.55
VAL_MIN_FAMILY_AUROC = 0.60
VAL_MIN_INCREMENTAL = 0.03
N_BOOTSTRAP = 5000
BOOTSTRAP_SEED = 2020

STATUS_AUTHORIZED = "phase20a_preanswer_physiology_authorized"
STATUS_GEN_DONE = "phase20a_preanswer_physiology_generation_complete_awaiting_gates"
STATUS_POP_HOLD = "phase20a_preanswer_physiology_population_hold"
STATUS_POP_PASS = (
    "phase20a_preanswer_physiology_population_pass_awaiting_discovery"
)
STATUS_DISC_HOLD = "phase20a_preanswer_physiology_discovery_hold"
STATUS_DISC_PASS = (
    "phase20a_preanswer_physiology_discovery_pass_awaiting_validation"
)
STATUS_VAL_HOLD = "phase20a_preanswer_physiology_validation_hold"
STATUS_PASS = "phase20a_preanswer_physiology_validation_pass_awaiting_audit"

GUARANTEE = (
    "PHASE 20A WAS THE FIRST PHYSIOLOGY TEST ON SAME-PROMPT STOCHASTIC POLICY "
    "TRAJECTORIES. PHYSIOLOGY MODEL SELECTION USED ONLY THE FROZEN ENRICHED TRAIN "
    "POPULATION. THE UNSEEN PHASE-19 VALIDATION FAMILIES WERE NOT USED FOR LAYER, "
    "POSITION, PROBE, BASELINE, OR THRESHOLD SELECTION. K0 WAS A NEGATIVE CONTROL "
    "ONLY. ALL PRIMARY TRAJECTORIES WERE MEASURED BEFORE THE FINAL RESPONSE "
    "PROPOSITION. NO LOCKED-FAMILY CALLS, POST-RESULT TUNING, PHASE-5 PROBE "
    "SCORING, OR CAUSAL INTERVENTIONS WERE PERFORMED."
)


def _sha_ids(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def _sha_json(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


TRAIN_IDS_SHA256 = _sha_ids(TRAIN_PROMPT_GROUP_IDS)
VAL_IDS_SHA256 = _sha_ids(VAL_PROMPT_GROUP_IDS)

assert TRAIN_IDS_SHA256 == (
    "bb44e97c7e8169d0437b4a41bae938183610537547d6e735c7f210dded673da6"
)
assert VAL_IDS_SHA256 == PHASE19_USABLE_IDS_SHA256


def sample_seed(prompt_index: int, sample_index: int) -> int:
    return SEED_BASE + prompt_index * N_SAMPLES + sample_index


def build_generation_schedule(prompts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(prompts, key=lambda p: p["prompt_group_id"])
    if len(ordered) != N_PROMPTS:
        raise ValueError(f"expected {N_PROMPTS} prompts, got {len(ordered)}")
    rows = []
    for pi, p in enumerate(ordered):
        split = "train" if p["prompt_group_id"] in TRAIN_PROMPT_GROUP_IDS else "validation"
        for si in range(N_SAMPLES):
            seed = sample_seed(pi, si)
            cid = f"p20|{p['prompt_group_id']}|t{TEMPERATURE}|s{si}|seed{seed}"
            rows.append(
                {
                    "continuation_id": cid,
                    "stage": "P20A1",
                    "prompt_group_id": p["prompt_group_id"],
                    "family": p["family"],
                    "split": split,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "sample_index": si,
                    "sample_seed": seed,
                    "prompt_index": pi,
                }
            )
    if len(rows) != N_CONTINUATIONS:
        raise ValueError("schedule size")
    if min(r["sample_seed"] for r in rows) < SEED_BASE:
        raise ValueError("seed collision")
    return rows


def threshold_manifest() -> dict[str, Any]:
    body = {
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "n_prompts": N_PROMPTS,
        "n_samples": N_SAMPLES,
        "n_continuations": N_CONTINUATIONS,
        "seed_base": SEED_BASE,
        "phase15_rule_version": PHASE15_RULE_VERSION,
        "phase15_rule_hash": PHASE15_RULE_HASH,
        "train_prompt_group_ids_sha256": TRAIN_IDS_SHA256,
        "validation_prompt_group_ids_sha256": VAL_IDS_SHA256,
        "eligibility": {
            "phase15_valid": True,
            "policy_record_or_alternate": True,
            "min_stage1_tokens": MIN_STAGE1_TOKENS,
        },
        "balanced_subset": {
            "n_record": N_BALANCED_PER_CLASS,
            "n_alternate": N_BALANCED_PER_CLASS,
            "sort": "sample_seed ascending",
        },
        "population_gates": {
            "train_min_eligible": TRAIN_MIN_ELIGIBLE,
            "train_min_per_family": TRAIN_MIN_PER_FAMILY,
            "val_min_eligible": VAL_MIN_ELIGIBLE,
            "val_min_per_family": VAL_MIN_PER_FAMILY,
        },
        "blocks": list(BLOCKS),
        "positions": list(POSITIONS),
        "probe_C": PROBE_C,
        "discovery_min_auroc": DISCOVERY_MIN_AUROC,
        "discovery_min_delta": DISCOVERY_MIN_DELTA,
        "validation_min_auroc": VAL_MIN_AUROC,
        "validation_min_auroc_ci_low": VAL_MIN_AUROC_CI_LOW,
        "validation_min_family_auroc": VAL_MIN_FAMILY_AUROC,
        "validation_min_incremental": VAL_MIN_INCREMENTAL,
        "n_bootstrap": N_BOOTSTRAP,
        "phase18_status": PHASE18_STATUS_REQUIRED,
        "phase19_status": PHASE19_STATUS_REQUIRED,
    }
    return {**body, "threshold_hash": _sha_json(body)}


THRESHOLD_MANIFEST = threshold_manifest()
THRESHOLD_HASH = THRESHOLD_MANIFEST["threshold_hash"]


def is_eligible(annotated: dict[str, Any], *, n_stage1_tokens: int) -> bool:
    if not annotated.get("phase15_valid_pre_answer"):
        return False
    if annotated.get("final_policy_label") not in (LABEL_RECORD, LABEL_ALTERNATE):
        return False
    if n_stage1_tokens < MIN_STAGE1_TOKENS:
        return False
    return True


def select_balanced_subset(
    eligible_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Per-prompt: first 8 record + first 8 alternate by sample_seed."""
    by_pg: dict[str, list] = defaultdict(list)
    for r in eligible_rows:
        by_pg[r["prompt_group_id"]].append(r)
    selected: list[dict[str, Any]] = []
    eligible_prompts: list[str] = []
    for pg in sorted(by_pg):
        rows = sorted(by_pg[pg], key=lambda r: int(r["sample_seed"]))
        rec = [r for r in rows if r["final_policy_label"] == LABEL_RECORD]
        alt = [r for r in rows if r["final_policy_label"] == LABEL_ALTERNATE]
        if len(rec) < N_BALANCED_PER_CLASS or len(alt) < N_BALANCED_PER_CLASS:
            continue
        eligible_prompts.append(pg)
        for r in rec[:N_BALANCED_PER_CLASS] + alt[:N_BALANCED_PER_CLASS]:
            selected.append(
                {
                    "continuation_id": r["continuation_id"],
                    "prompt_group_id": pg,
                    "family": r["family"],
                    "split": r["split"],
                    "sample_seed": r["sample_seed"],
                    "final_policy_label": r["final_policy_label"],
                }
            )
    selected.sort(key=lambda r: r["continuation_id"])
    ids = [r["continuation_id"] for r in selected]
    return {
        "eligible_prompt_group_ids": sorted(eligible_prompts),
        "n_eligible_prompts": len(eligible_prompts),
        "selected": selected,
        "selected_continuation_ids": ids,
        "balanced_subset_sha256": _sha_ids(ids),
        "n_selected": len(ids),
    }


def evaluate_population_gates(
    eligible_prompt_meta: list[dict[str, Any]],
) -> dict[str, Any]:
    """eligible_prompt_meta: [{prompt_group_id, family, split}, ...]"""
    train = [p for p in eligible_prompt_meta if p["split"] == "train"]
    val = [p for p in eligible_prompt_meta if p["split"] == "validation"]
    train_per = {f: 0 for f in TRAIN_FAMILIES}
    for p in train:
        train_per[p["family"]] += 1
    val_per = {f: 0 for f in VAL_FAMILIES}
    for p in val:
        val_per[p["family"]] += 1
    train_ok = (
        len(train) >= TRAIN_MIN_ELIGIBLE
        and all(v >= TRAIN_MIN_PER_FAMILY for v in train_per.values())
    )
    val_ok = (
        len(val) >= VAL_MIN_ELIGIBLE
        and all(v >= VAL_MIN_PER_FAMILY for v in val_per.values())
    )
    return {
        "train": {
            "n_eligible": len(train),
            "per_family": train_per,
            "min_overall": TRAIN_MIN_ELIGIBLE,
            "min_per_family": TRAIN_MIN_PER_FAMILY,
            "passed": train_ok,
        },
        "validation": {
            "n_eligible": len(val),
            "per_family": val_per,
            "min_overall": VAL_MIN_ELIGIBLE,
            "min_per_family": VAL_MIN_PER_FAMILY,
            "passed": val_ok,
        },
        "passed": train_ok and val_ok,
    }


def lofo_folds() -> list[dict[str, Any]]:
    """Leave-one-family-out over the three train families."""
    folds = []
    for held in TRAIN_FAMILIES:
        train_f = [f for f in TRAIN_FAMILIES if f != held]
        folds.append(
            {
                "held_out_family": held,
                "train_families": train_f,
                "name": f"hold_{held}",
            }
        )
    return folds


def select_discovery_candidate(
    rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """rows: [{block, position, act_auroc, best_baseline_auroc, best_baseline_type}]"""
    eligible = [
        r
        for r in rows
        if r["act_auroc"] >= DISCOVERY_MIN_AUROC
        and (r["act_auroc"] - r["best_baseline_auroc"]) >= DISCOVERY_MIN_DELTA
    ]
    if not eligible:
        return None

    def key(r: dict[str, Any]) -> tuple:
        delta = r["act_auroc"] - r["best_baseline_auroc"]
        return (
            -delta,
            -r["act_auroc"],
            POSITION_TIE_RANK[r["position"]],
            r["block"],
        )

    eligible.sort(key=key)
    chosen = eligible[0]
    path = (
        f"eligible={len(eligible)}; "
        f"max_delta={chosen['act_auroc'] - chosen['best_baseline_auroc']:.6f}; "
        f"max_auroc={chosen['act_auroc']:.6f}; "
        f"position={chosen['position']}; block={chosen['block']}"
    )
    return {**chosen, "tie_break_path": path, "n_eligible_candidates": len(eligible)}


__all__ = [
    "TRAIN_PROMPT_GROUP_IDS",
    "VAL_PROMPT_GROUP_IDS",
    "TRAIN_IDS_SHA256",
    "VAL_IDS_SHA256",
    "N_CONTINUATIONS",
    "N_PROMPTS",
    "N_SAMPLES",
    "TEMPERATURE",
    "TOP_P",
    "SEED_BASE",
    "BLOCKS",
    "POSITIONS",
    "THRESHOLD_MANIFEST",
    "THRESHOLD_HASH",
    "STATUS_AUTHORIZED",
    "STATUS_GEN_DONE",
    "STATUS_POP_HOLD",
    "STATUS_DISC_HOLD",
    "STATUS_VAL_HOLD",
    "STATUS_PASS",
    "GUARANTEE",
    "PHASE15_RULE_HASH",
    "build_generation_schedule",
    "is_eligible",
    "select_balanced_subset",
    "evaluate_population_gates",
    "lofo_folds",
    "select_discovery_candidate",
    "annotate_continuation",
    "_sha_ids",
    "_sha_json",
]
