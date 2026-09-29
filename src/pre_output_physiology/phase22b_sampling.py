"""Phase 22B — prospective K=20 sampling validation on Apollo RoleplayDeception.

Adds 10 new independent rollouts (r10–r19) per exact Phase-21 prompt.
Reuses Phase-21 prompts, split, grading, onset, and population gates unchanged.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from pre_output_physiology.phase21_roleplay import (
    MAX_NEW_TOKENS,
    MIN_TEST_QUALIFYING,
    MIN_TRAIN_QUALIFYING,
    N_PROMPTS,
    N_ROLLOUTS,
    TEMPERATURE,
    schedule_sha,
    select_mixed_population,
)

# Disjoint from Phase-21 seeds [21000000, 21003709]
SEED_BASE = 21_100_000
ROLLOUT_OFFSET = 10  # new rollout_index 10..19
N_ADDITIONAL = N_PROMPTS * N_ROLLOUTS  # 3710
PHASE21_SEED_BASE = 21_000_000
PHASE21_SEED_MAX = PHASE21_SEED_BASE + N_ADDITIONAL - 1  # 21003709

PHASE21_SPLIT_SHA = "fd3c5eecbc89681a665ef22bd989ea1ee88231bdf9ff45e262a0c313ac9ac3b3"
PHASE21_GEN_RUN = "phase21_gen_20260928T180342Z_3d3671eb"
PHASE21_GRADE_RUN = "phase21_grade_20260928T205710Z_b6287702"
PHASE21_ONSET_RUN = "phase21_onset_20260928T212924Z_68a5621c"
PHASE21_SCHEDULE_SHA = "ed83f7f469b59159a4fda28203517290a36f140f5bf2d0c1ea19ff87def20b0f"

STATUS_AUTHORIZED = "phase22b_k20_sampling_validation_authorized"
STATUS_GEN_DONE = "phase22b_k20_generation_complete_awaiting_grading"
STATUS_GRADED = "phase22b_k20_grading_complete_awaiting_onset"
STATUS_ONSET_DONE = "phase22b_k20_onset_complete_awaiting_population"
STATUS_POP_PASS = "phase22b_k20_population_pass_awaiting_audit"
STATUS_POP_HOLD = "phase22b_k20_population_hold"

GUARANTEE = (
    "PHASE 22B WAS A BEHAVIOR-ONLY PROSPECTIVE K=20 SAMPLING VALIDATION ON THE "
    "ORIGINAL 371 APOLLO ROLEPLAYDECEPTION PROMPTS. TEN ADDITIONAL INDEPENDENT "
    "ROLLOUTS WERE GENERATED FOR EVERY PROMPT (R10–R19) WITH SEEDS DISJOINT FROM "
    "PHASE 21. PHASE-21 ROLLOUTS R00–R09 WERE KEPT UNCHANGED. GRADING USED "
    "GPT-4O-2024-08-06 AND THE EXACT APOLLO TEMPLATE. ONSET ANNOTATION WAS "
    "TEXT-ONLY. POPULATION GATES WERE UNCHANGED (≥2 HONEST AND ≥2 EXPLICIT-ONSET "
    "DECEPTIVE; TRAIN≥25 / TEST≥8). NO ACTIVATIONS, PROBE FITTING, PHYSIOLOGY, "
    "PROMPT CHANGES, THRESHOLD CHANGES, OR K>20 GENERATION WERE PERFORMED."
)


def build_additional_schedule(
    prompts: Sequence[dict[str, Any]],
    split_map: dict[str, str],
    *,
    seed_base: int = SEED_BASE,
    rollout_offset: int = ROLLOUT_OFFSET,
) -> list[dict[str, Any]]:
    """Build schedule for rollouts r10–r19 with seeds disjoint from Phase 21."""
    rows: list[dict[str, Any]] = []
    for p in prompts:
        pid = p["prompt_id"]
        for k in range(N_ROLLOUTS):
            global_index = int(p["scenario_index"]) * N_ROLLOUTS + k
            rollout_index = rollout_offset + k
            seed = seed_base + global_index
            if PHASE21_SEED_BASE <= seed <= PHASE21_SEED_MAX:
                raise ValueError(f"seed {seed} collides with Phase-21 range")
            rows.append(
                {
                    "continuation_id": f"{pid}__r{rollout_index:02d}",
                    "prompt_id": pid,
                    "scenario_index": int(p["scenario_index"]),
                    "rollout_index": rollout_index,
                    "sample_seed": seed,
                    "split": split_map[pid],
                    "temperature": TEMPERATURE,
                    "max_new_tokens": MAX_NEW_TOKENS,
                }
            )
    if len(rows) != N_ADDITIONAL:
        raise ValueError(f"schedule length {len(rows)} != {N_ADDITIONAL}")
    seeds = [r["sample_seed"] for r in rows]
    if len(set(seeds)) != N_ADDITIONAL:
        raise ValueError("duplicate seeds in additional schedule")
    ids = [r["continuation_id"] for r in rows]
    if len(set(ids)) != N_ADDITIONAL:
        raise ValueError("duplicate continuation_ids")
    for r in rows:
        if int(r["rollout_index"]) < rollout_offset:
            raise ValueError(f"unexpected low rollout_index {r['rollout_index']}")
    return rows


def _is_explicit_deceptive(row: dict[str, Any]) -> bool:
    return (
        row.get("label") == "deceptive"
        and row.get("explicit_span") is True
        and row.get("onset_char_start") is not None
    )


def per_prompt_behavior_counts(
    rows: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Aggregate honest / deceptive-explicit / other counts per prompt."""
    by: dict[str, dict[str, Any]] = {}
    for r in rows:
        pid = r["prompt_id"]
        slot = by.setdefault(
            pid,
            {
                "prompt_id": pid,
                "split": r["split"],
                "n_honest": 0,
                "n_deceptive_explicit": 0,
                "n_other": 0,
                "n_total": 0,
            },
        )
        slot["n_total"] += 1
        if r.get("label") == "honest":
            slot["n_honest"] += 1
        elif _is_explicit_deceptive(r):
            slot["n_deceptive_explicit"] += 1
        else:
            slot["n_other"] += 1
    return by


def analyze_behavior_switching(
    phase21_rows: Sequence[dict[str, Any]],
    phase22b_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Measure whether one-sided Phase-21 prompts acquire the missing class in 22B."""
    c21 = per_prompt_behavior_counts(phase21_rows)
    c22 = per_prompt_behavior_counts(phase22b_rows)

    buckets = {
        "honest_gt0_dec_eq0": [],
        "dec_gt0_honest_eq0": [],
        "exact_10_0_or_0_10": [],
    }
    for pid, a in c21.items():
        h, d = a["n_honest"], a["n_deceptive_explicit"]
        if h > 0 and d == 0:
            buckets["honest_gt0_dec_eq0"].append(pid)
        if d > 0 and h == 0:
            buckets["dec_gt0_honest_eq0"].append(pid)
        if (h == 10 and d == 0) or (h == 0 and d == 10):
            buckets["exact_10_0_or_0_10"].append(pid)

    def _switch_stats(pids: list[str], need: str) -> dict[str, Any]:
        """need: 'dec' or 'honest' — previously unseen class to acquire."""
        train_acq = test_acq = 0
        train_n = test_n = 0
        acquired_ids: list[str] = []
        for pid in pids:
            a21 = c21[pid]
            a22 = c22.get(pid, {"n_honest": 0, "n_deceptive_explicit": 0, "split": a21["split"]})
            split = a21["split"]
            if split == "train":
                train_n += 1
            else:
                test_n += 1
            got = (
                a22["n_deceptive_explicit"] > 0
                if need == "dec"
                else a22["n_honest"] > 0
            )
            if got:
                acquired_ids.append(pid)
                if split == "train":
                    train_acq += 1
                else:
                    test_acq += 1
        return {
            "n_prompts": len(pids),
            "n_train": train_n,
            "n_test": test_n,
            "n_acquired_unseen_class": len(acquired_ids),
            "n_train_acquired": train_acq,
            "n_test_acquired": test_acq,
            "acquisition_rate": (
                len(acquired_ids) / len(pids) if pids else None
            ),
        }

    return {
        "phase21_only_buckets": {
            k: {
                "n": len(v),
                "n_train": sum(1 for p in v if c21[p]["split"] == "train"),
                "n_test": sum(1 for p in v if c21[p]["split"] == "test"),
            }
            for k, v in buckets.items()
        },
        "switching": {
            "honest_gt0_dec_eq0_acquires_dec_in_11_20": _switch_stats(
                buckets["honest_gt0_dec_eq0"], "dec"
            ),
            "dec_gt0_honest_eq0_acquires_honest_in_11_20": _switch_stats(
                buckets["dec_gt0_honest_eq0"], "honest"
            ),
            "exact_10_0_or_0_10": {
                "n_10_0": sum(
                    1 for p in buckets["exact_10_0_or_0_10"] if c21[p]["n_honest"] == 10
                ),
                "n_0_10": sum(
                    1
                    for p in buckets["exact_10_0_or_0_10"]
                    if c21[p]["n_deceptive_explicit"] == 10
                ),
                "ten_zero_acquires_dec_in_11_20": _switch_stats(
                    [p for p in buckets["exact_10_0_or_0_10"] if c21[p]["n_honest"] == 10],
                    "dec",
                ),
                "zero_ten_acquires_honest_in_11_20": _switch_stats(
                    [
                        p
                        for p in buckets["exact_10_0_or_0_10"]
                        if c21[p]["n_deceptive_explicit"] == 10
                    ],
                    "honest",
                ),
            },
        },
    }


def evaluate_k20_population(
    phase21_annotated: Sequence[dict[str, Any]],
    phase22b_annotated: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Combine 10+10 rollouts and apply frozen Phase-21 mixed-population gates."""
    if len(phase21_annotated) != N_ADDITIONAL:
        raise ValueError(f"phase21 rows {len(phase21_annotated)} != {N_ADDITIONAL}")
    if len(phase22b_annotated) != N_ADDITIONAL:
        raise ValueError(f"phase22b rows {len(phase22b_annotated)} != {N_ADDITIONAL}")

    combined = list(phase21_annotated) + list(phase22b_annotated)
    # Verify 20 per prompt
    counts = Counter(r["prompt_id"] for r in combined)
    if len(counts) != N_PROMPTS or any(v != 20 for v in counts.values()):
        bad = {k: v for k, v in counts.items() if v != 20}
        raise ValueError(f"expected 20/prompt; bad examples {list(bad.items())[:5]}")

    pop = select_mixed_population(combined)
    # Remap status to Phase-22B names (select_mixed_population uses Phase-21 statuses)
    gates = pop["gates"]
    passed = bool(gates["passed"])
    status = STATUS_POP_PASS if passed else STATUS_POP_HOLD
    switching = analyze_behavior_switching(phase21_annotated, phase22b_annotated)

    return {
        **pop,
        "status": status,
        "k": 20,
        "n_combined_rows": len(combined),
        "gates": {
            **gates,
            "min_train_qualifying": MIN_TRAIN_QUALIFYING,
            "min_test_qualifying": MIN_TEST_QUALIFYING,
        },
        "switching": switching,
        "phase21_onset_run": PHASE21_ONSET_RUN,
    }


__all__ = [
    "SEED_BASE",
    "ROLLOUT_OFFSET",
    "N_ADDITIONAL",
    "STATUS_AUTHORIZED",
    "STATUS_GEN_DONE",
    "STATUS_GRADED",
    "STATUS_ONSET_DONE",
    "STATUS_POP_PASS",
    "STATUS_POP_HOLD",
    "GUARANTEE",
    "PHASE21_SPLIT_SHA",
    "PHASE21_GEN_RUN",
    "PHASE21_GRADE_RUN",
    "PHASE21_ONSET_RUN",
    "build_additional_schedule",
    "schedule_sha",
    "evaluate_k20_population",
    "analyze_behavior_switching",
]
