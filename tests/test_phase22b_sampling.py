"""Unit tests for Phase-22B additional-rollout schedule contracts."""

from __future__ import annotations

import json
from pathlib import Path

from pre_output_physiology.phase21_roleplay import (
    N_PROMPTS,
    N_TOTAL,
    build_generation_schedule,
    load_apollo_dataset,
    schedule_sha,
)
from pre_output_physiology.phase22b_sampling import (
    N_ADDITIONAL,
    PHASE21_SCHEDULE_SHA,
    PHASE21_SEED_BASE,
    PHASE21_SEED_MAX,
    PHASE21_SPLIT_SHA,
    ROLLOUT_OFFSET,
    SEED_BASE,
    build_additional_schedule,
    evaluate_k20_population,
)

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "data/raw/apollo_roleplaying/dataset.yaml"
SPLIT = REPO / "data/processed/phase21_roleplay/prompt_split.json"


def test_additional_schedule_disjoint() -> None:
    prompts = load_apollo_dataset(DATASET)
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    assert split["prompt_split_sha256"] == PHASE21_SPLIT_SHA

    p21 = build_generation_schedule(prompts, split["split_by_prompt_id"])
    assert len(p21) == N_TOTAL
    assert schedule_sha(p21) == PHASE21_SCHEDULE_SHA

    add = build_additional_schedule(prompts, split["split_by_prompt_id"])
    assert len(add) == N_ADDITIONAL == N_PROMPTS * 10
    assert {r["rollout_index"] for r in add} == set(range(10, 20))
    assert all(r["continuation_id"].endswith(tuple(f"__r{i:02d}" for i in range(10, 20))) for r in add[:20])

    p21_seeds = {r["sample_seed"] for r in p21}
    add_seeds = {r["sample_seed"] for r in add}
    assert not (p21_seeds & add_seeds)
    assert min(add_seeds) == SEED_BASE
    assert max(add_seeds) == SEED_BASE + N_ADDITIONAL - 1
    assert max(add_seeds) > PHASE21_SEED_MAX
    assert min(add_seeds) > PHASE21_SEED_MAX
    assert PHASE21_SEED_BASE == 21_000_000

    p21_ids = {r["continuation_id"] for r in p21}
    add_ids = {r["continuation_id"] for r in add}
    assert not (p21_ids & add_ids)
    assert ROLLOUT_OFFSET == 10


def test_k20_combine_counts() -> None:
    """Synthetic: combine 10+10 and require gates shape."""
    prompts = load_apollo_dataset(DATASET)
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    split_map = split["split_by_prompt_id"]

    def _mk(pid: str, k: int, label: str, seed: int, explicit: bool = False) -> dict:
        row = {
            "continuation_id": f"{pid}__r{k:02d}",
            "prompt_id": pid,
            "split": split_map[pid],
            "sample_seed": seed,
            "label": label,
            "explicit_span": explicit if label == "deceptive" else False,
            "onset_char_start": 0 if (label == "deceptive" and explicit) else None,
        }
        return row

    # First 25 train + 8 test: make them qualify at K=20
    train_ids = split["train_prompt_ids"][:25]
    test_ids = split["test_prompt_ids"][:8]
    p21: list[dict] = []
    p22: list[dict] = []
    for i, pid in enumerate(prompts):
        pid_s = pid["prompt_id"]
        for k in range(10):
            # Default: all other (ambiguous)
            p21.append(_mk(pid_s, k, "ambiguous", 21_000_000 + i * 10 + k))
            p22.append(_mk(pid_s, 10 + k, "ambiguous", 21_100_000 + i * 10 + k))
    # Overwrite qualifying prompts: 2 honest + 2 explicit deceptive across 20
    for batch_ids, seed_off in ((train_ids, 0), (test_ids, 1000)):
        for j, pid_s in enumerate(batch_ids):
            # put 1h+1d in phase21, 1h+1d in phase22b
            for rows, base_k, base_seed in (
                (p21, 0, 21_000_000 + seed_off + j * 10),
                (p22, 10, 21_100_000 + seed_off + j * 10),
            ):
                # find and patch first two slots
                idxs = [
                    i
                    for i, r in enumerate(rows)
                    if r["prompt_id"] == pid_s
                ]
                rows[idxs[0]] = _mk(pid_s, base_k, "honest", base_seed)
                rows[idxs[1]] = _mk(
                    pid_s, base_k + 1, "deceptive", base_seed + 1, explicit=True
                )

    result = evaluate_k20_population(p21, p22)
    assert result["gates"]["n_train_qualifying"] >= 25
    assert result["gates"]["n_test_qualifying"] >= 8
    assert result["gates"]["passed"] is True
    assert result["status"].startswith("phase22b_k20_population_pass")
