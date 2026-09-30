"""Phase-24D collection schedule / sealing / integrity tests (no GPU)."""

from __future__ import annotations

import json
from pathlib import Path

from pre_output_physiology.phase24c_design import sha256_json
from pre_output_physiology.phase24d_collection import (
    N_NEW_TRAJECTORIES,
    N_REUSED_24B,
    N_TOTAL_TRAJECTORIES,
    SELECTED_K,
    assert_no_test_outcome_leakage,
    build_collection_schedule,
    capture_impl_source_sha,
    evaluate_integrity_gates_528,
    evaluate_realized_yield,
    final_status,
    realized_yield_counts,
    seal_test_summary,
    trajectory_seed_phase24c,
    verify_phase24c_artifact_hashes,
)

REPO = Path(__file__).resolve().parents[1]


def _prompts_by_id() -> dict:
    rows = [
        json.loads(x)
        for x in (REPO / "data/processed/phase21_roleplay/prompts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if x.strip()
    ]
    return {r["prompt_id"]: r for r in rows}


def _schedule():
    template = json.loads(
        (REPO / "artifacts/phase24c_design/generation_schedule_template.json").read_text(
            encoding="utf-8"
        )
    )
    pilot = json.loads(
        (REPO / "artifacts/phase24b_live/pilot_manifest.json").read_text(encoding="utf-8")
    )
    return build_collection_schedule(
        schedule_template=template,
        prompts_by_id=_prompts_by_id(),
        pilot_24b_ids=pilot["prompt_ids"],
    )


def test_phase24c_hashes_verify() -> None:
    ver = verify_phase24c_artifact_hashes(REPO)
    assert ver["verified"] is True
    assert ver["selected_k_check"]["pass"] is True


def test_exact_624_and_528_schedule() -> None:
    s = _schedule()
    assert s["n_new"] == N_NEW_TRAJECTORIES == 528
    assert s["n_reuse"] == N_REUSED_24B == 96
    assert s["n_total_after"] == N_TOTAL_TRAJECTORIES == 624
    assert s["selected_k"] == SELECTED_K == 16


def test_phase24b_reuse_and_no_duplicates() -> None:
    s = _schedule()
    pairs = {(r["prompt_id"], r["replicate_index"]) for r in s["new_rows"]}
    reuse = {(r["prompt_id"], r["replicate_index"]) for r in s["reuse_refs"]}
    assert pairs.isdisjoint(reuse)
    assert len(pairs | reuse) == 624
    # pilot new only 6..15
    pilot = json.loads(
        (REPO / "artifacts/phase24b_live/pilot_manifest.json").read_text(encoding="utf-8")
    )
    pilot_set = set(pilot["prompt_ids"])
    for r in s["new_rows"]:
        if r["prompt_id"] in pilot_set:
            assert 6 <= r["replicate_index"] <= 15
        else:
            assert 0 <= r["replicate_index"] <= 15
    for r in s["reuse_refs"]:
        assert r["prompt_id"] in pilot_set
        assert 0 <= r["replicate_index"] <= 5


def test_frozen_seeds_deterministic() -> None:
    assert trajectory_seed_phase24c("roleplay_091", 6) == trajectory_seed_phase24c(
        "roleplay_091", 6
    )
    s = _schedule()
    for r in s["new_rows"]:
        assert r["sample_seed"] == trajectory_seed_phase24c(
            r["prompt_id"], r["replicate_index"]
        )


def test_split_isolation() -> None:
    split = json.loads(
        (REPO / "artifacts/phase24c_design/split_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    s = _schedule()
    train, val, test = (
        set(split["train_prompt_ids"]),
        set(split["validation_prompt_ids"]),
        set(split["test_prompt_ids"]),
    )
    for r in s["new_rows"]:
        if r["split"] == "train":
            assert r["prompt_id"] in train
            assert not r["sealed"]
        elif r["split"] == "validation":
            assert r["prompt_id"] in val
            assert not r["sealed"]
        else:
            assert r["split"] == "test"
            assert r["prompt_id"] in test
            assert r["sealed"] is True


def test_test_sealing_summary_omits_outcomes() -> None:
    sealed = seal_test_summary(
        n_requested=176,
        n_completed=176,
        n_graded=176,
        integrity_ok=True,
        artifact_hashes={"labels_SEALED.json": "abc"},
    )
    blob = json.dumps(sealed)
    assert "honest" not in blob.lower() or "omitted" in blob
    assert "H_A_D_totals" in sealed["omitted"]
    assert sealed["sealed"] is True
    assert "per_prompt" not in sealed


def test_no_test_outcome_summaries_in_report_text() -> None:
    ok = assert_no_test_outcome_leakage(
        "# Report\n## LOCKED TEST\ntrajectories completed: 176\nintegrity ok\n"
    )
    assert ok is True
    bad = assert_no_test_outcome_leakage("Test H/A/D totals: 10/2/8\n")
    assert bad is False


def test_integrity_gates_and_yield() -> None:
    g = evaluate_integrity_gates_528(
        n_requested=528,
        n_completed=528,
        n_alignment_ok=528,
        n_activation_nan=0,
        n_logit_nan=0,
        n_artifact_ok=528,
        n_seed_unchanged=528,
    )
    assert g["passed"]
    train = realized_yield_counts(
        [
            {"prompt_id": f"p{i}", "open_class": "honest", "open_valid": True}
            for i in range(20)
            for _ in range(2)
        ]
        + [
            {"prompt_id": f"p{i}", "open_class": "deceptive", "open_valid": True}
            for i in range(20)
            for _ in range(2)
        ],
        prompt_ids=[f"p{i}" for i in range(20)],
    )
    assert train["n_prompts_ge2h_ge2d"] == 20
    val = realized_yield_counts(
        [
            {"prompt_id": f"v{i}", "open_class": "honest", "open_valid": True}
            for i in range(8)
            for _ in range(2)
        ]
        + [
            {"prompt_id": f"v{i}", "open_class": "deceptive", "open_valid": True}
            for i in range(8)
            for _ in range(2)
        ],
        prompt_ids=[f"v{i}" for i in range(8)],
    )
    y = evaluate_realized_yield(train_yield=train, val_yield=val)
    assert y["passed"]
    assert final_status(engineering_pass=True, yield_pass=True).startswith("phase24d_primary")


def test_capture_impl_equivalence_when_modal_present() -> None:
    p24b = REPO / "modal/phase24b_live_capture.py"
    p24d = REPO / "modal/phase24d_live_collection.py"
    if p24d.exists():
        assert capture_impl_source_sha(p24b) == capture_impl_source_sha(p24d)


def test_schedule_sha_stable() -> None:
    a = _schedule()
    b = _schedule()
    assert a["new_schedule_sha256"] == b["new_schedule_sha256"]
    assert a["new_schedule_sha256"] == sha256_json(a["new_rows"])
