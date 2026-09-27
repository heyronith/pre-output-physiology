"""Tests for Phase 16A sampling-scaling forecast."""

from __future__ import annotations

from pre_output_physiology.phase16_forecast import (
    DESIGN_HASH,
    N_GRID,
    STATUS_UNSUPPORTED,
    design_manifest,
    empirical_prompt_counts,
    interpret_forecast,
    run_optimistic_forecast,
    run_primary_forecast,
)


def test_design_hash_stable() -> None:
    assert design_manifest()["design_hash"] == DESIGN_HASH
    assert list(N_GRID) == [16, 32, 48, 64, 96, 128]


def test_empirical_prompt_counts_shape() -> None:
    rows = []
    for t in (0.7, 0.9, 1.1):
        for pg in range(32):
            for i in range(16):
                valid = i < 14
                alt = valid and i < 2
                rows.append(
                    {
                        "temperature": t,
                        "prompt_group_id": f"calib_{pg:02d}",
                        "phase15_valid_pre_answer": valid,
                        "final_policy_label": (
                            "goal_favored_alternate_choice"
                            if alt
                            else "record_choice"
                            if valid
                            else "pre_answer_leak_invalid"
                        ),
                    }
                )
    counts = empirical_prompt_counts(rows)
    assert set(counts.keys()) == {0.7, 0.9, 1.1}
    assert len(counts[0.9]) == 32
    assert counts[0.9][0]["alternate"] == 2
    assert counts[0.9][0]["record"] == 12
    assert counts[0.9][0]["invalid"] == 2


def test_interpret_unsupported_when_probs_low() -> None:
    primary = {
        str(t): {
            n: {
                "prob_groups_ge4_ge20": 0.1,
                "valid_fraction": {"median": 0.95},
            }
            for n in N_GRID
        }
        for t in (0.7, 0.9, 1.1)
    }
    out = interpret_forecast(primary)
    assert out["outcome"] == "sampling_only_rescue_unsupported"
    assert out["status"] == STATUS_UNSUPPORTED
    assert out["recommended_prospective_t_n"] is None


def test_interpret_plausible_at_n64() -> None:
    primary = {
        str(t): {
            n: {
                "prob_groups_ge4_ge20": 0.9 if n >= 64 else 0.1,
                "valid_fraction": {"median": 0.95},
            }
            for n in N_GRID
        }
        for t in (0.7, 0.9, 1.1)
    }
    out = interpret_forecast(primary)
    assert out["outcome"] == "sampling_only_rescue_plausible"
    assert out["recommended_prospective_t_n"]["n"] == 64


def test_interpret_expensive_only_at_96() -> None:
    primary = {
        str(t): {
            n: {
                "prob_groups_ge4_ge20": 0.9 if n >= 96 else 0.1,
                "valid_fraction": {"median": 0.95},
            }
            for n in N_GRID
        }
        for t in (0.7, 0.9, 1.1)
    }
    out = interpret_forecast(primary)
    assert out["outcome"] == "sampling_only_rescue_possible_but_expensive"
    assert out["recommended_prospective_t_n"] is None


def test_primary_deterministic() -> None:
    rows = []
    for t in (0.7, 0.9, 1.1):
        for pg in range(32):
            # Heterogeneous: half zero-alt, half mixed
            for i in range(16):
                if pg < 16:
                    alt = False
                    valid = True
                else:
                    alt = i < 4
                    valid = True
                rows.append(
                    {
                        "temperature": t,
                        "prompt_group_id": f"g{pg}",
                        "phase15_valid_pre_answer": valid,
                        "final_policy_label": (
                            "goal_favored_alternate_choice"
                            if alt
                            else "record_choice"
                        ),
                    }
                )
    counts = empirical_prompt_counts(rows)
    a = run_primary_forecast(counts, n_replicates=2000)
    b = run_primary_forecast(counts, n_replicates=2000)
    assert a["0.9"][32]["prob_groups_ge4_ge20"] == b["0.9"][32]["prob_groups_ge4_ge20"]
    opt = run_optimistic_forecast(counts, n_replicates=2000)
    # Optimistic should find more ge4 groups than heterogeneous primary at large N
    assert (
        opt["0.9"][64]["groups_ge4_each_class"]["median"]
        >= a["0.9"][64]["groups_ge4_each_class"]["median"]
    )
