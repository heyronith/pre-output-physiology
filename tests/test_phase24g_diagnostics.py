"""Phase-24G exploratory diagnostics unit tests (no GPU)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pre_output_physiology.phase24c_design import TEMPORAL_LANDMARKS
from pre_output_physiology.phase24f_confirmation import confirmatory_pass
from pre_output_physiology.phase24g_diagnostics import (
    ANALYSIS_SEED,
    EXPLORATORY_LABEL,
    NESTED_REPS,
    NESTED_REPS_TARGET_NOTE,
    assert_phase24f_immutable,
    candidate_in_neighborhood,
    classify_diagnostic_category,
    cosine_similarity,
    matrix_from_grid,
    power_projection,
    probe_stability_from_coefs,
    tag_exploratory,
)

REPO = Path(__file__).resolve().parents[1]


def test_assert_phase24f_immutable_passes() -> None:
    out = assert_phase24f_immutable(REPO)
    assert out["immutable"] is True
    assert out["primary_pass"] is False
    assert out["label"] == "PHASE24F_PRIMARY_IMMUTABLE"


def test_tag_exploratory_adds_label() -> None:
    tagged = tag_exploratory({"x": 1})
    assert tagged["analysis_label"] == EXPLORATORY_LABEL
    nested = tag_exploratory({"inner": {"y": 2}})
    assert nested["analysis_label"] == EXPLORATORY_LABEL
    assert nested["inner"]["analysis_label"] == EXPLORATORY_LABEL


def test_confirmatory_pass_still_false_for_frozen_24f() -> None:
    blob = json.loads(
        (REPO / "artifacts/phase24f_confirmation/primary_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert confirmatory_pass(
        delta_auroc=float(blob["delta_auroc"]),
        ci95_low=float(blob["delta_auroc_bootstrap"]["ci95"][0]),
    ) is False


def test_cosine_and_probe_stability() -> None:
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([1.0, 0.0, 0.0])
    assert cosine_similarity(u, v) == 1.0
    w = np.array([0.0, 1.0, 0.0])
    assert cosine_similarity(u, w) == 0.0
    coefs = [np.array([1.0, 2.0]), np.array([1.0, 2.1]), np.array([0.9, 2.0])]
    stab = probe_stability_from_coefs(coefs)
    assert stab["n"] == 3
    assert np.isfinite(stab["mean_pairwise_cosine"])
    assert stab["mean_pairwise_cosine"] > 0.99


def test_power_projection_shape() -> None:
    deltas = [0.01, -0.02, 0.05, 0.0, 0.03]
    effects = [0.05, 0.08, 0.10, 0.15]
    n_grid = [11, 20, 30, 40, 60, 80, 100]
    out = power_projection(
        observed_prompt_deltas=deltas,
        effect_sizes=effects,
        n_prompts_grid=n_grid,
        n_sims=50,
        seed=ANALYSIS_SEED,
    )
    assert len(out["rows"]) == len(effects) * len(n_grid)
    for row in out["rows"]:
        assert 0.0 <= row["power_estimate"] <= 1.0


def test_classify_diagnostic_category_abcd() -> None:
    assert (
        classify_diagnostic_category(
            {
                "topology_pearson": 0.05,
                "lopo_neighborhood_frac": 0.1,
                "test_same_prompt_frac_gt0": 0.2,
            }
        )
        == "D"
    )
    assert (
        classify_diagnostic_category(
            {
                "topology_pearson": 0.2,
                "lopo_exact_frac": 0.1,
                "lopo_neighborhood_frac": 0.3,
                "test_same_prompt_frac_gt0": 0.6,
            }
        )
        == "C"
    )
    assert (
        classify_diagnostic_category(
            {
                "topology_pearson": 0.4,
                "lopo_exact_frac": 0.5,
                "lopo_neighborhood_frac": 0.5,
                "test_same_prompt_frac_gt0": 0.75,
            }
        )
        == "B"
    )
    assert (
        classify_diagnostic_category(
            {
                "topology_pearson": 0.4,
                "lopo_exact_frac": 0.2,
                "lopo_neighborhood_frac": 0.4,
                "test_delta_auroc": 0.08,
                "test_ci_width": 0.3,
                "test_same_prompt_frac_gt0": 0.55,
            }
        )
        == "A"
    )


def test_exploratory_label_constant() -> None:
    assert EXPLORATORY_LABEL == "EXPLORATORY_POST_CONFIRMATION"


def test_nested_reps_documented() -> None:
    assert NESTED_REPS == 50
    assert NESTED_REPS_TARGET_NOTE == 1000
    assert NESTED_REPS < NESTED_REPS_TARGET_NOTE


def test_matrix_from_grid_shape_32x7() -> None:
    grid = {t: {L: float(t + L) for L in range(32)} for t in TEMPORAL_LANDMARKS}
    mat = matrix_from_grid(grid)
    assert mat.shape == (32, len(TEMPORAL_LANDMARKS))
    assert mat.shape == (32, 7)


def test_candidate_in_neighborhood() -> None:
    assert candidate_in_neighborhood(
        {"candidate_time": 1, "candidate_layer": 20}
    ) is True
    assert candidate_in_neighborhood(
        {"candidate_time": 32, "candidate_layer": 17}
    ) is False
    assert candidate_in_neighborhood(None) is False
