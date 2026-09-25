"""Probe normalization and fitting tests with synthetic activations."""

from __future__ import annotations

import numpy as np

from pre_output_physiology.metrics import evaluate_binary_classifier
from pre_output_physiology.probes import MeanLinearProbe, ProbeConfig
from pre_output_physiology.provenance import assert_not_main_revision, sha256_text


def test_probe_normalization_uses_train_statistics_only() -> None:
    rng = np.random.default_rng(0)
    x_train = rng.normal(loc=5.0, scale=2.0, size=(200, 16))
    y_train = (x_train[:, 0] > 5.0).astype(int)
    x_test = rng.normal(loc=50.0, scale=2.0, size=(40, 16))

    probe = MeanLinearProbe(ProbeConfig()).fit(x_train, y_train)
    mean = np.asarray(probe.transformation_mean)
    # Train mean should be near 5, not contaminated by test mean ~50
    assert np.allclose(mean.mean(), 5.0, atol=0.5)
    scores = probe.predict_proba(x_test)
    assert scores.shape == (40,)


def test_probe_hyperparams_match_phase2_contract() -> None:
    cfg = ProbeConfig()
    assert cfg.C == 0.01
    assert cfg.fit_intercept is True
    assert cfg.max_iter == 500
    assert cfg.random_state == 42
    assert cfg.normalize is True


def test_perfect_separation_metrics() -> None:
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = evaluate_binary_classifier(y, scores, n_bootstrap=50, bootstrap_seed=0)
    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0
    assert "auroc_ci_low" in metrics


def test_revision_cannot_be_main() -> None:
    assert_not_main_revision("model_revision", "abc123")
    try:
        assert_not_main_revision("model_revision", "main")
        raised = False
    except ValueError:
        raised = True
    assert raised
    assert sha256_text("x") == sha256_text("x")
