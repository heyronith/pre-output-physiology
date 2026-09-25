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
    assert "tpr_at_or_below_1pct_fpr" in metrics
    assert "actual_fpr_at_tpr_metric" in metrics
    assert metrics["actual_fpr_at_tpr_metric"] <= 0.01


def test_low_fpr_metric_never_uses_operating_point_above_target() -> None:
    """Construct scores where the first FPR>=1% point differs from best FPR<=1%."""
    from pre_output_physiology.metrics import tpr_at_or_below_fpr

    # 100 negatives, 100 positives. Create a score distribution where:
    # - a high-TPR point exists only after FPR exceeds 1%
    # - eligible FPR<=1% points have lower TPR
    y = np.array([0] * 100 + [1] * 100)
    scores = np.zeros(200, dtype=float)
    # Rank: top scores are mostly positives, but inject negatives early to force
    # ROC steps above/below 1% FPR.
    scores[100:] = np.linspace(1.0, 0.5, 100)  # positives
    scores[:100] = np.linspace(0.49, 0.0, 100)  # negatives
    # Move 5 negatives into the top of the ranking so FPR grows in coarse steps.
    scores[0:5] = np.linspace(0.99, 0.95, 5)

    result = tpr_at_or_below_fpr(y, scores, target_fpr=0.01)
    assert result["actual_fpr_at_tpr_metric"] <= 0.01 + 1e-12
    metrics = evaluate_binary_classifier(y, scores, n_bootstrap=20, bootstrap_seed=0)
    assert metrics["actual_fpr_at_tpr_metric"] <= 0.01 + 1e-12
    assert metrics["tpr_at_or_below_1pct_fpr"] == result["tpr_at_or_below_target_fpr"]

    # Explicitly verify no eligible-above-threshold selection:
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y, scores)
    above = fpr > 0.01
    if above.any():
        # Reported TPR must equal max TPR on fpr<=0.01, not on fpr>0.01-only points.
        eligible = fpr <= 0.01
        assert metrics["tpr_at_or_below_1pct_fpr"] == float(np.max(tpr[eligible]))
        assert metrics["tpr_at_or_below_1pct_fpr"] <= float(np.max(tpr[above])) + 1e-12


def test_revision_cannot_be_main() -> None:
    assert_not_main_revision("model_revision", "abc123")
    try:
        assert_not_main_revision("model_revision", "main")
        raised = False
    except ValueError:
        raised = True
    assert raised
    assert sha256_text("x") == sha256_text("x")
