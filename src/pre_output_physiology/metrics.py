"""Classification metrics for Phase 2 probe evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.01) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # First threshold where FPR >= target; if never, use last TPR.
    idxs = np.where(fpr >= target_fpr)[0]
    if len(idxs) == 0:
        return float(tpr[-1])
    return float(tpr[idxs[0]])


def bootstrap_auroc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_bootstrap: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        ys = y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        stats.append(float(roc_auc_score(yt, ys)))
    if not stats:
        raise ValueError("Bootstrap could not compute AUROC (class collapse in resamples)")
    arr = np.asarray(stats)
    lower = float(np.quantile(arr, alpha / 2))
    upper = float(np.quantile(arr, 1 - alpha / 2))
    return {
        "auroc_bootstrap_mean": float(arr.mean()),
        "auroc_ci_low": lower,
        "auroc_ci_high": upper,
        "n_bootstrap_effective": int(len(arr)),
    }


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray | None = None,
    *,
    n_bootstrap: int = 2000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if y_pred is None:
        y_pred = (y_score >= 0.5).astype(int)
    else:
        y_pred = np.asarray(y_pred).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    prevalence = float(y_true.mean()) if len(y_true) else float("nan")
    majority_acc = float(max(prevalence, 1 - prevalence))

    metrics: dict[str, Any] = {
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "n_negative": int((1 - y_true).sum()),
        "class_prevalence_positive": prevalence,
        "majority_baseline_accuracy": majority_acc,
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1),
        "tpr_at_1pct_fpr": tpr_at_fpr(y_true, y_score, 0.01),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }
    metrics.update(
        bootstrap_auroc_ci(
            y_true, y_score, n_bootstrap=n_bootstrap, seed=bootstrap_seed
        )
    )
    return metrics
