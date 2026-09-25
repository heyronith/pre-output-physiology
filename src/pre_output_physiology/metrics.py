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


def tpr_at_or_below_fpr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_fpr: float = 0.01,
) -> dict[str, float]:
    """Max TPR among ROC points with FPR <= target_fpr (no above-threshold points).

    Does not interpolate. If only the trivial (0,0)-style boundary is available at
    FPR <= target, that boundary is returned rather than jumping to FPR > target.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    eligible = np.where(fpr <= target_fpr)[0]
    if len(eligible) == 0:
        # Should not happen for sklearn roc_curve (always includes FPR=0), but be safe.
        return {
            "tpr_at_or_below_target_fpr": 0.0,
            "actual_fpr_at_tpr_metric": 0.0,
            "target_fpr": float(target_fpr),
        }
    # Among eligible points, take the maximum TPR; if ties, the lowest FPR among ties.
    eligible_tpr = tpr[eligible]
    best_local = int(np.argmax(eligible_tpr))
    # argmax returns first max; refine to lowest FPR among equal TPR
    max_tpr = eligible_tpr[best_local]
    tied = eligible[eligible_tpr == max_tpr]
    best_idx = int(tied[np.argmin(fpr[tied])])
    return {
        "tpr_at_or_below_target_fpr": float(tpr[best_idx]),
        "actual_fpr_at_tpr_metric": float(fpr[best_idx]),
        "target_fpr": float(target_fpr),
    }


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


def _group_indices(groups: np.ndarray) -> dict[object, np.ndarray]:
    groups = np.asarray(groups)
    out: dict[object, np.ndarray] = {}
    for gid in np.unique(groups):
        out[gid] = np.where(groups == gid)[0]
    return out


def bootstrap_auroc_ci_by_group(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    *,
    n_bootstrap: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Group-aware AUROC bootstrap (resample prompt/scenario groups)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    groups = np.asarray(groups)
    if len(y_true) != len(groups):
        raise ValueError("y_true and groups length mismatch")
    group_map = _group_indices(groups)
    unique_groups = np.asarray(list(group_map.keys()))
    rng = np.random.default_rng(seed)
    stats: list[float] = []
    for _ in range(n_bootstrap):
        chosen = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([group_map[g] for g in chosen])
        yt = y_true[idx]
        ys = y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        stats.append(float(roc_auc_score(yt, ys)))
    if not stats:
        raise ValueError("Group bootstrap could not compute AUROC")
    arr = np.asarray(stats)
    return {
        "auroc_bootstrap_mean": float(arr.mean()),
        "auroc_ci_low": float(np.quantile(arr, alpha / 2)),
        "auroc_ci_high": float(np.quantile(arr, 1 - alpha / 2)),
        "n_bootstrap_effective": int(len(arr)),
        "bootstrap_unit": "prompt_group",
    }


def paired_delta_auroc_ci_by_group(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    groups: np.ndarray,
    *,
    n_bootstrap: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Paired group-aware ΔAUROC = AUROC(A) − AUROC(B) on the same examples."""
    y_true = np.asarray(y_true)
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    groups = np.asarray(groups)
    if not (len(y_true) == len(scores_a) == len(scores_b) == len(groups)):
        raise ValueError("length mismatch")
    point = float(roc_auc_score(y_true, scores_a) - roc_auc_score(y_true, scores_b))
    group_map = _group_indices(groups)
    unique_groups = np.asarray(list(group_map.keys()))
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(n_bootstrap):
        chosen = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([group_map[g] for g in chosen])
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        deltas.append(
            float(roc_auc_score(yt, scores_a[idx]) - roc_auc_score(yt, scores_b[idx]))
        )
    if not deltas:
        raise ValueError("Paired group bootstrap could not compute ΔAUROC")
    arr = np.asarray(deltas)
    return {
        "delta_auroc": point,
        "delta_auroc_ci_low": float(np.quantile(arr, alpha / 2)),
        "delta_auroc_ci_high": float(np.quantile(arr, 1 - alpha / 2)),
        "n_bootstrap_effective": int(len(arr)),
        "bootstrap_unit": "prompt_group",
        "definition": "AUROC_activation - AUROC_surface",
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

    low_fpr = tpr_at_or_below_fpr(y_true, y_score, target_fpr=0.01)

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
        "tpr_at_or_below_1pct_fpr": low_fpr["tpr_at_or_below_target_fpr"],
        "actual_fpr_at_tpr_metric": low_fpr["actual_fpr_at_tpr_metric"],
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
