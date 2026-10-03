"""Phase 5C probe fitting, selection, and paired metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from pre_output_physiology.metrics import bootstrap_auroc_ci_by_group
from pre_output_physiology.phase4_scoring import paired_mean_diff_ci
from pre_output_physiology.phase5_physiology import (
    BOOTSTRAP_SEED,
    LOCKED_TEST_GATES,
    N_BOOTSTRAP,
    PROBE_C,
    PROBE_MAX_ITER,
    PROBE_SEED,
)


class ScaledLogisticProbe:
    """Train-only StandardScaler + LogisticRegression (fixed hyperparameters)."""

    def __init__(
        self,
        *,
        C: float = PROBE_C,
        max_iter: int = PROBE_MAX_ITER,
        seed: int = PROBE_SEED,
    ) -> None:
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(
            C=C,
            fit_intercept=True,
            max_iter=max_iter,
            random_state=seed,
        )
        self.C = C
        self.max_iter = max_iter
        self.seed = seed

    def fit(self, x: np.ndarray, y: np.ndarray) -> ScaledLogisticProbe:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y).astype(int)
        xs = self.scaler.fit_transform(x)
        self.clf.fit(xs, y)
        return self

    def decision_scores(self, x: np.ndarray) -> np.ndarray:
        xs = self.scaler.transform(np.asarray(x, dtype=np.float64))
        # Probability of positive class (S3).
        return self.clf.predict_proba(xs)[:, 1]

    def export_npz_arrays(self) -> dict[str, np.ndarray]:
        return {
            "mean": np.asarray(self.scaler.mean_, dtype=np.float64),
            "scale": np.asarray(self.scaler.scale_, dtype=np.float64),
            "coef": np.asarray(self.clf.coef_, dtype=np.float64).ravel(),
            "intercept": np.asarray([self.clf.intercept_[0]], dtype=np.float64),
        }


def auroc_with_paired_bootstrap(
    y: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=float)
    groups = np.asarray(groups)
    point = float(roc_auc_score(y, scores))
    ci = bootstrap_auroc_ci_by_group(
        y, scores, groups, n_bootstrap=n_bootstrap, seed=seed
    )
    return {
        "auroc": point,
        "auroc_ci_low": ci["auroc_ci_low"],
        "auroc_ci_high": ci["auroc_ci_high"],
        "n_bootstrap_effective": ci["n_bootstrap_effective"],
        "bootstrap_unit": "scenario_pair",
    }


def paired_s3_minus_s2(
    base_ids: list[str],
    condition_ids: list[str],
    scores: np.ndarray,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Paired mean(score_S3 - score_S2) with scenario bootstrap CI."""
    by_base: dict[str, dict[str, float]] = {}
    for bid, cid, sc in zip(base_ids, condition_ids, scores, strict=True):
        by_base.setdefault(bid, {})[cid] = float(sc)
    pos = []
    neg = []
    for _bid, d in sorted(by_base.items()):
        if "S3_strategic_deception" not in d or "S2_strategic_honesty" not in d:
            continue
        pos.append(d["S3_strategic_deception"])
        neg.append(d["S2_strategic_honesty"])
    return paired_mean_diff_ci(
        np.asarray(pos), np.asarray(neg), n_bootstrap=n_bootstrap, seed=seed
    )


def select_k1_candidate(
    layer_metrics: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Select k1 layer by val AUROC; tie-break worst-family AUROC then lower layer."""
    ranked = []
    for layer, m in layer_metrics.items():
        ranked.append(
            (
                -float(m["auroc"]),
                -float(m["worst_family_auroc"]),
                int(layer),
            )
        )
    ranked.sort()
    best_layer = ranked[0][2]
    return {
        "selected_endpoint": "k1",
        "selected_layer": best_layer,
        "selection_key": {
            "validation_auroc": layer_metrics[best_layer]["auroc"],
            "worst_family_auroc": layer_metrics[best_layer]["worst_family_auroc"],
            "layer": best_layer,
        },
        "ranking": [
            {
                "layer": layer,
                "validation_auroc": layer_metrics[layer]["auroc"],
                "worst_family_auroc": layer_metrics[layer]["worst_family_auroc"],
            }
            for _, _, layer in ranked
        ],
    }


def evaluate_locked_test_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    overall = float(metrics["auroc"])
    ci_low = float(metrics["auroc_ci_low"])
    per_family = metrics["per_family_auroc"]
    checks = {
        "overall_auroc_ge_0_70": overall >= LOCKED_TEST_GATES["overall_auroc_min"],
        "bootstrap_ci_low_gt_0_50": ci_low > LOCKED_TEST_GATES["bootstrap_ci_low_min"],
        "each_family_auroc_ge_0_60": all(
            float(v) >= LOCKED_TEST_GATES["per_family_auroc_min"]
            for v in per_family.values()
        ),
    }
    return {
        "gates": LOCKED_TEST_GATES,
        "checks": checks,
        "passed": all(checks.values()),
        "observed": {
            "overall_auroc": overall,
            "auroc_ci_low": ci_low,
            "per_family_auroc": per_family,
        },
    }
