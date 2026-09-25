"""Phase 4E frozen-probe scoring and scenario-paired contrast metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from pre_output_physiology.metrics import bootstrap_auroc_ci_by_group

FROZEN_PROBE_L12_K1_SHA256 = (
    "3bbf0c78e97d8ff14b028592e6e6ab3215651e973e0be1f004ce66e2b9b7aa65"
)
FROZEN_PROBE_L12_K0_SHA256 = (
    "dc14569a4e229cadc55fd3f6e6fef4e33f4eac4f58d00e83916238d5274f4f29"
)
PRIMARY_CONTRAST = "C3_vs_C2"
KEY_SECONDARY_CONTRAST = "C3_vs_C4"
EXPECTED_ELIGIBILITY = {
    "C3_vs_C2": {
        "n": 148,
        "sha256": "07dcb4567aaf98cea443327bca2cc336f082ab035c645ccd4c4b28dcc97503b2",
    },
    "C3_vs_C4": {
        "n": 139,
        "sha256": "c03c70d06cd5e487621781b8d0a5aa65354857c578a0842b7e216c894e045d22",
    },
}
N_BOOTSTRAP_MIN = 5000
CONTROLLED_PREFIX_TOKEN_ID = 12107


class FrozenProbe:
    """Immutable npz probe: train-only normalize + logistic decision score."""

    def __init__(self, path: Path) -> None:
        data = np.load(path)
        self.mean = np.asarray(data["mean"], dtype=np.float64).ravel()
        self.std = np.asarray(data["std"], dtype=np.float64).ravel()
        self.coef = np.asarray(data["coef"], dtype=np.float64).ravel()
        self.intercept = float(np.asarray(data["intercept"]).ravel()[0])
        self.layer = int(np.asarray(data["layer"]).ravel()[0]) if "layer" in data else None
        self.k = int(np.asarray(data["k"]).ravel()[0]) if "k" in data else None

    def decision_scores(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        x_n = (x - self.mean) / self.std
        logits = x_n @ self.coef + self.intercept
        # Keep logistic link consistent with MeanLinearProbe.predict_proba
        return 1.0 / (1.0 + np.exp(-logits))


def load_frozen_probe(path: Path, *, expected_sha256: str) -> FrozenProbe:
    import hashlib

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"probe hash mismatch for {path}: {digest} != {expected_sha256}")
    return FrozenProbe(path)


def paired_mean_diff_ci(
    scores_pos: np.ndarray,
    scores_neg: np.ndarray,
    *,
    n_bootstrap: int = N_BOOTSTRAP_MIN,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Scenario-paired mean(score_pos - score_neg) with bootstrap CI."""
    scores_pos = np.asarray(scores_pos, dtype=float)
    scores_neg = np.asarray(scores_neg, dtype=float)
    if scores_pos.shape != scores_neg.shape:
        raise ValueError("paired score length mismatch")
    diffs = scores_pos - scores_neg
    point = float(diffs.mean())
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boots = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(diffs[idx].mean())
    return {
        "paired_mean_diff": point,
        "paired_mean_diff_ci_low": float(np.quantile(boots, alpha / 2)),
        "paired_mean_diff_ci_high": float(np.quantile(boots, 1 - alpha / 2)),
        "n_pairs": int(n),
        "n_bootstrap": int(n_bootstrap),
        "bootstrap_unit": "scenario",
        "definition": "mean(score_positive - score_negative)",
    }


def contrast_metrics(
    scores_pos: np.ndarray,
    scores_neg: np.ndarray,
    scenario_ids: list[str],
    *,
    n_bootstrap: int = N_BOOTSTRAP_MIN,
    seed: int = 0,
) -> dict[str, Any]:
    """AUROC (pos=1) + scenario-paired bootstrap CI + paired mean diff CI."""
    scores_pos = np.asarray(scores_pos, dtype=float)
    scores_neg = np.asarray(scores_neg, dtype=float)
    if not (len(scores_pos) == len(scores_neg) == len(scenario_ids)):
        raise ValueError("contrast length mismatch")
    y = np.concatenate(
        [np.ones(len(scores_pos), dtype=int), np.zeros(len(scores_neg), dtype=int)]
    )
    scores = np.concatenate([scores_pos, scores_neg])
    groups = np.asarray(list(scenario_ids) + list(scenario_ids))
    auroc = float(roc_auc_score(y, scores))
    auroc_ci = bootstrap_auroc_ci_by_group(
        y, scores, groups, n_bootstrap=n_bootstrap, seed=seed
    )
    diff = paired_mean_diff_ci(
        scores_pos, scores_neg, n_bootstrap=n_bootstrap, seed=seed + 1
    )
    return {
        "n_pairs": int(len(scenario_ids)),
        "auroc": auroc,
        "auroc_ci_low": auroc_ci["auroc_ci_low"],
        "auroc_ci_high": auroc_ci["auroc_ci_high"],
        "auroc_bootstrap_mean": auroc_ci["auroc_bootstrap_mean"],
        "n_bootstrap_effective_auroc": auroc_ci["n_bootstrap_effective"],
        "bootstrap_unit": "scenario",
        "positive_class": "left_condition",
        **diff,
    }
