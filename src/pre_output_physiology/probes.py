"""Literature-aligned sklearn logistic probe for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class ProbeConfig:
    C: float = 0.01
    fit_intercept: bool = True
    normalize: bool = True
    max_iter: int = 500
    random_state: int = 42


class MeanLinearProbe:
    """Train-only normalization + LogisticRegression (LASR SklearnLogisticProbe-aligned)."""

    def __init__(self, config: ProbeConfig | None = None) -> None:
        self.config = config or ProbeConfig()
        self.classifier = LogisticRegression(
            C=self.config.C,
            fit_intercept=self.config.fit_intercept,
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,
        )
        self.transformation_mean: np.ndarray | float = 0.0
        self.transformation_std: np.ndarray | float = 1.0
        self.converged_: bool | None = None

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.transformation_mean) / self.transformation_std

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> MeanLinearProbe:
        x = np.asarray(x_train, dtype=np.float64)
        y = np.asarray(y_train)
        if self.config.normalize:
            self.transformation_mean = np.mean(x, axis=0, keepdims=True)
            std = np.std(x, axis=0, keepdims=True)
            # Avoid divide-by-zero on constant features.
            std = np.where(std == 0, 1.0, std)
            self.transformation_std = std
            x = self._normalize(x)
        else:
            self.transformation_mean = 0.0
            self.transformation_std = 1.0
        self.classifier.fit(x, y)
        self.converged_ = bool(self.classifier.n_iter_ < self.config.max_iter)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_n = self._normalize(np.asarray(x, dtype=np.float64))
        return self.classifier.predict(x_n)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x_n = self._normalize(np.asarray(x, dtype=np.float64))
        return self.classifier.predict_proba(x_n)[:, 1]

    def export_state(self) -> dict[str, Any]:
        return {
            "C": self.config.C,
            "fit_intercept": self.config.fit_intercept,
            "normalize": self.config.normalize,
            "max_iter": self.config.max_iter,
            "random_state": self.config.random_state,
            "converged": self.converged_,
            "n_iter": int(np.max(self.classifier.n_iter_))
            if hasattr(self.classifier, "n_iter_")
            else None,
            "coef_norm": float(np.linalg.norm(self.classifier.coef_))
            if hasattr(self.classifier, "coef_")
            else None,
        }
