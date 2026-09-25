"""Surface baselines and negative controls for Phase 3 H3 comparisons.

Phase 3A freezes the contract and provides constructors. Predictive fitting on
locked test is forbidden in Phase 3A.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

PREFIX_LENGTHS_FOR_BASELINES: list[int] = [0, 1, 2, 4, 8, 16]
TEXT_BASELINE_C_GRID: list[float] = [0.1, 1.0, 10.0]


@dataclass(frozen=True)
class SurfaceBaselineContract:
    """Frozen H3 surface-baseline specification (no test tuning)."""

    majority: bool = True
    prompt_text_only: bool = True
    prompt_plus_prefix: bool = True
    prefix_structural: bool = True
    logit_summaries_phase: str = "phase3b_optional"
    word_tfidf_ngram_range: tuple[int, int] = (1, 2)
    char_tfidf_ngram_range: tuple[int, int] = (3, 5)
    C_grid: tuple[float, ...] = (0.1, 1.0, 10.0)
    select_C_on: str = "phase3_validation_only"
    negative_controls: tuple[str, ...] = (
        "shuffled_label_probe",
        "random_activation_direction",
    )


def build_prompt_plus_prefix_text(
    prompt: str,
    generated_token_strings: list[str],
    k: int,
) -> str:
    """Exact prompt + first k generated tokens; nothing after k."""
    if k < 0:
        raise ValueError("k < 0")
    if k == 0:
        return prompt
    if k > len(generated_token_strings):
        raise ValueError("k exceeds available generated tokens")
    return prompt + "".join(generated_token_strings[:k])


def prefix_structural_features(prefix_text: str, k: int) -> dict[str, float]:
    return {
        "visible_prefix_token_count_k": float(k),
        "visible_prefix_char_count": float(len(prefix_text)),
        "punct_period": float(prefix_text.count(".")),
        "punct_comma": float(prefix_text.count(",")),
        "punct_question": float(prefix_text.count("?")),
        "punct_exclaim": float(prefix_text.count("!")),
        "response_started": float(1.0 if k > 0 else 0.0),
    }


def make_text_baseline_pipeline(C: float = 1.0, random_state: int = 42) -> Pipeline:
    """Word+char TF-IDF FeatureUnion → logistic regression."""
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2)
    char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2)
    return Pipeline(
        [
            ("features", FeatureUnion([("word", word), ("char", char)])),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    max_iter=1000,
                    random_state=random_state,
                    fit_intercept=True,
                ),
            ),
        ]
    )


@dataclass
class MajorityBaseline:
    prevalence_: float | None = None

    def fit(self, y: np.ndarray) -> MajorityBaseline:
        y = np.asarray(y)
        self.prevalence_ = float(y.mean()) if len(y) else 0.5
        return self

    def predict_proba(self, n: int) -> np.ndarray:
        if self.prevalence_ is None:
            raise RuntimeError("not fit")
        return np.full(n, self.prevalence_, dtype=float)


@dataclass
class RandomDirectionControl:
    """Fixed-seed random projection sanity control (not a scientific probe)."""

    dim: int
    seed: int = 0
    direction_: np.ndarray | None = field(default=None, repr=False)

    def fit(self) -> RandomDirectionControl:
        rng = np.random.default_rng(self.seed)
        v = rng.normal(size=self.dim)
        self.direction_ = v / (np.linalg.norm(v) + 1e-12)
        return self

    def scores(self, x: np.ndarray) -> np.ndarray:
        if self.direction_ is None:
            raise RuntimeError("not fit")
        return np.asarray(x) @ self.direction_


def shuffled_labels(y: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.asarray(y).copy()
    rng.shuffle(out)
    return out


def contract_as_dict(contract: SurfaceBaselineContract | None = None) -> dict[str, Any]:
    c = contract or SurfaceBaselineContract()
    return {
        "majority": c.majority,
        "prompt_text_only": c.prompt_text_only,
        "prompt_plus_prefix": c.prompt_plus_prefix,
        "prefix_structural": c.prefix_structural,
        "logit_summaries_phase": c.logit_summaries_phase,
        "word_tfidf_ngram_range": list(c.word_tfidf_ngram_range),
        "char_tfidf_ngram_range": list(c.char_tfidf_ngram_range),
        "C_grid": list(c.C_grid),
        "select_C_on": c.select_C_on,
        "negative_controls": list(c.negative_controls),
        "primary_surface_baseline": "prompt_plus_prefix",
    }
