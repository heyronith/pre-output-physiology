"""Phase 22 — zero-call Beta/Dirichlet–Multinomial sampling forecast from Phase 21.

No model calls. Forecasts whether expanding rollouts per prompt can pass the
frozen Phase-21 mixed-population gates (≥2 honest and ≥2 explicit-onset deceptive;
≥25 TRAIN / ≥8 TEST qualifying prompts).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

# Frozen Phase-21 gates (do not change)
MIN_HONEST = 2
MIN_DECEPTIVE_EXPLICIT = 2
MIN_TRAIN_QUALIFYING = 25
MIN_TEST_QUALIFYING = 8
N_OBSERVED = 10
N_PROMPTS = 371

FORECAST_TOTALS = (20, 40, 80, 160)
# Jeffreys Dirichlet prior for 3-way multinomial
DIRICHLET_PRIOR = (0.5, 0.5, 0.5)
N_MONTE_CARLO = 10_000
RNG_SEED = 22_000_000

ONSET_RUN_ID = "phase21_onset_20260928T212924Z_68a5621c"
GRADE_RUN_ID = "phase21_grade_20260928T205710Z_b6287702"
GEN_RUN_ID = "phase21_gen_20260928T180342Z_3d3671eb"

STATUS_SUPPORTED = "phase22_sampling_expansion_supported"
STATUS_UNSUPPORTED = "phase22_sampling_expansion_unsupported"
# Primary decision threshold for "credible" joint gate pass probability
CREDIBLE_P_BOTH = 0.5
STRONG_P_BOTH = 0.8

GUARANTEE = (
    "PHASE 22 WAS A ZERO-MODEL-CALL SAMPLING FORECAST USING ONLY FROZEN PHASE-21 "
    "GRADED AND ONSET-ANNOTATED ROLLOUTS. NO MISTRAL CALLS, OPENAI CALLS, "
    "ACTIVATIONS, OR PHYSIOLOGY WERE PERFORMED. PHASE-21 LABELS, SPLIT, PROMPTS, "
    "AND POPULATION THRESHOLDS WERE NOT CHANGED. NO NEW RESPONSES WERE GENERATED."
)


def _sha_json(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class PromptCounts:
    prompt_id: str
    split: str
    n_honest: int
    n_deceptive_explicit: int
    n_ambiguous: int
    n_exclude: int
    n_deceptive_no_explicit: int
    n_total: int

    @property
    def n_other(self) -> int:
        return self.n_ambiguous + self.n_exclude + self.n_deceptive_no_explicit

    @property
    def qualifies_observed(self) -> bool:
        return (
            self.n_honest >= MIN_HONEST
            and self.n_deceptive_explicit >= MIN_DECEPTIVE_EXPLICIT
        )

    def as_triple(self) -> tuple[int, int, int]:
        """(honest, deceptive_explicit, other)."""
        return (self.n_honest, self.n_deceptive_explicit, self.n_other)


def load_prompt_counts(annotated_rows: Sequence[dict[str, Any]]) -> list[PromptCounts]:
    by: dict[str, dict[str, Any]] = {}
    for r in annotated_rows:
        pid = r["prompt_id"]
        if pid not in by:
            by[pid] = {
                "split": r["split"],
                "honest": 0,
                "dec_exp": 0,
                "dec_no": 0,
                "amb": 0,
                "exc": 0,
                "n": 0,
            }
        b = by[pid]
        b["n"] += 1
        label = r.get("label")
        if label == "honest":
            b["honest"] += 1
        elif label == "deceptive":
            if (
                r.get("explicit_span") is True
                and r.get("onset_char_start") is not None
            ):
                b["dec_exp"] += 1
            else:
                b["dec_no"] += 1
        elif label == "ambiguous":
            b["amb"] += 1
        else:
            b["exc"] += 1

    out: list[PromptCounts] = []
    for pid in sorted(by):
        b = by[pid]
        if b["n"] != N_OBSERVED:
            raise ValueError(f"{pid} has {b['n']} rollouts, expected {N_OBSERVED}")
        out.append(
            PromptCounts(
                prompt_id=pid,
                split=b["split"],
                n_honest=b["honest"],
                n_deceptive_explicit=b["dec_exp"],
                n_ambiguous=b["amb"],
                n_exclude=b["exc"],
                n_deceptive_no_explicit=b["dec_no"],
                n_total=b["n"],
            )
        )
    if len(out) != N_PROMPTS:
        raise ValueError(f"expected {N_PROMPTS} prompts, got {len(out)}")
    return out


def stability_bucket(h: int, d: int) -> str:
    """Classify (honest, deceptive_explicit) count pairs among 10 rollouts."""
    pair = (h, d)
    if pair in {(10, 0), (0, 10)}:
        return "10/0_or_0/10"
    if pair in {(9, 1), (1, 9)}:
        return "9/1_or_1/9"
    if pair in {(8, 2), (2, 8)}:
        return "8/2_or_2/8"
    return "more_balanced_or_other"


def summarize_stability(counts: Sequence[PromptCounts]) -> dict[str, Any]:
    buckets = Counter(stability_bucket(c.n_honest, c.n_deceptive_explicit) for c in counts)
    # Also report extremes among usable (h+d) mass
    pure_one_sided = sum(
        1
        for c in counts
        if (c.n_honest + c.n_deceptive_explicit) >= 1
        and min(c.n_honest, c.n_deceptive_explicit) == 0
    )
    return {
        "n_prompts": len(counts),
        "buckets_by_honest_vs_dec_explicit_counts": dict(buckets),
        "n_pure_one_sided_usable": pure_one_sided,
        "n_qualifying_at_10": sum(1 for c in counts if c.qualifies_observed),
        "n_train_qualifying_at_10": sum(
            1 for c in counts if c.split == "train" and c.qualifies_observed
        ),
        "n_test_qualifying_at_10": sum(
            1 for c in counts if c.split == "test" and c.qualifies_observed
        ),
    }


def forecast_additional_sampling(
    counts: Sequence[PromptCounts],
    *,
    total_rollouts: int,
    n_mc: int = N_MONTE_CARLO,
    seed: int = RNG_SEED,
    prior: tuple[float, float, float] = DIRICHLET_PRIOR,
    structural_zeros: bool = False,
) -> dict[str, Any]:
    """Posterior predictive: keep observed 10; draw (K-10) more from Dirichlet-Multinomial.

    If structural_zeros=True, categories with observed count 0 are fixed at probability 0
    (no prior rescue of never-seen classes). Descriptive sensitivity only when flagged.
    """
    if total_rollouts < N_OBSERVED:
        raise ValueError("total_rollouts must be >= observed")
    extra_n = total_rollouts - N_OBSERVED
    rng = np.random.default_rng(seed + total_rollouts + int(structural_zeros) * 17)

    train_idx = [i for i, c in enumerate(counts) if c.split == "train"]
    test_idx = [i for i, c in enumerate(counts) if c.split == "test"]

    train_q = np.zeros(n_mc, dtype=np.int32)
    test_q = np.zeros(n_mc, dtype=np.int32)

    alpha0 = np.asarray(prior, dtype=np.float64)
    obs = np.stack([np.asarray(c.as_triple(), dtype=np.float64) for c in counts])
    posteriors = alpha0 + obs
    if structural_zeros:
        # Zero prior+data mass on never-observed categories; renorm handled by gamma draw
        posteriors = np.where(obs == 0, 0.0, posteriors)
        # If a row is all-zero somehow, fall back to prior (should not happen)
        row_sum = posteriors.sum(axis=1)
        bad = row_sum <= 0
        if bad.any():
            posteriors[bad] = alpha0

    for m in range(n_mc):
        g = rng.gamma(np.maximum(posteriors, 1e-300), 1.0)
        # structural zero categories: force gamma=0
        if structural_zeros:
            g = np.where(posteriors == 0, 0.0, g)
        p = g / g.sum(axis=1, keepdims=True)
        if extra_n == 0:
            extra = np.zeros((len(counts), 3), dtype=np.int64)
        else:
            extra = np.vstack(
                [rng.multinomial(extra_n, p[i]) for i in range(len(counts))]
            )
        h = np.asarray([c.n_honest for c in counts]) + extra[:, 0]
        d = np.asarray([c.n_deceptive_explicit for c in counts]) + extra[:, 1]
        qual = (h >= MIN_HONEST) & (d >= MIN_DECEPTIVE_EXPLICIT)
        train_q[m] = int(qual[train_idx].sum())
        test_q[m] = int(qual[test_idx].sum())

    both = (train_q >= MIN_TRAIN_QUALIFYING) & (test_q >= MIN_TEST_QUALIFYING)
    return {
        "total_rollouts_per_prompt": total_rollouts,
        "additional_rollouts_per_prompt": extra_n,
        "n_monte_carlo": n_mc,
        "dirichlet_prior": list(prior),
        "structural_zeros": structural_zeros,
        "method": (
            "dirichlet_multinomial_additional_draws_conditional_on_observed_10"
            + ("_structural_zeros" if structural_zeros else "")
        ),
        "expected_train_qualifying": float(train_q.mean()),
        "expected_test_qualifying": float(test_q.mean()),
        "train_qualifying_ci95": [
            float(np.quantile(train_q, 0.025)),
            float(np.quantile(train_q, 0.975)),
        ],
        "test_qualifying_ci95": [
            float(np.quantile(test_q, 0.025)),
            float(np.quantile(test_q, 0.975)),
        ],
        "p_train_ge_25": float(np.mean(train_q >= MIN_TRAIN_QUALIFYING)),
        "p_test_ge_8": float(np.mean(test_q >= MIN_TEST_QUALIFYING)),
        "p_both_gates": float(np.mean(both)),
    }


def decide_support(forecasts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    best = max(forecasts, key=lambda f: f["p_both_gates"])
    p = best["p_both_gates"]
    if p >= STRONG_P_BOTH:
        verdict = "supported_strong"
        status = STATUS_SUPPORTED
    elif p >= CREDIBLE_P_BOTH:
        verdict = "supported"
        status = STATUS_SUPPORTED
    else:
        verdict = "unsupported"
        status = STATUS_UNSUPPORTED
    return {
        "status": status,
        "verdict": verdict,
        "credible_p_both_threshold": CREDIBLE_P_BOTH,
        "strong_p_both_threshold": STRONG_P_BOTH,
        "best_total_rollouts": best["total_rollouts_per_prompt"],
        "best_p_both_gates": p,
        "decision_text": (
            "sampling expansion supported"
            if status == STATUS_SUPPORTED
            else "sampling-only rescue unsupported"
        ),
    }
