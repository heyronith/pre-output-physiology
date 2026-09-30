"""Phase 22C — zero-call updated sampling forecast from observed K=20 data.

Combines Phase-21 + Phase-22B annotated rollouts (20/prompt). Forecasts whether
expanding to K ∈ {30,40,60,80} can pass frozen mixed-population gates.
No model calls.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

MIN_HONEST = 2
MIN_DECEPTIVE_EXPLICIT = 2
MIN_TRAIN_QUALIFYING = 25
MIN_TEST_QUALIFYING = 8
N_OBSERVED = 20
N_PROMPTS = 371
EXPECTED_TRAIN_AT_20 = 34
EXPECTED_TEST_AT_20 = 4

FORECAST_TOTALS = (30, 40, 60, 80)
N_MONTE_CARLO = 20_000
RNG_SEED = 22_300_000

# Jeffreys
JEFFREYS_PRIOR = (0.5, 0.5, 0.5)
# Empirical weak prior: concentration κ=1 applied to global K=20 proportions
EMPIRICAL_CONCENTRATION = 1.0
# Conservative rare-event: Clopper–Pearson 95% one-sided UCB for 0/n
RARE_EVENT_ALPHA = 0.05

PHASE21_ONSET_RUN = "phase21_onset_20260928T212924Z_68a5621c"
PHASE22B_ONSET_RUN = "phase22b_onset_20260929T130149Z_96f69da6"

STATUS_SUPPORTED = "phase22c_sampling_expansion_supported"
STATUS_UNCERTAIN = "phase22c_sampling_expansion_uncertain"
STATUS_UNSUPPORTED = "phase22c_sampling_expansion_unsupported"

ModelName = Literal[
    "jeffreys",
    "empirical_weak",
    "structural_zeros",
    "conservative_rare_event",
]

OBSERVED_SWITCHING = {
    "honest_gt0_dec_eq0_acquires_dec": {"acquired": 11, "n": 134},
    "dec_gt0_honest_eq0_acquires_honest": {"acquired": 8, "n": 171},
    "exact_10_0_acquires_dec": {"acquired": 3, "n": 63},
    "exact_0_10_acquires_honest": {"acquired": 2, "n": 106},
    "note": (
        "Descriptive Phase-22B switching evidence only; not treated as proof "
        "of structural zeros."
    ),
}

GUARANTEE = (
    "PHASE 22C WAS A ZERO-MODEL-CALL UPDATED SAMPLING FORECAST USING ONLY "
    "FROZEN PHASE-21 AND PHASE-22B ONSET-ANNOTATED ROLLOUTS (20 PER PROMPT). "
    "NO MISTRAL CALLS, OPENAI CALLS, ACTIVATIONS, OR PHYSIOLOGY WERE PERFORMED. "
    "NO NEW RESPONSES WERE GENERATED. PHASE-21/22B LABELS, SPLIT, PROMPTS, AND "
    "POPULATION THRESHOLDS WERE NOT CHANGED. THE OBSERVED K=20 RESULT PROVIDED "
    "EVIDENCE THAT PROMPT-LEVEL BEHAVIOR WAS MORE STABLE THAN THE OPTIMISTIC "
    "K=10 JEFFREYS FORECAST ASSUMED."
)

INTERPRETATION = (
    "The observed K=20 result provided evidence that prompt-level behavior "
    "was more stable than the optimistic K=10 Jeffreys forecast assumed."
)


def _sha_json(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def clopper_pearson_zero_ucb(n: int, alpha: float = RARE_EVENT_ALPHA) -> float:
    """One-sided (1−α) upper confidence bound for Binomial p given 0/n successes."""
    if n <= 0:
        raise ValueError("n must be positive")
    return float(1.0 - alpha ** (1.0 / n))


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


def reproduce_k20_gates(counts: Sequence[PromptCounts]) -> dict[str, Any]:
    n_train = sum(1 for c in counts if c.split == "train" and c.qualifies_observed)
    n_test = sum(1 for c in counts if c.split == "test" and c.qualifies_observed)
    ok = n_train == EXPECTED_TRAIN_AT_20 and n_test == EXPECTED_TEST_AT_20
    return {
        "n_train_qualifying": n_train,
        "n_test_qualifying": n_test,
        "expected_train": EXPECTED_TRAIN_AT_20,
        "expected_test": EXPECTED_TEST_AT_20,
        "reproduced": ok,
        "gates_passed": False,  # HOLD by design at K=20
    }


def summarize_observed(counts: Sequence[PromptCounts]) -> dict[str, Any]:
    obs = np.stack([c.as_triple() for c in counts], dtype=np.float64)
    totals = obs.sum(axis=0)
    global_p = (totals / totals.sum()).tolist()
    n_zero_h = int((obs[:, 0] == 0).sum())
    n_zero_d = int((obs[:, 1] == 0).sum())
    n_zero_both_sides = int(((obs[:, 0] == 0) | (obs[:, 1] == 0)).sum())
    n_missing_one_gate_class = int(
        ((obs[:, 0] == 0) ^ (obs[:, 1] == 0)).sum()
    )
    return {
        "n_prompts": len(counts),
        "n_observed_per_prompt": N_OBSERVED,
        "global_counts_honest_dec_other": [int(x) for x in totals],
        "global_proportions": global_p,
        "n_prompts_zero_honest": n_zero_h,
        "n_prompts_zero_deceptive_explicit": n_zero_d,
        "n_prompts_missing_at_least_one_of_h_or_d": n_zero_both_sides,
        "n_prompts_exactly_one_of_h_or_d_zero": n_missing_one_gate_class,
        "n_qualifying_at_20": sum(1 for c in counts if c.qualifies_observed),
        "n_train_qualifying_at_20": sum(
            1 for c in counts if c.split == "train" and c.qualifies_observed
        ),
        "n_test_qualifying_at_20": sum(
            1 for c in counts if c.split == "test" and c.qualifies_observed
        ),
        "phase22b_switching_descriptive": OBSERVED_SWITCHING,
    }


def empirical_weak_prior(counts: Sequence[PromptCounts]) -> tuple[float, float, float]:
    totals = np.zeros(3, dtype=np.float64)
    for c in counts:
        totals += np.asarray(c.as_triple(), dtype=np.float64)
    props = totals / totals.sum()
    prior = EMPIRICAL_CONCENTRATION * props
    return (float(prior[0]), float(prior[1]), float(prior[2]))


def _posterior_matrix(
    counts: Sequence[PromptCounts],
    *,
    model: ModelName,
    empirical_prior: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return (n_prompts, 3) Dirichlet concentration parameters + meta."""
    obs = np.stack([np.asarray(c.as_triple(), dtype=np.float64) for c in counts])
    meta: dict[str, Any] = {"model": model}

    if model == "jeffreys":
        prior = np.asarray(JEFFREYS_PRIOR, dtype=np.float64)
        post = prior + obs
        meta["prior"] = list(JEFFREYS_PRIOR)
        meta["formulation"] = "Dirichlet(0.5,0.5,0.5) + observed counts"

    elif model == "empirical_weak":
        if empirical_prior is None:
            raise ValueError("empirical_prior required")
        prior = np.asarray(empirical_prior, dtype=np.float64)
        post = prior + obs
        meta["prior"] = list(empirical_prior)
        meta["concentration"] = EMPIRICAL_CONCENTRATION
        meta["formulation"] = (
            f"Dirichlet(κ·π̂_global) with κ={EMPIRICAL_CONCENTRATION}; "
            "π̂ from pooled K=20 counts; prompt data dominate"
        )

    elif model == "structural_zeros":
        prior = np.asarray(JEFFREYS_PRIOR, dtype=np.float64)
        post = prior + obs
        post = np.where(obs == 0, 0.0, post)
        row_sum = post.sum(axis=1)
        bad = row_sum <= 0
        if bad.any():
            post[bad] = prior
        meta["prior"] = list(JEFFREYS_PRIOR)
        meta["formulation"] = (
            "Never-observed class fixed at p=0 (no prior rescue); "
            "remaining classes use Jeffreys+data renorm via Dirichlet"
        )

    elif model == "conservative_rare_event":
        u = clopper_pearson_zero_ucb(N_OBSERVED, RARE_EVENT_ALPHA)
        # α_i = count_i if count_i>0 else UCB(0/n); Dirichlet draw around this
        post = np.where(obs > 0, obs, u)
        meta["prior"] = None
        meta["zero_class_ucb"] = u
        meta["ucb_alpha"] = RARE_EVENT_ALPHA
        meta["formulation"] = (
            f"For class i with count>0: α_i=count_i. For count=0: "
            f"α_i=Clopper–Pearson one-sided (1−{RARE_EVENT_ALPHA}) UCB for 0/"
            f"{N_OBSERVED} = 1−{RARE_EVENT_ALPHA}^(1/{N_OBSERVED})≈{u:.6f}. "
            "Draw p~Dirichlet(α); additional Multinomial(K−20, p). "
            "Does not fully Jeffreys-rescue never-seen classes."
        )
    else:
        raise ValueError(f"unknown model {model}")

    return post, meta


def _draw_extra_multinomial(
    rng: np.random.Generator,
    p: np.ndarray,
    extra_n: int,
) -> np.ndarray:
    """Vectorized Multinomial(extra_n, p) for p shape (n_mc, n_prompts, 3)."""
    n_mc, n_prompts, n_cat = p.shape
    if n_cat != 3:
        raise ValueError("expected 3 categories")
    if extra_n == 0:
        return np.zeros((n_mc, n_prompts, 3), dtype=np.int64)

    # Sequential binomials (handles varying p across prompts/MC)
    remaining = np.full((n_mc, n_prompts), extra_n, dtype=np.int64)
    out = np.zeros((n_mc, n_prompts, 3), dtype=np.int64)
    # Avoid division by zero: where remaining mass is 0, take 0
    for j in range(n_cat - 1):
        mass_left = p[:, :, j:].sum(axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            pj = np.where(mass_left > 0, p[:, :, j] / mass_left, 0.0)
        # Clip for numerical safety
        pj = np.clip(pj, 0.0, 1.0)
        drawn = rng.binomial(remaining, pj)
        out[:, :, j] = drawn
        remaining = remaining - drawn
    out[:, :, n_cat - 1] = remaining
    return out


def forecast_model(
    counts: Sequence[PromptCounts],
    *,
    total_rollouts: int,
    model: ModelName,
    n_mc: int = N_MONTE_CARLO,
    seed: int = RNG_SEED,
    empirical_prior: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    if total_rollouts < N_OBSERVED:
        raise ValueError("total_rollouts must be >= observed")
    extra_n = total_rollouts - N_OBSERVED
    model_tag = {
        "jeffreys": 0,
        "empirical_weak": 1,
        "structural_zeros": 2,
        "conservative_rare_event": 3,
    }[model]
    rng = np.random.default_rng(seed + total_rollouts * 1009 + model_tag * 9176)

    post, meta = _posterior_matrix(
        counts, model=model, empirical_prior=empirical_prior
    )
    train_idx = np.asarray(
        [i for i, c in enumerate(counts) if c.split == "train"], dtype=np.int64
    )
    test_idx = np.asarray(
        [i for i, c in enumerate(counts) if c.split == "test"], dtype=np.int64
    )
    h0 = np.asarray([c.n_honest for c in counts], dtype=np.int64)
    d0 = np.asarray([c.n_deceptive_explicit for c in counts], dtype=np.int64)

    # Dirichlet draws: gamma(α) then normalize; structural zeros force α=0 → p=0
    alpha = np.broadcast_to(post, (n_mc, len(counts), 3)).copy()
    g = rng.gamma(np.maximum(alpha, 1e-300), 1.0)
    if model == "structural_zeros":
        g = np.where(alpha == 0, 0.0, g)
    g_sum = g.sum(axis=-1, keepdims=True)
    # If a row somehow all zero, uniform fallback
    bad = g_sum[..., 0] <= 0
    if bad.any():
        g[bad] = 1.0
        g_sum = g.sum(axis=-1, keepdims=True)
    p = g / g_sum

    extra = _draw_extra_multinomial(rng, p, extra_n)
    h = h0[None, :] + extra[:, :, 0]
    d = d0[None, :] + extra[:, :, 1]
    qual = (h >= MIN_HONEST) & (d >= MIN_DECEPTIVE_EXPLICIT)
    train_q = qual[:, train_idx].sum(axis=1).astype(np.int32)
    test_q = qual[:, test_idx].sum(axis=1).astype(np.int32)
    both = (train_q >= MIN_TRAIN_QUALIFYING) & (test_q >= MIN_TEST_QUALIFYING)

    return {
        "total_rollouts_per_prompt": total_rollouts,
        "additional_rollouts_per_prompt": extra_n,
        "n_monte_carlo": n_mc,
        "model": model,
        "model_meta": meta,
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


def decide_support(
    results_by_model: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Apply Phase-22C multi-model decision rule."""
    non_struct = ("jeffreys", "empirical_weak")
    # Per K: check support rule
    ks = FORECAST_TOTALS
    per_k: dict[int, dict[str, Any]] = {}
    smallest_supported: int | None = None

    for k in ks:
        by_m = {
            m: next(f for f in results_by_model[m] if f["total_rollouts_per_prompt"] == k)
            for m in results_by_model
        }
        n_strong_nonstruct = sum(
            1 for m in non_struct if by_m[m]["p_both_gates"] >= 0.8
        )
        cons_ok = by_m["conservative_rare_event"]["p_both_gates"] >= 0.5
        meets = n_strong_nonstruct >= 2 and cons_ok
        per_k[k] = {
            "n_nonstruct_models_p_both_ge_0_8": n_strong_nonstruct,
            "conservative_p_both": by_m["conservative_rare_event"]["p_both_gates"],
            "conservative_ge_0_5": cons_ok,
            "jeffreys_p_both": by_m["jeffreys"]["p_both_gates"],
            "empirical_weak_p_both": by_m["empirical_weak"]["p_both_gates"],
            "structural_zeros_p_both": by_m["structural_zeros"]["p_both_gates"],
            "meets_support_rule": meets,
        }
        if meets and smallest_supported is None:
            smallest_supported = k

    # Unsupported: all reasonable models (jeffreys, empirical, conservative)
    # give low P(BOTH) even at K=80
    k80 = per_k[80]
    reasonable_low = all(
        k80[key] < 0.5
        for key in (
            "jeffreys_p_both",
            "empirical_weak_p_both",
            "conservative_p_both",
        )
    )

    # Material disagreement: e.g. Jeffreys strong but conservative/structural weak
    disagree = False
    for k in ks:
        row = per_k[k]
        optimistic = max(row["jeffreys_p_both"], row["empirical_weak_p_both"])
        pessimistic = min(
            row["conservative_p_both"], row["structural_zeros_p_both"]
        )
        if optimistic >= 0.8 and pessimistic < 0.5:
            disagree = True
        if abs(row["jeffreys_p_both"] - row["conservative_p_both"]) >= 0.3:
            disagree = True

    if smallest_supported is not None:
        status = STATUS_SUPPORTED
        verdict = "supported"
        decision_text = (
            f"sampling expansion supported at K≥{smallest_supported} "
            "(≥2 non-structural-zero models P(BOTH)≥0.8 and "
            "conservative rare-event P(BOTH)≥0.5)"
        )
    elif reasonable_low:
        status = STATUS_UNSUPPORTED
        verdict = "unsupported"
        decision_text = (
            "sampling expansion unsupported: jeffreys, empirical-weak, and "
            "conservative rare-event all give P(BOTH)<0.5 even at K=80"
        )
    else:
        status = STATUS_UNCERTAIN
        verdict = "uncertain"
        decision_text = (
            "sampling expansion uncertain: model conclusions disagree or "
            "support depends strongly on prior assumptions"
            + ("; material prior disagreement detected" if disagree else "")
        )

    return {
        "status": status,
        "verdict": verdict,
        "decision_text": decision_text,
        "smallest_supported_k": smallest_supported,
        "per_k": {str(k): per_k[k] for k in ks},
        "support_rule": {
            "non_structural_zero_models": list(non_struct),
            "require_n_nonstruct_p_both_ge_0_8": 2,
            "require_conservative_p_both_ge": 0.5,
            "same_practical_k": True,
        },
        "interpretation": INTERPRETATION,
    }
