"""Phase 6B frozen-probe factorial analysis (formulas frozen before results).

Per scenario the frozen Phase-5 probe yields s00, s10, s01, s11 (F00/F10/F01/F11).
All CIs resample the 720 base scenarios with replacement (scenario-paired).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from pre_output_physiology.metrics import bootstrap_auroc_ci_by_group

N_BOOTSTRAP = 5000
BOOTSTRAP_SEED = 0
ALPHA = 0.05

# (positive, negative, role)
CONTRASTS: dict[str, tuple[str, str, str]] = {
    "F11_minus_F10": ("F11", "F10", "primary"),
    "F01_minus_F00": ("F01", "F00", "key_replication"),
    "F10_minus_F00": ("F10", "F00", "operational_conflict_diagnostic"),
    "F11_minus_F01": ("F11", "F01", "additional_operational_diagnostic"),
}

EFFECT_FORMULAS: dict[str, str] = {
    "communication_main_effect": "0.5 * [(s01 - s00) + (s11 - s10)]",
    "operational_main_effect": "0.5 * [(s10 - s00) + (s11 - s01)]",
    "interaction": "(s11 - s10) - (s01 - s00)",
}

# Operational definitions for descriptive pattern flags (frozen pre-result).
# "Materially smaller": |operational main effect| < 0.5 * |communication main effect|.
MATERIALLY_SMALLER_RATIO = 0.5
PATTERN_DEFINITIONS: dict[str, str] = {
    "pattern_A_communication_dominant": (
        "F11-F10 CI > 0 AND F01-F00 CI > 0 AND communication main-effect CI > 0 AND "
        "|operational main effect| < 0.5*|communication main effect| AND "
        "max(|F10-F00|, |F11-F01|) < 0.5*|communication main effect|"
    ),
    "pattern_B_operational_dominant": (
        "(F10-F00 CI > 0 OR operational main-effect CI > 0) AND "
        "|operational main effect| >= |communication main effect| AND "
        "(F11-F10 CI includes 0 OR F01-F00 CI includes 0 OR the two differ in sign)"
    ),
    "pattern_C_interaction_mixed": (
        "interaction CI excludes 0 AND |interaction| >= 0.5*|communication main effect|, "
        "or exactly one of F11-F10 / F01-F00 has CI entirely > 0"
    ),
}


def effects_per_scenario(s: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    s00, s10, s01, s11 = s["F00"], s["F10"], s["F01"], s["F11"]
    return {
        "communication_main_effect": 0.5 * ((s01 - s00) + (s11 - s10)),
        "operational_main_effect": 0.5 * ((s10 - s00) + (s11 - s01)),
        "interaction": (s11 - s10) - (s01 - s00),
    }


def bootstrap_indices(n: int, *, n_bootstrap: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_bootstrap, n))


def mean_with_ci(values: np.ndarray, idx: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    boots = values[idx].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(boots, ALPHA / 2)),
        "ci_high": float(np.quantile(boots, 1 - ALPHA / 2)),
        "n_scenarios": int(len(values)),
    }


def ci_excludes_zero_positive(stat: dict[str, float]) -> bool:
    return stat["ci_low"] > 0


def ci_includes_zero(stat: dict[str, float]) -> bool:
    return stat["ci_low"] <= 0 <= stat["ci_high"]


def contrast_metrics(
    s: dict[str, np.ndarray],
    base_ids: list[str],
    pos: str,
    neg: str,
    idx: np.ndarray,
) -> dict[str, Any]:
    delta = mean_with_ci(s[pos] - s[neg], idx)
    y = np.concatenate([np.ones(len(base_ids)), np.zeros(len(base_ids))]).astype(int)
    scores = np.concatenate([s[pos], s[neg]])
    groups = np.asarray(list(base_ids) + list(base_ids))
    auroc = float(roc_auc_score(y, scores))
    ci = bootstrap_auroc_ci_by_group(
        y, scores, groups, n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED
    )
    return {
        "positive": pos,
        "negative": neg,
        "paired_mean_delta": delta["mean"],
        "paired_mean_delta_ci_low": delta["ci_low"],
        "paired_mean_delta_ci_high": delta["ci_high"],
        "fraction_scenarios_positive_delta": float(np.mean(s[pos] - s[neg] > 0)),
        "auroc": auroc,
        "auroc_ci_low": ci["auroc_ci_low"],
        "auroc_ci_high": ci["auroc_ci_high"],
        "auroc_bootstrap_unit": "scenario",
        "n_scenarios": len(base_ids),
    }


def pattern_flags(contrasts: dict[str, Any], effects: dict[str, Any]) -> dict[str, Any]:
    def d(name: str) -> dict[str, float]:
        c = contrasts[name]
        return {
            "mean": c["paired_mean_delta"],
            "ci_low": c["paired_mean_delta_ci_low"],
            "ci_high": c["paired_mean_delta_ci_high"],
        }

    comm = effects["communication_main_effect"]
    op = effects["operational_main_effect"]
    inter = effects["interaction"]
    f11_10, f01_00, f10_00, f11_01 = (
        d("F11_minus_F10"),
        d("F01_minus_F00"),
        d("F10_minus_F00"),
        d("F11_minus_F01"),
    )
    comm_mag = abs(comm["mean"])
    ratio = abs(op["mean"]) / comm_mag if comm_mag > 0 else float("inf")
    a = (
        ci_excludes_zero_positive(f11_10)
        and ci_excludes_zero_positive(f01_00)
        and ci_excludes_zero_positive(comm)
        and abs(op["mean"]) < MATERIALLY_SMALLER_RATIO * comm_mag
        and max(abs(f10_00["mean"]), abs(f11_01["mean"])) < MATERIALLY_SMALLER_RATIO * comm_mag
    )
    comm_weak = (
        ci_includes_zero(f11_10)
        or ci_includes_zero(f01_00)
        or np.sign(f11_10["mean"]) != np.sign(f01_00["mean"])
    )
    b = (
        (ci_excludes_zero_positive(f10_00) or ci_excludes_zero_positive(op))
        and abs(op["mean"]) >= comm_mag
        and comm_weak
    )
    one_sided = ci_excludes_zero_positive(f11_10) != ci_excludes_zero_positive(f01_00)
    c = (
        not ci_includes_zero(inter) and abs(inter["mean"]) >= MATERIALLY_SMALLER_RATIO * comm_mag
    ) or one_sided
    return {
        "definitions": PATTERN_DEFINITIONS,
        "operational_to_communication_ratio": ratio,
        "pattern_A_communication_dominant": bool(a),
        "pattern_B_operational_dominant": bool(b),
        "pattern_C_interaction_mixed": bool(c),
        "note": (
            "Descriptive flags under frozen operational definitions; interpret all "
            "contrasts and effects jointly. Not a pure deception claim."
        ),
    }


def analyze(
    s: dict[str, np.ndarray],
    base_ids: list[str],
    families: list[str],
) -> dict[str, Any]:
    """s: condition -> per-scenario scores aligned with base_ids/families."""
    n = len(base_ids)
    for cid in ("F00", "F10", "F01", "F11"):
        if len(s[cid]) != n:
            raise ValueError(f"{cid} length mismatch")
    idx = bootstrap_indices(n)
    contrasts = {
        name: {**contrast_metrics(s, base_ids, pos, neg, idx), "role": role}
        for name, (pos, neg, role) in CONTRASTS.items()
    }
    per = effects_per_scenario(s)
    effects = {name: mean_with_ci(v, idx) for name, v in per.items()}
    fam_arr = np.asarray(families)
    by_family: dict[str, Any] = {}
    for fam in sorted(set(families)):
        mask = fam_arr == fam
        fidx = bootstrap_indices(int(mask.sum()))
        by_family[fam] = {
            **{name: mean_with_ci(v[mask], fidx) for name, v in per.items()},
            **{
                f"{name}_delta": mean_with_ci((s[pos] - s[neg])[mask], fidx)
                for name, (pos, neg, _role) in CONTRASTS.items()
            },
        }
    condition_means = {cid: float(np.mean(v)) for cid, v in s.items()}
    return {
        "n_scenarios": n,
        "n_bootstrap": N_BOOTSTRAP,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_unit": "base_scenario (paired across all four conditions)",
        "condition_mean_scores": condition_means,
        "contrasts": contrasts,
        "effect_formulas": EFFECT_FORMULAS,
        "effects": effects,
        "effects_by_family": by_family,
        "pattern_flags": pattern_flags(contrasts, effects),
    }
