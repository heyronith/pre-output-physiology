"""Phase 16A — same-prompt sampling-scaling posterior-predictive forecast.

Statistical design forecast only. Uses frozen Phase-14/15 calibration annotations.
No language-model calls.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any

import numpy as np

from pre_output_physiology.phase14_design import TEMPERATURE_GRID
from pre_output_physiology.phase15_onset import (
    LABEL_ALTERNATE,
    LABEL_RECORD,
    N_PROMPT_GROUPS,
    PHASE14_RAW_CONTINUATIONS_SHA256,
    PHASE14_RUN_ID,
    RULE_HASH,
    RULE_VERSION,
)

PHASE15_RULE_HASH = RULE_HASH
PHASE15_RULE_VERSION = RULE_VERSION

PHASE14_STATUS = "phase14a_same_prompt_trajectory_calibration_hold"
PHASE15_STATUS = "phase15a_same_prompt_onset_reanalysis_hold"

N_GRID: tuple[int, ...] = (16, 32, 48, 64, 96, 128)
N_REPLICATES = 50_000
MONTE_CARLO_SEED = 1601

STATUS_AUTHORIZED = "phase16a_same_prompt_scaling_forecast_authorized"
STATUS_PLAUSIBLE = "phase16a_same_prompt_scaling_forecast_plausible_awaiting_audit"
STATUS_EXPENSIVE = "phase16a_same_prompt_scaling_forecast_expensive_awaiting_audit"
STATUS_UNSUPPORTED = "phase16a_same_prompt_scaling_forecast_unsupported_awaiting_audit"

# Frozen interpretation thresholds (do not change after results).
RESCUE_MIN_GROUPS_GE4 = 20
RESCUE_MIN_PROB = 0.80
RESCUE_MIN_MEDIAN_VALID_FRAC = 0.90
RESCUE_PLAUSIBLE_MAX_N = 64

GUARANTEE = (
    "PHASE 16A WAS A STATISTICAL DESIGN FORECAST USING ONLY EXISTING PHASE-14/15 "
    "CALIBRATION DATA. PHASE 14A AND PHASE 15A REMAIN HOLDS. NO LANGUAGE-MODEL "
    "CALLS, FINAL/LOCKED PROMPT CALLS, ACTIVATIONS, PROBES, RESAMPLING FROM THE "
    "MODEL, PROMPT CHANGES, TEMPERATURE EXTENSION, OR CAUSAL INTERVENTIONS WERE "
    "PERFORMED. POSTERIOR-PREDICTIVE RESULTS ARE DESIGN FORECASTS, NOT NEW "
    "EMPIRICAL MODEL EVIDENCE."
)


def design_manifest() -> dict[str, Any]:
    body = {
        "phase14_run_id": PHASE14_RUN_ID,
        "raw_continuations_sha256": PHASE14_RAW_CONTINUATIONS_SHA256,
        "phase15_rule_version": PHASE15_RULE_VERSION,
        "phase15_rule_hash": PHASE15_RULE_HASH,
        "temperature_grid": list(TEMPERATURE_GRID),
        "n_grid": list(N_GRID),
        "n_replicates": N_REPLICATES,
        "monte_carlo_seed": MONTE_CARLO_SEED,
        "jeffreys_policy": "p_alt ~ Beta(alt+0.5, record+0.5)",
        "jeffreys_validity": "q_valid ~ Beta(valid+0.5, invalid+0.5)",
        "bootstrap": "32 prompt groups with replacement from empirical calibration prompts",
        "interpretation": {
            "plausible": (
                f"any T with N<={RESCUE_PLAUSIBLE_MAX_N}: "
                f"P(groups>={RESCUE_MIN_GROUPS_GE4} with >=4/class) "
                f">={RESCUE_MIN_PROB} AND median valid_frac "
                f">={RESCUE_MIN_MEDIAN_VALID_FRAC}"
            ),
            "expensive": "fails N<=64 but succeeds at N=96 or 128",
            "unsupported": "fails even at N=128",
        },
        "phase14_status_unchanged": PHASE14_STATUS,
        "phase15_status_unchanged": PHASE15_STATUS,
    }
    return {
        **body,
        "design_hash": hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


DESIGN_MANIFEST = design_manifest()
DESIGN_HASH = DESIGN_MANIFEST["design_hash"]


def empirical_prompt_counts(
    annotated_rows: list[dict[str, Any]],
) -> dict[float, list[dict[str, Any]]]:
    """Per temperature: list of 32 prompt-group count dicts."""
    out: dict[float, list[dict[str, Any]]] = {}
    for t in TEMPERATURE_GRID:
        by: dict[str, dict[str, int]] = defaultdict(
            lambda: {"record": 0, "alternate": 0, "invalid": 0, "n_observed": 0}
        )
        for r in annotated_rows:
            if float(r["temperature"]) != float(t):
                continue
            pg = r["prompt_group_id"]
            by[pg]["n_observed"] += 1
            if r["phase15_valid_pre_answer"] and r["final_policy_label"] == LABEL_RECORD:
                by[pg]["record"] += 1
            elif (
                r["phase15_valid_pre_answer"]
                and r["final_policy_label"] == LABEL_ALTERNATE
            ):
                by[pg]["alternate"] += 1
            else:
                by[pg]["invalid"] += 1
        groups = sorted(by.keys())
        if len(groups) != N_PROMPT_GROUPS:
            raise ValueError(f"expected {N_PROMPT_GROUPS} groups at T={t}, got {len(groups)}")
        out[float(t)] = [
            {
                "prompt_group_id": pg,
                "record": by[pg]["record"],
                "alternate": by[pg]["alternate"],
                "invalid": by[pg]["invalid"],
                "valid": by[pg]["record"] + by[pg]["alternate"],
                "n_observed": by[pg]["n_observed"],
            }
            for pg in groups
        ]
    return out


def heterogeneity_diagnostic(
    counts_by_t: dict[float, list[dict[str, Any]]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for t, groups in counts_by_t.items():
        alt_counts = [g["alternate"] for g in groups]
        rates = []
        for g in groups:
            v = g["valid"]
            rates.append((g["alternate"] / v) if v else None)
        finite = [x for x in rates if x is not None]
        out[str(t)] = {
            "n_prompt_groups": len(groups),
            "zero_alt_groups": sum(1 for a in alt_counts if a == 0),
            "one_to_two_alt_groups": sum(1 for a in alt_counts if 1 <= a <= 2),
            "ge3_alt_groups": sum(1 for a in alt_counts if a >= 3),
            "empirical_alternate_rate_median": (
                float(np.median(finite)) if finite else None
            ),
            "empirical_alternate_rate_iqr": (
                [
                    float(np.percentile(finite, 25)),
                    float(np.percentile(finite, 75)),
                ]
                if finite
                else None
            ),
            "empirical_alternate_rate_min": min(finite) if finite else None,
            "empirical_alternate_rate_max": max(finite) if finite else None,
            "n_groups_with_any_valid": sum(1 for g in groups if g["valid"] > 0),
            "pooled_record": sum(g["record"] for g in groups),
            "pooled_alternate": sum(g["alternate"] for g in groups),
            "pooled_invalid": sum(g["invalid"] for g in groups),
        }
    return out


def _summarize_metric(samples: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.median(samples)),
        "ci95_low": float(np.percentile(samples, 2.5)),
        "ci95_high": float(np.percentile(samples, 97.5)),
        "mean": float(np.mean(samples)),
    }


def _run_forecast_for_temperature(
    groups: list[dict[str, Any]],
    *,
    n_values: tuple[int, ...],
    n_replicates: int,
    rng: np.random.Generator,
    pooled: bool,
) -> dict[int, dict[str, Any]]:
    """Posterior-predictive forecast for one temperature.

    If pooled=False: bootstrap prompt groups, draw per-group Jeffreys params.
    If pooled=True: optimistic — every prompt shares pooled Beta posteriors.
    """
    n_g = len(groups)
    rec = np.array([g["record"] for g in groups], dtype=np.float64)
    alt = np.array([g["alternate"] for g in groups], dtype=np.float64)
    inv = np.array([g["invalid"] for g in groups], dtype=np.float64)

    if pooled:
        pool_rec = float(rec.sum())
        pool_alt = float(alt.sum())
        pool_inv = float(inv.sum())
        pool_valid = pool_rec + pool_alt
        # Draw once per replicate then broadcast to all 32 synthetic prompts.
        q = rng.beta(pool_valid + 0.5, pool_inv + 0.5, size=n_replicates)
        p = rng.beta(pool_alt + 0.5, pool_rec + 0.5, size=n_replicates)
        # Shape (replicates, 32) by broadcasting
        q_mat = np.broadcast_to(q[:, None], (n_replicates, n_g))
        p_mat = np.broadcast_to(p[:, None], (n_replicates, n_g))
    else:
        # Bootstrap indices: (replicates, 32)
        boot_idx = rng.integers(0, n_g, size=(n_replicates, n_g))
        rec_b = rec[boot_idx]
        alt_b = alt[boot_idx]
        inv_b = inv[boot_idx]
        valid_b = rec_b + alt_b
        q_mat = rng.beta(valid_b + 0.5, inv_b + 0.5)
        p_mat = rng.beta(alt_b + 0.5, rec_b + 0.5)

    results: dict[int, dict[str, Any]] = {}
    for n in n_values:
        v = rng.binomial(n, q_mat)
        a = rng.binomial(v, p_mat)
        r = v - a
        ge1 = np.sum((r >= 1) & (a >= 1), axis=1)
        ge3 = np.sum((r >= 3) & (a >= 3), axis=1)
        ge4 = np.sum((r >= 4) & (a >= 4), axis=1)
        ge6 = np.sum((r >= 6) & (a >= 6), axis=1)
        ge8 = np.sum((r >= 8) & (a >= 8), axis=1)
        tot_r = np.sum(r, axis=1)
        tot_a = np.sum(a, axis=1)
        tot_v = tot_r + tot_a
        valid_frac = tot_v / (n * n_g)
        with np.errstate(divide="ignore", invalid="ignore"):
            alt_frac = np.where(tot_v > 0, tot_a / tot_v, np.nan)

        results[int(n)] = {
            "n_per_prompt": int(n),
            "groups_ge1_each_class": _summarize_metric(ge1.astype(np.float64)),
            "groups_ge3_each_class": _summarize_metric(ge3.astype(np.float64)),
            "groups_ge4_each_class": _summarize_metric(ge4.astype(np.float64)),
            "groups_ge6_each_class": _summarize_metric(ge6.astype(np.float64)),
            "groups_ge8_each_class": _summarize_metric(ge8.astype(np.float64)),
            "total_valid_record": _summarize_metric(tot_r.astype(np.float64)),
            "total_valid_alternate": _summarize_metric(tot_a.astype(np.float64)),
            "valid_fraction": _summarize_metric(valid_frac.astype(np.float64)),
            "alternate_fraction_among_valid": {
                "median": float(np.nanmedian(alt_frac)),
                "ci95_low": float(np.nanpercentile(alt_frac, 2.5)),
                "ci95_high": float(np.nanpercentile(alt_frac, 97.5)),
                "mean": float(np.nanmean(alt_frac)),
            },
            "prob_groups_ge4_ge16": float(np.mean(ge4 >= 16)),
            "prob_groups_ge4_ge20": float(np.mean(ge4 >= 20)),
            "prob_groups_ge6_ge16": float(np.mean(ge6 >= 16)),
            "prob_groups_ge6_ge20": float(np.mean(ge6 >= 20)),
        }
    return results


def run_primary_forecast(
    counts_by_t: dict[float, list[dict[str, Any]]],
    *,
    seed: int = MONTE_CARLO_SEED,
    n_replicates: int = N_REPLICATES,
    n_values: tuple[int, ...] = N_GRID,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {}
    for t in TEMPERATURE_GRID:
        # Independent stream per temperature via spawning for reproducibility
        # while keeping a single frozen seed documented.
        child = np.random.default_rng(rng.integers(0, 2**63 - 1))
        out[str(t)] = _run_forecast_for_temperature(
            counts_by_t[float(t)],
            n_values=n_values,
            n_replicates=n_replicates,
            rng=child,
            pooled=False,
        )
    return out


def run_optimistic_forecast(
    counts_by_t: dict[float, list[dict[str, Any]]],
    *,
    seed: int = MONTE_CARLO_SEED,
    n_replicates: int = N_REPLICATES,
    n_values: tuple[int, ...] = N_GRID,
) -> dict[str, Any]:
    """Clearly labeled optimistic forecast ignoring prompt heterogeneity."""
    # Offset seed so optimistic stream is distinct but deterministic.
    rng = np.random.default_rng(seed + 10_000_019)
    out: dict[str, Any] = {}
    for t in TEMPERATURE_GRID:
        child = np.random.default_rng(rng.integers(0, 2**63 - 1))
        out[str(t)] = _run_forecast_for_temperature(
            counts_by_t[float(t)],
            n_values=n_values,
            n_replicates=n_replicates,
            rng=child,
            pooled=True,
        )
    return out


def interpret_forecast(primary: dict[str, Any]) -> dict[str, Any]:
    """Frozen interpretation. Do not change thresholds after seeing results."""
    plausible_hits: list[dict[str, Any]] = []
    expensive_hits: list[dict[str, Any]] = []
    for t in TEMPERATURE_GRID:
        for n in N_GRID:
            cell = primary[str(t)][n]
            ok_prob = cell["prob_groups_ge4_ge20"] >= RESCUE_MIN_PROB
            ok_valid = cell["valid_fraction"]["median"] >= RESCUE_MIN_MEDIAN_VALID_FRAC
            hit = {
                "temperature": float(t),
                "n": int(n),
                "prob_groups_ge4_ge20": cell["prob_groups_ge4_ge20"],
                "median_valid_fraction": cell["valid_fraction"]["median"],
                "meets_rule": bool(ok_prob and ok_valid),
            }
            if hit["meets_rule"]:
                if n <= RESCUE_PLAUSIBLE_MAX_N:
                    plausible_hits.append(hit)
                else:
                    expensive_hits.append(hit)

    if plausible_hits:
        outcome = "sampling_only_rescue_plausible"
        status = STATUS_PLAUSIBLE
        recommended = min(
            plausible_hits,
            key=lambda h: (h["n"], h["temperature"]),
        )
    elif expensive_hits:
        outcome = "sampling_only_rescue_possible_but_expensive"
        status = STATUS_EXPENSIVE
        recommended = None
    else:
        outcome = "sampling_only_rescue_unsupported"
        status = STATUS_UNSUPPORTED
        recommended = None

    return {
        "outcome": outcome,
        "status": status,
        "plausible_hits": plausible_hits,
        "expensive_hits": expensive_hits,
        "recommended_prospective_t_n": recommended,
        "rule": {
            "min_groups_ge4": RESCUE_MIN_GROUPS_GE4,
            "min_prob": RESCUE_MIN_PROB,
            "min_median_valid_fraction": RESCUE_MIN_MEDIAN_VALID_FRAC,
            "plausible_max_n": RESCUE_PLAUSIBLE_MAX_N,
        },
    }


def heterogeneity_penalty(
    primary: dict[str, Any], optimistic: dict[str, Any]
) -> dict[str, Any]:
    """Quantify how much prompt heterogeneity hurts vs pooled optimistic."""
    out: dict[str, Any] = {}
    for t in TEMPERATURE_GRID:
        per_n = {}
        for n in N_GRID:
            p = primary[str(t)][n]
            o = optimistic[str(t)][n]
            per_n[str(n)] = {
                "delta_prob_ge4_ge20": o["prob_groups_ge4_ge20"]
                - p["prob_groups_ge4_ge20"],
                "delta_prob_ge4_ge16": o["prob_groups_ge4_ge16"]
                - p["prob_groups_ge4_ge16"],
                "delta_median_groups_ge4": o["groups_ge4_each_class"]["median"]
                - p["groups_ge4_each_class"]["median"],
                "delta_median_alt_fraction": o["alternate_fraction_among_valid"][
                    "median"
                ]
                - p["alternate_fraction_among_valid"]["median"],
                "primary_prob_ge4_ge20": p["prob_groups_ge4_ge20"],
                "optimistic_prob_ge4_ge20": o["prob_groups_ge4_ge20"],
            }
        out[str(t)] = per_n
    return out
