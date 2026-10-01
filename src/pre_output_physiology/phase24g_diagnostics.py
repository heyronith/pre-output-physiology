"""Phase 24G — exploratory post-confirmation failure diagnostics.

The Phase-24F primary FAIL is immutable. All new TEST analyses are labeled
EXPLORATORY_POST_CONFIRMATION.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from pre_output_physiology.phase24c_design import (
    TEMPORAL_LANDMARKS,
    sha256_file,
)
from pre_output_physiology.phase24e_discovery import (
    safe_auroc,
    sample_weights_for_fit,
)
from pre_output_physiology.phase24f_confirmation import (
    CANDIDATE_LAYER,
    CANDIDATE_TIME,
    STATUS_NOT_CONFIRMED,
    confirmatory_pass,
)

STARTING_SHA = "05d6257e4ebb2b5e9951c91bbec297c843506cf0"
ANALYSIS_SEED = 2407
EXPLORATORY_LABEL = "EXPLORATORY_POST_CONFIRMATION"

STATUS = "phase24g_confirmatory_failure_diagnostics_complete"

GUARANTEE = (
    "PHASE 24G WAS AN EXPLORATORY POST-CONFIRMATION DIAGNOSTIC ANALYSIS. "
    "THE PHASE-24F PRIMARY CONFIRMATORY FAILURE REMAINS FINAL AND UNCHANGED. "
    "NO NEW TRAJECTORIES, SAE ANALYSIS, CAUSAL INTERVENTION, LABEL CHANGES, "
    "OR PRIMARY-ENDPOINT REDEFINITION WAS PERFORMED."
)

# Frozen Phase-24F primary artifact hash (must never change)
PHASE24F_PRIMARY_HASH = (
    "f23fe5c78d9df1f70a24e302ebc163b49cfa71a96179162aee3e766527a116f0"
)

NESTED_REPS = 50  # proxy-delta LOPO/nested budget; example target 1000
NESTED_REPS_TARGET_NOTE = 1000
PROBE_BOOTSTRAP_REPS = 30
LOPO_NEIGHBORHOOD = {
    "times": {1, 2},
    "layers": set(range(8, 24)),  # candidate band neighborhood
}


class ImmutableViolationError(RuntimeError):
    """Raised if Phase-24F primary artifacts are mutated."""


def assert_phase24f_immutable(repo_root: Path) -> dict[str, Any]:
    """Verify Phase-24F primary FAIL artifact is unchanged."""
    path = repo_root / "artifacts/phase24f_confirmation/primary_metrics.json"
    got = sha256_file(str(path))
    if got != PHASE24F_PRIMARY_HASH:
        raise ImmutableViolationError(
            f"Phase-24F primary_metrics.json hash changed: {got}"
        )
    blob = json.loads(path.read_text(encoding="utf-8"))
    if blob.get("primary_pass") is not False:
        raise ImmutableViolationError("primary_pass must remain false")
    if blob.get("status") != STATUS_NOT_CONFIRMED:
        raise ImmutableViolationError(f"status mutated: {blob.get('status')}")
    if blob.get("coordinate") != {"time": CANDIDATE_TIME, "layer": CANDIDATE_LAYER}:
        raise ImmutableViolationError("coordinate mutated")
    if not confirmatory_pass(
        delta_auroc=float(blob["delta_auroc"]),
        ci95_low=float(blob["delta_auroc_bootstrap"]["ci95"][0]),
    ):
        # expected: FAIL remains FAIL
        pass
    else:
        raise ImmutableViolationError("confirmatory_pass flipped to True")
    return {
        "immutable": True,
        "hash": got,
        "primary_pass": False,
        "status": blob["status"],
        "delta_auroc": blob["delta_auroc"],
        "ci95": blob["delta_auroc_bootstrap"]["ci95"],
        "label": "PHASE24F_PRIMARY_IMMUTABLE",
    }


def tag_exploratory(obj: Any) -> Any:
    """Recursively ensure EXPLORATORY_POST_CONFIRMATION labeling on dicts."""
    if isinstance(obj, dict):
        out = {k: tag_exploratory(v) for k, v in obj.items()}
        out.setdefault("analysis_label", EXPLORATORY_LABEL)
        return out
    if isinstance(obj, list):
        return [tag_exploratory(x) for x in obj]
    return obj


def load_val_delta_grid(repo_root: Path) -> dict[int, dict[int, float]]:
    rows = json.loads(
        (
            repo_root / "artifacts/phase24e_discovery/coordinate_metrics.json"
        ).read_text(encoding="utf-8")
    )["rows"]
    grid: dict[int, dict[int, float]] = {t: {} for t in TEMPORAL_LANDMARKS}
    for r in rows:
        grid[int(r["time"])][int(r["layer"])] = float(r["delta_auroc"])
    return grid


def load_val_metric_maps(
    repo_root: Path,
) -> dict[str, dict[int, dict[int, float]]]:
    rows = json.loads(
        (
            repo_root / "artifacts/phase24e_discovery/coordinate_metrics.json"
        ).read_text(encoding="utf-8")
    )["rows"]
    keys = (
        "surface_auroc",
        "activation_auroc",
        "surface_plus_activation_auroc",
        "delta_auroc",
    )
    maps: dict[str, dict[int, dict[int, float]]] = {
        k: {t: {} for t in TEMPORAL_LANDMARKS} for k in keys
    }
    for r in rows:
        t, L = int(r["time"]), int(r["layer"])
        for k in keys:
            maps[k][t][L] = float(r[k])
    return maps


def matrix_from_grid(
    grid: dict[int, dict[int, float]],
) -> np.ndarray:
    """Shape (n_layers=32, n_times=7)."""
    times = list(TEMPORAL_LANDMARKS)
    mat = np.full((32, len(times)), np.nan)
    for ti, t in enumerate(times):
        for L in range(32):
            mat[L, ti] = float(grid.get(t, {}).get(L, float("nan")))
    return mat


def corr_224(
    a: dict[int, dict[int, float]], b: dict[int, dict[int, float]]
) -> dict[str, float]:
    va, vb = [], []
    for t in TEMPORAL_LANDMARKS:
        for L in range(32):
            x = a[t][L]
            y = b[t][L]
            if np.isfinite(x) and np.isfinite(y):
                va.append(x)
                vb.append(y)
    va_a = np.asarray(va)
    vb_a = np.asarray(vb)
    if len(va_a) < 3:
        return {"pearson": float("nan"), "spearman": float("nan"), "n": len(va_a)}
    pr = float(stats.pearsonr(va_a, vb_a).statistic)
    sr = float(stats.spearmanr(va_a, vb_a).statistic)
    return {"pearson": pr, "spearman": sr, "n": int(len(va_a))}


def layerwise_corr(
    a: dict[int, dict[int, float]], b: dict[int, dict[int, float]]
) -> list[dict[str, float]]:
    out = []
    for L in range(32):
        va = [a[t][L] for t in TEMPORAL_LANDMARKS]
        vb = [b[t][L] for t in TEMPORAL_LANDMARKS]
        mask = [np.isfinite(x) and np.isfinite(y) for x, y in zip(va, vb, strict=True)]
        va_a = np.asarray([x for x, m in zip(va, mask, strict=True) if m])
        vb_a = np.asarray([y for y, m in zip(vb, mask, strict=True) if m])
        if len(va_a) < 3:
            out.append({"layer": L, "pearson": float("nan"), "spearman": float("nan")})
        else:
            out.append(
                {
                    "layer": L,
                    "pearson": float(stats.pearsonr(va_a, vb_a).statistic),
                    "spearman": float(stats.spearmanr(va_a, vb_a).statistic),
                }
            )
    return out


def timewise_corr(
    a: dict[int, dict[int, float]], b: dict[int, dict[int, float]]
) -> list[dict[str, float]]:
    out = []
    for t in TEMPORAL_LANDMARKS:
        va = [a[t][L] for L in range(32)]
        vb = [b[t][L] for L in range(32)]
        mask = [np.isfinite(x) and np.isfinite(y) for x, y in zip(va, vb, strict=True)]
        va_a = np.asarray([x for x, m in zip(va, mask, strict=True) if m])
        vb_a = np.asarray([y for y, m in zip(vb, mask, strict=True) if m])
        if len(va_a) < 3:
            out.append({"time": t, "pearson": float("nan"), "spearman": float("nan")})
        else:
            out.append(
                {
                    "time": t,
                    "pearson": float(stats.pearsonr(va_a, vb_a).statistic),
                    "spearman": float(stats.spearmanr(va_a, vb_a).statistic),
                }
            )
    return out


def high_region_overlap(
    a: dict[int, dict[int, float]],
    b: dict[int, dict[int, float]],
    *,
    quantile: float = 0.75,
) -> dict[str, Any]:
    """Overlap of coordinates above each map's quantile threshold."""
    vals_a = [a[t][L] for t in TEMPORAL_LANDMARKS for L in range(32)]
    vals_b = [b[t][L] for t in TEMPORAL_LANDMARKS for L in range(32)]
    ta = float(np.nanquantile(vals_a, quantile))
    tb = float(np.nanquantile(vals_b, quantile))
    set_a = {
        (t, L)
        for t in TEMPORAL_LANDMARKS
        for L in range(32)
        if a[t][L] >= ta
    }
    set_b = {
        (t, L)
        for t in TEMPORAL_LANDMARKS
        for L in range(32)
        if b[t][L] >= tb
    }
    inter = set_a & set_b
    union = set_a | set_b
    return {
        "quantile": quantile,
        "threshold_a": ta,
        "threshold_b": tb,
        "n_a": len(set_a),
        "n_b": len(set_b),
        "n_intersection": len(inter),
        "jaccard": (len(inter) / len(union)) if union else float("nan"),
        "intersection_examples": sorted(inter)[:20],
    }


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < 1e-12 or nv < 1e-12:
        return float("nan")
    return float(np.dot(u, v) / (nu * nv))


def probe_stability_from_coefs(
    coefs: Sequence[np.ndarray],
) -> dict[str, Any]:
    """Aggregate cosine similarity / norm / sign consistency across resamples."""
    mats = [np.asarray(c, dtype=np.float64).ravel() for c in coefs if c is not None]
    if len(mats) < 2:
        return {
            "n": len(mats),
            "mean_pairwise_cosine": float("nan"),
            "mean_norm": float("nan"),
            "sign_consistency": float("nan"),
        }
    cosines = []
    for i in range(len(mats)):
        for j in range(i + 1, len(mats)):
            cosines.append(cosine_similarity(mats[i], mats[j]))
    stacked = np.stack(mats, axis=0)
    sign_mode = np.sign(np.sum(np.sign(stacked), axis=0))
    # fraction of coefs matching majority sign (ignore zeros)
    agree = []
    for j in range(stacked.shape[1]):
        col = stacked[:, j]
        if abs(sign_mode[j]) < 0.5:
            continue
        agree.append(float(np.mean(np.sign(col) == sign_mode[j])))
    return {
        "n": len(mats),
        "mean_pairwise_cosine": float(np.nanmean(cosines)),
        "median_pairwise_cosine": float(np.nanmedian(cosines)),
        "mean_norm": float(np.mean([np.linalg.norm(m) for m in mats])),
        "sign_consistency": float(np.mean(agree)) if agree else float("nan"),
    }


def power_projection(
    *,
    observed_prompt_deltas: Sequence[float],
    effect_sizes: Sequence[float],
    n_prompts_grid: Sequence[int],
    n_sims: int = 2000,
    seed: int = ANALYSIS_SEED,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Estimate power to detect ΔAUROC effects via prompt-level bootstrap sims.

    Uses the empirical distribution of prompt-level contributions / residuals
    scaled so the mean matches each target effect size.
    """
    rng = np.random.default_rng(seed)
    base = np.asarray(list(observed_prompt_deltas), dtype=np.float64)
    if len(base) < 2:
        return {"error": "insufficient prompt deltas", "n_prompts_obs": len(base)}
    # center then we'll shift to target mean
    centered = base - base.mean()
    rows = []
    for effect in effect_sizes:
        for n_p in n_prompts_grid:
            hits = 0
            for _ in range(n_sims):
                # resample residual structure
                idx = rng.integers(0, len(centered), size=n_p)
                sample = centered[idx] + float(effect)
                # percentile CI of mean
                boots = []
                for _b in range(500):
                    bidx = rng.integers(0, n_p, size=n_p)
                    boots.append(float(sample[bidx].mean()))
                lo = float(np.quantile(boots, alpha / 2))
                if lo > 0:
                    hits += 1
            rows.append(
                {
                    "effect_size_delta_auroc": float(effect),
                    "n_prompts": int(n_p),
                    "power_estimate": hits / n_sims,
                    "n_sims": n_sims,
                    "criterion": "CI_low(mean)>0",
                }
            )
    return {
        "method": (
            "prompt-level residual bootstrap; shift mean to target effect; "
            "power = P(percentile CI lower bound of mean > 0)"
        ),
        "n_observed_prompts": int(len(base)),
        "observed_mean": float(base.mean()),
        "observed_std": float(base.std(ddof=1)),
        "rows": rows,
        "analysis_label": EXPLORATORY_LABEL,
    }


def candidate_in_neighborhood(
    cand: dict[str, Any] | None,
    *,
    times: set[int] | None = None,
    layers: set[int] | None = None,
) -> bool:
    if cand is None:
        return False
    times = times or LOPO_NEIGHBORHOOD["times"]
    layers = layers or LOPO_NEIGHBORHOOD["layers"]
    return int(cand["candidate_time"]) in times and int(cand["candidate_layer"]) in layers


def classify_diagnostic_category(evidence: dict[str, Any]) -> str:
    """Descriptive A/B/C/D classification from aggregated diagnostics."""
    # Heuristics documented in report; not a primary endpoint.
    topo = evidence.get("topology_pearson", float("nan"))
    lopo_exact = evidence.get("lopo_exact_frac", 0.0)
    lopo_near = evidence.get("lopo_neighborhood_frac", 0.0)
    optimism = evidence.get("optimism_gap", float("nan"))
    same_prompt_frac = evidence.get("test_same_prompt_frac_gt0", float("nan"))
    probe_cos = evidence.get("probe_mean_cosine", float("nan"))
    test_delta = evidence.get("test_delta_auroc", float("nan"))
    ci_width = evidence.get("test_ci_width", float("nan"))

    # D: little robust evidence
    if (
        (not np.isfinite(topo) or topo < 0.1)
        and lopo_near < 0.25
        and (not np.isfinite(same_prompt_frac) or same_prompt_frac < 0.5)
    ):
        return "D"
    # C: selection instability
    if lopo_exact < 0.15 and lopo_near < 0.4 and (
        not np.isfinite(topo) or topo < 0.35
    ):
        return "C"
    # B: prompt-specific / heterogeneous
    if (
        np.isfinite(same_prompt_frac)
        and same_prompt_frac >= 0.7
        and (not np.isfinite(topo) or topo < 0.45)
    ):
        return "B"
    # A: stable region, underpowered
    if (
        (np.isfinite(topo) and topo >= 0.35)
        and lopo_near >= 0.35
        and np.isfinite(test_delta)
        and test_delta > 0
        and np.isfinite(ci_width)
        and ci_width > 0.25
    ):
        return "A"
    # Fallback blending
    if np.isfinite(optimism) and optimism > 0.05 and lopo_exact < 0.3:
        return "C"
    if np.isfinite(same_prompt_frac) and same_prompt_frac >= 0.6:
        return "B"
    if np.isfinite(probe_cos) and probe_cos < 0.3:
        return "C"
    return "D"


def fit_linear_probe_coef(
    X: np.ndarray,
    y: np.ndarray,
    labels: Sequence[str],
    prompts: Sequence[str],
    C: float,
    seed: int = ANALYSIS_SEED,
) -> np.ndarray:
    """Return activation-probe coefficient vector (TRAIN-fit standardization)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    w = sample_weights_for_fit(labels, prompts)
    clf = LogisticRegression(
        C=C,
        solver="liblinear",
        class_weight=None,
        max_iter=2000,
        random_state=seed,
    )
    clf.fit(Xs, y, sample_weight=w)
    return np.asarray(clf.coef_).ravel()


def prompt_level_delta_contributions(
    *,
    y: np.ndarray,
    surface: np.ndarray,
    combined: np.ndarray,
    prompt_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Leave-one-prompt AUROC drop approximation of prompt influence + local sep."""
    pid_arr = np.asarray(list(prompt_ids))
    out = []
    for p in sorted(set(pid_arr.tolist())):
        mask = pid_arr == p
        yh, sh, ch = y[mask], surface[mask], combined[mask]
        n_h = int((yh == 0).sum())
        n_d = int((yh == 1).sum())
        local_delta = float("nan")
        if n_h >= 1 and n_d >= 1 and len(np.unique(yh)) == 2:
            local_delta = safe_auroc(yh, ch) - safe_auroc(yh, sh)
        sep_s = (
            float(sh[yh == 1].mean() - sh[yh == 0].mean())
            if n_h and n_d
            else float("nan")
        )
        sep_c = (
            float(ch[yh == 1].mean() - ch[yh == 0].mean())
            if n_h and n_d
            else float("nan")
        )
        out.append(
            {
                "prompt_id": p,
                "n_honest": n_h,
                "n_deceptive": n_d,
                "local_delta_auroc": local_delta,
                "surface_sep_D_minus_H": sep_s,
                "combined_sep_D_minus_H": sep_c,
            }
        )
    return out
