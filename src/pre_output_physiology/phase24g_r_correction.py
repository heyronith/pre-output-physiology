"""Phase 24G-R — corrected LOPO + nested diagnostics (full stacked pipelines).

Phase-24F confirmatory FAIL remains immutable. Prior Phase-24G exploratory
artifacts are referenced by hash and not overwritten.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pre_output_physiology.phase24c_design import sha256_file
from pre_output_physiology.phase24f_confirmation import (
    CANDIDATE_LAYER,
    CANDIDATE_TIME,
    STATUS_NOT_CONFIRMED,
    confirmatory_pass,
)
from pre_output_physiology.phase24g_diagnostics import (
    PHASE24F_PRIMARY_HASH,
    classify_diagnostic_category,
)

STARTING_SHA = "8f2cbb67d0ad6a8ae4d0c8afb304729166f11a17"
ANALYSIS_SEED = 2408
NESTED_REPS = 1000  # exact requirement — do not reduce
LOPO_N = 28
PROXY_COMPARE_NESTED = 50

STATUS = "phase24g_r_corrected_diagnostics_complete_awaiting_independent_code_audit"
STATUS_RUNTIME_STOP = "phase24g_r_stopped_full_pipeline_1000_nested_runtime_prohibitive"

GUARANTEE = (
    "PHASE 24G-R CORRECTED THE PHASE-24G LOPO AND NESTED DIAGNOSTICS USING "
    "THE FULL FROZEN STACKED PIPELINES AND 1,000 NESTED PROMPT-LEVEL "
    "REPETITIONS. THE PHASE-24F PRIMARY CONFIRMATORY FAILURE REMAINS FINAL "
    "AND UNCHANGED. NO NEW MODEL DATA, LABEL CHANGES, SAE ANALYSIS, CAUSAL "
    "INTERVENTION, OR PRIMARY-ENDPOINT REDEFINITION WAS PERFORMED."
)

# Prior Phase-24G artifacts that must not be overwritten
PRIOR_24G_ARTIFACTS = (
    "candidate_stability_by_split.json",
    "diagnostic_category.json",
    "diagnostics_manifest.json",
    "exploratory_test_coordinate_metrics.json",
    "heatmap_test_activation.svg",
    "heatmap_test_combined.svg",
    "heatmap_test_delta.svg",
    "heatmap_test_surface.svg",
    "heatmap_val_delta.svg",
    "lopo_candidate_stability.json",
    "nested_resampling_optimism.json",
    "phase24f_immutability.json",
    "power_projections.json",
    "probe_direction_stability.json",
    "prompt_heterogeneity.json",
    "same_prompt_robustness.json",
    "surface_baseline_diagnostics.json",
    "val_test_topology.json",
    "artifact_hashes.json",
)


def assert_nested_reps_exact() -> None:
    if NESTED_REPS != 1000:
        raise RuntimeError(
            f"NESTED_REPS must be exactly 1000 for Phase 24G-R; got {NESTED_REPS}"
        )


def assert_no_proxy_in_primary(proxy_delta: bool) -> None:
    if proxy_delta:
        raise RuntimeError(
            "Phase 24G-R primary diagnostics forbid proxy_delta=True "
            "(must use full SURFACE+ACTIVATION − SURFACE)"
        )


def snapshot_prior_24g_hashes(repo_root: Path) -> dict[str, Any]:
    """Record hashes of prior Phase-24G artifacts; do not modify them."""
    root = repo_root / "artifacts/phase24g_diagnostics"
    out: dict[str, Any] = {"artifacts": {}, "all_present": True}
    for name in PRIOR_24G_ARTIFACTS:
        path = root / name
        if not path.exists():
            out["all_present"] = False
            out["artifacts"][name] = {"exists": False}
            continue
        out["artifacts"][name] = {
            "exists": True,
            "sha256": sha256_file(str(path)),
            "path": str(path.relative_to(repo_root)),
        }
    return out


def verify_prior_24g_unchanged(
    repo_root: Path, snapshot: dict[str, Any]
) -> dict[str, Any]:
    """Confirm prior Phase-24G files still match the recorded snapshot hashes."""
    root = repo_root / "artifacts/phase24g_diagnostics"
    mismatches = []
    for name, meta in snapshot.get("artifacts", {}).items():
        if not meta.get("exists"):
            continue
        path = root / name
        got = sha256_file(str(path))
        if got != meta["sha256"]:
            mismatches.append({"name": name, "expected": meta["sha256"], "observed": got})
    return {"unchanged": len(mismatches) == 0, "mismatches": mismatches}


def assert_phase24f_immutable(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "artifacts/phase24f_confirmation/primary_metrics.json"
    got = sha256_file(str(path))
    if got != PHASE24F_PRIMARY_HASH:
        raise RuntimeError(f"Phase-24F primary hash changed: {got}")
    blob = json.loads(path.read_text(encoding="utf-8"))
    if blob.get("primary_pass") is not False:
        raise RuntimeError("Phase-24F primary_pass mutated")
    if blob.get("status") != STATUS_NOT_CONFIRMED:
        raise RuntimeError("Phase-24F status mutated")
    if not confirmatory_pass(
        delta_auroc=float(blob["delta_auroc"]),
        ci95_low=float(blob["delta_auroc_bootstrap"]["ci95"][0]),
    ):
        pass
    else:
        raise RuntimeError("Phase-24F confirmatory_pass flipped")
    return {
        "immutable": True,
        "hash": got,
        "primary_pass": False,
        "status": blob["status"],
        "delta_auroc": blob["delta_auroc"],
        "ci95": blob["delta_auroc_bootstrap"]["ci95"],
        "coordinate": {"time": CANDIDATE_TIME, "layer": CANDIDATE_LAYER},
    }


def summarize_lopo(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from collections import Counter

    cands = [r["candidate"] for r in rows if r.get("candidate")]
    return {
        "n": len(rows),
        "frac_candidate_exists": float(np_mean([r.get("candidate") is not None for r in rows])),
        "frac_time_1": float(
            np_mean([bool(c and c["candidate_time"] == 1) for c in cands])
        )
        if cands
        else float("nan"),
        "frac_layer_20": float(
            np_mean([bool(c and c["candidate_layer"] == 20) for c in cands])
        )
        if cands
        else float("nan"),
        "frac_exact_t1_l20": float(
            np_mean(
                [
                    bool(
                        c
                        and c["candidate_time"] == 1
                        and c["candidate_layer"] == 20
                    )
                    for c in cands
                ]
            )
        )
        if cands
        else float("nan"),
        "frac_neighborhood": float(
            np_mean([bool(r.get("in_neighborhood")) for r in rows])
        ),
        "time_hist": dict(
            Counter(c["candidate_time"] for c in cands)
        ),
        "layer_hist": dict(
            Counter(c["candidate_layer"] for c in cands)
        ),
        "delta_auroc_mean": float(
            np_mean([r.get("delta_auroc") for r in rows if _finite(r.get("delta_auroc"))])
        ),
        "pipeline": "full_stacked_SURFACE_plus_ACTIVATION_minus_SURFACE",
        "proxy_used": False,
    }


def summarize_nested(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from collections import Counter

    import numpy as np

    disc = [r["discovery_delta"] for r in rows if _finite(r.get("discovery_delta"))]
    hold = [r["heldout_delta"] for r in rows if _finite(r.get("heldout_delta"))]
    opt = [r["optimism"] for r in rows if _finite(r.get("optimism"))]
    cands = [r["candidate"] for r in rows if r.get("candidate")]
    return {
        "n_reps_required": NESTED_REPS,
        "n_reps_completed": len(rows),
        "frac_candidate_exists": float(np.mean([r.get("candidate") is not None for r in rows])),
        "expected_discovery_delta_mean": float(np.mean(disc)) if disc else float("nan"),
        "expected_discovery_delta_median": float(np.median(disc)) if disc else float("nan"),
        "expected_heldout_delta_mean": float(np.mean(hold)) if hold else float("nan"),
        "expected_heldout_delta_median": float(np.median(hold)) if hold else float("nan"),
        "optimism_gap_mean": float(np.mean(opt)) if opt else float("nan"),
        "optimism_gap_median": float(np.median(opt)) if opt else float("nan"),
        # Empirical 2.5–97.5% percentiles of the optimism-gap sample
        # (discovery − held-out across nested reps). NOT a CI on the mean.
        "optimism_gap_empirical_percentile_025_975": (
            [float(np.quantile(opt, 0.025)), float(np.quantile(opt, 0.975))]
            if len(opt) >= 2
            else [float("nan"), float("nan")]
        ),

        "frac_exact_t1_l20": float(
            np.mean(
                [
                    bool(
                        c
                        and c.get("time") == 1
                        and c.get("layer") == 20
                    )
                    for c in cands
                ]
            )
        )
        if cands
        else float("nan"),
        "time_hist": dict(Counter(c["time"] for c in cands)),
        "layer_hist": dict(Counter(c["layer"] for c in cands)),
        "pipeline": "full_stacked_SURFACE_plus_ACTIVATION_minus_SURFACE",
        "proxy_used": False,
        "analysis_seed": ANALYSIS_SEED,
    }


def compare_proxy_vs_full(
    proxy_rows: list[dict[str, Any]], full_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compare proxy vs full on paired runs (same splits)."""
    import numpy as np

    n = min(len(proxy_rows), len(full_rows))
    d_proxy, d_full = [], []
    time_agree = layer_agree = exists_agree = 0
    for i in range(n):
        p, f = proxy_rows[i], full_rows[i]
        pc, fc = p.get("candidate"), f.get("candidate")
        exists_agree += int((pc is None) == (fc is None))
        if pc and fc:
            pt = pc.get("candidate_time", pc.get("time"))
            ft = fc.get("candidate_time", fc.get("time"))
            pl = pc.get("candidate_layer", pc.get("layer"))
            fl = fc.get("candidate_layer", fc.get("layer"))
            time_agree += int(pt == ft)
            layer_agree += int(pl == fl)
        dp = p.get("delta_auroc", p.get("discovery_delta"))
        df = f.get("delta_auroc", f.get("discovery_delta"))
        if _finite(dp) and _finite(df):
            d_proxy.append(float(dp))
            d_full.append(float(df))
    corr = float("nan")
    if len(d_proxy) >= 3:
        corr = float(np.corrcoef(d_proxy, d_full)[0, 1])
    return {
        "n_paired": n,
        "delta_auroc_pearson": corr,
        "frac_time_agree": time_agree / max(1, n),
        "frac_layer_agree": layer_agree / max(1, n),
        "frac_candidate_exists_agree": exists_agree / max(1, n),
        "n_delta_pairs": len(d_proxy),
    }


def category_change_statement(old: str, new: str) -> str:
    """Whether corrected analysis preserves/weakens/changes Category C.

    - preserves: still Category C
    - weakens: still C-adjacent instability narrative but category letter moved
      toward less selection-instability emphasis (C → D), or same letter with
      explicitly softer instability metrics (handled by caller via evidence)
    - changes: different supportive category (C → A or C → B)
    """
    if old == new:
        return "preserves"
    if old == "C" and new == "D":
        return "weakens"
    if old == "C" and new in ("A", "B"):
        return "changes"
    return "changes"


def _finite(x: Any) -> bool:
    import math

    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def np_mean(xs: list[Any]) -> float:
    import numpy as np

    arr = [float(x) for x in xs]
    return float(np.mean(arr)) if arr else float("nan")


# Re-export classify for convenience
__all__ = [
    "ANALYSIS_SEED",
    "GUARANTEE",
    "LOPO_N",
    "NESTED_REPS",
    "PRIOR_24G_ARTIFACTS",
    "PROXY_COMPARE_NESTED",
    "STARTING_SHA",
    "STATUS",
    "STATUS_RUNTIME_STOP",
    "assert_nested_reps_exact",
    "assert_no_proxy_in_primary",
    "assert_phase24f_immutable",
    "category_change_statement",
    "classify_diagnostic_category",
    "compare_proxy_vs_full",
    "snapshot_prior_24g_hashes",
    "summarize_lopo",
    "summarize_nested",
    "verify_prior_24g_unchanged",
]
