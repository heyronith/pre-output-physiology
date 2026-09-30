"""Phase 24F — single locked-TEST confirmatory analysis at frozen t=1/L20."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from pre_output_physiology.phase24c_design import sha256_file
from pre_output_physiology.phase24e_discovery import (
    prompt_cluster_bootstrap_ci,
    sample_weights_for_fit,
)

STARTING_SHA = "4bbc6c28f53298cb3dbcd9b909c08b498e7414ea"
ANALYSIS_SEED = 2406

# Frozen Phase-24E candidate (immutable)
CANDIDATE_TIME = 1
CANDIDATE_LAYER = 20

FROZEN_C = {
    "text": 0.01,
    "logits": 0.01,
    "surface": 0.1,
    "activation": 0.1,
    "surface_plus_activation": 10.0,
}

TEST_BOOTSTRAP_REPS = 10_000

STATUS_CONFIRMED = "phase24f_locked_test_confirmed_primary_precursor_supported"
STATUS_NOT_CONFIRMED = "phase24f_locked_test_not_confirmed_primary_precursor_unsupported"

GUARANTEE = (
    "PHASE 24F PERFORMED ONE PROSPECTIVELY FROZEN LOCKED-TEST CONFIRMATORY "
    "ANALYSIS AT t=1, LAYER=20. NO TEST-SET LAYER/TIME SEARCH, HYPERPARAMETER "
    "TUNING, SAE ANALYSIS, CAUSAL INTERVENTION, OR POST-HOC CANDIDATE "
    "SUBSTITUTION WAS PERFORMED."
)

# Confirmatory code must refuse evaluating any other coordinate
ALLOWED_COORDINATE = (CANDIDATE_TIME, CANDIDATE_LAYER)


class CoordinateViolationError(RuntimeError):
    """Raised if confirmatory analysis attempts a non-frozen coordinate."""


def assert_frozen_coordinate(time: int, layer: int) -> None:
    if (int(time), int(layer)) != ALLOWED_COORDINATE:
        raise CoordinateViolationError(
            f"only frozen coordinate t={CANDIDATE_TIME}/L{CANDIDATE_LAYER} "
            f"allowed; got t={time}/L{layer}"
        )


def confirmatory_pass(*, delta_auroc: float, ci95_low: float) -> bool:
    """Frozen criterion: ΔAUROC > 0 AND CI lower bound strictly > 0."""
    if math.isnan(delta_auroc) or math.isnan(ci95_low):
        return False
    return float(delta_auroc) > 0.0 and float(ci95_low) > 0.0


def verify_frozen_provenance(repo_root: Path) -> dict[str, Any]:
    """Verify 24C/24D/24E hashes and frozen candidate before opening TEST."""
    checks: list[dict[str, Any]] = []
    ok = True

    def _check(key: str, path: Path, expected: str) -> None:
        nonlocal ok
        got = sha256_file(str(path))
        match = got == expected
        if not match:
            ok = False
        checks.append(
            {
                "key": key,
                "path": str(path.relative_to(repo_root)),
                "pass": match,
                "expected": expected,
                "observed": got,
            }
        )

    c24 = json.loads(
        (repo_root / "artifacts/phase24c_design/artifact_hashes.json").read_text(
            encoding="utf-8"
        )
    )
    for key, rel, hash_key in (
        (
            "phase24c_design_matrix",
            "artifacts/phase24c_design/design_matrix.json",
            "design_matrix.json",
        ),
        (
            "phase24c_split",
            "artifacts/phase24c_design/split_manifest.json",
            "split_manifest.json",
        ),
        (
            "phase24c_analysis_spec",
            "artifacts/phase24c_design/analysis_spec.json",
            "analysis_spec.json",
        ),
        (
            "phase24c_config",
            "configs/experiments/phase24c_full_trajectory_design.yaml",
            "config",
        ),
        (
            "phase24c_schedule_template",
            "artifacts/phase24c_design/generation_schedule_template.json",
            "generation_schedule_template.json",
        ),
    ):
        _check(key, repo_root / rel, c24[hash_key])

    d24 = json.loads(
        (repo_root / "artifacts/phase24d_collection/artifact_hashes.json").read_text(
            encoding="utf-8"
        )
    )
    for key, rel, hash_key in (
        ("phase24d_summary", "artifacts/phase24d_collection/summary.json", "summary.json"),
        ("phase24d_schedule", "artifacts/phase24d_collection/schedule.json", "schedule.json"),
        (
            "phase24d_train_labels",
            "artifacts/phase24d_collection/by_split/TRAIN/labels.json",
            "TRAIN_labels.json",
        ),
        (
            "phase24d_val_labels",
            "artifacts/phase24d_collection/by_split/VALIDATION/labels.json",
            "VALIDATION_labels.json",
        ),
        (
            "phase24d_test_labels_sealed",
            "artifacts/phase24d_collection/by_split/LOCKED_TEST/labels_SEALED.json",
            "LOCKED_TEST_labels_SEALED.json",
        ),
        (
            "phase24d_test_sealed_summary",
            "artifacts/phase24d_collection/by_split/LOCKED_TEST/sealed_summary.json",
            "LOCKED_TEST_sealed_summary.json",
        ),
    ):
        _check(key, repo_root / rel, d24[hash_key])

    e24 = json.loads(
        (repo_root / "artifacts/phase24e_discovery/artifact_hashes.json").read_text(
            encoding="utf-8"
        )
    )
    cand_path = repo_root / "artifacts/phase24e_discovery/candidate_selection.json"
    _check("phase24e_candidate_selection", cand_path, e24["candidate_selection.json"])
    _check(
        "phase24e_analysis_manifest",
        repo_root / "artifacts/phase24e_discovery/analysis_manifest.json",
        e24["analysis_manifest.json"],
    )
    _check(
        "phase24e_selected_C",
        repo_root / "artifacts/phase24e_discovery/selected_C.json",
        e24["selected_C.json"],
    )

    cand = json.loads(cand_path.read_text(encoding="utf-8"))
    t_ok = cand["candidate"]["candidate_time"] == CANDIDATE_TIME
    L_ok = cand["candidate"]["candidate_layer"] == CANDIDATE_LAYER
    if not (t_ok and L_ok):
        ok = False
    checks.append(
        {
            "key": "candidate_time_layer",
            "pass": t_ok and L_ok,
            "expected": {"time": CANDIDATE_TIME, "layer": CANDIDATE_LAYER},
            "observed": {
                "time": cand["candidate"]["candidate_time"],
                "layer": cand["candidate"]["candidate_layer"],
            },
        }
    )

    cm = cand["candidate_metrics"]
    expected_C = {
        "C_text": FROZEN_C["text"],
        "C_logits": FROZEN_C["logits"],
        "C_surface": FROZEN_C["surface"],
        "C_activation": FROZEN_C["activation"],
        "C_surface_plus_activation": FROZEN_C["surface_plus_activation"],
    }
    c_ok = all(float(cm[k]) == float(v) for k, v in expected_C.items())
    if not c_ok:
        ok = False
    checks.append(
        {
            "key": "frozen_C_values",
            "pass": c_ok,
            "expected": expected_C,
            "observed": {k: cm[k] for k in expected_C},
        }
    )

    split = json.loads(
        (repo_root / "artifacts/phase24c_design/split_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    n_ok = (
        len(split["train_prompt_ids"]) == 20
        and len(split["validation_prompt_ids"]) == 8
        and len(split["test_prompt_ids"]) == 11
    )
    if not n_ok:
        ok = False
    checks.append(
        {
            "key": "split_prompt_counts",
            "pass": n_ok,
            "observed": {
                "train": len(split["train_prompt_ids"]),
                "validation": len(split["validation_prompt_ids"]),
                "test": len(split["test_prompt_ids"]),
            },
        }
    )

    return {"verified": ok, "checks": checks}


def binary_classification_metrics(
    y: np.ndarray, proba: np.ndarray, *, threshold: float = 0.5
) -> dict[str, float]:
    """Sensitivity/specificity/precision/recall/Brier/log-loss at fixed threshold."""
    y = np.asarray(y, dtype=np.int32)
    proba = np.clip(np.asarray(proba, dtype=np.float64), 1e-6, 1 - 1e-6)
    pred = (proba >= threshold).astype(np.int32)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = sens
    prev = float(y.mean()) if len(y) else float("nan")
    try:
        brier = float(brier_score_loss(y, proba))
    except ValueError:
        brier = float("nan")
    try:
        ll = float(log_loss(y, proba, labels=[0, 1]))
    except ValueError:
        ll = float("nan")
    return {
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "recall": rec,
        "brier": brier,
        "log_loss": ll,
        "prevalence_deceptive": prev,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "threshold": threshold,
    }


def safe_auroc(y: np.ndarray, scores: np.ndarray) -> float:
    if len(y) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def safe_auprc(y: np.ndarray, scores: np.ndarray) -> float:
    if len(y) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, scores))


def same_prompt_score_diffs(
    *,
    y: np.ndarray,
    scores: np.ndarray,
    prompt_ids: Sequence[str],
    min_h: int = 2,
    min_d: int = 2,
) -> dict[str, Any]:
    """Secondary: mean_score_D - mean_score_H per prompt with ≥min_h H and ≥min_d D."""
    pid_arr = np.asarray(list(prompt_ids))
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=np.float64)
    diffs: list[dict[str, Any]] = []
    for p in sorted(set(pid_arr.tolist())):
        mask = pid_arr == p
        yh = y[mask]
        sh = scores[mask]
        n_h = int((yh == 0).sum())
        n_d = int((yh == 1).sum())
        if n_h < min_h or n_d < min_d:
            continue
        mean_h = float(sh[yh == 0].mean())
        mean_d = float(sh[yh == 1].mean())
        diffs.append(
            {
                "prompt_id": p,
                "n_honest": n_h,
                "n_deceptive": n_d,
                "mean_score_H": mean_h,
                "mean_score_D": mean_d,
                "diff_D_minus_H": mean_d - mean_h,
            }
        )
    vals = [d["diff_D_minus_H"] for d in diffs]
    if not vals:
        return {
            "n_qualifying_prompts": 0,
            "diffs": [],
            "median_diff": float("nan"),
            "fraction_gt_0": float("nan"),
            "mean_diff": float("nan"),
            "ci95": [float("nan"), float("nan")],
        }
    arr = np.asarray(vals, dtype=np.float64)
    # percentile bootstrap across prompts
    rng = np.random.default_rng(ANALYSIS_SEED)
    boots = []
    for _ in range(TEST_BOOTSTRAP_REPS):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boots.append(float(np.median(sample)))
    boots_a = np.asarray(boots, dtype=np.float64)
    return {
        "n_qualifying_prompts": len(diffs),
        "diffs": diffs,
        "median_diff": float(np.median(arr)),
        "mean_diff": float(np.mean(arr)),
        "fraction_gt_0": float(np.mean(arr > 0)),
        "ci95": [
            float(np.quantile(boots_a, 0.025)),
            float(np.quantile(boots_a, 0.975)),
        ],
        "bootstrap_reps": TEST_BOOTSTRAP_REPS,
        "bootstrap_statistic": "median_diff",
    }


def delta_auroc_cluster_bootstrap(
    *,
    y: np.ndarray,
    surface_scores: np.ndarray,
    combined_scores: np.ndarray,
    prompt_ids: Sequence[str],
    n_reps: int = TEST_BOOTSTRAP_REPS,
    seed: int = ANALYSIS_SEED,
) -> dict[str, Any]:
    """10k prompt-cluster bootstrap for ΔAUROC = comb − surface."""
    out = prompt_cluster_bootstrap_ci(
        y=y,
        scores_a=surface_scores,
        scores_b=combined_scores,
        prompt_ids=prompt_ids,
        n_reps=n_reps,
        seed=seed,
    )
    assert out["n_reps"] == n_reps
    return out


# Re-export weighting helper for tests
__all__ = [
    "ALLOWED_COORDINATE",
    "ANALYSIS_SEED",
    "CANDIDATE_LAYER",
    "CANDIDATE_TIME",
    "CoordinateViolationError",
    "FROZEN_C",
    "GUARANTEE",
    "STATUS_CONFIRMED",
    "STATUS_NOT_CONFIRMED",
    "STARTING_SHA",
    "TEST_BOOTSTRAP_REPS",
    "assert_frozen_coordinate",
    "binary_classification_metrics",
    "confirmatory_pass",
    "delta_auroc_cluster_bootstrap",
    "safe_auprc",
    "safe_auroc",
    "same_prompt_score_diffs",
    "sample_weights_for_fit",
    "verify_frozen_provenance",
]
