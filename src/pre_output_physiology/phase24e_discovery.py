"""Phase 24E — TRAIN+VALIDATION discovery analysis (TEST sealed)."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from pre_output_physiology.phase24c_design import (
    CANDIDATE_MIN_CONSEC_LAYERS,
    CANDIDATE_MIN_MEDIAN_DELTA,
    LOGISTIC_C_GRID,
    N_LAYERS,
    TEMPORAL_LANDMARKS,
    VAL_BOOTSTRAP_REPS,
    prompt_equal_balanced_weights,
    prompt_group_folds,
    sha256_file,
)

STARTING_SHA = "cc1c011734e975d8ae8bcb71bba732110910b696"
ANALYSIS_SEED = 2405
N_FOLDS = 5

STATUS_CANDIDATE = "phase24e_candidate_discovered_awaiting_locked_test_authorization"
STATUS_NO_CANDIDATE = "phase24e_no_validation_candidate_primary_precursor_not_discovered"

GUARANTEE = (
    "PHASE 24E USED TRAIN AND VALIDATION DATA ONLY. LOCKED TEST LABELS, "
    "ACTIVATIONS, AND SCIENTIFIC OUTCOMES REMAINED SEALED. NO SAE ANALYSIS, "
    "CAUSAL INTERVENTION, OR LOCKED-TEST CONFIRMATORY ANALYSIS WAS PERFORMED."
)

FORBIDDEN_TEST_MARKERS = (
    "LOCKED_TEST",
    "labels_SEALED",
    "new_trajectories_SEALED",
    "/test/",
    "split=test",
)


class SealViolationError(RuntimeError):
    """Raised if discovery code attempts to touch sealed TEST artifacts."""


def assert_not_test_path(path: str | Path) -> None:
    s = str(path).replace("\\", "/")
    for marker in FORBIDDEN_TEST_MARKERS:
        if marker in s:
            raise SealViolationError(f"TEST seal violation: path touches {marker}: {s}")
    lower = s.lower()
    if "locked_test" in lower or "labels_sealed" in lower:
        raise SealViolationError(f"TEST seal violation: {s}")


def assert_split_allowed(split: str) -> None:
    if split not in ("train", "validation"):
        raise SealViolationError(f"split {split!r} not allowed in Phase 24E discovery")


def verify_frozen_input_hashes(repo_root: Path) -> dict[str, Any]:
    """Verify Phase-24C and key Phase-24D freeze hashes."""
    checks: list[dict[str, Any]] = []
    ok = True

    def _check(key: str, path: Path, expected: str) -> None:
        nonlocal ok
        assert_not_test_path(path)
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
    mapping_c = {
        "phase24c_design_matrix": "artifacts/phase24c_design/design_matrix.json",
        "phase24c_split": "artifacts/phase24c_design/split_manifest.json",
        "phase24c_analysis_spec": "artifacts/phase24c_design/analysis_spec.json",
        "phase24c_config": "configs/experiments/phase24c_full_trajectory_design.yaml",
        "phase24c_schedule_template": (
            "artifacts/phase24c_design/generation_schedule_template.json"
        ),
    }
    key_map = {
        "phase24c_design_matrix": "design_matrix.json",
        "phase24c_split": "split_manifest.json",
        "phase24c_analysis_spec": "analysis_spec.json",
        "phase24c_config": "config",
        "phase24c_schedule_template": "generation_schedule_template.json",
    }
    for k, rel in mapping_c.items():
        _check(k, repo_root / rel, c24[key_map[k]])

    d24 = json.loads(
        (repo_root / "artifacts/phase24d_collection/artifact_hashes.json").read_text(
            encoding="utf-8"
        )
    )
    for k, rel in (
        ("phase24d_summary", "artifacts/phase24d_collection/summary.json"),
        ("phase24d_schedule", "artifacts/phase24d_collection/schedule.json"),
        ("phase24d_train_labels", "artifacts/phase24d_collection/by_split/TRAIN/labels.json"),
        (
            "phase24d_val_labels",
            "artifacts/phase24d_collection/by_split/VALIDATION/labels.json",
        ),
    ):
        hash_key = {
            "phase24d_summary": "summary.json",
            "phase24d_schedule": "schedule.json",
            "phase24d_train_labels": "TRAIN_labels.json",
            "phase24d_val_labels": "VALIDATION_labels.json",
        }[k]
        _check(k, repo_root / rel, d24[hash_key])

    # Confirm TEST seal files exist but are not opened for science
    test_label = (
        repo_root
        / "artifacts/phase24d_collection/by_split/LOCKED_TEST/labels_SEALED.json"
    )
    test_exists = test_label.exists()
    checks.append(
        {
            "key": "locked_test_exists_unopened",
            "path": str(test_label.relative_to(repo_root)),
            "pass": test_exists,
            "note": "existence only; discovery must not deserialize contents",
        }
    )
    if not test_exists:
        ok = False

    design = json.loads(
        (repo_root / "artifacts/phase24c_design/design_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    k_ok = design.get("selected_k") == 16
    if not k_ok:
        ok = False
    checks.append(
        {
            "key": "selected_k_16",
            "pass": k_ok,
            "observed": design.get("selected_k"),
        }
    )
    return {"verified": ok, "checks": checks}


def bf16_u16_to_float32(arr_u16: np.ndarray) -> np.ndarray:
    """Interpret uint16 bit patterns as bfloat16 → float32."""
    u16 = np.asarray(arr_u16, dtype=np.uint16)
    # bfloat16: top 16 bits of float32
    as_u32 = u16.astype(np.uint32) << 16
    return as_u32.view(np.float32)


def delta_h_vector(h_t: np.ndarray, h_0: np.ndarray) -> np.ndarray:
    if h_t.shape != h_0.shape:
        raise ValueError("Δh shape mismatch")
    return h_t.astype(np.float32) - h_0.astype(np.float32)


def prefix_token_ids(
    generated_token_ids: Sequence[int], prediction_step: int
) -> list[int]:
    """Surface tokens available before sampling token at step t: 0..t-1."""
    if prediction_step < 0:
        raise ValueError("prediction_step < 0")
    if prediction_step > len(generated_token_ids):
        raise ValueError("future token leakage: step exceeds generated length")
    return list(generated_token_ids[:prediction_step])


def survives_landmark(n_generated: int, landmark: int) -> bool:
    """Prediction step `landmark` exists iff n_generated > landmark."""
    return int(n_generated) > int(landmark)


def make_text_pipeline(C: float, random_state: int = ANALYSIS_SEED) -> Pipeline:
    """Frozen TEXT: word 1–2 + char-within-word 3–5 TF-IDF → L2 logistic."""
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)
    char_wb = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    return Pipeline(
        [
            ("features", FeatureUnion([("word", word), ("char_wb", char_wb)])),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    solver="liblinear",
                    class_weight=None,  # sample weights carry balanced+prompt-equal
                    max_iter=2000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_logit_or_act_clf(C: float, random_state: int = ANALYSIS_SEED) -> LogisticRegression:
    return LogisticRegression(
        C=C,
        solver="liblinear",
        class_weight=None,
        max_iter=2000,
        random_state=random_state,
    )


def _as_binary_labels(classes: Sequence[str]) -> np.ndarray:
    """honest=0, deceptive=1."""
    out = []
    for c in classes:
        if c == "honest":
            out.append(0)
        elif c == "deceptive":
            out.append(1)
        else:
            raise ValueError(f"non-primary label {c}")
    return np.asarray(out, dtype=np.int32)


def sample_weights_for_fit(labels: Sequence[str], prompt_ids: Sequence[str]) -> np.ndarray:
    """Documented Phase-24C weighting: balanced class × equal total weight/prompt."""
    return np.asarray(
        prompt_equal_balanced_weights(labels, prompt_ids), dtype=np.float64
    )


def select_C_prompt_group_cv(
    *,
    X: Any,
    y: np.ndarray,
    prompt_ids: Sequence[str],
    labels_str: Sequence[str],
    kind: str,
    C_grid: Sequence[float] = LOGISTIC_C_GRID,
    n_folds: int = N_FOLDS,
    seed: int = ANALYSIS_SEED,
) -> dict[str, Any]:
    """Select C by prompt-group CV AUROC on TRAIN only."""
    prompts = sorted(set(prompt_ids))
    folds = prompt_group_folds(prompts, n_folds=min(n_folds, len(prompts)))
    # drop empty folds
    folds = [f for f in folds if f]
    best_C = float(C_grid[0])
    best_score = -1.0
    per_C: dict[str, float] = {}
    pid_arr = np.asarray(list(prompt_ids))
    for C in C_grid:
        oof = np.full(len(y), np.nan, dtype=np.float64)
        for hold in folds:
            hold_set = set(hold)
            te_mask = np.array([p in hold_set for p in pid_arr])
            tr_mask = ~te_mask
            if tr_mask.sum() < 2 or te_mask.sum() < 1:
                continue
            if len(np.unique(y[tr_mask])) < 2:
                continue
            w = sample_weights_for_fit(
                [labels_str[i] for i in range(len(y)) if tr_mask[i]],
                [prompt_ids[i] for i in range(len(y)) if tr_mask[i]],
            )
            if kind == "text":
                pipe = make_text_pipeline(float(C), random_state=seed)
                X_tr = [X[i] for i in range(len(y)) if tr_mask[i]]
                X_te = [X[i] for i in range(len(y)) if te_mask[i]]
                pipe.fit(X_tr, y[tr_mask], clf__sample_weight=w)
                oof[te_mask] = pipe.predict_proba(X_te)[:, 1]
            else:
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X[tr_mask])
                X_te = scaler.transform(X[te_mask])
                clf = make_logit_or_act_clf(float(C), random_state=seed)
                clf.fit(X_tr, y[tr_mask], sample_weight=w)
                oof[te_mask] = clf.predict_proba(X_te)[:, 1]
        valid = np.isfinite(oof)
        if valid.sum() < 2 or len(np.unique(y[valid])) < 2:
            score = float("nan")
        else:
            score = float(roc_auc_score(y[valid], oof[valid]))
        per_C[str(C)] = score
        if not math.isnan(score) and score > best_score:
            best_score = score
            best_C = float(C)
    return {"selected_C": best_C, "cv_auroc_by_C": per_C, "best_cv_auroc": best_score}


def oof_log_odds(
    *,
    X: Any,
    y: np.ndarray,
    prompt_ids: Sequence[str],
    labels_str: Sequence[str],
    kind: str,
    C: float,
    n_folds: int = N_FOLDS,
    seed: int = ANALYSIS_SEED,
) -> np.ndarray:
    """Out-of-fold predicted log-odds for stacking (TRAIN leakage prevention)."""
    prompts = sorted(set(prompt_ids))
    folds = [f for f in prompt_group_folds(prompts, n_folds=min(n_folds, len(prompts))) if f]
    pid_arr = np.asarray(list(prompt_ids))
    oof = np.full(len(y), np.nan, dtype=np.float64)
    for hold in folds:
        hold_set = set(hold)
        te_mask = np.array([p in hold_set for p in pid_arr])
        tr_mask = ~te_mask
        if tr_mask.sum() < 2 or te_mask.sum() < 1 or len(np.unique(y[tr_mask])) < 2:
            continue
        w = sample_weights_for_fit(
            [labels_str[i] for i in range(len(y)) if tr_mask[i]],
            [prompt_ids[i] for i in range(len(y)) if tr_mask[i]],
        )
        if kind == "text":
            pipe = make_text_pipeline(C, random_state=seed)
            X_tr = [X[i] for i in range(len(y)) if tr_mask[i]]
            X_te = [X[i] for i in range(len(y)) if te_mask[i]]
            pipe.fit(X_tr, y[tr_mask], clf__sample_weight=w)
            proba = pipe.predict_proba(X_te)[:, 1]
        else:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr_mask])
            X_te = scaler.transform(X[te_mask])
            clf = make_logit_or_act_clf(C, random_state=seed)
            clf.fit(X_tr, y[tr_mask], sample_weight=w)
            proba = clf.predict_proba(X_te)[:, 1]
        # convert to log-odds
        proba = np.clip(proba, 1e-6, 1 - 1e-6)
        oof[te_mask] = np.log(proba / (1 - proba))
    return oof


def fit_predict_log_odds(
    *,
    X_train: Any,
    y_train: np.ndarray,
    labels_train: Sequence[str],
    prompts_train: Sequence[str],
    X_eval: Any,
    kind: str,
    C: float,
    seed: int = ANALYSIS_SEED,
) -> np.ndarray:
    """Fit on TRAIN, return eval log-odds. Preprocessing TRAIN-only."""
    w = sample_weights_for_fit(labels_train, prompts_train)
    if kind == "text":
        pipe = make_text_pipeline(C, random_state=seed)
        pipe.fit(list(X_train), y_train, clf__sample_weight=w)
        proba = pipe.predict_proba(list(X_eval))[:, 1]
    else:
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train)
        Xev = scaler.transform(X_eval)
        clf = make_logit_or_act_clf(C, random_state=seed)
        clf.fit(Xtr, y_train, sample_weight=w)
        proba = clf.predict_proba(Xev)[:, 1]
    proba = np.clip(proba, 1e-6, 1 - 1e-6)
    return np.log(proba / (1 - proba))


def select_C_meta_cv(
    *,
    Z_oof: np.ndarray,
    y: np.ndarray,
    prompt_ids: Sequence[str],
    labels_str: Sequence[str],
    C_grid: Sequence[float] = LOGISTIC_C_GRID,
    n_folds: int = N_FOLDS,
    seed: int = ANALYSIS_SEED,
) -> dict[str, Any]:
    """Select meta L2 C on OOF base features with prompt-group CV."""
    valid = np.all(np.isfinite(Z_oof), axis=1)
    if valid.sum() < 4:
        return {"selected_C": 1.0, "cv_auroc_by_C": {}, "best_cv_auroc": float("nan")}
    return select_C_prompt_group_cv(
        X=Z_oof,
        y=y,
        prompt_ids=prompt_ids,
        labels_str=labels_str,
        kind="matrix",
        C_grid=C_grid,
        n_folds=n_folds,
        seed=seed,
    )


def safe_auroc(y: np.ndarray, scores: np.ndarray) -> float:
    if len(y) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def safe_auprc(y: np.ndarray, scores: np.ndarray) -> float:
    if len(y) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, scores))


def prompt_cluster_bootstrap_ci(
    *,
    y: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray | None,
    prompt_ids: Sequence[str],
    n_reps: int = VAL_BOOTSTRAP_REPS,
    seed: int = ANALYSIS_SEED,
) -> dict[str, Any]:
    """Bootstrap prompts (not trajectories). scores_b optional for ΔAUROC."""
    rng = np.random.default_rng(seed)
    prompts = sorted(set(prompt_ids))
    pid_arr = np.asarray(list(prompt_ids))
    by_p = {p: np.where(pid_arr == p)[0] for p in prompts}
    stats = []
    for _ in range(n_reps):
        sampled = rng.choice(prompts, size=len(prompts), replace=True)
        idx = np.concatenate([by_p[p] for p in sampled])
        ya = y[idx]
        if scores_b is None:
            stats.append(safe_auroc(ya, scores_a[idx]))
        else:
            a = safe_auroc(ya, scores_a[idx])
            b = safe_auroc(ya, scores_b[idx])
            stats.append(b - a if not (math.isnan(a) or math.isnan(b)) else float("nan"))
    arr = np.asarray(stats, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return {
            "n_reps": n_reps,
            "mean": float("nan"),
            "ci95": [float("nan"), float("nan")],
        }
    return {
        "n_reps": n_reps,
        "mean": float(np.mean(finite)),
        "ci95": [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))],
        "n_finite": int(len(finite)),
    }


def svg_heatmap(
    matrix: np.ndarray,
    *,
    times: Sequence[int],
    layers: Sequence[int],
    title: str,
    path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Dependency-free SVG heatmap (x=time, y=layer)."""
    mat = np.asarray(matrix, dtype=np.float64)
    # mat shape (n_layers, n_times)
    finite = mat[np.isfinite(mat)]
    if vmin is None:
        vmin = float(np.min(finite)) if len(finite) else 0.0
    if vmax is None:
        vmax = float(np.max(finite)) if len(finite) else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6
    cell_w, cell_h = 36, 12
    pad_l, pad_t, pad_r, pad_b = 48, 40, 20, 40
    width = pad_l + cell_w * len(times) + pad_r
    height = pad_t + cell_h * len(layers) + pad_b
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width // 2}" y="22" text-anchor="middle" '
        f'font-family="sans-serif" font-size="13">{title}</text>',
    ]

    def color(v: float) -> str:
        if not np.isfinite(v):
            return "#dddddd"
        t = (v - vmin) / (vmax - vmin)
        t = max(0.0, min(1.0, t))
        # blue → white → red
        if t < 0.5:
            u = t * 2
            r = int(255 * u)
            g = int(255 * u)
            b = 255
        else:
            u = (t - 0.5) * 2
            r = 255
            g = int(255 * (1 - u))
            b = int(255 * (1 - u))
        return f"#{r:02x}{g:02x}{b:02x}"

    for yi, L in enumerate(layers):
        for xi, _t in enumerate(times):
            v = mat[yi, xi]
            x = pad_l + xi * cell_w
            y = pad_t + yi * cell_h
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" '
                f'fill="{color(v)}" stroke="#eee" stroke-width="0.5"/>'
            )
        parts.append(
            f'<text x="{pad_l - 4}" y="{pad_t + yi * cell_h + cell_h - 2}" '
            f'text-anchor="end" font-family="sans-serif" font-size="8">L{L}</text>'
        )
    for xi, t in enumerate(times):
        parts.append(
            f'<text x="{pad_l + xi * cell_w + cell_w / 2}" y="{height - 16}" '
            f'text-anchor="middle" font-family="sans-serif" font-size="10">{t}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def list_qualifying_bands(
    delta_auroc_by_time_layer: dict[int, dict[int, float]],
) -> list[dict[str, Any]]:
    """All bands satisfying contiguous Δ>0 and median≥0.05 (for reporting)."""
    out: list[dict[str, Any]] = []
    for t in TEMPORAL_LANDMARKS:
        layer_map = delta_auroc_by_time_layer.get(t, {})
        start = None
        for L in range(N_LAYERS):
            d = float(layer_map.get(L, float("-inf")))
            if d > 0:
                if start is None:
                    start = L
            else:
                if start is not None:
                    end = L - 1
                    if end - start + 1 >= CANDIDATE_MIN_CONSEC_LAYERS:
                        vals = [float(layer_map[i]) for i in range(start, end + 1)]
                        med = float(np.median(vals))
                        if med >= CANDIDATE_MIN_MEDIAN_DELTA:
                            out.append(
                                {
                                    "time": t,
                                    "band": [start, end],
                                    "median_delta_auroc": med,
                                }
                            )
                    start = None
        if start is not None:
            end = N_LAYERS - 1
            if end - start + 1 >= CANDIDATE_MIN_CONSEC_LAYERS:
                vals = [float(layer_map[i]) for i in range(start, end + 1)]
                med = float(np.median(vals))
                if med >= CANDIDATE_MIN_MEDIAN_DELTA:
                    out.append(
                        {"time": t, "band": [start, end], "median_delta_auroc": med}
                    )
    return out
