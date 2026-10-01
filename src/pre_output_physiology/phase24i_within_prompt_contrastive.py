"""Phase 24I — within-prompt contrastive analysis of one frozen Phase-24H lead.

DEVELOPMENT ONLY (TRAIN+VALIDATION = 28 prompts, 388 honest/deceptive rollouts).
Lead representation (frozen, the ONLY activation model):
    C_MULTILAYER | DELTA | t1   with  Δh(1, L) = h(1, L) - h(0, L),  L = 0..31.

Question: among stochastic rollouts of the SAME prompt, is there a common t=1
residual-stream pattern that ranks deceptive above honest rollouts on entirely unseen
prompts, and does it add value beyond the visible SURFACE baseline?

Training is purely within-prompt pairwise (every H x D pair of one prompt contributes
both orientations); the primary estimand is the prompt-equal mean of salt-averaged
within-prompt ΔAUROC (NOT a pooled trajectory AUROC).

Phase-24F stays the final confirmatory FAIL. No GPU, no model generation. The library
never writes files.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from pre_output_physiology import phase24h_internal_signal_breadth as H
from pre_output_physiology.phase24c_design import prompt_group_folds, sha256_file
from pre_output_physiology.phase24e_discovery import SealViolationError, safe_auroc

STATUS = "phase24i_within_prompt_contrastive_complete_awaiting_independent_code_audit"
STARTING_SHA = "baec005ac60208adb13f43e17f9e1b8c915e9af3"
EXPLORATORY_LABEL = "EXPLORATORY_DEVELOPMENT_ONLY_PHASE24I"
ANALYSIS_SEED = H.ANALYSIS_SEED  # 2408: identical classifier seed => identical 24H SURFACE
GUARANTEE = (
    "PHASE 24I WAS A WITHIN-PROMPT CONTRASTIVE ANALYSIS OF ONE FROZEN PHASE-24H LEAD ON "
    "DEVELOPMENT (TRAIN+VALIDATION) DATA ONLY. LOCKED TEST LABELS/ACTIVATIONS AND "
    "PHASE-24F TEST ACTIVATIONS WERE NOT ACCESSED. PHASE-24F REMAINS THE FINAL "
    "CONFIRMATORY FAIL. NO GPU, MODEL GENERATION, SAE, OR CAUSAL WORK WAS PERFORMED."
)

# ---- frozen protocol constants (mirrored in the YAML config) ----
LEAD_KEY = "C_MULTILAYER|DELTA|t1"
LEAD_TIME = 1
LEAD_VARIANT = "DELTA"
N_LAYERS = 32
N_DEV_PROMPTS = H.N_DEV_PROMPTS  # 28
N_EXPECTED_RECORDS = 388
N_OUTER_FOLDS = H.N_OUTER_FOLDS  # 7
PROMPTS_PER_EVAL_FOLD = 4
OUTER_SALTS: tuple[int, ...] = H.OUTER_SALTS  # (0, 1, 2)
INNER_FOLDS = H.INNER_FOLDS  # 5
LAYER_C = float(H.PROBE_C)  # 0.1 frozen
META_C = 1.0  # frozen
COMBINED_C = float(H.COMBINED_C)  # 10.0 frozen
SURFACE_C_TEXT = float(H.SURFACE_C_TEXT)
SURFACE_C_LOGITS = float(H.SURFACE_C_LOGITS)
SURFACE_C_META = float(H.SURFACE_C_META)
MIN_PER_CLASS_WITHIN_PROMPT = 2
N_BOOTSTRAP = 10000
N_SHUFFLES = 100
SHUFFLE_SALTS: tuple[int, ...] = (0,)
SHUFFLE_ALPHA = 0.05
MIN_FRACTION_POSITIVE_PROMPTS = 0.60
SCALE_FLOOR = 1e-12

PHASE24H_FOLDS_REL = "artifacts/phase24h_internal_signal_breadth/outer_folds.json"
PHASE24H_FOLDS_SHA256 = "dee54b0f39fc3a7c79afcbe5a90bccf9383e51da1f2ab21cf83b74c2f0d85eb6"
PHASE24H_INPUT_HASHES_REL = "artifacts/phase24h_internal_signal_breadth/input_hashes.json"
PHASE24H_INPUT_AGGREGATE_SHA256 = (
    "d94982adb6db42b8fd290b36e228cb522edfc08b6a797a73236f2e712fb55465"
)
PHASE24H_ROW_METRICS_REL = "artifacts/phase24h_internal_signal_breadth/row_metrics.json"

PROMOTED = H.PROMOTED
NOT_PROMOTED = H.NOT_PROMOTED
REVOKED = H.REVOKED


# ---- guards ----
def assert_dev_input_path(path: str | Path) -> None:
    """Raise SealViolationError if path touches TEST / sealed / Phase-24F TEST."""
    H.assert_dev_input_path(path)


def assert_dev_activation_path(path: str | Path, repo_root: Path) -> None:
    """Activation files must live under activations_local/{TRAIN,VALIDATION}."""
    H.assert_dev_activation_path(path, repo_root)


def derive_seed(*parts: Any) -> int:
    h = hashlib.sha256("|".join(["phase24i", *map(str, parts)]).encode()).hexdigest()
    return int(h[:8], 16)


# ---- frozen Phase-24H outer folds (reused verbatim, never regenerated) ----
def load_phase24h_folds(
    repo_root: Path, dev_prompts: list[str] | None = None
) -> dict[int, list[list[str]]]:
    """Load the exact Phase-24H outer folds; verify SHA256 and structure."""
    path = repo_root / PHASE24H_FOLDS_REL
    assert_dev_input_path(path)
    digest = sha256_file(str(path))
    if digest != PHASE24H_FOLDS_SHA256:
        raise RuntimeError(
            f"Phase-24H outer_folds.json hash mismatch: {digest} != {PHASE24H_FOLDS_SHA256}"
        )
    blob = json.loads(path.read_text(encoding="utf-8"))
    folds = {int(s): [list(f) for f in v] for s, v in blob.items()}
    if tuple(sorted(folds)) != OUTER_SALTS:
        raise RuntimeError(f"unexpected salts in Phase-24H folds: {sorted(folds)}")
    for s, fl in folds.items():
        if len(fl) != N_OUTER_FOLDS or any(len(f) != PROMPTS_PER_EVAL_FOLD for f in fl):
            raise RuntimeError(f"salt {s}: fold structure is not 7 x 4")
        flat = [p for f in fl for p in f]
        if len(set(flat)) != len(flat):
            raise RuntimeError(f"salt {s}: prompt held out more than once")
        if dev_prompts is not None and sorted(flat) != sorted(dev_prompts):
            raise RuntimeError(f"salt {s}: folds do not partition the DEV prompts")
    return folds


def train_eval_split(
    folds: dict[int, list[list[str]]], dev_prompts: list[str], salt: int, fold: int
) -> tuple[list[str], list[str]]:
    """(train, eval) prompts for one frozen Phase-24H outer fold."""
    ev = sorted(folds[salt][fold])
    ev_set = set(ev)
    tr = sorted(p for p in dev_prompts if p not in ev_set)
    if set(tr) & ev_set:
        raise AssertionError("prompt leakage across outer fold")
    return tr, ev


def aggregate_input_hash(file_hashes: dict[str, str]) -> str:
    """Aggregate used by Phase 24H: sha256 of 'SPLIT/name hash' lines in listed order."""
    return hashlib.sha256(
        "\n".join(f"{k} {v}" for k, v in file_hashes.items()).encode()
    ).hexdigest()


def verify_against_phase24h_hashes(
    file_hashes: dict[str, str], repo_root: Path
) -> dict[str, Any]:
    """Compare freshly computed DEV NPZ hashes with the Phase-24H manifest."""
    path = repo_root / PHASE24H_INPUT_HASHES_REL
    manifest = json.loads(path.read_text(encoding="utf-8"))
    ref = manifest["files"]
    agg = aggregate_input_hash(file_hashes)
    missing = sorted(set(ref) - set(file_hashes))
    extra = sorted(set(file_hashes) - set(ref))
    differing = sorted(k for k in set(ref) & set(file_hashes) if ref[k] != file_hashes[k])
    return {
        "n_files_now": len(file_hashes),
        "n_files_manifest": len(ref),
        "aggregate_now": agg,
        "aggregate_manifest": manifest["aggregate_sha256"],
        "aggregate_expected_frozen": PHASE24H_INPUT_AGGREGATE_SHA256,
        "missing_vs_manifest": missing,
        "extra_vs_manifest": extra,
        "differing_vs_manifest": differing,
        "verified": bool(
            not missing
            and not extra
            and not differing
            and agg == manifest["aggregate_sha256"] == PHASE24H_INPUT_AGGREGATE_SHA256
        ),
    }


# ---- Δh construction ----
def delta_layer(
    data: H.DevData, rows: np.ndarray, layer: int, act_map: np.ndarray | None = None
) -> np.ndarray:
    """Δh(1, L) = h(1, L) - h(0, L) for one layer. act_map moves BOTH h0 and h1 together."""
    src = rows if act_map is None else act_map[rows]
    h1 = data.h[LEAD_TIME][src, layer, :]
    h0 = data.h0[src, layer, :]
    return (h1 - h0).astype(np.float32, copy=False)


def delta_all_layers(
    data: H.DevData, rows: np.ndarray, act_map: np.ndarray | None = None
) -> np.ndarray:
    return H.get_activations(data, LEAD_TIME, rows, LEAD_VARIANT, act_map)


# ---- within-prompt pair construction ----
@dataclass
class PairIndex:
    d_idx: np.ndarray  # positions (into the supplied rows) of deceptive members
    h_idx: np.ndarray  # positions of honest members
    prompts: np.ndarray  # prompt id of every pair

    @property
    def n_pairs(self) -> int:
        return int(len(self.d_idx))

    def per_prompt_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for p in self.prompts.tolist():
            out[p] = out.get(p, 0) + 1
        return out


def build_pair_index(y: np.ndarray, prompts: Any) -> PairIndex:
    """All H x D pairs, WITHIN a single prompt only (no between-prompt pairs)."""
    pid = np.asarray(prompts)
    y = np.asarray(y)
    d_parts: list[np.ndarray] = []
    h_parts: list[np.ndarray] = []
    p_parts: list[np.ndarray] = []
    for p in sorted(set(pid.tolist())):
        idx = np.where(pid == p)[0]
        hi = idx[y[idx] == 0]
        di = idx[y[idx] == 1]
        if len(hi) == 0 or len(di) == 0:
            continue
        dd, hh = np.meshgrid(di, hi, indexing="ij")
        d_parts.append(dd.ravel())
        h_parts.append(hh.ravel())
        p_parts.append(np.full(dd.size, p, dtype=object))
    if not d_parts:
        e = np.zeros(0, dtype=np.int64)
        return PairIndex(e, e.copy(), np.zeros(0, dtype=object))
    return PairIndex(
        np.concatenate(d_parts), np.concatenate(h_parts), np.concatenate(p_parts)
    )


def pair_weights(pair_prompts: np.ndarray) -> np.ndarray:
    """Per-pair weight so that each prompt has equal TOTAL weight over both orientations.

    Each unordered pair yields two samples (label 1 and label 0) that each get the SAME
    weight w = 1 / (2 * n_pairs_in_prompt): prompt total = 1 and, because every pair
    contributes one sample to each class, the two classes also have equal total weight.
    """
    counts: dict[str, int] = {}
    for p in pair_prompts.tolist():
        counts[p] = counts.get(p, 0) + 1
    return np.asarray([1.0 / (2.0 * counts[p]) for p in pair_prompts.tolist()], dtype=np.float64)


def pairwise_design(
    X: np.ndarray, y: np.ndarray, prompts: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray, PairIndex]:
    """Both orientations: (x_D - x_H, 1) and (x_H - x_D, 0); prompt/class-equal weights."""
    pairs = build_pair_index(y, prompts)
    if pairs.n_pairs == 0:
        raise ValueError("no within-prompt H x D pairs available")
    xd = X[pairs.d_idx].astype(np.float64)
    xh = X[pairs.h_idx].astype(np.float64)
    Xp = np.concatenate([xd - xh, xh - xd], axis=0)
    yp = np.concatenate([np.ones(pairs.n_pairs), np.zeros(pairs.n_pairs)]).astype(np.int32)
    w = pair_weights(pairs.prompts)
    wp = np.concatenate([w, w])
    return Xp, yp, wp, pairs


def has_pairs(y: np.ndarray, prompts: Any) -> bool:
    return build_pair_index(y, prompts).n_pairs > 0


@dataclass
class PairwiseLinear:
    """Linear L2 pairwise logistic probe. score(x) = (x / scale) . coef (no intercept)."""

    C: float
    scale: np.ndarray
    coef: np.ndarray
    n_pairs: int
    n_prompts: int

    def score(self, X: np.ndarray) -> np.ndarray:
        return np.asarray((np.asarray(X, dtype=np.float64) / self.scale) @ self.coef)


def fit_pairwise(
    X: np.ndarray, y: np.ndarray, prompts: Any, C: float, seed: int = ANALYSIS_SEED
) -> PairwiseLinear:
    """Fit on within-prompt pair differences of the supplied (training) rows ONLY.

    Non-finite rows are dropped. Features are scaled (no centering; pair differences are
    symmetric, mean zero) by the RMS of the training pair differences, so the scale
    depends on training pairs only.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    pid = np.asarray(prompts)
    ok = np.all(np.isfinite(X), axis=1)
    Xp, yp, wp, pairs = pairwise_design(X[ok], y[ok], pid[ok])
    n_pairs = pairs.n_pairs
    scale = np.sqrt(np.mean(Xp[:n_pairs] ** 2, axis=0))
    scale = np.where(scale < SCALE_FLOOR, 1.0, scale)
    clf = LogisticRegression(
        C=float(C), solver="liblinear", fit_intercept=False, class_weight=None,
        max_iter=2000, random_state=seed,
    )
    clf.fit(Xp / scale, yp, sample_weight=wp)
    return PairwiseLinear(
        C=float(C), scale=scale, coef=clf.coef_.ravel().copy(),
        n_pairs=n_pairs, n_prompts=len(pairs.per_prompt_counts()),
    )


def pairwise_oof_scores(
    X: np.ndarray, y: np.ndarray, prompts: Any, C: float,
    seed: int = ANALYSIS_SEED, n_folds: int = INNER_FOLDS,
) -> np.ndarray:
    """Inner prompt-group OOF scores: a row is scored by a model that never saw its prompt."""
    pid = np.asarray(prompts)
    y = np.asarray(y)
    uniq = sorted(set(pid.tolist()))
    oof = np.full(len(y), np.nan, dtype=np.float64)
    if len(uniq) < 2:
        return oof
    folds = [f for f in prompt_group_folds(uniq, n_folds=min(n_folds, len(uniq))) if f]
    for hold in folds:
        te = np.isin(pid, list(hold))
        tr = ~te
        if not has_pairs(y[tr], pid[tr]):
            continue
        model = fit_pairwise(X[tr], y[tr], pid[tr], C, seed)
        oof[te] = model.score(X[te])
    return oof


# ---- the 32-layer pairwise probe bank ----
def layer_scores(
    data: H.DevData,
    tr: np.ndarray,
    ev: np.ndarray,
    act_map: np.ndarray | None = None,
    seed: int = ANALYSIS_SEED,
    n_layers: int = N_LAYERS,
) -> tuple[np.ndarray, np.ndarray]:
    """All 32 layers, C=0.1, no layer selection.

    Train rows: inner prompt-group OOF scores. Eval rows: probe fit on ALL eligible
    outer-train prompts, applied to unseen eval prompts.
    """
    y_tr = data.y[tr]
    pid_tr = np.asarray(data.prompt_ids)[tr]
    Z_oof = np.full((len(tr), n_layers), np.nan)
    Z_ev = np.zeros((len(ev), n_layers))
    for layer in range(n_layers):
        X_tr = delta_layer(data, tr, layer, act_map)
        X_ev = delta_layer(data, ev, layer, act_map)
        Z_oof[:, layer] = pairwise_oof_scores(X_tr, y_tr, pid_tr, LAYER_C, seed)
        Z_ev[:, layer] = fit_pairwise(X_tr, y_tr, pid_tr, LAYER_C, seed).score(X_ev)
    return Z_oof, Z_ev


# ---- outer fold ----
@dataclass
class FoldResult:
    salt: int
    fold: int
    eval_rows: np.ndarray
    surf: np.ndarray
    act: np.ndarray
    comb: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)
    surface: H.SurfaceOut | None = None
    timings: dict[str, float] = field(default_factory=dict)


def pair_summary(data: H.DevData, rows: np.ndarray) -> dict[str, Any]:
    pairs = build_pair_index(data.y[rows], np.asarray(data.prompt_ids)[rows])
    counts = pairs.per_prompt_counts()
    return {
        "n_rollouts": int(len(rows)),
        "n_pair_bearing_prompts": len(counts),
        "n_hxd_pairs": int(pairs.n_pairs),
        "n_training_samples_both_orientations": int(2 * pairs.n_pairs),
        "pairs_per_prompt": dict(sorted(counts.items())),
    }


def run_outer_fold(
    data: H.DevData,
    train_prompts: list[str],
    eval_prompts: list[str],
    *,
    salt: int = -1,
    fold: int = -1,
    act_map: np.ndarray | None = None,
    surface: H.SurfaceOut | None = None,
    seed: int = ANALYSIS_SEED,
) -> FoldResult:
    """All fitting uses `train_prompts` only; scores are produced for `eval_prompts` only."""
    tp, ep = set(train_prompts), set(eval_prompts)
    if tp & ep:
        raise AssertionError("train/eval prompt overlap in outer fold")
    pid = np.asarray(data.prompt_ids)
    surv = data.survivors(LEAD_TIME)
    tr = np.where(surv & np.isin(pid, list(tp)))[0]
    ev = np.where(surv & np.isin(pid, list(ep)))[0]
    if len(tr) < 8 or len(ev) == 0 or len(np.unique(data.y[tr])) < 2:
        raise RuntimeError("insufficient rows for outer fold")
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    sf = surface if surface is not None else H.fit_surface(data, LEAD_TIME, tr, ev, seed)
    if not (np.array_equal(sf.train_rows, tr) and np.array_equal(sf.eval_rows, ev)):
        raise AssertionError("cached SURFACE rows differ from activation rows")
    timings["surface"] = time.perf_counter() - t0

    y_tr = data.y[tr]
    pid_tr = pid[tr]
    t0 = time.perf_counter()
    Z_oof, Z_ev = layer_scores(data, tr, ev, act_map, seed)
    timings["layer_probes"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    meta = fit_pairwise(Z_oof, y_tr, pid_tr, META_C, seed)
    act_oof = pairwise_oof_scores(Z_oof, y_tr, pid_tr, META_C, seed)
    act_ev = meta.score(Z_ev)
    comb_model = fit_pairwise(
        np.stack([sf.oof, act_oof], axis=1), y_tr, pid_tr, COMBINED_C, seed
    )
    comb_ev = comb_model.score(np.stack([sf.ev, act_ev], axis=1))
    timings["meta_combined"] = time.perf_counter() - t0

    info = {
        "salt": salt,
        "fold": fold,
        "eval_prompts": sorted(eval_prompts),
        "n_train_rollouts": int(len(tr)),
        "n_eval_rollouts": int(len(ev)),
        "train_pairs": pair_summary(data, tr),
        "meta_n_pairs": meta.n_pairs,
        "meta_coef": [float(c) for c in meta.coef],
        "combined_coef_surface_activation": [float(c) for c in comb_model.coef],
        "n_layers": int(Z_oof.shape[1]),
        "layers_selected": "none (all 32 layers enter the meta)",
    }
    return FoldResult(
        salt=salt, fold=fold, eval_rows=ev, surf=sf.ev, act=act_ev, comb=comb_ev,
        info=info, surface=sf, timings=timings,
    )


def assemble_predictions(
    data: H.DevData, results: list[FoldResult]
) -> dict[int, dict[str, np.ndarray]]:
    """salt -> arrays. Every trajectory is predicted at most once per salt."""
    by_salt: dict[int, list[FoldResult]] = {}
    for r in results:
        by_salt.setdefault(r.salt, []).append(r)
    pid = np.asarray(data.prompt_ids)
    out: dict[int, dict[str, np.ndarray]] = {}
    for salt, rs in sorted(by_salt.items()):
        rs = sorted(rs, key=lambda r: r.fold)
        rows = np.concatenate([r.eval_rows for r in rs])
        if len(set(rows.tolist())) != len(rows):
            raise AssertionError(f"duplicate predictions for salt {salt}")
        out[salt] = {
            "rows": rows,
            "y": data.y[rows],
            "prompts": pid[rows],
            "surf": np.concatenate([r.surf for r in rs]),
            "act": np.concatenate([r.act for r in rs]),
            "comb": np.concatenate([r.comb for r in rs]),
        }
    return out


# ---- activation-bundle shuffle (negative control) ----
def shuffle_permutation(prompt_ids: list[str], strata: np.ndarray, rep: int) -> np.ndarray:
    """Deterministic permutation within (prompt, stratum): act_map[i] = source row.

    The map redirects BOTH h0 and h1 together (one complete activation bundle)."""
    n = len(prompt_ids)
    act_map = np.arange(n)
    groups: dict[tuple[str, int], list[int]] = {}
    for i, (p, s) in enumerate(zip(prompt_ids, strata, strict=True)):
        groups.setdefault((p, int(s)), []).append(i)
    for (p, s), idx in sorted(groups.items()):
        if len(idx) < 2:
            continue
        rng = np.random.default_rng(derive_seed("shuffle", rep, p, s))
        act_map[idx] = np.asarray(idx)[rng.permutation(len(idx))]
    return act_map


def shuffle_control_verdict(
    real_delta: float, shuffle_deltas: list[float], alpha: float = SHUFFLE_ALPHA
) -> dict[str, Any]:
    """Survive iff one-sided p <= alpha AND real > 95th percentile of shuffled Δ."""
    return H.shuffle_control_verdict(real_delta, shuffle_deltas, alpha)


# ---- within-prompt metrics ----
def _prompt_index(p: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, list[int]] = {}
    for i, q in enumerate(p["prompts"].tolist()):
        out.setdefault(q, []).append(i)
    return {k: np.asarray(v) for k, v in out.items()}


def qualifies(y: np.ndarray) -> bool:
    return bool(
        int((y == 0).sum()) >= MIN_PER_CLASS_WITHIN_PROMPT
        and int((y == 1).sum()) >= MIN_PER_CLASS_WITHIN_PROMPT
    )


def per_prompt_salt_table(
    per_salt: dict[int, dict[str, np.ndarray]],
) -> dict[str, dict[str, Any]]:
    """prompt -> {n_h, n_d, salts: {salt: (surface, combined, activation AUROCs)}}."""
    table: dict[str, dict[str, Any]] = {}
    for s, p in sorted(per_salt.items()):
        for q, idx in _prompt_index(p).items():
            y = p["y"][idx]
            if not qualifies(y):
                continue
            a_s = safe_auroc(y, p["surf"][idx])
            a_c = safe_auroc(y, p["comb"][idx])
            a_a = safe_auroc(y, p["act"][idx])
            if not (math.isfinite(a_s) and math.isfinite(a_c)):
                continue
            ent = table.setdefault(
                q, {"n_h": int((y == 0).sum()), "n_d": int((y == 1).sum()), "salts": {}}
            )
            ent["salts"][s] = (a_s, a_c, a_a)
    return table


def primary_analysis(
    per_salt: dict[int, dict[str, np.ndarray]], n_boot: int, seed: int
) -> dict[str, Any]:
    """PRIMARY estimand: prompt-equal mean of salt-averaged within-prompt ΔAUROC.

    Uses stored cross-fitted predictions only (no model fitting). Each prompt gets one
    salt-averaged Δ; the primary statistic is the mean over qualifying prompts."""
    salts = sorted(per_salt)
    table = per_prompt_salt_table(per_salt)
    prompts = sorted(table)
    rows: list[dict[str, Any]] = []
    for q in prompts:
        ent = table[q]
        sv = ent["salts"]
        surf = float(np.mean([v[0] for v in sv.values()]))
        comb = float(np.mean([v[1] for v in sv.values()]))
        act_vals = [v[2] for v in sv.values() if math.isfinite(v[2])]
        rows.append({
            "prompt_id": q, "n_honest": ent["n_h"], "n_deceptive": ent["n_d"],
            "n_salts": len(sv),
            "surface_auroc": surf, "combined_auroc": comb, "delta_auroc": comb - surf,
            "activation_auroc_descriptive_only": (
                float(np.mean(act_vals)) if act_vals else float("nan")
            ),
            "delta_by_salt": {str(s): sv[s][1] - sv[s][0] for s in sorted(sv)},
        })
    d = np.asarray([r["delta_auroc"] for r in rows], dtype=np.float64)
    per_salt_mean = {}
    for s in salts:
        ds = [table[q]["salts"][s][1] - table[q]["salts"][s][0]
              for q in prompts if s in table[q]["salts"]]
        per_salt_mean[str(s)] = float(np.mean(ds)) if ds else float("nan")
    nan2 = [float("nan")] * 2
    if len(d) == 0:
        return {
            "n_qualifying": 0, "mean_delta": float("nan"), "median_delta": float("nan"),
            "fraction_gt_0": float("nan"), "per_salt_mean_delta": per_salt_mean,
            "rows": [], "bootstrap": {"n_reps": n_boot, "mean_ci95": nan2,
                                      "median_ci95": nan2},
        }
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx]
    means = boots.mean(axis=1)
    meds = np.median(boots, axis=1)
    act = [r["activation_auroc_descriptive_only"] for r in rows]
    return {
        "n_qualifying": int(len(d)),
        "mean_delta": float(np.mean(d)),
        "median_delta": float(np.median(d)),
        "fraction_gt_0": float(np.mean(d > 0)),
        "per_salt_mean_delta": per_salt_mean,
        "mean_surface_auroc": float(np.mean([r["surface_auroc"] for r in rows])),
        "mean_combined_auroc": float(np.mean([r["combined_auroc"] for r in rows])),
        "mean_activation_auroc_descriptive_only": float(np.nanmean(act)),
        "rows": rows,
        "bootstrap": {
            "n_reps": int(n_boot),
            "mean_of_means": float(means.mean()),
            "mean_ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
            "median_ci95": [float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))],
        },
    }


def salt_prompt_equal_delta(per_salt: dict[int, dict[str, np.ndarray]], salt: int) -> float:
    """Prompt-equal mean within-prompt ΔAUROC for ONE salt (shuffle-control statistic)."""
    table = per_prompt_salt_table({salt: per_salt[salt]})
    ds = [v["salts"][salt][1] - v["salts"][salt][0] for v in table.values()]
    return float(np.mean(ds)) if ds else float("nan")


def promotion_decision(
    *,
    mean_delta: float,
    ci_low: float,
    per_salt_delta: list[float],
    median_delta: float,
    fraction_positive: float,
    n_salts_expected: int = len(OUTER_SALTS),
) -> dict[str, Any]:
    """Frozen five-part rule: ALL must hold."""

    def _gt0(x: float) -> bool:
        return bool(math.isfinite(x) and x > 0)

    crit = {
        "prompt_equal_mean_delta_gt_0": _gt0(mean_delta),
        "bootstrap_lower_gt_0": _gt0(ci_low),
        "mean_delta_gt_0_in_all_salts": bool(
            len(per_salt_delta) == n_salts_expected and all(_gt0(x) for x in per_salt_delta)
        ),
        "median_prompt_delta_gt_0": _gt0(median_delta),
        "fraction_positive_prompts_ge_0.60": bool(
            math.isfinite(fraction_positive)
            and fraction_positive >= MIN_FRACTION_POSITIVE_PROMPTS
        ),
    }
    promoted = all(crit.values())
    return {"criteria": crit, "promoted": promoted,
            "label": PROMOTED if promoted else NOT_PROMOTED}


def promotion_from_primary(primary: dict[str, Any]) -> dict[str, Any]:
    return promotion_decision(
        mean_delta=primary["mean_delta"],
        ci_low=primary["bootstrap"]["mean_ci95"][0],
        per_salt_delta=[primary["per_salt_mean_delta"][str(s)] for s in OUTER_SALTS
                        if str(s) in primary["per_salt_mean_delta"]],
        median_delta=primary["median_delta"],
        fraction_positive=primary["fraction_gt_0"],
    )


def pooled_descriptive(per_salt: dict[int, dict[str, np.ndarray]]) -> dict[str, Any]:
    """Pooled-trajectory AUROCs. DESCRIPTIVE ONLY: never the primary estimand."""
    out: dict[str, Any] = {}
    deltas = []
    for s, p in sorted(per_salt.items()):
        a_s = safe_auroc(p["y"], p["surf"])
        a_c = safe_auroc(p["y"], p["comb"])
        out[str(s)] = {
            "n": int(len(p["y"])), "surface_auroc": a_s, "combined_auroc": a_c,
            "delta_auroc": a_c - a_s,
            "activation_auroc": safe_auroc(p["y"], p["act"]),
        }
        deltas.append(a_c - a_s)
    out["mean_delta_auroc_over_salts"] = float(np.mean(deltas)) if deltas else float("nan")
    return out


def full_metrics(
    per_salt: dict[int, dict[str, np.ndarray]], n_boot: int, seed: int
) -> dict[str, Any]:
    primary = primary_analysis(per_salt, n_boot, seed)
    return {
        "primary": primary,
        "promotion": promotion_from_primary(primary),
        "pooled_descriptive": pooled_descriptive(per_salt),
    }


# ---- 24H vs 24I descriptive comparison (24H artifacts are read-only) ----
def compare_with_phase24h(
    repo_root: Path, metrics_24i: dict[str, Any]
) -> dict[str, Any]:
    blob = json.loads((repo_root / PHASE24H_ROW_METRICS_REL).read_text(encoding="utf-8"))
    row = blob["rows"][LEAD_KEY]
    sp = row["same_prompt"]
    p = metrics_24i["primary"]
    h_by_prompt = {r["prompt_id"]: r["delta_auroc"] for r in row["same_prompt_rows"]}
    i_by_prompt = {r["prompt_id"]: r["delta_auroc"] for r in p["rows"]}
    shared = sorted(set(h_by_prompt) & set(i_by_prompt))
    return {
        "descriptive_only": True,
        "lead_key": LEAD_KEY,
        "phase24h": {
            "global_pooled_delta_auroc": row["pooled_delta_auroc"],
            "global_pooled_delta_by_salt": {
                s: v["delta_auroc"] for s, v in row["per_salt"].items()
            },
            "global_pooled_bootstrap_ci95": row["bootstrap"]["ci95"],
            "within_prompt_mean_delta": sp["mean_delta"],
            "within_prompt_median_delta": sp["median_delta"],
            "within_prompt_fraction_gt_0": sp["fraction_gt_0"],
            "n_qualifying": sp["n_qualifying"],
            "model": "pointwise logistic meta over per-layer pointwise probes (C grid)",
        },
        "phase24i": {
            "global_pooled_delta_auroc": metrics_24i["pooled_descriptive"][
                "mean_delta_auroc_over_salts"],
            "global_pooled_delta_by_salt": {
                s: metrics_24i["pooled_descriptive"][s]["delta_auroc"]
                for s in map(str, OUTER_SALTS) if s in metrics_24i["pooled_descriptive"]
            },
            "within_prompt_mean_delta": p["mean_delta"],
            "within_prompt_median_delta": p["median_delta"],
            "within_prompt_fraction_gt_0": p["fraction_gt_0"],
            "n_qualifying": p["n_qualifying"],
            "model": "within-prompt pairwise logistic probes and meta (frozen C)",
        },
        "shared_qualifying_prompts": len(shared),
        "per_prompt": [
            {"prompt_id": q, "delta_24h": h_by_prompt[q], "delta_24i": i_by_prompt[q]}
            for q in shared
        ],
    }


def phase24f_tree_hash(repo_root: Path) -> str:
    return H.phase24f_tree_hash(repo_root)


__all__ = [
    "SealViolationError",
    "PairIndex",
    "PairwiseLinear",
    "FoldResult",
]
