"""Phase 24H — exploratory internal-signal-breadth representation discovery.

DEVELOPMENT ONLY (TRAIN+VALIDATION = 28 prompts). Honest/deceptive only.
Phase-24F stays the final confirmatory FAIL. No GPU, no model generation.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pre_output_physiology.phase24c_design import prompt_group_folds, sha256_file
from pre_output_physiology.phase24e_discovery import (
    FORBIDDEN_TEST_MARKERS,
    SealViolationError,
    assert_not_test_path,
    bf16_u16_to_float32,
    fit_predict_log_odds,
    make_logit_or_act_clf,
    oof_log_odds,
    safe_auprc,
    safe_auroc,
    sample_weights_for_fit,
)
from pre_output_physiology.phase24f_confirmation import FROZEN_C

STATUS = "phase24h_internal_signal_breadth_complete_awaiting_independent_code_audit"
STARTING_SHA = "f03c8eb006cf8cceb15ffbcb44baea5779599096"
EXPLORATORY_LABEL = "EXPLORATORY_DEVELOPMENT_ONLY_PHASE24H"
ANALYSIS_SEED = 2408
GUARANTEE = (
    "PHASE 24H WAS EXPLORATORY REPRESENTATION DISCOVERY ON DEVELOPMENT "
    "(TRAIN+VALIDATION) DATA ONLY. LOCKED TEST LABELS/ACTIVATIONS AND PHASE-24F "
    "TEST ACTIVATIONS WERE NOT ACCESSED. PHASE-24F REMAINS THE FINAL CONFIRMATORY "
    "FAIL. NO GPU, MODEL GENERATION, SAE, OR CAUSAL WORK WAS PERFORMED."
)

# ---- frozen protocol constants (mirrored in the YAML config) ----
EARLY_TIMES: tuple[int, ...] = (1, 2, 4, 8)
N_LAYERS = 32
N_DEV_PROMPTS = 28
N_OUTER_FOLDS = 7
OUTER_SALTS: tuple[int, ...] = (0, 1, 2)
INNER_FOLDS = 5
VARIANTS: tuple[str, ...] = ("RAW", "DELTA")
PROBE_C = float(FROZEN_C["activation"])  # 0.1
SURFACE_C_TEXT = float(FROZEN_C["text"])
SURFACE_C_LOGITS = float(FROZEN_C["logits"])
SURFACE_C_META = float(FROZEN_C["surface"])
COMBINED_C = float(FROZEN_C["surface_plus_activation"])
META_C_GRID: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
PCA_DIMS: tuple[int, ...] = (4, 8, 16)
PCA_LOGIT_C = 1.0
SVM_C_GRID: tuple[float, ...] = (0.1, 1.0, 10.0)
N_BOOTSTRAP = 2000
N_SHUFFLES = 100
SHUFFLE_SALTS: tuple[int, ...] = (0,)
SHUFFLE_ALPHA = 0.05
MIN_FRACTION_POSITIVE_PROMPTS = 0.60
MIN_PER_CLASS_WITHIN_PROMPT = 2

PROMOTED = "promising_for_independent_replication"
NOT_PROMOTED = "not_promoted"
REVOKED = "revoked_by_shuffle_control"

FAM_A = "A_RAW_SINGLE"
FAM_B = "B_DELTA_SINGLE"
FAM_C = "C_MULTILAYER"
FAM_D = "D_MULTITIME"
FAM_E = "E_LOWDIM"
FAM_F = "F_NONLINEAR"

ACT_DEV_REL = "artifacts/phase24e_discovery/activations_local"
ALLOWED_ACT_SUBDIRS = ("TRAIN", "VALIDATION")
EXTRA_FORBIDDEN_MARKERS = (
    "phase24f_confirmation/activations_local",
    "labels_sealed",
    "new_trajectories_sealed",
)
ALLOWED_SPLITS = ("train", "validation")


# ---- guards ----
def assert_dev_input_path(path: str | Path) -> None:
    """Raise SealViolationError if path touches TEST / sealed / Phase-24F TEST."""
    assert_not_test_path(path)
    s = str(path).replace("\\", "/")
    low = s.lower()
    for marker in (*FORBIDDEN_TEST_MARKERS, *EXTRA_FORBIDDEN_MARKERS):
        if marker.lower() in low:
            raise SealViolationError(f"Phase 24H seal violation: {marker} in {s}")
    if "phase24f_confirmation" in low and "activation" in low:
        raise SealViolationError(f"Phase 24H: Phase-24F activation path forbidden: {s}")


def assert_dev_activation_path(path: str | Path, repo_root: Path) -> None:
    """Activation files must live under activations_local/{TRAIN,VALIDATION}."""
    assert_dev_input_path(path)
    p = Path(path).resolve()
    base = (repo_root / ACT_DEV_REL).resolve()
    try:
        rel = p.relative_to(base)
    except ValueError as exc:
        raise SealViolationError(f"activation path outside DEV dir: {p}") from exc
    if not rel.parts or rel.parts[0] not in ALLOWED_ACT_SUBDIRS:
        raise SealViolationError(f"activation path not TRAIN/VALIDATION: {p}")


def assert_split_allowed(split: str) -> None:
    if split not in ALLOWED_SPLITS:
        raise SealViolationError(f"split {split!r} not allowed in Phase 24H")


def assert_dev_prompts(dev_prompts: list[str], split: dict[str, Any]) -> None:
    """Exactly 28 DEV prompts = TRAIN(20) + VALIDATION(8); disjoint from TEST."""
    train = set(split["train_prompt_ids"])
    val = set(split["validation_prompt_ids"])
    test = set(split["test_prompt_ids"])
    if len(set(dev_prompts)) != N_DEV_PROMPTS or len(dev_prompts) != N_DEV_PROMPTS:
        raise SealViolationError(f"expected exactly {N_DEV_PROMPTS} DEV prompts")
    if set(dev_prompts) != (train | val):
        raise SealViolationError("DEV prompts differ from TRAIN+VALIDATION manifest")
    if set(dev_prompts) & test:
        raise SealViolationError("DEV prompts intersect TEST prompts")


def load_dev_npz_arrays(
    path: str | Path, repo_root: Path, times: tuple[int, ...] = EARLY_TIMES
) -> dict[str, Any]:
    """Guarded DEV NPZ read: h0, h[t], logits[t] only (bf16 -> float32)."""
    assert_dev_activation_path(path, repo_root)
    z = np.load(path)
    post = z["post_block_bf16_u16"]
    logits = z["logits_f16"]
    n_gen = int(len(z["generated_token_ids"]))
    out: dict[str, Any] = {
        "n_generated": n_gen,
        "h0": bf16_u16_to_float32(post[0]),
        "h": {},
        "logits": {},
    }
    for t in times:
        if n_gen > t:
            out["h"][t] = bf16_u16_to_float32(post[t])
            out["logits"][t] = logits[t].astype(np.float32)
    return out


def derive_seed(*parts: Any) -> int:
    h = hashlib.sha256("|".join(["phase24h", *map(str, parts)]).encode()).hexdigest()
    return int(h[:8], 16)


# ---- outer folds ----
def outer_fold_key(salt: int, prompt_id: str) -> str:
    return hashlib.sha256(f"phase24h|outer|{salt}|{prompt_id}".encode()).hexdigest()


def outer_folds(
    prompt_ids: list[str], salt: int, n_folds: int = N_OUTER_FOLDS
) -> list[list[str]]:
    """Sort by SHA256 key, round-robin into n_folds (exactly equal sized)."""
    ids = list(prompt_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate prompt ids")
    if len(ids) % n_folds != 0:
        raise ValueError(f"{len(ids)} prompts not divisible into {n_folds} folds")
    ordered = sorted(ids, key=lambda p: (outer_fold_key(salt, p), p))
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    for i, pid in enumerate(ordered):
        folds[i % n_folds].append(pid)
    return folds


def train_eval_split(
    prompt_ids: list[str], salt: int, fold: int, n_folds: int = N_OUTER_FOLDS
) -> tuple[list[str], list[str]]:
    folds = outer_folds(prompt_ids, salt, n_folds)
    ev = sorted(folds[fold])
    tr = sorted(p for p in prompt_ids if p not in set(ev))
    if set(tr) & set(ev):
        raise AssertionError("prompt leakage across outer fold")
    return tr, ev


# ---- data container ----
@dataclass
class DevData:
    tids: list[str]
    prompt_ids: list[str]
    labels: list[str]  # honest / deceptive only
    n_generated: np.ndarray
    texts: dict[int, list[str]]
    logits: dict[int, np.ndarray]
    h0: np.ndarray  # (n, layers, dim)
    h: dict[int, np.ndarray]  # t -> (n, layers, dim); rows invalid if not survivor

    def __post_init__(self) -> None:
        for lab in self.labels:
            if lab not in ("honest", "deceptive"):
                raise ValueError(f"non-primary label in DevData: {lab}")

    @property
    def n(self) -> int:
        return len(self.tids)

    @property
    def y(self) -> np.ndarray:
        return np.asarray([0 if lab == "honest" else 1 for lab in self.labels], dtype=np.int32)

    @property
    def n_layers(self) -> int:
        return int(self.h0.shape[1])

    def survivors(self, t: int) -> np.ndarray:
        return np.asarray(self.n_generated) > t

    def survival_level(self) -> np.ndarray:
        lvl = np.zeros(self.n, dtype=np.int32)
        for t in EARLY_TIMES:
            lvl[self.survivors(t)] = t
        return lvl


def get_activations(
    data: DevData,
    t: int,
    rows: np.ndarray,
    variant: str,
    act_map: np.ndarray | None = None,
) -> np.ndarray:
    """RAW: h(t,L). DELTA: h(t,L) - h(0,L). act_map redirects activation source rows."""
    src = rows if act_map is None else act_map[rows]
    h_t = data.h[t][src]
    if variant == "RAW":
        return h_t.astype(np.float32, copy=False)
    if variant == "DELTA":
        return (h_t - data.h0[src]).astype(np.float32, copy=False)
    raise ValueError(f"unknown variant {variant}")


# ---- activation shuffle negative control ----
def shuffle_permutation(
    prompt_ids: list[str], strata: np.ndarray, rep: int
) -> np.ndarray:
    """Deterministic permutation within (prompt, stratum): act_map[i] = source row."""
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
    arr = np.asarray([d for d in shuffle_deltas if math.isfinite(d)], dtype=np.float64)
    if len(arr) == 0 or not math.isfinite(real_delta):
        return {"n_shuffles": int(len(arr)), "revoked": True, "reason": "no finite control"}
    n_ge = int(np.sum(arr >= real_delta))
    p = (1 + n_ge) / (1 + len(arr))
    q95 = float(np.quantile(arr, 0.95))
    exceeds = bool(p <= alpha and real_delta > q95)
    return {
        "n_shuffles": int(len(arr)),
        "real_delta": float(real_delta),
        "shuffle_mean": float(arr.mean()),
        "shuffle_sd": float(arr.std(ddof=1)) if len(arr) > 1 else float("nan"),
        "shuffle_q95": q95,
        "shuffle_max": float(arr.max()),
        "n_shuffle_ge_real": n_ge,
        "one_sided_p": float(p),
        "revoked": not exceeds,
        "reason": "ok" if exceeds else "real does not clearly exceed shuffle distribution",
    }


# ---- meta models (train-only scaling / PCA) ----
@dataclass
class FittedMeta:
    kind: str
    hp: float
    scaler: StandardScaler
    pca: PCA | None
    clf: Any

    def score(self, Z: np.ndarray) -> np.ndarray:
        Zs = self.scaler.transform(Z)
        if self.pca is not None:
            Zs = self.pca.transform(Zs)
        return np.asarray(self.clf.decision_function(Zs), dtype=np.float64)


def _valid_rows(Z: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(Z), axis=1)


def fit_meta(
    kind: str,
    hp: float,
    Z: np.ndarray,
    y: np.ndarray,
    labels: list[str],
    prompts: list[str],
    seed: int = ANALYSIS_SEED,
) -> FittedMeta:
    """Fit scaler (+PCA) + classifier on TRAINING rows only (non-finite rows dropped)."""
    ok = _valid_rows(Z)
    Zv = Z[ok]
    yv = y[ok]
    lv = [labels[i] for i in range(len(labels)) if ok[i]]
    pv = [prompts[i] for i in range(len(prompts)) if ok[i]]
    w = sample_weights_for_fit(lv, pv)
    scaler = StandardScaler().fit(Zv)
    Zs = scaler.transform(Zv)
    pca: PCA | None = None
    if kind == "logit":
        clf: Any = make_logit_or_act_clf(float(hp), random_state=seed)
    elif kind == "pca_logit":
        k = int(hp)
        pca = PCA(n_components=k, svd_solver="full", random_state=seed).fit(Zs)
        Zs = pca.transform(Zs)
        clf = make_logit_or_act_clf(PCA_LOGIT_C, random_state=seed)
    elif kind == "rbf_svm":
        clf = SVC(kernel="rbf", gamma="scale", C=float(hp), random_state=seed)
    else:
        raise ValueError(f"unknown meta kind {kind}")
    clf.fit(Zs, yv, sample_weight=w)
    return FittedMeta(kind=kind, hp=float(hp), scaler=scaler, pca=pca, clf=clf)


def meta_oof_scores(
    kind: str,
    hp: float,
    Z: np.ndarray,
    y: np.ndarray,
    labels: list[str],
    prompts: list[str],
    seed: int = ANALYSIS_SEED,
    n_folds: int = INNER_FOLDS,
) -> np.ndarray:
    """Prompt-group OOF scores of a meta model; NaN where unavailable."""
    ok = _valid_rows(Z)
    pid = np.asarray(prompts)
    oof = np.full(len(y), np.nan, dtype=np.float64)
    uniq = sorted(set(pid[ok].tolist()))
    if len(uniq) < 2:
        return oof
    folds = [f for f in prompt_group_folds(uniq, n_folds=min(n_folds, len(uniq))) if f]
    for hold in folds:
        te = np.isin(pid, list(hold)) & ok
        tr = (~np.isin(pid, list(hold))) & ok
        if tr.sum() < 4 or te.sum() < 1 or len(np.unique(y[tr])) < 2:
            continue
        kk = kind
        h = hp
        if kind == "pca_logit":
            h = min(int(hp), Z.shape[1], int(tr.sum()) - 1)
        fm = fit_meta(
            kk,
            h,
            Z[tr],
            y[tr],
            [labels[i] for i in range(len(labels)) if tr[i]],
            [prompts[i] for i in range(len(prompts)) if tr[i]],
            seed,
        )
        oof[te] = fm.score(Z[te])
    return oof


def select_meta_hp(
    kind: str,
    grid: tuple[float, ...],
    Z: np.ndarray,
    y: np.ndarray,
    labels: list[str],
    prompts: list[str],
    seed: int = ANALYSIS_SEED,
) -> tuple[float, dict[str, float]]:
    """Inner prompt-group CV AUROC selection; ties -> earliest grid entry."""
    best = grid[0]
    best_score = -1.0
    per: dict[str, float] = {}
    n_feat = Z.shape[1]
    for hp in grid:
        if kind == "pca_logit" and int(hp) > n_feat:
            continue
        oof = meta_oof_scores(kind, hp, Z, y, labels, prompts, seed)
        ok = np.isfinite(oof)
        s = safe_auroc(y[ok], oof[ok]) if ok.sum() >= 2 else float("nan")
        per[str(hp)] = s
        if math.isfinite(s) and s > best_score:
            best, best_score = hp, s
    return float(best), per


def combine_scores(
    surf_tr: np.ndarray,
    act_tr: np.ndarray,
    y_tr: np.ndarray,
    labels_tr: list[str],
    prompts_tr: list[str],
    surf_ev: np.ndarray,
    act_ev: np.ndarray,
    seed: int = ANALYSIS_SEED,
) -> np.ndarray:
    """SURFACE+ACTIVATION meta: logistic on [surface log-odds, activation score]."""
    fm = fit_meta(
        "logit",
        COMBINED_C,
        np.stack([surf_tr, act_tr], axis=1),
        y_tr,
        labels_tr,
        prompts_tr,
        seed,
    )
    return fm.score(np.stack([surf_ev, act_ev], axis=1))


# ---- SURFACE (TEXT + LOGITS) ----
@dataclass
class SurfaceOut:
    t: int
    train_rows: np.ndarray
    eval_rows: np.ndarray
    oof: np.ndarray
    ev: np.ndarray


def fit_surface(
    data: DevData,
    t: int,
    train_rows: np.ndarray,
    eval_rows: np.ndarray,
    seed: int = ANALYSIS_SEED,
) -> SurfaceOut:
    y_tr = data.y[train_rows]
    labs = [data.labels[i] for i in train_rows]
    pids = [data.prompt_ids[i] for i in train_rows]
    txt_tr = [data.texts[t][i] for i in train_rows]
    txt_ev = [data.texts[t][i] for i in eval_rows]
    lg_tr = data.logits[t][train_rows]
    lg_ev = data.logits[t][eval_rows]
    oof_text = oof_log_odds(
        X=txt_tr, y=y_tr, prompt_ids=pids, labels_str=labs, kind="text",
        C=SURFACE_C_TEXT, seed=seed,
    )
    oof_logit = oof_log_odds(
        X=lg_tr, y=y_tr, prompt_ids=pids, labels_str=labs, kind="matrix",
        C=SURFACE_C_LOGITS, seed=seed,
    )
    ev_text = fit_predict_log_odds(
        X_train=txt_tr, y_train=y_tr, labels_train=labs, prompts_train=pids,
        X_eval=txt_ev, kind="text", C=SURFACE_C_TEXT, seed=seed,
    )
    ev_logit = fit_predict_log_odds(
        X_train=lg_tr, y_train=y_tr, labels_train=labs, prompts_train=pids,
        X_eval=lg_ev, kind="matrix", C=SURFACE_C_LOGITS, seed=seed,
    )
    Z = np.stack([oof_text, oof_logit], axis=1)
    surf_oof = meta_oof_scores("logit", SURFACE_C_META, Z, y_tr, labs, pids, seed)
    fm = fit_meta("logit", SURFACE_C_META, Z, y_tr, labs, pids, seed)
    surf_ev = fm.score(np.stack([ev_text, ev_logit], axis=1))
    return SurfaceOut(
        t=t, train_rows=train_rows, eval_rows=eval_rows, oof=surf_oof, ev=surf_ev
    )


# ---- per-(variant,time) layer probes ----
def layer_scores(
    X_tr: np.ndarray,
    X_ev: np.ndarray,
    y_tr: np.ndarray,
    labels: list[str],
    prompts: list[str],
    seed: int = ANALYSIS_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-layer probes: inner-OOF log-odds (train) and full-train log-odds (eval)."""
    n_layers = X_tr.shape[1]
    Z_oof = np.full((X_tr.shape[0], n_layers), np.nan)
    Z_ev = np.zeros((X_ev.shape[0], n_layers))
    for layer in range(n_layers):
        Z_oof[:, layer] = oof_log_odds(
            X=np.ascontiguousarray(X_tr[:, layer, :]), y=y_tr, prompt_ids=prompts,
            labels_str=labels, kind="matrix", C=PROBE_C, seed=seed,
        )
        Z_ev[:, layer] = fit_predict_log_odds(
            X_train=np.ascontiguousarray(X_tr[:, layer, :]), y_train=y_tr,
            labels_train=labels, prompts_train=prompts,
            X_eval=np.ascontiguousarray(X_ev[:, layer, :]), kind="matrix",
            C=PROBE_C, seed=seed,
        )
    return Z_oof, Z_ev


@dataclass
class LayerState:
    tr: np.ndarray
    ev: np.ndarray
    Z_oof: np.ndarray
    Z_ev: np.ndarray
    c_oof: np.ndarray | None = None
    c_ev: np.ndarray | None = None


@dataclass
class RowPrediction:
    row_key: str
    eval_rows: np.ndarray
    surf: np.ndarray
    act: np.ndarray
    comb: np.ndarray
    surface_time: int
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class FoldOutput:
    rows: dict[str, RowPrediction]
    surface: dict[int, SurfaceOut]
    timings: dict[str, float]


def single_key(variant: str, scope: str) -> str:
    fam = FAM_A if variant == "RAW" else FAM_B
    return f"{fam}|{scope}"


def row_keys_all() -> list[str]:
    keys: list[str] = []
    for v in VARIANTS:
        for t in EARLY_TIMES:
            keys.append(single_key(v, f"t{t}"))
        keys.append(single_key(v, "search"))
    for v in VARIANTS:
        for t in EARLY_TIMES:
            keys.append(f"{FAM_C}|{v}|t{t}")
    for v in VARIANTS:
        keys.append(f"{FAM_D}|{v}")
    for fam in (FAM_E, FAM_F):
        for v in VARIANTS:
            for t in EARLY_TIMES:
                keys.append(f"{fam}|{v}|t{t}")
    return keys


def parse_row_key(key: str) -> dict[str, Any]:
    parts = key.split("|")
    fam = parts[0]
    if fam in (FAM_A, FAM_B):
        variant = "RAW" if fam == FAM_A else "DELTA"
        return {"family": fam, "variant": variant, "scope": parts[1]}
    variant = parts[1]
    scope = parts[2] if len(parts) > 2 else "multitime"
    return {"family": fam, "variant": variant, "scope": scope}


def required_scope(key: str) -> tuple[str, tuple[int, ...]]:
    """(variant, times) needed to compute a row (used by the shuffle control)."""
    info = parse_row_key(key)
    if info["scope"] in ("search", "multitime"):
        return info["variant"], EARLY_TIMES
    return info["variant"], (int(info["scope"][1:]),)


def _auroc_delta(y: np.ndarray, surf: np.ndarray, comb: np.ndarray) -> float:
    ok = np.isfinite(surf) & np.isfinite(comb)
    if ok.sum() < 2:
        return float("nan")
    return safe_auroc(y[ok], comb[ok]) - safe_auroc(y[ok], surf[ok])


def _pos(index_rows: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    lookup = {int(r): i for i, r in enumerate(index_rows)}
    return np.asarray([lookup[int(r)] for r in wanted], dtype=np.int64)


def run_outer_fold(
    data: DevData,
    train_prompts: list[str],
    eval_prompts: list[str],
    *,
    act_map: np.ndarray | None = None,
    variants: tuple[str, ...] = VARIANTS,
    times: tuple[int, ...] = EARLY_TIMES,
    surface_cache: dict[int, SurfaceOut] | None = None,
    seed: int = ANALYSIS_SEED,
) -> FoldOutput:
    """All fitting happens on `train_prompts` only; predictions only for `eval_prompts`."""
    tp, ep = set(train_prompts), set(eval_prompts)
    if tp & ep:
        raise AssertionError("train/eval prompt overlap in outer fold")
    pid = np.asarray(data.prompt_ids)
    y_all = data.y
    cache: dict[int, SurfaceOut] = surface_cache if surface_cache is not None else {}
    timings: dict[str, float] = {}
    rows_out: dict[str, RowPrediction] = {}
    states: dict[tuple[str, int], LayerState] = {}
    single_best: dict[tuple[str, int], dict[str, Any]] = {}

    def _emit(
        key: str, ev_rows: np.ndarray, surf_ev: np.ndarray, act_ev: np.ndarray,
        comb_ev: np.ndarray, t: int, meta: dict[str, Any],
    ) -> None:
        rows_out[key] = RowPrediction(
            row_key=key, eval_rows=ev_rows, surf=surf_ev, act=act_ev, comb=comb_ev,
            surface_time=t, meta=meta,
        )

    for t in times:
        surv = data.survivors(t)
        tr = np.where(surv & np.isin(pid, list(tp)))[0]
        ev = np.where(surv & np.isin(pid, list(ep)))[0]
        if len(tr) < 8 or len(ev) == 0 or len(np.unique(y_all[tr])) < 2:
            continue
        t0 = time.perf_counter()
        if t not in cache:
            cache[t] = fit_surface(data, t, tr, ev, seed)
        sf = cache[t]
        if not (np.array_equal(sf.train_rows, tr) and np.array_equal(sf.eval_rows, ev)):
            raise AssertionError("cached SURFACE rows differ from activation rows")
        timings[f"surface_t{t}"] = time.perf_counter() - t0
        y_tr = y_all[tr]
        labs = [data.labels[i] for i in tr]
        pids = [data.prompt_ids[i] for i in tr]
        for v in variants:
            t1 = time.perf_counter()
            X_tr = get_activations(data, t, tr, v, act_map)
            X_ev = get_activations(data, t, ev, v, act_map)
            Z_oof, Z_ev = layer_scores(X_tr, X_ev, y_tr, labs, pids, seed)
            del X_tr, X_ev
            st = LayerState(tr=tr, ev=ev, Z_oof=Z_oof, Z_ev=Z_ev)
            states[(v, t)] = st
            timings[f"probes_{v}_t{t}"] = time.perf_counter() - t1

            # ---- single-layer (A RAW / B DELTA): layer chosen by inner-OOF DeltaAUROC
            n_l = Z_oof.shape[1]
            deltas = np.full(n_l, np.nan)
            for layer in range(n_l):
                comb_oof = meta_oof_scores(
                    "logit", COMBINED_C, np.stack([sf.oof, Z_oof[:, layer]], axis=1),
                    y_tr, labs, pids, seed,
                )
                deltas[layer] = _auroc_delta(y_tr, sf.oof, comb_oof)
            finite = np.where(np.isfinite(deltas))[0]
            best_l = int(finite[np.argmax(deltas[finite])]) if len(finite) else 0
            best_d = float(deltas[best_l]) if len(finite) else float("-inf")
            comb_ev = combine_scores(
                sf.oof, Z_oof[:, best_l], y_tr, labs, pids, sf.ev, Z_ev[:, best_l], seed
            )
            single_best[(v, t)] = {
                "inner_delta": best_d, "layer": best_l, "t": t,
                "act_ev": Z_ev[:, best_l], "comb_ev": comb_ev, "ev": ev, "surf_ev": sf.ev,
            }
            _emit(
                single_key(v, f"t{t}"), ev, sf.ev, Z_ev[:, best_l], comb_ev, t,
                {"layer": best_l, "inner_oof_delta": best_d},
            )

            # ---- C / E / F meta families on OOF layer scores
            specs = (
                (FAM_C, "logit", META_C_GRID),
                (FAM_E, "pca_logit", PCA_DIMS),
                (FAM_F, "rbf_svm", SVM_C_GRID),
            )
            for fam, kind, grid in specs:
                hp, per = select_meta_hp(
                    kind, tuple(map(float, grid)), Z_oof, y_tr, labs, pids, seed
                )
                if kind == "pca_logit":
                    hp = float(min(int(hp), n_l, len(tr) - 1))
                act_oof = meta_oof_scores(kind, hp, Z_oof, y_tr, labs, pids, seed)
                fm = fit_meta(kind, hp, Z_oof, y_tr, labs, pids, seed)
                act_ev = fm.score(Z_ev)
                comb = combine_scores(sf.oof, act_oof, y_tr, labs, pids, sf.ev, act_ev, seed)
                _emit(
                    f"{fam}|{v}|t{t}", ev, sf.ev, act_ev, comb, t,
                    {"hp": hp, "inner_cv_auroc": per},
                )
                if fam == FAM_C:
                    st.c_oof, st.c_ev = act_oof, act_ev
            timings[f"variant_{v}_t{t}"] = time.perf_counter() - t1

    # ---- search over (t, layer): choose by inner-OOF DeltaAUROC (ties -> earliest t)
    for v in variants:
        cands = [single_best[(v, t)] for t in times if (v, t) in single_best]
        if not cands:
            continue
        best = max(cands, key=lambda c: (c["inner_delta"], -c["t"]))
        _emit(
            single_key(v, "search"), best["ev"], best["surf_ev"], best["act_ev"],
            best["comb_ev"], best["t"],
            {"layer": best["layer"], "selected_time": best["t"],
             "inner_oof_delta": best["inner_delta"]},
        )

    # ---- D multitime: t8 survivors, baseline SURFACE@t8
    if 8 in times and all(t in times for t in EARLY_TIMES):
        for v in variants:
            if not all((v, t) in states and states[(v, t)].c_oof is not None for t in EARLY_TIMES):
                continue
            st8 = states[(v, 8)]
            sf8 = cache[8]
            W_tr = np.stack(
                [states[(v, t)].c_oof[_pos(states[(v, t)].tr, st8.tr)] for t in EARLY_TIMES],
                axis=1,
            )
            W_ev = np.stack(
                [states[(v, t)].c_ev[_pos(states[(v, t)].ev, st8.ev)] for t in EARLY_TIMES],
                axis=1,
            )
            y8 = y_all[st8.tr]
            labs8 = [data.labels[i] for i in st8.tr]
            pid8 = [data.prompt_ids[i] for i in st8.tr]
            hp, per = select_meta_hp("logit", META_C_GRID, W_tr, y8, labs8, pid8, seed)
            act_oof = meta_oof_scores("logit", hp, W_tr, y8, labs8, pid8, seed)
            act_ev = fit_meta("logit", hp, W_tr, y8, labs8, pid8, seed).score(W_ev)
            comb = combine_scores(sf8.oof, act_oof, y8, labs8, pid8, sf8.ev, act_ev, seed)
            _emit(
                f"{FAM_D}|{v}", st8.ev, sf8.ev, act_ev, comb, 8,
                {"hp": hp, "inner_cv_auroc": per, "baseline": "SURFACE@t8"},
            )
    return FoldOutput(rows=rows_out, surface=cache, timings=timings)


# ---- assembling cross-fitted predictions ----
def assemble_predictions(
    data: DevData, fold_outputs: list[tuple[int, int, FoldOutput]]
) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    """row_key -> salt -> arrays. A trajectory appears at most once per (row, salt)."""
    acc: dict[str, dict[int, list[RowPrediction]]] = {}
    for salt, _fold, out in fold_outputs:
        for key, rp in out.rows.items():
            acc.setdefault(key, {}).setdefault(salt, []).append(rp)
    y_all = data.y
    pid = np.asarray(data.prompt_ids)
    result: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for key, by_salt in acc.items():
        result[key] = {}
        for salt, rps in by_salt.items():
            rows = np.concatenate([r.eval_rows for r in rps])
            if len(set(rows.tolist())) != len(rows):
                raise AssertionError(f"duplicate predictions for {key} salt {salt}")
            result[key][salt] = {
                "rows": rows,
                "y": y_all[rows],
                "prompts": pid[rows],
                "surf": np.concatenate([r.surf for r in rps]),
                "act": np.concatenate([r.act for r in rps]),
                "comb": np.concatenate([r.comb for r in rps]),
                "surface_time": np.concatenate(
                    [np.full(len(r.eval_rows), r.surface_time) for r in rps]
                ),
            }
    return result


# ---- metrics: pooled, bootstrap, same-prompt, promotion ----
def _pooled_delta_one(p: dict[str, np.ndarray]) -> tuple[float, float, float, float, float]:
    a_s = safe_auroc(p["y"], p["surf"])
    a_c = safe_auroc(p["y"], p["comb"])
    return a_s, a_c, a_c - a_s, safe_auprc(p["y"], p["surf"]), safe_auprc(p["y"], p["comb"])


def _prompt_index(p: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, list[int]] = {}
    for i, q in enumerate(p["prompts"].tolist()):
        out.setdefault(q, []).append(i)
    return {k: np.asarray(v) for k, v in out.items()}


def prompt_bootstrap_delta(
    per_salt: dict[int, dict[str, np.ndarray]], n_boot: int, seed: int
) -> dict[str, Any]:
    """Prompt-cluster bootstrap of mean-over-salts pooled DeltaAUROC."""
    prompts = sorted({q for p in per_salt.values() for q in p["prompts"].tolist()})
    idx_by_salt = {s: _prompt_index(p) for s, p in per_salt.items()}
    rng = np.random.default_rng(seed)
    stats: list[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(prompts, size=len(prompts), replace=True)
        ds = []
        for s, p in per_salt.items():
            parts = [idx_by_salt[s][q] for q in sampled if q in idx_by_salt[s]]
            if not parts:
                continue
            idx = np.concatenate(parts)
            a = safe_auroc(p["y"][idx], p["surf"][idx])
            b = safe_auroc(p["y"][idx], p["comb"][idx])
            if math.isfinite(a) and math.isfinite(b):
                ds.append(b - a)
        if ds:
            stats.append(float(np.mean(ds)))
    arr = np.asarray(stats)
    if len(arr) == 0:
        return {"n_reps": n_boot, "n_finite": 0, "ci95": [float("nan")] * 2, "mean": float("nan")}
    return {
        "n_reps": n_boot,
        "n_finite": int(len(arr)),
        "mean": float(arr.mean()),
        "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
    }


def within_prompt_analysis(
    per_salt: dict[int, dict[str, np.ndarray]], n_boot: int, seed: int
) -> dict[str, Any]:
    """Same-prompt analysis on stored cross-fitted predictions ONLY (no model fitting)."""
    per_prompt: dict[str, list[tuple[float, float]]] = {}
    for _s, p in sorted(per_salt.items()):
        for q, idx in _prompt_index(p).items():
            y = p["y"][idx]
            n_h, n_d = int((y == 0).sum()), int((y == 1).sum())
            if n_h < MIN_PER_CLASS_WITHIN_PROMPT or n_d < MIN_PER_CLASS_WITHIN_PROMPT:
                continue
            a = safe_auroc(y, p["surf"][idx])
            b = safe_auroc(y, p["comb"][idx])
            if math.isfinite(a) and math.isfinite(b):
                per_prompt.setdefault(q, []).append((a, b))
    prompts = sorted(per_prompt)
    rows = []
    for q in prompts:
        surf_a = float(np.mean([x[0] for x in per_prompt[q]]))
        comb_a = float(np.mean([x[1] for x in per_prompt[q]]))
        rows.append(
            {"prompt_id": q, "surface_auroc": surf_a, "combined_auroc": comb_a,
             "delta_auroc": comb_a - surf_a, "n_salts": len(per_prompt[q])}
        )
    d = np.asarray([r["delta_auroc"] for r in rows], dtype=np.float64)
    if len(d) == 0:
        return {
            "n_qualifying": 0, "median_delta": float("nan"), "mean_delta": float("nan"),
            "fraction_gt_0": float("nan"), "rows": [],
            "bootstrap": {"median_ci95": [float("nan")] * 2, "mean_ci95": [float("nan")] * 2},
        }
    rng = np.random.default_rng(seed)
    meds, means = [], []
    for _ in range(n_boot):
        b = d[rng.integers(0, len(d), size=len(d))]
        meds.append(np.median(b))
        means.append(np.mean(b))
    return {
        "n_qualifying": int(len(d)),
        "median_delta": float(np.median(d)),
        "mean_delta": float(np.mean(d)),
        "fraction_gt_0": float(np.mean(d > 0)),
        "mean_surface_auroc": float(np.mean([r["surface_auroc"] for r in rows])),
        "mean_combined_auroc": float(np.mean([r["combined_auroc"] for r in rows])),
        "rows": rows,
        "bootstrap": {
            "n_reps": n_boot,
            "median_ci95": [float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))],
            "mean_ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        },
    }


def promotion_decision(
    *,
    pooled_delta: float,
    ci_low: float,
    per_salt_delta: list[float],
    within_median: float,
    fraction_positive: float,
    n_salts_expected: int = len(OUTER_SALTS),
) -> dict[str, Any]:
    """Frozen promotion rule: ALL five criteria must hold."""
    def _gt0(x: float) -> bool:
        return bool(math.isfinite(x) and x > 0)

    crit = {
        "pooled_delta_gt_0": _gt0(pooled_delta),
        "bootstrap_lower_gt_0": _gt0(ci_low),
        "delta_gt_0_in_all_salts": bool(
            len(per_salt_delta) == n_salts_expected and all(_gt0(x) for x in per_salt_delta)
        ),
        "within_prompt_median_gt_0": _gt0(within_median),
        "fraction_positive_prompts_ge_0.60": bool(
            math.isfinite(fraction_positive)
            and fraction_positive >= MIN_FRACTION_POSITIVE_PROMPTS
        ),
    }
    promoted = all(crit.values())
    return {"criteria": crit, "promoted": promoted,
            "label": PROMOTED if promoted else NOT_PROMOTED}


def row_metrics(
    per_salt: dict[int, dict[str, np.ndarray]], n_boot: int, seed: int
) -> dict[str, Any]:
    """Full metric block for one row (activation-alone is descriptive only)."""
    salts = sorted(per_salt)
    per: dict[str, Any] = {}
    deltas, delta_pr = [], []
    for s in salts:
        a_s, a_c, d, ap_s, ap_c = _pooled_delta_one(per_salt[s])
        a_act = safe_auroc(per_salt[s]["y"], per_salt[s]["act"])
        per[str(s)] = {
            "n": int(len(per_salt[s]["y"])),
            "n_prompts": int(len(set(per_salt[s]["prompts"].tolist()))),
            "surface_auroc": a_s, "combined_auroc": a_c, "delta_auroc": d,
            "surface_auprc": ap_s, "combined_auprc": ap_c, "delta_auprc": ap_c - ap_s,
            "activation_auroc_descriptive_only": a_act,
        }
        deltas.append(d)
        delta_pr.append(ap_c - ap_s)
    pooled = float(np.mean(deltas)) if deltas else float("nan")
    boot = prompt_bootstrap_delta(per_salt, n_boot, seed)
    wp = within_prompt_analysis(per_salt, n_boot, seed + 1)
    promo = promotion_decision(
        pooled_delta=pooled, ci_low=boot["ci95"][0], per_salt_delta=deltas,
        within_median=wp["median_delta"], fraction_positive=wp["fraction_gt_0"],
    )
    return {
        "per_salt": per,
        "pooled_delta_auroc": pooled,
        "pooled_surface_auroc": float(np.mean([per[str(s)]["surface_auroc"] for s in salts])),
        "pooled_combined_auroc": float(np.mean([per[str(s)]["combined_auroc"] for s in salts])),
        "pooled_delta_auprc": float(np.mean(delta_pr)),
        "bootstrap": boot,
        "same_prompt": {k: v for k, v in wp.items() if k != "rows"},
        "same_prompt_rows": wp["rows"],
        "promotion": promo,
    }


def salt_delta(per_salt: dict[int, dict[str, np.ndarray]], salt: int) -> float:
    return _pooled_delta_one(per_salt[salt])[2]


def phase24f_tree_hash(repo_root: Path) -> str:
    """Hash all non-activation Phase-24F files (activation dir never traversed)."""
    base = repo_root / "artifacts/phase24f_confirmation"
    h = hashlib.sha256()
    for p in sorted(base.iterdir()):
        if p.is_file():
            h.update(p.name.encode())
            h.update(sha256_file(str(p)).encode())
    return h.hexdigest()
