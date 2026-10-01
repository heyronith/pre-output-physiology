#!/usr/bin/env python3
"""Phase 24G: exploratory post-confirmation failure diagnostics.

Phase-24F primary FAIL is immutable. New TEST analyses are EXPLORATORY only.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase24c_design import (  # noqa: E402
    TEMPORAL_LANDMARKS,
    prompt_group_folds,
    select_candidate_region,
    sha256_file,
)
from pre_output_physiology.phase24e_discovery import (  # noqa: E402
    _as_binary_labels,
    bf16_u16_to_float32,
    delta_h_vector,
    fit_predict_log_odds,
    make_logit_or_act_clf,
    oof_log_odds,
    prefix_token_ids,
    prompt_cluster_bootstrap_ci,
    safe_auprc,
    safe_auroc,
    sample_weights_for_fit,
    survives_landmark,
    svg_heatmap,
)
from pre_output_physiology.phase24f_confirmation import (  # noqa: E402
    CANDIDATE_LAYER,
    CANDIDATE_TIME,
    FROZEN_C,
)
from pre_output_physiology.phase24g_diagnostics import (  # noqa: E402
    ANALYSIS_SEED,
    EXPLORATORY_LABEL,
    GUARANTEE,
    NESTED_REPS,
    NESTED_REPS_TARGET_NOTE,
    STARTING_SHA,
    STATUS,
    assert_phase24f_immutable,
    candidate_in_neighborhood,
    classify_diagnostic_category,
    corr_224,
    fit_linear_probe_coef,
    high_region_overlap,
    layerwise_corr,
    load_val_metric_maps,
    matrix_from_grid,
    power_projection,
    probe_stability_from_coefs,
    prompt_level_delta_contributions,
    tag_exploratory,
    timewise_corr,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

OUT = REPO_ROOT / "artifacts/phase24g_diagnostics"
ACT_DEV = REPO_ROOT / "artifacts/phase24e_discovery/activations_local"
ACT_TEST = (
    REPO_ROOT / "artifacts/phase24f_confirmation/activations_local" / "LOCKED_TEST"
)
SPLIT_PATH = REPO_ROOT / "artifacts/phase24c_design/split_manifest.json"
LABELS_TRAIN = REPO_ROOT / "artifacts/phase24d_collection/by_split/TRAIN/labels.json"
LABELS_VAL = REPO_ROOT / "artifacts/phase24d_collection/by_split/VALIDATION/labels.json"
LABELS_TEST = (
    REPO_ROOT / "artifacts/phase24d_collection/by_split/LOCKED_TEST/labels_SEALED.json"
)
META24D = REPO_ROOT / "artifacts/phase24d_collection/capture_meta.json"
TRAJ24B = REPO_ROOT / "artifacts/phase24b_live/trajectories.jsonl"
REUSE = REPO_ROOT / "artifacts/phase24d_collection/by_split/TRAIN/phase24b_reuse_index.json"
SELECTED_C = REPO_ROOT / "artifacts/phase24e_discovery/selected_C.json"
CFG = REPO_ROOT / "configs/experiments/phase24g_failure_diagnostics.yaml"
REPORT = REPO_ROOT / "reports/phase24g_confirmatory_failure_diagnostics.md"
DECISION_LOG = REPO_ROOT / "docs/decision_log.md"

BOOT_QUICK = 500
METRIC_KEYS = (
    "surface_auroc",
    "activation_auroc",
    "combined_auroc",
    "delta_auroc",
    "text_auroc",
    "logits_auroc",
)


def _load_tokenizer():
    from transformers import AutoTokenizer

    from pre_output_physiology.phase24b_live import MODEL_ID, MODEL_REVISION

    return AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, use_fast=True
    )


def _decode_prefix(tok, prompt_text: str, gen_ids: list[int], t: int) -> str:
    pref = prefix_token_ids(gen_ids, t)
    if not pref:
        return prompt_text
    return prompt_text + tok.decode(pref, skip_special_tokens=True)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    # mmap keeps post_block on disk until sliced; handle must stay alive.
    z = np.load(path, mmap_mode="r")
    return {
        "post_block_u16": z["post_block_bf16_u16"],
        "logits_f16": z["logits_f16"],
        "generated_token_ids": z["generated_token_ids"],
        "_npz_handle": z,
    }


def _label_map(path: Path, allowed: set[str]) -> dict[str, dict]:
    blob = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for r in blob["rows"]:
        if r["prompt_id"] not in allowed:
            continue
        if r.get("open_valid") and r.get("open_class") in (
            "honest",
            "deceptive",
            "ambiguous",
        ):
            out[r["trajectory_id"]] = {
                "label": r["open_class"],
                "prompt_id": r["prompt_id"],
                "primary": r["open_class"] in ("honest", "deceptive"),
            }
    return out


def _build_all_records(tok) -> dict[str, list[dict]]:
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    train_set = set(split["train_prompt_ids"])
    val_set = set(split["validation_prompt_ids"])
    test_set = set(split["test_prompt_ids"])

    labels = {}
    labels.update(_label_map(LABELS_TRAIN, train_set))
    labels.update(_label_map(LABELS_VAL, val_set))
    labels.update(_label_map(LABELS_TEST, test_set))

    meta = json.loads(META24D.read_text(encoding="utf-8"))
    meta_by = {m["trajectory_id"]: m for m in meta["meta_rows"] if m.get("completed")}
    reuse_ids = set(json.loads(REUSE.read_text(encoding="utf-8"))["trajectory_ids"])
    traj24b = {
        json.loads(x)["trajectory_id"]: json.loads(x)
        for x in TRAJ24B.read_text(encoding="utf-8").splitlines()
        if x.strip()
    }

    by_split: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for tid, lab in labels.items():
        pid = lab["prompt_id"]
        if pid in train_set:
            sp, folder = "train", "TRAIN"
            npz = ACT_DEV / folder / f"{tid}.npz"
        elif pid in val_set:
            sp, folder = "validation", "VALIDATION"
            npz = ACT_DEV / folder / f"{tid}.npz"
        else:
            sp, folder = "test", "LOCKED_TEST"
            npz = ACT_TEST / f"{tid}.npz"
        if not npz.exists():
            raise FileNotFoundError(npz)
        if tid in meta_by:
            m = meta_by[tid]
            gen_ids = list(m["generated_token_ids"])
            prompt_text = tok.decode(m["prompt_token_ids"], skip_special_tokens=False)
        elif tid in reuse_ids:
            m = traj24b[tid]
            gen_ids = list(m["generated_token_ids"])
            prompt_text = tok.decode(m["prompt_token_ids"], skip_special_tokens=False)
        else:
            raise KeyError(tid)
        by_split[sp].append(
            {
                "trajectory_id": tid,
                "prompt_id": pid,
                "split": sp,
                "label": lab["label"],
                "primary": lab["primary"],
                "generated_token_ids": gen_ids,
                "prompt_text": prompt_text,
                "n_generated": len(gen_ids),
                "npz_path": str(npz),
            }
        )
    return by_split


def _selected_C(t: int, layer: int | None, kind: str) -> float:
    """Reuse Phase-24E selected Cs; fall back to frozen confirmatory Cs."""
    sel = json.loads(SELECTED_C.read_text(encoding="utf-8"))
    if kind in ("TEXT", "LOGITS", "SURFACE"):
        key = f"t{t}_{kind}"
        if key in sel:
            return float(sel[key]["selected_C"])
        return float(
            FROZEN_C[{"TEXT": "text", "LOGITS": "logits", "SURFACE": "surface"}[kind]]
        )
    assert layer is not None
    key = f"t{t}_L{layer}_{kind}"
    if key in sel:
        return float(sel[key]["selected_C"])
    return float(
        FROZEN_C[
            {
                "ACTIVATION": "activation",
                "SURFACE_PLUS_ACTIVATION": "surface_plus_activation",
            }[kind]
        ]
    )


def _oof_meta(
    Z_oof: np.ndarray,
    y: np.ndarray,
    labels: list[str],
    prompts: list[str],
    C: float,
) -> np.ndarray:
    valid = np.all(np.isfinite(Z_oof), axis=1)
    oof = np.full(len(y), np.nan)
    folds = [f for f in prompt_group_folds(sorted(set(prompts)), n_folds=5) if f]
    pid_arr = np.asarray(prompts)
    for hold in folds:
        hold_set = set(hold)
        te = np.array([p in hold_set for p in pid_arr])
        trm = (~te) & valid
        tem = te & valid
        if trm.sum() < 2 or tem.sum() < 1 or len(np.unique(y[trm])) < 2:
            continue
        sc = StandardScaler()
        Z1 = sc.fit_transform(Z_oof[trm])
        Z2 = sc.transform(Z_oof[tem])
        w = sample_weights_for_fit(
            [labels[i] for i in range(len(labels)) if trm[i]],
            [prompts[i] for i in range(len(prompts)) if trm[i]],
        )
        clf = make_logit_or_act_clf(C, random_state=ANALYSIS_SEED)
        clf.fit(Z1, y[trm], sample_weight=w)
        pr = np.clip(clf.predict_proba(Z2)[:, 1], 1e-6, 1 - 1e-6)
        oof[tem] = np.log(pr / (1 - pr))
    return oof


def _fit_meta_proba(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    labels_train: list[str],
    prompts_train: list[str],
    Z_eval: np.ndarray,
    C: float,
) -> np.ndarray:
    valid = np.all(np.isfinite(Z_train), axis=1)
    sc = StandardScaler()
    Ztr = sc.fit_transform(Z_train[valid])
    w = sample_weights_for_fit(
        [labels_train[i] for i in range(len(labels_train)) if valid[i]],
        [prompts_train[i] for i in range(len(prompts_train)) if valid[i]],
    )
    clf = make_logit_or_act_clf(C, random_state=ANALYSIS_SEED)
    clf.fit(Ztr, y_train[valid], sample_weight=w)
    return np.clip(clf.predict_proba(sc.transform(Z_eval))[:, 1], 1e-6, 1 - 1e-6)


def _log_odds_to_proba(lo: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-lo))


def _primary(recs: list[dict]) -> list[dict]:
    return [r for r in recs if r["primary"]]


def _feat_bundle(
    recs: list[dict],
    cache: dict[str, dict],
    tok,
    t: int,
    layer: int | None,
) -> dict[str, Any]:
    """Build TEXT/LOGITS/(optional ACT) features for survivors at t."""
    texts, logits, acts, labels, prompts, tids = [], [], [], [], [], []
    for r in _primary(recs):
        if not survives_landmark(r["n_generated"], t):
            continue
        arr = cache[r["trajectory_id"]]
        texts.append(_decode_prefix(tok, r["prompt_text"], r["generated_token_ids"], t))
        logits.append(arr["logits_f16"][t].astype(np.float32))
        if layer is not None:
            # Convert only the two time slices needed for Δh (numerically identical).
            post_u16 = arr["post_block_u16"]
            h_t = bf16_u16_to_float32(np.asarray(post_u16[t, layer]))
            h_0 = bf16_u16_to_float32(np.asarray(post_u16[0, layer]))
            acts.append(delta_h_vector(h_t, h_0))
        labels.append(r["label"])
        prompts.append(r["prompt_id"])
        tids.append(r["trajectory_id"])
    if not labels:
        return {"empty": True}
    y = _as_binary_labels(labels)
    out: dict[str, Any] = {
        "empty": False,
        "texts": texts,
        "logits": np.stack(logits),
        "labels": labels,
        "prompts": prompts,
        "y": y,
        "tids": tids,
    }
    if layer is not None:
        out["acts"] = np.stack(acts)
    return out


def _delta_h_all_layers(
    recs: list[dict], cache: dict[str, dict], t: int
) -> tuple[np.ndarray | None, list[dict]]:
    """Stack Δh for all 32 layers: shape (n, 32, dim)."""
    mats: list[np.ndarray] = []
    order: list[dict] = []
    for r in _primary(recs):
        if not survives_landmark(r["n_generated"], t):
            continue
        # Convert only t=0 and landmark slices (not the full 17-step tensor).
        post_u16 = cache[r["trajectory_id"]]["post_block_u16"]
        h_t = bf16_u16_to_float32(np.asarray(post_u16[t]))
        h_0 = bf16_u16_to_float32(np.asarray(post_u16[0]))
        mats.append(delta_h_vector(h_t, h_0))
        order.append(r)
    if not mats:
        return None, []
    return np.stack(mats), order


def _surface_path_shared(
    tr: dict[str, Any],
    ev: dict[str, Any],
    t: int,
    *,
    logits_only: bool = False,
) -> dict[str, Any]:
    """TEXT/LOGITS/SURFACE scores shared across layers at time t.

    logits_only=True: skip TF-IDF TEXT (LOPO/nested budget proxy).
    """
    C_logit = _selected_C(t, None, "LOGITS")
    C_surf = _selected_C(t, None, "SURFACE")

    oof_logit = oof_log_odds(
        X=tr["logits"],
        y=tr["y"],
        prompt_ids=tr["prompts"],
        labels_str=tr["labels"],
        kind="matrix",
        C=C_logit,
        seed=ANALYSIS_SEED,
    )
    lo_logit_ev = fit_predict_log_odds(
        X_train=tr["logits"],
        y_train=tr["y"],
        labels_train=tr["labels"],
        prompts_train=tr["prompts"],
        X_eval=ev["logits"],
        kind="matrix",
        C=C_logit,
        seed=ANALYSIS_SEED,
    )
    if logits_only:
        # proxy surface = logits-only scores (no TEXT / no meta stack)
        surf_p = _log_odds_to_proba(lo_logit_ev)
        oof_surf = oof_logit
        lo_surf_ev = lo_logit_ev
        text_auroc = float("nan")
        text_p = np.full(len(ev["y"]), np.nan)
        logit_p = surf_p
    else:
        C_text = _selected_C(t, None, "TEXT")
        oof_text = oof_log_odds(
            X=tr["texts"],
            y=tr["y"],
            prompt_ids=tr["prompts"],
            labels_str=tr["labels"],
            kind="text",
            C=C_text,
            seed=ANALYSIS_SEED,
        )
        lo_text_ev = fit_predict_log_odds(
            X_train=tr["texts"],
            y_train=tr["y"],
            labels_train=tr["labels"],
            prompts_train=tr["prompts"],
            X_eval=ev["texts"],
            kind="text",
            C=C_text,
            seed=ANALYSIS_SEED,
        )
        Z_oof = np.stack([oof_text, oof_logit], axis=1)
        surf_p = _fit_meta_proba(
            Z_oof,
            tr["y"],
            tr["labels"],
            tr["prompts"],
            np.stack([lo_text_ev, lo_logit_ev], axis=1),
            C_surf,
        )
        oof_surf = _oof_meta(Z_oof, tr["y"], tr["labels"], tr["prompts"], C_surf)
        lo_surf_ev = np.log(
            np.clip(surf_p, 1e-6, 1 - 1e-6)
            / (1.0 - np.clip(surf_p, 1e-6, 1 - 1e-6))
        )
        text_p = _log_odds_to_proba(lo_text_ev)
        logit_p = _log_odds_to_proba(lo_logit_ev)
        text_auroc = safe_auroc(ev["y"], text_p)

    if logits_only:
        logit_p = _log_odds_to_proba(lo_logit_ev)

    return {
        "y_ev": ev["y"],
        "surf_auroc": safe_auroc(ev["y"], surf_p),
        "text_auroc": text_auroc,
        "logits_auroc": safe_auroc(ev["y"], logit_p),
        "oof_surf": oof_surf,
        "lo_surf_ev": lo_surf_ev,
        "surf_p": surf_p,
        "text_p": text_p,
        "logit_p": logit_p,
        "ev_labels": ev["labels"],
        "ev_prompts": ev["prompts"],
    }


def discovery_maps_fast(
    train_recs: list[dict],
    eval_recs: list[dict],
    cache: dict[str, dict],
    tok,
    *,
    proxy_delta: bool = False,
) -> dict[str, dict[int, dict[int, float]]]:
    """7×32 AUROC grids: fit train, score eval; SURFACE built once per t.

    If proxy_delta=True (LOPO/nested budget mode): skip activation OOF + meta
    stacking; use Δ ≈ ACTIVATION_AUROC − SURFACE_AUROC. Documented as
    selection-stability proxy only — not used for the exploratory TEST grid.
    """
    maps: dict[str, dict[int, dict[int, float]]] = {
        k: {t: {} for t in TEMPORAL_LANDMARKS} for k in METRIC_KEYS
    }
    for t in TEMPORAL_LANDMARKS:
        tr = _feat_bundle(train_recs, cache, tok, t, layer=0)
        ev = _feat_bundle(eval_recs, cache, tok, t, layer=0)
        if tr["empty"] or ev["empty"]:
            for L in range(32):
                for k in METRIC_KEYS:
                    maps[k][t][L] = float("nan")
            print(f"  maps_fast t={t} skipped (no survivors)", flush=True)
            continue

        shared = _surface_path_shared(tr, ev, t, logits_only=proxy_delta)
        tr_acts, tr_order = _delta_h_all_layers(train_recs, cache, t)
        ev_acts, ev_order = _delta_h_all_layers(eval_recs, cache, t)
        if tr_acts is None or ev_acts is None:
            for L in range(32):
                for k in METRIC_KEYS:
                    maps[k][t][L] = float("nan")
            continue

        tr_lab = [r["label"] for r in tr_order]
        tr_pid = [r["prompt_id"] for r in tr_order]
        y_tr = _as_binary_labels(tr_lab)
        y_ev = shared["y_ev"]
        surf_au = shared["surf_auroc"]
        text_au = shared["text_auroc"]
        logit_au = shared["logits_auroc"]

        for L in range(32):
            C_act = _selected_C(t, L, "ACTIVATION")
            Xtr = tr_acts[:, L, :]
            Xev = ev_acts[:, L, :]
            lo_act_ev = fit_predict_log_odds(
                X_train=Xtr,
                y_train=y_tr,
                labels_train=tr_lab,
                prompts_train=tr_pid,
                X_eval=Xev,
                kind="matrix",
                C=C_act,
                seed=ANALYSIS_SEED,
            )
            act_p = _log_odds_to_proba(lo_act_ev)
            act_au = safe_auroc(y_ev, act_p)
            if proxy_delta:
                comb_au = act_au
                d = act_au - surf_au
            else:
                C_comb = _selected_C(t, L, "SURFACE_PLUS_ACTIVATION")
                oof_act = oof_log_odds(
                    X=Xtr,
                    y=y_tr,
                    prompt_ids=tr_pid,
                    labels_str=tr_lab,
                    kind="matrix",
                    C=C_act,
                    seed=ANALYSIS_SEED,
                )
                Zc = np.stack([shared["oof_surf"], oof_act], axis=1)
                comb_p = _fit_meta_proba(
                    Zc,
                    y_tr,
                    tr_lab,
                    tr_pid,
                    np.stack([shared["lo_surf_ev"], lo_act_ev], axis=1),
                    C_comb,
                )
                comb_au = safe_auroc(y_ev, comb_p)
                d = comb_au - surf_au
            maps["surface_auroc"][t][L] = float(surf_au)
            maps["activation_auroc"][t][L] = float(act_au)
            maps["combined_auroc"][t][L] = float(comb_au)
            maps["delta_auroc"][t][L] = float(d) if np.isfinite(d) else float("nan")
            maps["text_auroc"][t][L] = float(text_au)
            maps["logits_auroc"][t][L] = float(logit_au)

        mode = "proxy" if proxy_delta else "full"
        print(
            f"  maps_fast[{mode}] t={t} surf={surf_au:.3f} text={text_au:.3f}",
            flush=True,
        )
    return maps


def discovery_delta_grid_fast(
    train_recs: list[dict],
    eval_recs: list[dict],
    cache: dict[str, dict],
    tok,
    *,
    proxy_delta: bool = True,
) -> dict[int, dict[int, float]]:
    """ΔAUROC grid with -inf where discovery coordinate is invalid.

    Default proxy_delta=True for LOPO/nested computational budget.
    """
    maps = discovery_maps_fast(
        train_recs, eval_recs, cache, tok, proxy_delta=proxy_delta
    )
    grid: dict[int, dict[int, float]] = {t: {} for t in TEMPORAL_LANDMARKS}
    for t in TEMPORAL_LANDMARKS:
        for L in range(32):
            d = maps["delta_auroc"][t].get(L, float("nan"))
            grid[t][L] = float(d) if np.isfinite(d) else float("-inf")
    return grid


def eval_coordinate(
    train_recs: list[dict],
    eval_recs: list[dict],
    cache: dict[str, dict],
    tok,
    *,
    t: int,
    layer: int,
    use_oof_for_train_eval: bool = False,
) -> dict[str, Any]:
    """Single (t, layer) metrics: TRAIN-fit or OOF when eval is train."""
    tr = _feat_bundle(train_recs, cache, tok, t, layer)
    ev = _feat_bundle(eval_recs, cache, tok, t, layer)
    if tr["empty"] or ev["empty"]:
        return {"skipped": True, "time": t, "layer": layer}

    if use_oof_for_train_eval:
        C_text = _selected_C(t, None, "TEXT")
        C_logit = _selected_C(t, None, "LOGITS")
        C_surf = _selected_C(t, None, "SURFACE")
        C_act = _selected_C(t, layer, "ACTIVATION")
        C_comb = _selected_C(t, layer, "SURFACE_PLUS_ACTIVATION")

        oof_text = oof_log_odds(
            X=tr["texts"],
            y=tr["y"],
            prompt_ids=tr["prompts"],
            labels_str=tr["labels"],
            kind="text",
            C=C_text,
            seed=ANALYSIS_SEED,
        )
        oof_logit = oof_log_odds(
            X=tr["logits"],
            y=tr["y"],
            prompt_ids=tr["prompts"],
            labels_str=tr["labels"],
            kind="matrix",
            C=C_logit,
            seed=ANALYSIS_SEED,
        )
        oof_surf = _oof_meta(
            np.stack([oof_text, oof_logit], axis=1),
            tr["y"],
            tr["labels"],
            tr["prompts"],
            C_surf,
        )
        oof_act = oof_log_odds(
            X=tr["acts"],
            y=tr["y"],
            prompt_ids=tr["prompts"],
            labels_str=tr["labels"],
            kind="matrix",
            C=C_act,
            seed=ANALYSIS_SEED,
        )
        surf_p = _log_odds_to_proba(oof_surf)
        act_p = _log_odds_to_proba(oof_act)
        comb_p = _log_odds_to_proba(
            _oof_meta(
                np.stack([oof_surf, oof_act], axis=1),
                tr["y"],
                tr["labels"],
                tr["prompts"],
                C_comb,
            )
        )
        y = tr["y"]
        prompts = tr["prompts"]
        text_p = _log_odds_to_proba(oof_text)
        logit_p = _log_odds_to_proba(oof_logit)
        labels = tr["labels"]
        acts = tr["acts"]
    else:
        shared = _surface_path_shared(tr, ev, t)
        C_act = _selected_C(t, layer, "ACTIVATION")
        C_comb = _selected_C(t, layer, "SURFACE_PLUS_ACTIVATION")
        oof_act = oof_log_odds(
            X=tr["acts"],
            y=tr["y"],
            prompt_ids=tr["prompts"],
            labels_str=tr["labels"],
            kind="matrix",
            C=C_act,
            seed=ANALYSIS_SEED,
        )
        lo_act_ev = fit_predict_log_odds(
            X_train=tr["acts"],
            y_train=tr["y"],
            labels_train=tr["labels"],
            prompts_train=tr["prompts"],
            X_eval=ev["acts"],
            kind="matrix",
            C=C_act,
            seed=ANALYSIS_SEED,
        )
        surf_p = shared["surf_p"]
        act_p = _log_odds_to_proba(lo_act_ev)
        comb_p = _fit_meta_proba(
            np.stack([shared["oof_surf"], oof_act], axis=1),
            tr["y"],
            tr["labels"],
            tr["prompts"],
            np.stack([shared["lo_surf_ev"], lo_act_ev], axis=1),
            C_comb,
        )
        y = ev["y"]
        prompts = ev["prompts"]
        text_p = shared["text_p"]
        logit_p = shared["logit_p"]
        labels = ev["labels"]
        acts = ev["acts"]

    surf_auroc = safe_auroc(y, surf_p)
    act_auroc = safe_auroc(y, act_p)
    comb_auroc = safe_auroc(y, comb_p)
    delta = comb_auroc - surf_auroc
    boot = prompt_cluster_bootstrap_ci(
        y=y,
        scores_a=surf_p,
        scores_b=comb_p,
        prompt_ids=prompts,
        n_reps=BOOT_QUICK,
        seed=ANALYSIS_SEED,
    )
    return {
        "skipped": False,
        "time": t,
        "layer": layer,
        "surface_auroc": surf_auroc,
        "activation_auroc": act_auroc,
        "combined_auroc": comb_auroc,
        "delta_auroc": delta,
        "text_auroc": safe_auroc(y, text_p),
        "logits_auroc": safe_auroc(y, logit_p),
        "text_auprc": safe_auprc(y, text_p),
        "logits_auprc": safe_auprc(y, logit_p),
        "surface_auprc": safe_auprc(y, surf_p),
        "activation_auprc": safe_auprc(y, act_p),
        "combined_auprc": safe_auprc(y, comb_p),
        "delta_auroc_bootstrap": boot,
        "n": int(len(y)),
        "n_honest": int((y == 0).sum()),
        "n_deceptive": int((y == 1).sum()),
        "n_prompts": len(set(prompts)),
        "prompt_contributions": prompt_level_delta_contributions(
            y=y, surface=surf_p, combined=comb_p, prompt_ids=prompts
        ),
        "scores": {
            "y": y.tolist(),
            "prompts": list(prompts),
            "labels": list(labels),
            "surface": surf_p.tolist(),
            "activation": act_p.tolist(),
            "combined": comb_p.tolist(),
            "text": text_p.tolist(),
            "logits": logit_p.tolist(),
        },
        "acts_for_probe": acts,
    }


def same_prompt_table(scores_blob: dict[str, Any], min_h: int = 1, min_d: int = 1):
    y = np.asarray(scores_blob["y"])
    prompts = scores_blob["prompts"]
    out = []
    for name in ("surface", "activation", "combined"):
        s = np.asarray(scores_blob[name])
        pid = np.asarray(prompts)
        for p in sorted(set(pid.tolist())):
            mask = pid == p
            yh, sh = y[mask], s[mask]
            n_h = int((yh == 0).sum())
            n_d = int((yh == 1).sum())
            if n_h < min_h or n_d < min_d:
                continue
            out.append(
                {
                    "model": name,
                    "prompt_id": p,
                    "n_honest": n_h,
                    "n_deceptive": n_d,
                    "diff_D_minus_H": float(sh[yh == 1].mean() - sh[yh == 0].mean()),
                }
            )
    return out


def prompt_heterogeneity(recs_by_split: dict[str, list[dict]]) -> dict[str, Any]:
    rows = []
    for sp, recs in recs_by_split.items():
        by_p: dict[str, Counter] = {}
        lens: dict[str, list[int]] = {}
        for r in recs:
            by_p.setdefault(r["prompt_id"], Counter())[r["label"]] += 1
            lens.setdefault(r["prompt_id"], []).append(r["n_generated"])
        for p, c in sorted(by_p.items()):
            rows.append(
                {
                    "split": sp,
                    "prompt_id": p,
                    "honest": int(c.get("honest", 0)),
                    "ambiguous": int(c.get("ambiguous", 0)),
                    "deceptive": int(c.get("deceptive", 0)),
                    "mean_n_generated": float(np.mean(lens[p])),
                    "median_n_generated": float(np.median(lens[p])),
                    "prevalence_D_among_HD": (
                        c.get("deceptive", 0)
                        / max(1, c.get("honest", 0) + c.get("deceptive", 0))
                    ),
                }
            )
    return {"rows": rows, "analysis_label": EXPLORATORY_LABEL}


def _strip_heavy(m: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in m.items() if k not in ("scores", "acts_for_probe")}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Asserting Phase-24F immutability…", flush=True)
    imm = assert_phase24f_immutable(REPO_ROOT)
    write_json(OUT / "phase24f_immutability.json", imm)

    print("Loading tokenizer + records…", flush=True)
    tok = _load_tokenizer()
    by_split = _build_all_records(tok)
    train = by_split["train"]
    val = by_split["validation"]
    test = by_split["test"]
    dev = train + val
    print(
        f"records train={len(train)} val={len(val)} test={len(test)} "
        f"(incl. ambiguous)",
        flush=True,
    )

    print("Loading NPZs…", flush=True)
    cache: dict[str, dict] = {}
    for r in train + val + test:
        cache[r["trajectory_id"]] = _load_npz(Path(r["npz_path"]))

    t, L = CANDIDATE_TIME, CANDIDATE_LAYER
    print("=== 1. Candidate stability t=1/L20 ===", flush=True)
    train_m = eval_coordinate(
        train, train, cache, tok, t=t, layer=L, use_oof_for_train_eval=True
    )
    val_m = eval_coordinate(train, val, cache, tok, t=t, layer=L)
    test_m = eval_coordinate(dev, test, cache, tok, t=t, layer=L)
    test_m["analysis_label"] = EXPLORATORY_LABEL

    p24f = json.loads(
        (REPO_ROOT / "artifacts/phase24f_confirmation/primary_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    cand_stability = {
        "coordinate": {"time": t, "layer": L},
        "train_oof": _strip_heavy(train_m),
        "validation": _strip_heavy(val_m),
        "test_exploratory": _strip_heavy(test_m),
        "phase24f_frozen_reference": {
            "surface_auroc": p24f["surface_auroc"],
            "activation_auroc": p24f["activation_auroc"],
            "combined_auroc": p24f["surface_plus_activation_auroc"],
            "delta_auroc": p24f["delta_auroc"],
            "ci95": p24f["delta_auroc_bootstrap"]["ci95"],
            "primary_pass": False,
        },
    }
    write_json(OUT / "candidate_stability_by_split.json", tag_exploratory(cand_stability))

    surface_diag = {
        "train_oof": {
            "text_auroc": train_m.get("text_auroc"),
            "logits_auroc": train_m.get("logits_auroc"),
            "surface_auroc": train_m.get("surface_auroc"),
        },
        "validation": {
            "text_auroc": val_m.get("text_auroc"),
            "logits_auroc": val_m.get("logits_auroc"),
            "surface_auroc": val_m.get("surface_auroc"),
        },
        "test_exploratory": {
            "text_auroc": test_m.get("text_auroc"),
            "logits_auroc": test_m.get("logits_auroc"),
            "surface_auroc": test_m.get("surface_auroc"),
            "analysis_label": EXPLORATORY_LABEL,
        },
        "note": (
            "TEST SURFACE ~chance may inflate ΔAUROC interpretation; "
            "inspect TEXT vs LOGITS separately."
        ),
    }
    write_json(OUT / "surface_baseline_diagnostics.json", tag_exploratory(surface_diag))

    print("=== 6/7. Heterogeneity + same-prompt ===", flush=True)
    het = prompt_heterogeneity(by_split)
    write_json(OUT / "prompt_heterogeneity.json", het)
    same = {
        "train": same_prompt_table(train_m["scores"], min_h=1, min_d=1),
        "validation": same_prompt_table(val_m["scores"], min_h=1, min_d=1),
        "test_exploratory": same_prompt_table(test_m["scores"], min_h=1, min_d=1),
        "test_ge2": same_prompt_table(test_m["scores"], min_h=2, min_d=2),
        "analysis_label": EXPLORATORY_LABEL,
        "note": "Secondary only; cannot replace Phase-24F primary FAIL",
    }
    for key in ("train", "validation", "test_exploratory", "test_ge2"):
        comb = [r for r in same[key] if r["model"] == "combined"]
        if comb:
            diffs = [r["diff_D_minus_H"] for r in comb]
            same[f"{key}_combined_summary"] = {
                "n": len(diffs),
                "median": float(np.median(diffs)),
                "mean": float(np.mean(diffs)),
                "fraction_gt_0": float(np.mean(np.asarray(diffs) > 0)),
            }
    write_json(OUT / "same_prompt_robustness.json", same)

    print("=== 4. EXPLORATORY TEST 7×32 grid (maps_fast) ===", flush=True)
    test_maps = discovery_maps_fast(dev, test, cache, tok, proxy_delta=False)
    test_delta = {
        t: {
            L: (
                float(test_maps["delta_auroc"][t][L])
                if np.isfinite(test_maps["delta_auroc"][t].get(L, float("nan")))
                else float("-inf")
            )
            for L in range(32)
        }
        for t in TEMPORAL_LANDMARKS
    }

    test_rows = []
    for t in TEMPORAL_LANDMARKS:
        for L in range(32):
            test_rows.append(
                {
                    "time": t,
                    "layer": L,
                    "surface_auroc": test_maps["surface_auroc"][t][L],
                    "activation_auroc": test_maps["activation_auroc"][t][L],
                    "combined_auroc": test_maps["combined_auroc"][t][L],
                    "delta_auroc": test_maps["delta_auroc"][t][L],
                    "text_auroc": test_maps["text_auroc"][t][L],
                    "logits_auroc": test_maps["logits_auroc"][t][L],
                    "analysis_label": EXPLORATORY_LABEL,
                }
            )
    finite_test = [r for r in test_rows if np.isfinite(r["delta_auroc"])]
    best_test = max(finite_test, key=lambda r: r["delta_auroc"]) if finite_test else None
    write_json(
        OUT / "exploratory_test_coordinate_metrics.json",
        tag_exploratory(
            {
                "rows": test_rows,
                "best_delta_coordinate": best_test,
                "n_coordinates": 224,
                "note": "NOT confirmatory",
            }
        ),
    )

    val_maps = load_val_metric_maps(REPO_ROOT)
    for name, grid in (
        ("val_delta", val_maps["delta_auroc"]),
        ("test_delta", test_delta),
        ("test_surface", test_maps["surface_auroc"]),
        ("test_activation", test_maps["activation_auroc"]),
        ("test_combined", test_maps["combined_auroc"]),
    ):
        svg_heatmap(
            matrix_from_grid(grid),
            times=list(TEMPORAL_LANDMARKS),
            layers=list(range(32)),
            title=f"{EXPLORATORY_LABEL}: {name}",
            path=OUT / f"heatmap_{name}.svg",
        )

    print("=== 5. VAL↔TEST topology ===", flush=True)
    topo = {
        "overall": corr_224(val_maps["delta_auroc"], test_delta),
        "layerwise": layerwise_corr(val_maps["delta_auroc"], test_delta),
        "timewise": timewise_corr(val_maps["delta_auroc"], test_delta),
        "high_region_overlap_q75": high_region_overlap(
            val_maps["delta_auroc"], test_delta, quantile=0.75
        ),
        "high_region_overlap_q90": high_region_overlap(
            val_maps["delta_auroc"], test_delta, quantile=0.90
        ),
        "analysis_label": EXPLORATORY_LABEL,
    }
    write_json(OUT / "val_test_topology.json", topo)

    print("=== 2. Leave-one-prompt-out discovery ===", flush=True)
    all_dev_prompts = sorted({r["prompt_id"] for r in dev})
    train_pids = {r["prompt_id"] for r in train}
    lopo_rows = []
    for i, left in enumerate(all_dev_prompts):
        print(f"LOPO {i + 1}/28 leave {left}", flush=True)
        tr_r = [r for r in train if r["prompt_id"] != left]
        va_r = [r for r in val if r["prompt_id"] != left]
        if len({r["prompt_id"] for r in tr_r}) < 4 or len(
            {r["prompt_id"] for r in va_r}
        ) < 2:
            rem = [p for p in all_dev_prompts if p != left]
            rng = np.random.default_rng(ANALYSIS_SEED + i)
            rng.shuffle(rem)
            n_tr = max(4, int(0.7 * len(rem)))
            tr_set, va_set = set(rem[:n_tr]), set(rem[n_tr:])
            tr_r = [r for r in dev if r["prompt_id"] in tr_set]
            va_r = [r for r in dev if r["prompt_id"] in va_set]
        grid = discovery_delta_grid_fast(tr_r, va_r, cache, tok)
        cand = select_candidate_region(grid)
        lopo_rows.append(
            {
                "left_out_prompt": left,
                "left_out_split": "train" if left in train_pids else "validation",
                "candidate": cand,
                "exact_t1_l20": bool(
                    cand
                    and cand["candidate_time"] == 1
                    and cand["candidate_layer"] == 20
                ),
                "in_neighborhood": candidate_in_neighborhood(cand),
            }
        )
    lopo_art = {
        "n": len(lopo_rows),
        "rows": lopo_rows,
        "frac_exact_t1_l20": float(np.mean([r["exact_t1_l20"] for r in lopo_rows])),
        "frac_neighborhood": float(np.mean([r["in_neighborhood"] for r in lopo_rows])),
        "time_hist": dict(
            Counter(
                r["candidate"]["candidate_time"]
                for r in lopo_rows
                if r["candidate"]
            )
        ),
        "layer_hist": dict(
            Counter(
                r["candidate"]["candidate_layer"]
                for r in lopo_rows
                if r["candidate"]
            )
        ),
    }
    write_json(OUT / "lopo_candidate_stability.json", lopo_art)

    print(f"=== 3. Nested optimism ({NESTED_REPS} reps) ===", flush=True)
    rng = np.random.default_rng(ANALYSIS_SEED)
    nested_rows = []
    for rep in range(NESTED_REPS):
        prompts = list(all_dev_prompts)
        rng.shuffle(prompts)
        tr_set = set(prompts[:14])
        va_set = set(prompts[14:21])
        ho_set = set(prompts[21:28])
        tr_r = [r for r in dev if r["prompt_id"] in tr_set]
        va_r = [r for r in dev if r["prompt_id"] in va_set]
        ho_r = [r for r in dev if r["prompt_id"] in ho_set]
        grid = discovery_delta_grid_fast(tr_r, va_r, cache, tok)
        cand = select_candidate_region(grid)
        if cand is None:
            nested_rows.append(
                {
                    "rep": rep,
                    "candidate": None,
                    "discovery_delta": float("nan"),
                    "heldout_delta": float("nan"),
                }
            )
        else:
            disc_delta = float(cand["layer_delta_auroc"])
            hm = eval_coordinate(
                tr_r + va_r,
                ho_r,
                cache,
                tok,
                t=int(cand["candidate_time"]),
                layer=int(cand["candidate_layer"]),
            )
            held = (
                float(hm["delta_auroc"])
                if not hm.get("skipped")
                else float("nan")
            )
            nested_rows.append(
                {
                    "rep": rep,
                    "candidate": {
                        "time": cand["candidate_time"],
                        "layer": cand["candidate_layer"],
                        "band": cand["band"],
                    },
                    "discovery_delta": disc_delta,
                    "heldout_delta": held,
                    "optimism": (
                        disc_delta - held
                        if np.isfinite(disc_delta) and np.isfinite(held)
                        else float("nan")
                    ),
                }
            )
        if (rep + 1) % 10 == 0:
            print(f"  nested {rep + 1}/{NESTED_REPS}", flush=True)

    disc_vals = [
        r["discovery_delta"]
        for r in nested_rows
        if np.isfinite(r.get("discovery_delta", np.nan))
    ]
    hold_vals = [
        r["heldout_delta"]
        for r in nested_rows
        if np.isfinite(r.get("heldout_delta", np.nan))
    ]
    opt_vals = [
        r["optimism"]
        for r in nested_rows
        if np.isfinite(r.get("optimism", np.nan))
    ]
    nested_art = {
        "n_reps_requested_example": NESTED_REPS_TARGET_NOTE,
        "n_reps_completed": NESTED_REPS,
        "note": (
            f"Computational budget: completed {NESTED_REPS} nested reps "
            f"(example target {NESTED_REPS_TARGET_NOTE}) with frozen Phase-24E Cs "
            "and proxy Δ=ACTIVATION−SURFACE for selection stability "
            "(full stacked Δ used for exploratory TEST grid and candidate "
            "stability sections)."
        ),
        "expected_discovery_delta": float(np.mean(disc_vals)) if disc_vals else float("nan"),
        "expected_heldout_delta": float(np.mean(hold_vals)) if hold_vals else float("nan"),
        "optimism_gap": float(np.mean(opt_vals)) if opt_vals else float("nan"),
        "discovery_delta_ci95": (
            [
                float(np.quantile(disc_vals, 0.025)),
                float(np.quantile(disc_vals, 0.975)),
            ]
            if disc_vals
            else [float("nan"), float("nan")]
        ),
        "heldout_delta_ci95": (
            [
                float(np.quantile(hold_vals, 0.025)),
                float(np.quantile(hold_vals, 0.975)),
            ]
            if hold_vals
            else [float("nan"), float("nan")]
        ),
        "frac_candidate_none": float(np.mean([r["candidate"] is None for r in nested_rows])),
        "frac_exact_t1_l20": float(
            np.mean(
                [
                    bool(
                        r["candidate"]
                        and r["candidate"]["time"] == 1
                        and r["candidate"]["layer"] == 20
                    )
                    for r in nested_rows
                ]
            )
        ),
        "rows": nested_rows,
    }
    write_json(OUT / "nested_resampling_optimism.json", nested_art)

    print("=== 8. Probe-direction stability ===", flush=True)
    coords: list[tuple[str, int, int]] = [
        ("frozen_candidate", 1, 20),
        ("best_val", 32, 17),
    ]
    if best_test:
        coords.append(
            (
                "best_exploratory_test",
                int(best_test["time"]),
                int(best_test["layer"]),
            )
        )
    inter = topo["high_region_overlap_q75"]["intersection_examples"]
    if inter:
        coords.append(("shared_high_q75", int(inter[0][0]), int(inter[0][1])))

    probe_art = {"coordinates": {}, "analysis_label": EXPLORATORY_LABEL}
    rng_p = np.random.default_rng(ANALYSIS_SEED)
    dev_primary = _primary(dev)
    pids = sorted({r["prompt_id"] for r in dev_primary})
    for name, tt, LL in coords:
        coefs = []
        for _ in range(30):
            sampled = rng_p.choice(pids, size=len(pids), replace=True)
            recs = []
            for p in sampled:
                recs.extend([r for r in dev_primary if r["prompt_id"] == p])
            feat = _feat_bundle(recs, cache, tok, tt, LL)
            if feat["empty"] or len(np.unique(feat["y"])) < 2:
                continue
            coefs.append(
                fit_linear_probe_coef(
                    feat["acts"],
                    feat["y"],
                    feat["labels"],
                    feat["prompts"],
                    _selected_C(tt, LL, "ACTIVATION"),
                )
            )
        probe_art["coordinates"][name] = {
            "time": tt,
            "layer": LL,
            "stability": probe_stability_from_coefs(coefs),
        }
    write_json(OUT / "probe_direction_stability.json", probe_art)

    print("=== 10. Power projections ===", flush=True)
    test_prompt_deltas = [
        r["local_delta_auroc"]
        for r in test_m["prompt_contributions"]
        if np.isfinite(r["local_delta_auroc"])
    ]
    if len(test_prompt_deltas) < 3:
        test_prompt_deltas = [
            r["combined_sep_D_minus_H"]
            for r in test_m["prompt_contributions"]
            if np.isfinite(r["combined_sep_D_minus_H"])
        ]
    power = power_projection(
        observed_prompt_deltas=test_prompt_deltas,
        effect_sizes=[0.05, 0.08, 0.10, 0.15],
        n_prompts_grid=[11, 20, 30, 40, 60, 80, 100],
        n_sims=2000,
        seed=ANALYSIS_SEED,
    )
    write_json(OUT / "power_projections.json", power)

    test_same_frac = same.get("test_ge2_combined_summary", {}).get(
        "fraction_gt_0", float("nan")
    )
    evidence = {
        "topology_pearson": topo["overall"]["pearson"],
        "lopo_exact_frac": lopo_art["frac_exact_t1_l20"],
        "lopo_neighborhood_frac": lopo_art["frac_neighborhood"],
        "optimism_gap": nested_art["optimism_gap"],
        "test_same_prompt_frac_gt0": test_same_frac,
        "probe_mean_cosine": probe_art["coordinates"]
        .get("frozen_candidate", {})
        .get("stability", {})
        .get("mean_pairwise_cosine", float("nan")),
        "test_delta_auroc": p24f["delta_auroc"],
        "test_ci_width": float(
            p24f["delta_auroc_bootstrap"]["ci95"][1]
            - p24f["delta_auroc_bootstrap"]["ci95"][0]
        ),
    }
    category = classify_diagnostic_category(evidence)
    category_art = {
        "category": category,
        "evidence": evidence,
        "definitions": {
            "A": "Stable region, underpowered confirmation",
            "B": "Real but prompt-specific / heterogeneous signal",
            "C": "Candidate-selection instability",
            "D": "Little robust evidence",
        },
        "phase24f_remains_fail": True,
        "analysis_label": EXPLORATORY_LABEL,
    }
    write_json(OUT / "diagnostic_category.json", category_art)

    git_commit = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
        .decode()
        .strip()
    )
    manifest = {
        "created_at": utc_now_iso(),
        "starting_sha": STARTING_SHA,
        "git_commit": git_commit,
        "status": STATUS,
        "category": category,
        "phase24f_immutable": True,
        "guarantee": GUARANTEE,
        "nested_reps": NESTED_REPS,
        "exploratory_label": EXPLORATORY_LABEL,
    }
    write_json(OUT / "diagnostics_manifest.json", manifest)

    CFG.write_text(
        f"""# Phase 24G — confirmatory failure diagnostics (exploratory)

experiment_id: phase24g_failure_diagnostics
phase: phase24g
status: {STATUS}
starting_sha: {STARTING_SHA}
diagnostic_category: {category}
phase24f_primary_pass: false
phase24f_immutable: true

authorizations:
  exploratory_post_confirmation_diagnostics: true
  primary_endpoint_redefinition: false
  sae_analysis_authorized: false
  causal_intervention_authorized: false
  new_trajectories_authorized: false

notes: >
  EXPLORATORY_POST_CONFIRMATION diagnostics only.
  Phase-24F FAIL remains final and unchanged.
""",
        encoding="utf-8",
    )

    def _stab_row(label: str, surf, act, comb, delta) -> str:
        return (
            f"| {label} | {surf:.4f} | {act:.4f} | {comb:.4f} | {delta:.4f} |"
        )

    row_train = _stab_row(
        "TRAIN (OOF)",
        train_m["surface_auroc"],
        train_m["activation_auroc"],
        train_m["combined_auroc"],
        train_m["delta_auroc"],
    )
    row_val = _stab_row(
        "VALIDATION",
        val_m["surface_auroc"],
        val_m["activation_auroc"],
        val_m["combined_auroc"],
        val_m["delta_auroc"],
    )
    row_test = _stab_row(
        "TEST (exploratory refit)",
        test_m["surface_auroc"],
        test_m["activation_auroc"],
        test_m["combined_auroc"],
        test_m["delta_auroc"],
    )
    row_frozen = _stab_row(
        "TEST Phase-24F frozen",
        p24f["surface_auroc"],
        p24f["activation_auroc"],
        p24f["surface_plus_activation_auroc"],
        p24f["delta_auroc"],
    )

    report = f"""# Phase 24G — Confirmatory failure diagnostics

**Status:** `{STATUS}`

**Diagnostic category:** **{category}** — {category_art['definitions'][category]}

{GUARANTEE}

## Phase-24F immutability

Verified hash match and `primary_pass=false`. Frozen ΔAUROC={p24f['delta_auroc']:.4f},
CI95={p24f['delta_auroc_bootstrap']['ci95']}.

## 1. Candidate stability (t=1, L20)

| Split | SURFACE | ACTIVATION | COMBINED | ΔAUROC |
|---|---|---|---|---|
{row_train}
{row_val}
{row_test}
{row_frozen}

## 2. LOPO candidate stability

Exact t=1/L20: **{lopo_art['frac_exact_t1_l20']:.3f}**
Neighborhood: **{lopo_art['frac_neighborhood']:.3f}**
Time hist: `{json.dumps(lopo_art['time_hist'])}`
Layer hist: `{json.dumps(lopo_art['layer_hist'])}`

## 3. Nested resampling optimism ({NESTED_REPS} reps)

Expected discovery ΔAUROC: **{nested_art['expected_discovery_delta']:.4f}**
Expected held-out ΔAUROC: **{nested_art['expected_heldout_delta']:.4f}**
Optimism gap: **{nested_art['optimism_gap']:.4f}**
Frac exact t=1/L20: **{nested_art['frac_exact_t1_l20']:.3f}**

## 4. Exploratory TEST best ΔAUROC region

`{json.dumps(best_test)}`

Label: `{EXPLORATORY_LABEL}` — not confirmed.

## 5. VAL↔TEST topology

Overall Pearson/Spearman: `{json.dumps(topo['overall'])}`
Jaccard high-q75 overlap: **{topo['high_region_overlap_q75']['jaccard']:.3f}**

## 6–7. Heterogeneity / same-prompt

TEST ≥2H+≥2D combined fraction>0: **{test_same_frac}**
Summaries in `same_prompt_robustness.json`.

## 8. Probe-direction stability

`{json.dumps(probe_art['coordinates'], indent=2)}`

## 9. TEXT vs LOGITS

`{json.dumps(surface_diag, indent=2)}`

## 10. Power projections

See `power_projections.json` (planning only; no new data).

## 11. Category

**{category}**: {category_art['definitions'][category]}

Phase-24F FAIL is unchanged.
"""
    REPORT.write_text(report, encoding="utf-8")

    hash_files = [
        p for p in OUT.iterdir() if p.is_file() and p.name != "artifact_hashes.json"
    ]
    hash_files.extend([CFG, REPORT])
    hashes = {p.name: sha256_file(str(p)) for p in sorted(set(hash_files))}
    write_json(OUT / "artifact_hashes.json", hashes)

    dlog = DECISION_LOG.read_text(encoding="utf-8")
    if "### D164 — Phase 24G" not in dlog:
        entry = (
            "\n### D164 — Phase 24G confirmatory failure diagnostics\n\n"
            "- **Type:** **OUR RESEARCH DECISION**\n"
            f"- **Decision:** Starting from Phase-24F FAIL `{STARTING_SHA[:7]}…`, "
            "Phase 24G ran exploratory post-confirmation diagnostics "
            f"(LOPO, nested optimism n={NESTED_REPS}, TEST 7×32 exploratory grid, "
            f"topology, heterogeneity, probe stability, power). "
            f"Diagnostic category **{category}**. "
            "Phase-24F primary FAIL remains final and unchanged. "
            "No SAE, causal intervention, new trajectories, or endpoint redefinition.\n"
            "- **Date:** 2026-09-30\n"
        )
        DECISION_LOG.write_text(dlog.rstrip() + "\n" + entry, encoding="utf-8")

    print(
        json.dumps(
            {
                "status": STATUS,
                "category": category,
                "optimism_gap": nested_art["optimism_gap"],
                "lopo_exact": lopo_art["frac_exact_t1_l20"],
                "topo_pearson": topo["overall"]["pearson"],
                "best_test": best_test,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
