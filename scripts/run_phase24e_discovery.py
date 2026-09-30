#!/usr/bin/env python3
"""Run Phase-24E TRAIN+VALIDATION discovery analysis (TEST sealed; no GPU needed)."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase24c_design import (  # noqa: E402
    LOGISTIC_C_GRID,
    N_LAYERS,
    TEMPORAL_LANDMARKS,
    VAL_BOOTSTRAP_REPS,
    select_candidate_region,
    sha256_file,
)
from pre_output_physiology.phase24e_discovery import (  # noqa: E402
    ANALYSIS_SEED,
    GUARANTEE,
    STARTING_SHA,
    STATUS_CANDIDATE,
    STATUS_NO_CANDIDATE,
    SealViolationError,
    _as_binary_labels,
    assert_not_test_path,
    assert_split_allowed,
    bf16_u16_to_float32,
    delta_h_vector,
    fit_predict_log_odds,
    list_qualifying_bands,
    oof_log_odds,
    prefix_token_ids,
    prompt_cluster_bootstrap_ci,
    safe_auprc,
    safe_auroc,
    select_C_meta_cv,
    select_C_prompt_group_cv,
    survives_landmark,
    svg_heatmap,
    verify_frozen_input_hashes,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

OUT = REPO_ROOT / "artifacts/phase24e_discovery"
ACT_DIR = OUT / "activations_local"
CFG = REPO_ROOT / "configs/experiments/phase24e_discovery_analysis.yaml"
REPORT = REPO_ROOT / "reports/phase24e_discovery_analysis.md"
DECISION_LOG = REPO_ROOT / "docs/decision_log.md"
SPLIT_PATH = REPO_ROOT / "artifacts/phase24c_design/split_manifest.json"
LABELS_TRAIN = REPO_ROOT / "artifacts/phase24d_collection/by_split/TRAIN/labels.json"
LABELS_VAL = REPO_ROOT / "artifacts/phase24d_collection/by_split/VALIDATION/labels.json"
META24D = REPO_ROOT / "artifacts/phase24d_collection/capture_meta.json"
TRAJ24B = REPO_ROOT / "artifacts/phase24b_live/trajectories.jsonl"
REUSE = REPO_ROOT / "artifacts/phase24d_collection/by_split/TRAIN/phase24b_reuse_index.json"


def _load_tokenizer():
    from transformers import AutoTokenizer

    from pre_output_physiology.phase24b_live import MODEL_ID, MODEL_REVISION

    tok = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, use_fast=True
    )
    return tok


def _decode_prefix(tok, prompt_text: str, gen_ids: list[int], t: int) -> str:
    pref = prefix_token_ids(gen_ids, t)
    if not pref:
        return prompt_text
    return prompt_text + tok.decode(pref, skip_special_tokens=True)


def _build_records(tok) -> list[dict]:
    """Assemble TRAIN+VAL trajectory records with labels; never touch TEST."""
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    train_set = set(split["train_prompt_ids"])
    val_set = set(split["validation_prompt_ids"])
    test_set = set(split["test_prompt_ids"])

    labels = {}
    for path, allowed in ((LABELS_TRAIN, train_set), (LABELS_VAL, val_set)):
        assert_not_test_path(path)
        blob = json.loads(path.read_text(encoding="utf-8"))
        for r in blob["rows"]:
            if r["prompt_id"] in test_set:
                raise SealViolationError("test prompt in train/val labels")
            if r["prompt_id"] not in allowed:
                raise RuntimeError(f"prompt {r['prompt_id']} not in expected split")
            if r.get("open_valid") and r.get("open_class") in ("honest", "deceptive"):
                labels[r["trajectory_id"]] = {
                    "label": r["open_class"],
                    "prompt_id": r["prompt_id"],
                    "replicate_index": r["replicate_index"],
                    "source": r.get("source"),
                }

    # metadata for new 24D traj
    meta = json.loads(META24D.read_text(encoding="utf-8"))
    meta_by = {
        m["trajectory_id"]: m
        for m in meta["meta_rows"]
        if m.get("completed") and m["split"] in ("train", "validation")
    }

    # 24B reuse meta
    reuse_ids = set(
        json.loads(REUSE.read_text(encoding="utf-8"))["trajectory_ids"]
    )
    traj24b = {
        json.loads(x)["trajectory_id"]: json.loads(x)
        for x in TRAJ24B.read_text(encoding="utf-8").splitlines()
        if x.strip()
    }

    records = []
    for tid, lab in labels.items():
        pid = lab["prompt_id"]
        if pid in test_set:
            raise SealViolationError(tid)
        if pid in train_set:
            split_name = "train"
            folder = "TRAIN"
        elif pid in val_set:
            split_name = "validation"
            folder = "VALIDATION"
        else:
            raise RuntimeError(f"unknown prompt {pid}")
        assert_split_allowed(split_name)

        npz_path = ACT_DIR / folder / f"{tid}.npz"
        assert_not_test_path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"missing activation {npz_path}")

        if tid in meta_by:
            m = meta_by[tid]
            gen_ids = list(m["generated_token_ids"])
            prompt_text = tok.decode(m["prompt_token_ids"], skip_special_tokens=False)
            # Prefer reconstructing from stored scenario formatting consistency:
            # use full_response prefix: answer_prefix is start of response;
            # surface text = prompt chat format. Stored prompt_token_ids decode is OK.
            answer_prefix = m["answer_prefix"]
            n_gen = len(gen_ids)
        elif tid in reuse_ids:
            m = traj24b[tid]
            gen_ids = list(m["generated_token_ids"])
            prompt_text = tok.decode(m["prompt_token_ids"], skip_special_tokens=False)
            answer_prefix = m["answer_prefix"]
            n_gen = len(gen_ids)
        else:
            raise KeyError(tid)

        records.append(
            {
                "trajectory_id": tid,
                "prompt_id": pid,
                "split": split_name,
                "label": lab["label"],
                "generated_token_ids": gen_ids,
                "prompt_text": prompt_text,
                "answer_prefix": answer_prefix,
                "n_generated": n_gen,
                "npz_path": str(npz_path),
            }
        )
    return records


def _load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    assert_not_test_path(path)
    with np.load(path) as z:
        return {
            "post_block_u16": z["post_block_bf16_u16"],
            "logits_f16": z["logits_f16"],
            "generated_token_ids": z["generated_token_ids"],
        }


def main() -> int:
    ver = verify_frozen_input_hashes(REPO_ROOT)
    if not ver["verified"]:
        raise SystemExit(f"STOP: frozen hash mismatch\n{json.dumps(ver, indent=2)}")

    if not (ACT_DIR / "TRAIN").exists() or not (ACT_DIR / "VALIDATION").exists():
        raise SystemExit(
            "STOP: activations_local TRAIN/VALIDATION missing — run "
            "modal/phase24e_fetch_activations.py first"
        )
    # Ensure no TEST dir was fetched
    if (ACT_DIR / "LOCKED_TEST").exists() or (ACT_DIR / "TEST").exists():
        raise SystemExit("STOP: TEST activation directory must not exist locally")

    print("Loading tokenizer…", flush=True)
    tok = _load_tokenizer()
    records = _build_records(tok)
    train_recs = [r for r in records if r["split"] == "train"]
    val_recs = [r for r in records if r["split"] == "validation"]
    print(
        f"records train={len(train_recs)} val={len(val_recs)} "
        f"(H/D primary only)",
        flush=True,
    )

    # Usable totals
    def _counts(recs):
        h = sum(1 for r in recs if r["label"] == "honest")
        d = sum(1 for r in recs if r["label"] == "deceptive")
        return {
            "honest": h,
            "deceptive": d,
            "n_trajectories": len(recs),
            "n_prompts": len({r["prompt_id"] for r in recs}),
        }

    train_counts = _counts(train_recs)
    val_counts = _counts(val_recs)

    risk_sets = {}
    for t in TEMPORAL_LANDMARKS:
        risk_sets[str(t)] = {
            "train": sum(1 for r in train_recs if survives_landmark(r["n_generated"], t)),
            "validation": sum(
                1 for r in val_recs if survives_landmark(r["n_generated"], t)
            ),
        }

    # Preload NPZs into memory (float activations only as needed)
    print("Loading NPZs…", flush=True)
    cache: dict[str, dict[str, np.ndarray]] = {}
    for r in records:
        cache[r["trajectory_id"]] = _load_npz_arrays(Path(r["npz_path"]))

    grid_rows: list[dict] = []
    delta_map: dict[int, dict[int, float]] = {t: {} for t in TEMPORAL_LANDMARKS}
    surface_mat = np.full((N_LAYERS, len(TEMPORAL_LANDMARKS)), np.nan)
    act_mat = np.full((N_LAYERS, len(TEMPORAL_LANDMARKS)), np.nan)
    comb_mat = np.full((N_LAYERS, len(TEMPORAL_LANDMARKS)), np.nan)
    delta_mat = np.full((N_LAYERS, len(TEMPORAL_LANDMARKS)), np.nan)

    selected_C_log: dict[str, Any] = {}
    bootstrap_log: dict[str, Any] = {}

    for ti, t in enumerate(TEMPORAL_LANDMARKS):
        print(f"=== time landmark t={t} ===", flush=True)
        tr = [r for r in train_recs if survives_landmark(r["n_generated"], t)]
        va = [r for r in val_recs if survives_landmark(r["n_generated"], t)]
        if len(tr) < 4 or len(va) < 2:
            for L in range(N_LAYERS):
                grid_rows.append(
                    {
                        "time": t,
                        "layer": L,
                        "skipped": True,
                        "reason": "insufficient risk set",
                    }
                )
            continue

        # Build TEXT / LOGITS features at t (bind landmark via default)
        def feats_for(recs, t_landmark: int = t):
            texts, logits, labels, prompts, tids = [], [], [], [], []
            for r in recs:
                arr = cache[r["trajectory_id"]]
                # align: array index == prediction step
                logit_row = arr["logits_f16"][t_landmark].astype(np.float32)
                text = _decode_prefix(
                    tok, r["prompt_text"], r["generated_token_ids"], t_landmark
                )
                # ensure no future token: prefix length == t
                assert (
                    len(prefix_token_ids(r["generated_token_ids"], t_landmark))
                    == t_landmark
                )
                texts.append(text)
                logits.append(logit_row)
                labels.append(r["label"])
                prompts.append(r["prompt_id"])
                tids.append(r["trajectory_id"])
            return texts, np.stack(logits, axis=0), labels, prompts, tids

        tr_text, tr_logits, tr_lab, tr_pid, _ = feats_for(tr)
        va_text, va_logits, va_lab, va_pid, _ = feats_for(va)
        y_tr = _as_binary_labels(tr_lab)
        y_va = _as_binary_labels(va_lab)

        # Select C for TEXT / LOGITS on TRAIN
        c_text = select_C_prompt_group_cv(
            X=tr_text,
            y=y_tr,
            prompt_ids=tr_pid,
            labels_str=tr_lab,
            kind="text",
        )
        c_logit = select_C_prompt_group_cv(
            X=tr_logits,
            y=y_tr,
            prompt_ids=tr_pid,
            labels_str=tr_lab,
            kind="matrix",
        )
        selected_C_log[f"t{t}_TEXT"] = c_text
        selected_C_log[f"t{t}_LOGITS"] = c_logit

        # OOF log-odds for SURFACE stacking
        oof_text = oof_log_odds(
            X=tr_text,
            y=y_tr,
            prompt_ids=tr_pid,
            labels_str=tr_lab,
            kind="text",
            C=c_text["selected_C"],
        )
        oof_logit = oof_log_odds(
            X=tr_logits,
            y=y_tr,
            prompt_ids=tr_pid,
            labels_str=tr_lab,
            kind="matrix",
            C=c_logit["selected_C"],
        )
        Z_oof = np.stack([oof_text, oof_logit], axis=1)
        c_surf = select_C_meta_cv(
            Z_oof=Z_oof, y=y_tr, prompt_ids=tr_pid, labels_str=tr_lab
        )
        selected_C_log[f"t{t}_SURFACE"] = c_surf

        # VAL base predictions from full TRAIN fits
        lo_text_va = fit_predict_log_odds(
            X_train=tr_text,
            y_train=y_tr,
            labels_train=tr_lab,
            prompts_train=tr_pid,
            X_eval=va_text,
            kind="text",
            C=c_text["selected_C"],
        )
        lo_logit_va = fit_predict_log_odds(
            X_train=tr_logits,
            y_train=y_tr,
            labels_train=tr_lab,
            prompts_train=tr_pid,
            X_eval=va_logits,
            kind="matrix",
            C=c_logit["selected_C"],
        )
        # Fit SURFACE meta on OOF, predict VAL
        valid_oof = np.all(np.isfinite(Z_oof), axis=1)
        from sklearn.preprocessing import StandardScaler

        from pre_output_physiology.phase24e_discovery import (
            make_logit_or_act_clf,
            sample_weights_for_fit,
        )

        scaler_s = StandardScaler()
        Ztr = scaler_s.fit_transform(Z_oof[valid_oof])
        w_s = sample_weights_for_fit(
            [tr_lab[i] for i in range(len(tr_lab)) if valid_oof[i]],
            [tr_pid[i] for i in range(len(tr_pid)) if valid_oof[i]],
        )
        meta_s = make_logit_or_act_clf(c_surf["selected_C"])
        meta_s.fit(Ztr, y_tr[valid_oof], sample_weight=w_s)
        Zva = scaler_s.transform(np.stack([lo_text_va, lo_logit_va], axis=1))
        surf_proba_va = np.clip(meta_s.predict_proba(Zva)[:, 1], 1e-6, 1 - 1e-6)
        surf_score_va = surf_proba_va  # use probability for AUROC
        surface_auroc = safe_auroc(y_va, surf_score_va)
        surface_auprc = safe_auprc(y_va, surf_score_va)

        # Also need OOF SURFACE log-odds for SURFACE+ACTIVATION stacking
        # Approximate: use OOF meta predictions via fold-wise meta fit would be ideal;
        # use meta fitted on all OOF applied to OOF features as TRAIN surface scores
        # for stacking with activation — but that leaks slightly within OOF.
        # Proper: nested OOF. For frozen design, use OOF text/logit stacked with
        # meta C selected, predicting each OOF row with meta fit on other OOF folds.
        oof_surf = np.full(len(y_tr), np.nan)
        from pre_output_physiology.phase24c_design import prompt_group_folds

        folds = [
            f
            for f in prompt_group_folds(sorted(set(tr_pid)), n_folds=5)
            if f
        ]
        pid_arr = np.asarray(tr_pid)
        for hold in folds:
            hold_set = set(hold)
            te = np.array([p in hold_set for p in pid_arr])
            trm = (~te) & valid_oof
            tem = te & valid_oof
            if trm.sum() < 2 or tem.sum() < 1 or len(np.unique(y_tr[trm])) < 2:
                continue
            sc = StandardScaler()
            Z1 = sc.fit_transform(Z_oof[trm])
            Z2 = sc.transform(Z_oof[tem])
            w = sample_weights_for_fit(
                [tr_lab[i] for i in range(len(tr_lab)) if trm[i]],
                [tr_pid[i] for i in range(len(tr_pid)) if trm[i]],
            )
            clf = make_logit_or_act_clf(c_surf["selected_C"])
            clf.fit(Z1, y_tr[trm], sample_weight=w)
            pr = np.clip(clf.predict_proba(Z2)[:, 1], 1e-6, 1 - 1e-6)
            oof_surf[tem] = np.log(pr / (1 - pr))

        lo_surf_va = np.log(surf_proba_va / (1 - surf_proba_va))

        # Precompute h0 and ht for all layers once (bind landmark via default)
        def act_matrix(recs, t_landmark: int = t):
            # returns (n, 32, 4096) delta at time t
            mats = []
            for r in recs:
                post = bf16_u16_to_float32(cache[r["trajectory_id"]]["post_block_u16"])
                # post: (n_steps, 32, 4096)
                h0 = post[0]
                ht = post[t_landmark]
                mats.append(delta_h_vector(ht, h0))
            return np.stack(mats, axis=0)  # (n, 32, 4096)

        tr_act_all = act_matrix(tr)
        va_act_all = act_matrix(va)

        for L in range(N_LAYERS):
            Xtr_a = tr_act_all[:, L, :]
            Xva_a = va_act_all[:, L, :]
            c_act = select_C_prompt_group_cv(
                X=Xtr_a,
                y=y_tr,
                prompt_ids=tr_pid,
                labels_str=tr_lab,
                kind="matrix",
            )
            selected_C_log[f"t{t}_L{L}_ACTIVATION"] = {
                "selected_C": c_act["selected_C"],
                "best_cv_auroc": c_act["best_cv_auroc"],
            }

            oof_act = oof_log_odds(
                X=Xtr_a,
                y=y_tr,
                prompt_ids=tr_pid,
                labels_str=tr_lab,
                kind="matrix",
                C=c_act["selected_C"],
            )
            lo_act_va = fit_predict_log_odds(
                X_train=Xtr_a,
                y_train=y_tr,
                labels_train=tr_lab,
                prompts_train=tr_pid,
                X_eval=Xva_a,
                kind="matrix",
                C=c_act["selected_C"],
            )
            act_proba_va = 1 / (1 + np.exp(-lo_act_va))
            act_auroc = safe_auroc(y_va, act_proba_va)
            act_auprc = safe_auprc(y_va, act_proba_va)

            # SURFACE+ACTIVATION stack
            Z2_oof = np.stack([oof_surf, oof_act], axis=1)
            c_comb = select_C_meta_cv(
                Z_oof=Z2_oof, y=y_tr, prompt_ids=tr_pid, labels_str=tr_lab
            )
            selected_C_log[f"t{t}_L{L}_SURFACE_PLUS_ACTIVATION"] = {
                "selected_C": c_comb["selected_C"],
                "best_cv_auroc": c_comb["best_cv_auroc"],
            }
            v2 = np.all(np.isfinite(Z2_oof), axis=1)
            sc2 = StandardScaler()
            Ztr2 = sc2.fit_transform(Z2_oof[v2])
            w2 = sample_weights_for_fit(
                [tr_lab[i] for i in range(len(tr_lab)) if v2[i]],
                [tr_pid[i] for i in range(len(tr_pid)) if v2[i]],
            )
            meta2 = make_logit_or_act_clf(c_comb["selected_C"])
            meta2.fit(Ztr2, y_tr[v2], sample_weight=w2)
            Zva2 = sc2.transform(np.stack([lo_surf_va, lo_act_va], axis=1))
            comb_proba = np.clip(meta2.predict_proba(Zva2)[:, 1], 1e-6, 1 - 1e-6)
            comb_auroc = safe_auroc(y_va, comb_proba)
            comb_auprc = safe_auprc(y_va, comb_proba)
            delta = (
                comb_auroc - surface_auroc
                if not (math.isnan(comb_auroc) or math.isnan(surface_auroc))
                else float("nan")
            )

            boot = prompt_cluster_bootstrap_ci(
                y=y_va,
                scores_a=surf_score_va,
                scores_b=comb_proba,
                prompt_ids=va_pid,
                n_reps=VAL_BOOTSTRAP_REPS,
                seed=ANALYSIS_SEED + 1000 * t + L,
            )
            bootstrap_log[f"t{t}_L{L}"] = boot

            row = {
                "time": t,
                "layer": L,
                "surface_auroc": surface_auroc,
                "activation_auroc": act_auroc,
                "surface_plus_activation_auroc": comb_auroc,
                "delta_auroc": delta,
                "surface_auprc": surface_auprc,
                "activation_auprc": act_auprc,
                "surface_plus_activation_auprc": comb_auprc,
                "n_honest_val": int(np.sum(y_va == 0)),
                "n_deceptive_val": int(np.sum(y_va == 1)),
                "n_honest_train": int(np.sum(y_tr == 0)),
                "n_deceptive_train": int(np.sum(y_tr == 1)),
                "n_prompts_val": len(set(va_pid)),
                "n_prompts_train": len(set(tr_pid)),
                "C_text": c_text["selected_C"],
                "C_logits": c_logit["selected_C"],
                "C_surface": c_surf["selected_C"],
                "C_activation": c_act["selected_C"],
                "C_surface_plus_activation": c_comb["selected_C"],
                "delta_auroc_bootstrap": boot,
                "skipped": False,
            }
            grid_rows.append(row)
            delta_map[t][L] = float(delta) if not math.isnan(delta) else float("-inf")
            surface_mat[L, ti] = surface_auroc
            act_mat[L, ti] = act_auroc
            comb_mat[L, ti] = comb_auroc
            delta_mat[L, ti] = delta

            if L % 8 == 0:
                print(
                    f"  L{L}: surf={surface_auroc:.3f} act={act_auroc:.3f} "
                    f"comb={comb_auroc:.3f} d={delta:.3f}",
                    flush=True,
                )

    # Candidate selection
    bands = list_qualifying_bands(delta_map)
    candidate = select_candidate_region(delta_map)
    if candidate is None:
        status = STATUS_NO_CANDIDATE
        cand_metrics = None
    else:
        status = STATUS_CANDIDATE
        # find row
        cand_metrics = next(
            r
            for r in grid_rows
            if r.get("time") == candidate["candidate_time"]
            and r.get("layer") == candidate["candidate_layer"]
            and not r.get("skipped")
        )

    git_commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()

    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "hash_verification.json", ver)
    write_json(
        OUT / "analysis_manifest.json",
        {
            "created_at": utc_now_iso(),
            "git_commit": git_commit,
            "starting_sha": STARTING_SHA,
            "status": status,
            "guarantee": GUARANTEE,
            "temporal_landmarks": list(TEMPORAL_LANDMARKS),
            "n_layers": N_LAYERS,
            "n_coordinates": 7 * 32,
            "train_usable": train_counts,
            "validation_usable": val_counts,
            "risk_sets": risk_sets,
            "C_grid": list(LOGISTIC_C_GRID),
            "bootstrap_reps": VAL_BOOTSTRAP_REPS,
            "analysis_seed": ANALYSIS_SEED,
            "weighting": (
                "class-balanced via prompt_equal_balanced_weights: "
                "w = (n/(n_classes*n_c)) * (1/n_prompt_usable); "
                "ambiguous excluded; sklearn class_weight=None"
            ),
            "test_sealed": True,
        },
    )
    write_json(OUT / "coordinate_metrics.json", {"rows": grid_rows})
    write_json(OUT / "selected_C.json", selected_C_log)
    write_json(OUT / "bootstrap_results.json", bootstrap_log)
    write_json(
        OUT / "candidate_selection.json",
        {
            "status": status,
            "qualifying_bands": bands,
            "candidate": candidate,
            "candidate_metrics": cand_metrics,
            "rule": {
                "min_consecutive_layers_delta_gt_0": 3,
                "min_band_median_delta_auroc": 0.05,
                "choose_earliest_time_then_best_band_then_best_layer": True,
                "tie_break": "lower_layer_number",
            },
        },
    )

    svg_heatmap(
        surface_mat,
        times=TEMPORAL_LANDMARKS,
        layers=list(range(N_LAYERS)),
        title="VALIDATION SURFACE AUROC",
        path=OUT / "heatmap_surface_auroc.svg",
        vmin=0.4,
        vmax=1.0,
    )
    svg_heatmap(
        act_mat,
        times=TEMPORAL_LANDMARKS,
        layers=list(range(N_LAYERS)),
        title="VALIDATION ACTIVATION AUROC",
        path=OUT / "heatmap_activation_auroc.svg",
        vmin=0.4,
        vmax=1.0,
    )
    svg_heatmap(
        comb_mat,
        times=TEMPORAL_LANDMARKS,
        layers=list(range(N_LAYERS)),
        title="VALIDATION SURFACE+ACTIVATION AUROC",
        path=OUT / "heatmap_surface_plus_activation_auroc.svg",
        vmin=0.4,
        vmax=1.0,
    )
    svg_heatmap(
        delta_mat,
        times=TEMPORAL_LANDMARKS,
        layers=list(range(N_LAYERS)),
        title="VALIDATION ΔAUROC",
        path=OUT / "heatmap_delta_auroc.svg",
        vmin=-0.2,
        vmax=0.2,
    )

    # Summaries for report
    def _best(key):
        vals = [
            (r[key], r["time"], r["layer"])
            for r in grid_rows
            if not r.get("skipped") and np.isfinite(r.get(key, float("nan")))
        ]
        if not vals:
            return None
        return max(vals, key=lambda x: x[0])

    best_surf = _best("surface_auroc")
    best_act = _best("activation_auroc")
    best_comb = _best("surface_plus_activation_auroc")
    best_delta = _best("delta_auroc")

    cfg = f"""# Phase 24E — discovery analysis (TRAIN+VALIDATION only)

experiment_id: phase24e_discovery_analysis
phase: phase24e
status: {status}
starting_sha: {STARTING_SHA}
selected_k: 16
test_sealed: true

candidate_time: {None if candidate is None else candidate['candidate_time']}
candidate_layer: {None if candidate is None else candidate['candidate_layer']}

authorizations:
  locked_test_analysis_authorized: false
  sae_analysis_authorized: false
  causal_intervention_authorized: false
  physiology_test_open_authorized: false

notes: >
  Discovery on VALIDATION after TRAIN fit. TEST remains sealed.
"""
    CFG.write_text(cfg, encoding="utf-8")

    cand_block = "NONE"
    if candidate is not None and cand_metrics is not None:
        cand_block = json.dumps(
            {
                "candidate_time": candidate["candidate_time"],
                "candidate_layer": candidate["candidate_layer"],
                "band": candidate["band"],
                "surface_auroc": cand_metrics["surface_auroc"],
                "activation_auroc": cand_metrics["activation_auroc"],
                "surface_plus_activation_auroc": cand_metrics[
                    "surface_plus_activation_auroc"
                ],
                "delta_auroc": cand_metrics["delta_auroc"],
                "delta_auroc_bootstrap": cand_metrics["delta_auroc_bootstrap"],
            },
            indent=2,
        )

    def _metric_row(name: str, best: tuple | None) -> str:
        if not best:
            return f"| {name} | None | None | None |"
        return f"| {name} | {best[0]:.4f} | {best[1]} | {best[2]} |"

    train_row = (
        f"| TRAIN | {train_counts['honest']} | {train_counts['deceptive']} | "
        f"{train_counts['n_trajectories']} | {train_counts['n_prompts']} |"
    )
    val_row = (
        f"| VALIDATION | {val_counts['honest']} | {val_counts['deceptive']} | "
        f"{val_counts['n_trajectories']} | {val_counts['n_prompts']} |"
    )
    n_done = sum(1 for r in grid_rows if not r.get("skipped"))
    report = f"""# Phase 24E — Discovery analysis (TRAIN + VALIDATION)

**Status:** `{status}`

{GUARANTEE}

## Input verification

Verified: **{ver['verified']}**

## TEST seal

LOCKED TEST was not read for labels, activations, or scientific metrics.
Existence of sealed artifacts was verified only.

## Usable primary H/D counts

| Split | Honest | Deceptive | Trajectories | Prompts |
|---|---|---|---|---|
{train_row}
{val_row}

## Risk sets (survive landmark t)

```json
{json.dumps(risk_sets, indent=2)}
```

## Hyperparameters

C grid `{list(LOGISTIC_C_GRID)}` selected by prompt-group CV on TRAIN only.
Weighting: balanced × prompt-equal (`prompt_equal_balanced_weights`).

## Grid

224 coordinates completed: **{n_done}** /
224 evaluated rows (including skips: {len(grid_rows)}).

| Metric | Best value | time | layer |
|---|---|---|---|
{_metric_row("SURFACE AUROC", best_surf)}
{_metric_row("ACTIVATION AUROC", best_act)}
{_metric_row("SURFACE+ACTIVATION AUROC", best_comb)}
{_metric_row("ΔAUROC", best_delta)}

## Qualifying candidate bands

```json
{json.dumps(bands, indent=2)}
```

## Selected candidate

```json
{cand_block}
```

## Heatmaps

- `heatmap_surface_auroc.svg`
- `heatmap_activation_auroc.svg`
- `heatmap_surface_plus_activation_auroc.svg`
- `heatmap_delta_auroc.svg`

## Authorizations

Locked TEST analysis / SAE / causal: **false**.
"""
    # Leakage check
    low = report.lower()
    if "test h/a/d" in low or "locked test labels:" in low:
        raise SystemExit("STOP: report leaks TEST outcomes")
    REPORT.write_text(report, encoding="utf-8")

    hashes = {
        p.name: sha256_file(str(p))
        for p in [
            OUT / "hash_verification.json",
            OUT / "analysis_manifest.json",
            OUT / "coordinate_metrics.json",
            OUT / "selected_C.json",
            OUT / "bootstrap_results.json",
            OUT / "candidate_selection.json",
            OUT / "heatmap_surface_auroc.svg",
            OUT / "heatmap_activation_auroc.svg",
            OUT / "heatmap_surface_plus_activation_auroc.svg",
            OUT / "heatmap_delta_auroc.svg",
            CFG,
            REPORT,
        ]
    }
    write_json(OUT / "artifact_hashes.json", hashes)

    log = DECISION_LOG.read_text(encoding="utf-8")
    if "### D162 —" not in log:
        if candidate is None:
            decision = (
                "Phase 24E discovery on TRAIN+VALIDATION found no coordinate "
                "satisfying the frozen candidate rule; status "
                f"`{status}`. TEST remained sealed."
            )
        else:
            decision = (
                "Phase 24E discovery selected candidate "
                f"(t={candidate['candidate_time']}, L={candidate['candidate_layer']}) "
                f"with ΔAUROC={cand_metrics['delta_auroc']:.4f}; status `{status}`. "
                "TEST remained sealed; no confirmatory TEST analysis."
            )
        entry = (
            "\n### D162 — Phase 24E discovery analysis\n\n"
            "- **Type:** **OUR RESEARCH DECISION**\n"
            f"- **Decision:** {decision}\n"
            "- **Date:** 2026-09-30\n"
        )
        DECISION_LOG.write_text(log.rstrip() + "\n" + entry, encoding="utf-8")

    print(
        json.dumps(
            {
                "status": status,
                "candidate": candidate,
                "best_delta": best_delta,
                "train_counts": train_counts,
                "val_counts": val_counts,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
