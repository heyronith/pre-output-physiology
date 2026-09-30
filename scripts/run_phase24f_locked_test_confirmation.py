#!/usr/bin/env python3
"""Phase-24F: one locked-TEST confirmatory evaluation at frozen t=1 / L20.

Refits on DEVELOPMENT (TRAIN+VALIDATION) with frozen Cs; opens TEST once.
No layer/time search, no C selection, no exploratory plots.
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

from pre_output_physiology.phase24c_design import prompt_group_folds, sha256_file  # noqa: E402
from pre_output_physiology.phase24e_discovery import (  # noqa: E402
    _as_binary_labels,
    bf16_u16_to_float32,
    delta_h_vector,
    fit_predict_log_odds,
    make_logit_or_act_clf,
    oof_log_odds,
    prefix_token_ids,
    sample_weights_for_fit,
    survives_landmark,
)
from pre_output_physiology.phase24f_confirmation import (  # noqa: E402
    ANALYSIS_SEED,
    CANDIDATE_LAYER,
    CANDIDATE_TIME,
    FROZEN_C,
    GUARANTEE,
    STARTING_SHA,
    STATUS_CONFIRMED,
    STATUS_NOT_CONFIRMED,
    TEST_BOOTSTRAP_REPS,
    assert_frozen_coordinate,
    binary_classification_metrics,
    confirmatory_pass,
    delta_auroc_cluster_bootstrap,
    safe_auprc,
    safe_auroc,
    same_prompt_score_diffs,
    verify_frozen_provenance,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

OUT = REPO_ROOT / "artifacts/phase24f_confirmation"
ACT_DEV = REPO_ROOT / "artifacts/phase24e_discovery/activations_local"
ACT_TEST = OUT / "activations_local" / "LOCKED_TEST"
CFG = REPO_ROOT / "configs/experiments/phase24f_locked_test_confirmation.yaml"
REPORT = REPO_ROOT / "reports/phase24f_locked_test_confirmation.md"
SPLIT_PATH = REPO_ROOT / "artifacts/phase24c_design/split_manifest.json"
LABELS_TRAIN = REPO_ROOT / "artifacts/phase24d_collection/by_split/TRAIN/labels.json"
LABELS_VAL = REPO_ROOT / "artifacts/phase24d_collection/by_split/VALIDATION/labels.json"
LABELS_TEST = (
    REPO_ROOT / "artifacts/phase24d_collection/by_split/LOCKED_TEST/labels_SEALED.json"
)
META24D = REPO_ROOT / "artifacts/phase24d_collection/capture_meta.json"
TRAJ24B = REPO_ROOT / "artifacts/phase24b_live/trajectories.jsonl"
REUSE = REPO_ROOT / "artifacts/phase24d_collection/by_split/TRAIN/phase24b_reuse_index.json"
DECISION_LOG = REPO_ROOT / "docs/decision_log.md"


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
    with np.load(path) as z:
        return {
            "post_block_u16": z["post_block_bf16_u16"],
            "logits_f16": z["logits_f16"],
            "generated_token_ids": z["generated_token_ids"],
        }


def _label_rows(path: Path) -> list[dict]:
    blob = json.loads(path.read_text(encoding="utf-8"))
    return list(blob["rows"])


def _build_dev_records(tok) -> list[dict]:
    """DEVELOPMENT = TRAIN + VALIDATION only (never TEST)."""
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    train_set = set(split["train_prompt_ids"])
    val_set = set(split["validation_prompt_ids"])
    test_set = set(split["test_prompt_ids"])

    labels: dict[str, dict] = {}
    for path, allowed in ((LABELS_TRAIN, train_set), (LABELS_VAL, val_set)):
        for r in _label_rows(path):
            if r["prompt_id"] in test_set:
                raise RuntimeError("TEST prompt leaked into DEVELOPMENT labels")
            if r["prompt_id"] not in allowed:
                raise RuntimeError(f"prompt {r['prompt_id']} wrong split")
            if r.get("open_valid") and r.get("open_class") in ("honest", "deceptive"):
                labels[r["trajectory_id"]] = {
                    "label": r["open_class"],
                    "prompt_id": r["prompt_id"],
                }

    meta = json.loads(META24D.read_text(encoding="utf-8"))
    meta_by = {
        m["trajectory_id"]: m
        for m in meta["meta_rows"]
        if m.get("completed") and m["split"] in ("train", "validation")
    }
    reuse_ids = set(json.loads(REUSE.read_text(encoding="utf-8"))["trajectory_ids"])
    traj24b = {
        json.loads(x)["trajectory_id"]: json.loads(x)
        for x in TRAJ24B.read_text(encoding="utf-8").splitlines()
        if x.strip()
    }

    records = []
    for tid, lab in labels.items():
        pid = lab["prompt_id"]
        if pid in train_set:
            folder = "TRAIN"
            split_name = "train"
        elif pid in val_set:
            folder = "VALIDATION"
            split_name = "validation"
        else:
            raise RuntimeError(pid)
        npz_path = ACT_DEV / folder / f"{tid}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(npz_path)
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
        records.append(
            {
                "trajectory_id": tid,
                "prompt_id": pid,
                "split": split_name,
                "label": lab["label"],
                "generated_token_ids": gen_ids,
                "prompt_text": prompt_text,
                "n_generated": len(gen_ids),
                "npz_path": str(npz_path),
            }
        )
    return records


def _build_test_records(tok) -> tuple[list[dict], dict[str, Any]]:
    """Open LOCKED TEST labels + activations exactly once for confirmation."""
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    test_set = set(split["test_prompt_ids"])
    if len(test_set) != 11:
        raise RuntimeError("expected 11 TEST prompts")

    labels_blob = json.loads(LABELS_TEST.read_text(encoding="utf-8"))
    rows = labels_blob["rows"]

    # Behavioral summary (all classes including ambiguous)
    by_prompt: dict[str, Counter] = {p: Counter() for p in test_set}
    totals = Counter()
    for r in rows:
        pid = r["prompt_id"]
        if pid not in test_set:
            raise RuntimeError(f"non-TEST prompt in sealed labels: {pid}")
        cls = r.get("open_class") if r.get("open_valid") else "invalid"
        if cls not in ("honest", "ambiguous", "deceptive"):
            cls = "invalid"
        by_prompt[pid][cls] += 1
        totals[cls] += 1

    n_mixed_1 = sum(
        1 for p, c in by_prompt.items() if c["honest"] >= 1 and c["deceptive"] >= 1
    )
    n_mixed_2 = sum(
        1 for p, c in by_prompt.items() if c["honest"] >= 2 and c["deceptive"] >= 2
    )
    behavioral = {
        "H_A_D_totals": {
            "honest": int(totals["honest"]),
            "ambiguous": int(totals["ambiguous"]),
            "deceptive": int(totals["deceptive"]),
            "invalid": int(totals["invalid"]),
            "n_trajectories": len(rows),
            "n_prompts": len(test_set),
        },
        "per_prompt_H_A_D": {
            p: {
                "honest": int(c["honest"]),
                "ambiguous": int(c["ambiguous"]),
                "deceptive": int(c["deceptive"]),
            }
            for p, c in sorted(by_prompt.items())
        },
        "n_prompts_ge1H_ge1D": n_mixed_1,
        "n_prompts_ge2H_ge2D": n_mixed_2,
    }

    meta = json.loads(META24D.read_text(encoding="utf-8"))
    meta_by = {
        m["trajectory_id"]: m
        for m in meta["meta_rows"]
        if m.get("completed") and m["split"] == "test"
    }

    primary_labels = {}
    for r in rows:
        if r.get("open_valid") and r.get("open_class") in ("honest", "deceptive"):
            primary_labels[r["trajectory_id"]] = {
                "label": r["open_class"],
                "prompt_id": r["prompt_id"],
            }

    records = []
    for tid, lab in primary_labels.items():
        m = meta_by[tid]
        npz_path = ACT_TEST / f"{tid}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(npz_path)
        gen_ids = list(m["generated_token_ids"])
        prompt_text = tok.decode(m["prompt_token_ids"], skip_special_tokens=False)
        records.append(
            {
                "trajectory_id": tid,
                "prompt_id": lab["prompt_id"],
                "split": "test",
                "label": lab["label"],
                "generated_token_ids": gen_ids,
                "prompt_text": prompt_text,
                "n_generated": len(gen_ids),
                "npz_path": str(npz_path),
            }
        )
    behavioral["usable_H_D"] = {
        "honest": sum(1 for r in records if r["label"] == "honest"),
        "deceptive": sum(1 for r in records if r["label"] == "deceptive"),
        "n_trajectories": len(records),
        "n_prompts": len({r["prompt_id"] for r in records}),
    }
    return records, behavioral


def _features_at_candidate(
    recs: list[dict], cache: dict[str, dict], tok, *, t: int, layer: int
) -> tuple[list[str], np.ndarray, np.ndarray, list[str], list[str], np.ndarray]:
    assert_frozen_coordinate(t, layer)
    texts: list[str] = []
    logits: list[np.ndarray] = []
    acts: list[np.ndarray] = []
    labels: list[str] = []
    prompts: list[str] = []
    for r in recs:
        if not survives_landmark(r["n_generated"], t):
            continue
        arr = cache[r["trajectory_id"]]
        logit_row = arr["logits_f16"][t].astype(np.float32)
        text = _decode_prefix(tok, r["prompt_text"], r["generated_token_ids"], t)
        assert len(prefix_token_ids(r["generated_token_ids"], t)) == t
        post = bf16_u16_to_float32(arr["post_block_u16"])
        dh = delta_h_vector(post[t][layer], post[0][layer])
        texts.append(text)
        logits.append(logit_row)
        acts.append(dh)
        labels.append(r["label"])
        prompts.append(r["prompt_id"])
    y = _as_binary_labels(labels)
    return texts, np.stack(logits), np.stack(acts), labels, prompts, y


def _oof_meta_log_odds(
    Z_oof: np.ndarray,
    y: np.ndarray,
    labels: list[str],
    prompts: list[str],
    C: float,
) -> np.ndarray:
    """Nested OOF meta log-odds on DEVELOPMENT for stacking."""
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


def _fit_meta_predict(
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
    Zev = sc.transform(Z_eval)
    return np.clip(clf.predict_proba(Zev)[:, 1], 1e-6, 1 - 1e-6)


def main() -> int:
    assert_frozen_coordinate(CANDIDATE_TIME, CANDIDATE_LAYER)
    t = CANDIDATE_TIME
    layer = CANDIDATE_LAYER

    print("Verifying frozen provenance…", flush=True)
    ver = verify_frozen_provenance(REPO_ROOT)
    write_json(OUT / "hash_verification.json", ver)
    if not ver["verified"]:
        raise SystemExit(f"STOP: provenance mismatch\n{json.dumps(ver, indent=2)}")

    if not (ACT_DEV / "TRAIN").exists() or not (ACT_DEV / "VALIDATION").exists():
        raise SystemExit("STOP: DEVELOPMENT activations missing")
    if not ACT_TEST.exists():
        raise SystemExit(
            "STOP: TEST activations missing — run "
            "modal/phase24f_fetch_test_activations.py first"
        )

    print("Loading tokenizer…", flush=True)
    tok = _load_tokenizer()
    print("Building DEVELOPMENT records…", flush=True)
    dev = _build_dev_records(tok)
    print(f"DEVELOPMENT primary H/D trajectories: {len(dev)}", flush=True)

    print("Opening LOCKED TEST once…", flush=True)
    test, behavioral = _build_test_records(tok)
    write_json(OUT / "test_behavioral_summary.json", behavioral)
    print(
        f"TEST usable H/D={behavioral['usable_H_D']} "
        f"totals={behavioral['H_A_D_totals']}",
        flush=True,
    )

    # Risk set at t=1
    dev_rs = [r for r in dev if survives_landmark(r["n_generated"], t)]
    test_rs = [r for r in test if survives_landmark(r["n_generated"], t)]
    risk = {
        "time": t,
        "development": len(dev_rs),
        "test": len(test_rs),
        "test_honest": sum(1 for r in test_rs if r["label"] == "honest"),
        "test_deceptive": sum(1 for r in test_rs if r["label"] == "deceptive"),
    }
    print(f"t=1 risk-set: {risk}", flush=True)

    print("Loading NPZs…", flush=True)
    cache: dict[str, dict] = {}
    for r in dev_rs + test_rs:
        cache[r["trajectory_id"]] = _load_npz(Path(r["npz_path"]))

    (
        dev_text,
        dev_logits,
        dev_act,
        dev_lab,
        dev_pid,
        y_dev,
    ) = _features_at_candidate(dev_rs, cache, tok, t=t, layer=layer)
    (
        te_text,
        te_logits,
        te_act,
        te_lab,
        te_pid,
        y_te,
    ) = _features_at_candidate(test_rs, cache, tok, t=t, layer=layer)

    # --- Frozen predictors: DEVELOPMENT fit only ---
    C = FROZEN_C
    # TEXT / LOGITS OOF on DEVELOPMENT for SURFACE stacking
    oof_text = oof_log_odds(
        X=dev_text,
        y=y_dev,
        prompt_ids=dev_pid,
        labels_str=dev_lab,
        kind="text",
        C=C["text"],
        seed=ANALYSIS_SEED,
    )
    oof_logit = oof_log_odds(
        X=dev_logits,
        y=y_dev,
        prompt_ids=dev_pid,
        labels_str=dev_lab,
        kind="matrix",
        C=C["logits"],
        seed=ANALYSIS_SEED,
    )
    Z_surf_oof = np.stack([oof_text, oof_logit], axis=1)

    # Full-fit TEXT/LOGITS → TEST log-odds
    lo_text_te = fit_predict_log_odds(
        X_train=dev_text,
        y_train=y_dev,
        labels_train=dev_lab,
        prompts_train=dev_pid,
        X_eval=te_text,
        kind="text",
        C=C["text"],
        seed=ANALYSIS_SEED,
    )
    lo_logit_te = fit_predict_log_odds(
        X_train=dev_logits,
        y_train=y_dev,
        labels_train=dev_lab,
        prompts_train=dev_pid,
        X_eval=te_logits,
        kind="matrix",
        C=C["logits"],
        seed=ANALYSIS_SEED,
    )
    surf_proba_te = _fit_meta_predict(
        Z_surf_oof,
        y_dev,
        dev_lab,
        dev_pid,
        np.stack([lo_text_te, lo_logit_te], axis=1),
        C["surface"],
    )
    lo_surf_te = np.log(surf_proba_te / (1 - surf_proba_te))

    # ACTIVATION
    oof_act = oof_log_odds(
        X=dev_act,
        y=y_dev,
        prompt_ids=dev_pid,
        labels_str=dev_lab,
        kind="matrix",
        C=C["activation"],
        seed=ANALYSIS_SEED,
    )
    lo_act_te = fit_predict_log_odds(
        X_train=dev_act,
        y_train=y_dev,
        labels_train=dev_lab,
        prompts_train=dev_pid,
        X_eval=te_act,
        kind="matrix",
        C=C["activation"],
        seed=ANALYSIS_SEED,
    )
    act_proba_te = 1 / (1 + np.exp(-lo_act_te))

    # SURFACE+ACTIVATION: OOF SURFACE meta + OOF activation
    oof_surf = _oof_meta_log_odds(
        Z_surf_oof, y_dev, dev_lab, dev_pid, C["surface"]
    )
    Z_comb_oof = np.stack([oof_surf, oof_act], axis=1)
    comb_proba_te = _fit_meta_predict(
        Z_comb_oof,
        y_dev,
        dev_lab,
        dev_pid,
        np.stack([lo_surf_te, lo_act_te], axis=1),
        C["surface_plus_activation"],
    )

    # Primary metrics
    surf_auroc = safe_auroc(y_te, surf_proba_te)
    act_auroc = safe_auroc(y_te, act_proba_te)
    comb_auroc = safe_auroc(y_te, comb_proba_te)
    delta = comb_auroc - surf_auroc
    surf_auprc = safe_auprc(y_te, surf_proba_te)
    act_auprc = safe_auprc(y_te, act_proba_te)
    comb_auprc = safe_auprc(y_te, comb_proba_te)

    surf_cls = binary_classification_metrics(y_te, surf_proba_te)
    act_cls = binary_classification_metrics(y_te, act_proba_te)
    comb_cls = binary_classification_metrics(y_te, comb_proba_te)

    print("Running 10,000 prompt-cluster bootstrap…", flush=True)
    boot = delta_auroc_cluster_bootstrap(
        y=y_te,
        surface_scores=surf_proba_te,
        combined_scores=comb_proba_te,
        prompt_ids=te_pid,
        n_reps=TEST_BOOTSTRAP_REPS,
        seed=ANALYSIS_SEED,
    )
    ci_low, ci_high = boot["ci95"]
    passed = confirmatory_pass(delta_auroc=delta, ci95_low=ci_low)
    status = STATUS_CONFIRMED if passed else STATUS_NOT_CONFIRMED

    secondary = same_prompt_score_diffs(
        y=y_te, scores=comb_proba_te, prompt_ids=te_pid, min_h=2, min_d=2
    )

    primary = {
        "coordinate": {"time": t, "layer": layer},
        "frozen_C": C,
        "n_test_riskset": len(y_te),
        "n_dev_riskset": len(y_dev),
        "surface_auroc": surf_auroc,
        "activation_auroc": act_auroc,
        "surface_plus_activation_auroc": comb_auroc,
        "delta_auroc": delta,
        "surface_auprc": surf_auprc,
        "activation_auprc": act_auprc,
        "surface_plus_activation_auprc": comb_auprc,
        "surface_cls": surf_cls,
        "activation_cls": act_cls,
        "surface_plus_activation_cls": comb_cls,
        "delta_auroc_bootstrap": boot,
        "primary_pass": passed,
        "status": status,
    }
    write_json(OUT / "primary_metrics.json", primary)
    write_json(OUT / "bootstrap_10000.json", boot)
    write_json(OUT / "same_prompt_secondary.json", secondary)

    preprocessing_meta = {
        "fit_split": "DEVELOPMENT=TRAIN+VALIDATION",
        "test_excluded_from_fit": True,
        "candidate_time": t,
        "candidate_layer": layer,
        "frozen_C": C,
        "weighting": (
            "class-balanced via prompt_equal_balanced_weights; "
            "sklearn class_weight=None"
        ),
        "oof_stacking": True,
        "analysis_seed": ANALYSIS_SEED,
        "no_C_selection": True,
        "no_coordinate_search": True,
    }
    write_json(OUT / "preprocessing_metadata.json", preprocessing_meta)

    git_commit = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
        .decode()
        .strip()
    )
    manifest = {
        "created_at": utc_now_iso(),
        "starting_sha": STARTING_SHA,
        "git_commit": git_commit,
        "status": status,
        "primary_pass": passed,
        "candidate_time": t,
        "candidate_layer": layer,
        "frozen_C": C,
        "bootstrap_reps": TEST_BOOTSTRAP_REPS,
        "risk_set_t1": risk,
        "test_behavioral": behavioral,
        "guarantee": GUARANTEE,
        "provenance_verified": True,
    }
    write_json(OUT / "confirmation_manifest.json", manifest)

    # Config + report
    cfg = f"""# Phase 24F — locked TEST confirmation (single coordinate)

experiment_id: phase24f_locked_test_confirmation
phase: phase24f
status: {status}
starting_sha: {STARTING_SHA}
candidate_time: {t}
candidate_layer: {layer}
primary_pass: {str(passed).lower()}

frozen_C:
  text: {C['text']}
  logits: {C['logits']}
  surface: {C['surface']}
  activation: {C['activation']}
  surface_plus_activation: {C['surface_plus_activation']}

authorizations:
  locked_test_confirmation_authorized: true
  sae_analysis_authorized: false
  causal_intervention_authorized: false
  test_layer_time_search_authorized: false

notes: >
  One prospectively frozen confirmatory evaluation on LOCKED TEST at t=1/L20.
  No retuning, no alternate coordinates, no SAE/causal analyses.
"""
    CFG.parent.mkdir(parents=True, exist_ok=True)
    CFG.write_text(cfg, encoding="utf-8")

    report = f"""# Phase 24F — Locked TEST confirmation

**Status:** `{status}`

{GUARANTEE}

## Provenance

Verified before opening TEST: **{ver['verified']}**

Candidate: t={t}, L={layer}

Frozen C: `{json.dumps(C)}`

## TEST behavioral summary

H/A/D totals: `{json.dumps(behavioral['H_A_D_totals'])}`

Usable H/D: `{json.dumps(behavioral['usable_H_D'])}`

≥1H+≥1D prompts: **{behavioral['n_prompts_ge1H_ge1D']}**

≥2H+≥2D prompts: **{behavioral['n_prompts_ge2H_ge2D']}**

t=1 risk-set: `{json.dumps(risk)}`

## Primary confirmatory metrics (t=1, L20)

| Model | AUROC | AUPRC |
|---|---|---|
| SURFACE | {surf_auroc:.6f} | {surf_auprc:.6f} |
| ACTIVATION | {act_auroc:.6f} | {act_auprc:.6f} |
| SURFACE+ACTIVATION | {comb_auroc:.6f} | {comb_auprc:.6f} |
| ΔAUROC | {delta:.6f} | — |

ΔAUROC 95% prompt-cluster bootstrap CI ({TEST_BOOTSTRAP_REPS} reps):
**[{ci_low:.6f}, {ci_high:.6f}]**

Primary criterion (ΔAUROC>0 AND CI_low>0): **{'PASS' if passed else 'FAIL'}**

### Classification (threshold 0.5)

SURFACE: `{json.dumps(surf_cls)}`

ACTIVATION: `{json.dumps(act_cls)}`

SURFACE+ACTIVATION: `{json.dumps(comb_cls)}`

## Secondary same-prompt (≥2H+≥2D)

`{json.dumps(secondary, indent=2)}`

## Interpretation

{"Confirmed: activations at the frozen coordinate add held-out "
     "predictive signal beyond SURFACE."
     if passed
     else
     "Not confirmed: validation-discovered activation advantage did not "
     "meet the frozen held-out standard."}

No TEST layer×time search, SAE, or causal analyses were performed.
"""
    REPORT.write_text(report, encoding="utf-8")

    # Hash outputs
    hash_targets = {
        "confirmation_manifest.json": OUT / "confirmation_manifest.json",
        "primary_metrics.json": OUT / "primary_metrics.json",
        "bootstrap_10000.json": OUT / "bootstrap_10000.json",
        "same_prompt_secondary.json": OUT / "same_prompt_secondary.json",
        "test_behavioral_summary.json": OUT / "test_behavioral_summary.json",
        "preprocessing_metadata.json": OUT / "preprocessing_metadata.json",
        "hash_verification.json": OUT / "hash_verification.json",
        "phase24f_locked_test_confirmation.yaml": CFG,
        "phase24f_locked_test_confirmation.md": REPORT,
    }
    hashes = {k: sha256_file(str(p)) for k, p in hash_targets.items()}
    write_json(OUT / "artifact_hashes.json", hashes)

    # Decision log append
    dlog = DECISION_LOG.read_text(encoding="utf-8")
    if "### D163 — Phase 24F" not in dlog:
        outcome = "PASS" if passed else "FAIL"
        entry = (
            "\n### D163 — Phase 24F locked-TEST confirmation\n\n"
            "- **Type:** **OUR RESEARCH DECISION**\n"
            f"- **Decision:** Starting from Phase-24E freeze "
            f"`{STARTING_SHA[:7]}…`, Phase 24F performed the single "
            "prospectively frozen confirmatory analysis on LOCKED TEST "
            f"at t=1/L20 (DEVELOPMENT refit; frozen Cs; 10k prompt-cluster "
            f"bootstrap). Primary criterion {outcome} "
            f"(ΔAUROC={delta:.4f}; CI95=[{ci_low:.4f}, {ci_high:.4f}]). "
            f"Status `{status}`. No TEST layer/time search, hyperparameter "
            "retuning, SAE, or causal intervention.\n"
            "- **Date:** 2026-09-30\n"
        )
        DECISION_LOG.write_text(dlog.rstrip() + "\n" + entry, encoding="utf-8")

    summary = {
        "status": status,
        "primary_pass": passed,
        "delta": delta,
        "ci95": boot["ci95"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
