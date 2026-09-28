#!/usr/bin/env python3
"""Phase-21 physiology analysis: L12 k1 probe vs text/logit baselines on held-out prompts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.metrics import (  # noqa: E402
    bootstrap_auroc_ci_by_group,
    paired_delta_auroc_ci_by_group,
)
from pre_output_physiology.phase21_roleplay import (  # noqa: E402
    BOOTSTRAP_SEED,
    GUARANTEE,
    N_BOOTSTRAP,
    PRIMARY_OFFSET,
    PROBE_C,
    PROBE_MAX_ITER,
    PROBE_SEED,
    SECONDARY_OFFSETS,
    STATUS_HOLD,
    STATUS_PASS,
)
from pre_output_physiology.probes import MeanLinearProbe, ProbeConfig  # noqa: E402
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402


def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def _auprc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def _fit_text(texts: list[str], y: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, Any]:
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, max_features=8000)
    tr_texts = [texts[i] for i in range(len(texts)) if train_mask[i]]
    vec.fit(tr_texts)
    x = vec.transform(texts).toarray().astype(np.float64)
    clf = LogisticRegression(
        C=PROBE_C, max_iter=PROBE_MAX_ITER, random_state=PROBE_SEED
    )
    clf.fit(x[train_mask], y[train_mask])
    scores = clf.decision_function(x)
    return scores, (vec, clf)


def _fit_logit(
    logits: np.ndarray, y: np.ndarray, train_mask: np.ndarray
) -> tuple[np.ndarray, Any]:
    x = np.asarray(logits, dtype=np.float64)
    scaler = StandardScaler()
    scaler.fit(x[train_mask])
    xs = scaler.transform(x)
    clf = LogisticRegression(
        C=PROBE_C, max_iter=PROBE_MAX_ITER, random_state=PROBE_SEED
    )
    clf.fit(xs[train_mask], y[train_mask])
    return clf.decision_function(xs), (scaler, clf)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-run-id", required=True)
    ap.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "artifacts/phase21a_physiology"),
    )
    args = ap.parse_args()

    extract_dir = REPO_ROOT / "artifacts/runs" / args.extract_run_id
    tens = load_file(str(extract_dir / "activations.safetensors"))
    meta = json.loads((extract_dir / "extract_meta.json").read_text(encoding="utf-8"))
    texts_npz = np.load(extract_dir / "prefix_texts.npz", allow_pickle=True)

    if not meta.get("k0_identity_ok"):
        raise SystemExit("k0 identity failed — pipeline error")

    y = np.asarray(tens["labels_deceptive"], dtype=int)
    rows = meta["rows"]
    splits = np.asarray([r["split"] for r in rows])
    prompt_ids = np.asarray([r["prompt_id"] for r in rows])
    train = splits == "train"
    test = splits == "test"

    # Availability mask for primary k1
    avail_k1 = np.asarray(
        [bool(r.get("available", {}).get("k1")) for r in rows], dtype=bool
    )
    # Pair availability: both honest and deceptive of a pair must be available
    pair_ids = [r["pair_id"] for r in rows]
    pair_ok: dict[str, bool] = {}
    for pid in set(pair_ids):
        idxs = [i for i, p in enumerate(pair_ids) if p == pid]
        pair_ok[pid] = all(avail_k1[i] for i in idxs)
    use = np.asarray([pair_ok[p] for p in pair_ids], dtype=bool)

    # k0 chance check on usable primary set
    k0 = tens["act_k0"]
    probe_k0 = MeanLinearProbe(
        ProbeConfig(C=PROBE_C, max_iter=PROBE_MAX_ITER, random_state=PROBE_SEED)
    )
    tr_u = train & use
    te_u = test & use
    if len(np.unique(y[tr_u])) < 2 or len(np.unique(y[te_u])) < 2:
        raise SystemExit("insufficient class balance after availability filter")
    probe_k0.fit(k0[tr_u], y[tr_u])
    k0_test_scores = probe_k0.predict_proba(k0[te_u])
    k0_auroc = _auroc(y[te_u], k0_test_scores)
    if abs(k0_auroc - 0.5) > 0.05:
        raise SystemExit(f"k0 AUROC not chance: {k0_auroc}")

    results: dict[str, Any] = {
        "created_at": utc_now_iso(),
        "extract_run_id": args.extract_run_id,
        "k0_identity_ok": True,
        "k0_test_auroc": k0_auroc,
        "n_train_used": int(tr_u.sum()),
        "n_test_used": int(te_u.sum()),
        "offsets": {},
    }

    # Primary + secondary offsets
    for offset in [PRIMARY_OFFSET, *SECONDARY_OFFSETS]:
        avail = np.asarray(
            [bool(r.get("available", {}).get(offset)) for r in rows], dtype=bool
        )
        pair_ok_o: dict[str, bool] = {}
        for pid in set(pair_ids):
            idxs = [i for i, p in enumerate(pair_ids) if p == pid]
            pair_ok_o[pid] = all(avail[i] for i in idxs)
        use_o = np.asarray([pair_ok_o[p] for p in pair_ids], dtype=bool)
        tr = train & use_o
        te = test & use_o
        if tr.sum() == 0 or te.sum() == 0 or len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            results["offsets"][offset] = {"status": "unavailable"}
            continue

        x_act = tens[f"act_{offset}"]
        probe = MeanLinearProbe(
            ProbeConfig(C=PROBE_C, max_iter=PROBE_MAX_ITER, random_state=PROBE_SEED)
        )
        probe.fit(x_act[tr], y[tr])
        act_scores = probe.predict_proba(x_act)

        texts = [str(x) for x in texts_npz[offset].tolist()]
        text_scores, _ = _fit_text(texts, y, tr)

        logits = tens[f"logits_{offset}"]
        # Drop rows with all-nan logits
        logit_ok = np.isfinite(logits).all(axis=1)
        tr_l = tr & logit_ok
        logit_scores = np.full(len(y), np.nan)
        if tr_l.sum() and len(np.unique(y[tr_l])) >= 2:
            ls, _ = _fit_logit(logits, y, tr_l)
            logit_scores = ls

        # Combined: stack act? No — surface combined is text+logit scores via LR on train
        comb_scores = np.full(len(y), np.nan)
        valid_comb = te if False else (tr | te)  # noqa: F841
        x_comb_tr = np.stack([text_scores[tr], logit_scores[tr]], axis=1)
        if np.isfinite(x_comb_tr).all() and len(np.unique(y[tr])) >= 2:
            comb = LogisticRegression(
                C=PROBE_C, max_iter=PROBE_MAX_ITER, random_state=PROBE_SEED
            )
            comb.fit(x_comb_tr, y[tr])
            x_all = np.stack([text_scores, logit_scores], axis=1)
            finite = np.isfinite(x_all).all(axis=1)
            comb_scores[finite] = comb.decision_function(x_all[finite])

        act_auroc = _auroc(y[te], act_scores[te])
        text_auroc = _auroc(y[te], text_scores[te])
        logit_auroc = _auroc(y[te], logit_scores[te])
        comb_auroc = _auroc(y[te], comb_scores[te])
        baselines = {
            "text": text_auroc,
            "logit": logit_auroc,
            "combined": comb_auroc,
        }
        best_name = max(
            baselines, key=lambda k: baselines[k] if not math.isnan(baselines[k]) else -1
        )
        best_base = baselines[best_name]
        delta = act_auroc - best_base

        act_ci = bootstrap_auroc_ci_by_group(
            y[te],
            act_scores[te],
            prompt_ids[te],
            n_bootstrap=N_BOOTSTRAP,
            seed=BOOTSTRAP_SEED,
        )
        base_scores = {
            "text": text_scores,
            "logit": logit_scores,
            "combined": comb_scores,
        }[best_name]
        delta_ci = paired_delta_auroc_ci_by_group(
            y[te],
            act_scores[te],
            base_scores[te],
            prompt_ids[te],
            n_bootstrap=N_BOOTSTRAP,
            seed=BOOTSTRAP_SEED,
        )

        results["offsets"][offset] = {
            "n_train": int(tr.sum()),
            "n_test": int(te.sum()),
            "activation_auroc": act_auroc,
            "activation_auprc": _auprc(y[te], act_scores[te]),
            "activation_auroc_ci": act_ci,
            "text_auroc": text_auroc,
            "logit_auroc": logit_auroc,
            "combined_baseline_auroc": comb_auroc,
            "best_baseline": best_name,
            "best_baseline_auroc": best_base,
            "delta": delta,
            "delta_ci": delta_ci,
            "probe_state": probe.export_state(),
        }

    primary = results["offsets"].get(PRIMARY_OFFSET, {})
    pass_ok = (
        primary.get("activation_auroc", 0) >= 0.65
        and primary.get("activation_auroc_ci", {}).get("auroc_ci_low", 0) > 0.55
        and primary.get("delta", -1) >= 0.03
        and primary.get("delta_ci", {}).get("delta_auroc_ci_low", -1) > 0
        and abs(k0_auroc - 0.5) <= 0.05
    )
    results["status"] = STATUS_PASS if pass_ok else STATUS_HOLD
    results["primary_offset"] = PRIMARY_OFFSET
    results["guarantee"] = GUARANTEE

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "physiology_summary.json", results)
    print(
        json.dumps(
            {
                "status": results["status"],
                "k1": {
                    k: primary.get(k)
                    for k in (
                        "activation_auroc",
                        "activation_auroc_ci",
                        "best_baseline",
                        "best_baseline_auroc",
                        "delta",
                        "delta_ci",
                    )
                },
                "k0_test_auroc": k0_auroc,
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
