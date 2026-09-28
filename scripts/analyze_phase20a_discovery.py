#!/usr/bin/env python3
"""Phase 20A2 discovery analysis: LOFO probes, baselines, freeze candidate.

Train balanced subset only. No validation activations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from safetensors.numpy import load_file

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase14_design import (  # noqa: E402
    CONSIDERATION_PREFIX,
    RESPONSE_CONTINUATION,
)
from pre_output_physiology.phase15_onset import LABEL_ALTERNATE, LABEL_RECORD  # noqa: E402
from pre_output_physiology.phase20_physiology import (  # noqa: E402
    BLOCKS,
    DISCOVERY_MIN_AUROC,
    DISCOVERY_MIN_DELTA,
    GUARANTEE,
    K0_AUROC_TOL,
    POSITIONS,
    PROBE_C,
    PROBE_MAX_ITER,
    PROBE_SEED,
    STATUS_DISC_HOLD,
    STATUS_DISC_PASS,
    TRAIN_FAMILIES,
    _sha_json,
    lofo_folds,
    select_discovery_candidate,
)
from pre_output_physiology.phase5_probes import ScaledLogisticProbe  # noqa: E402
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402
from pre_output_physiology.phase8_design import format_chat  # noqa: E402


def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def _fit_lr(x: np.ndarray, y: np.ndarray) -> ScaledLogisticProbe:
    p = ScaledLogisticProbe(C=PROBE_C, max_iter=PROBE_MAX_ITER, seed=PROBE_SEED)
    p.fit(x, y)
    return p


def _pooled_oof_auroc(
    x: np.ndarray,
    y: np.ndarray,
    families: list[str],
) -> dict[str, Any]:
    """Leave-one-family-out; return pooled OOF AUROC + per-family."""
    oof = np.full(len(y), np.nan)
    per_fam = {}
    for fold in lofo_folds():
        held = fold["held_out_family"]
        tr = np.array([f != held for f in families])
        te = np.array([f == held for f in families])
        if tr.sum() == 0 or te.sum() == 0:
            continue
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            per_fam[held] = None
            continue
        probe = _fit_lr(x[tr], y[tr])
        scores = probe.decision_scores(x[te])
        oof[te] = scores
        per_fam[held] = _auroc(y[te], scores)
    valid = np.isfinite(oof)
    pooled = _auroc(y[valid], oof[valid]) if valid.any() else float("nan")
    return {"pooled_auroc": pooled, "per_family_auroc": per_fam, "oof_scores": oof}


def _consideration_prefix_text(full_text: str, body_ids: list[int], off: int, tok) -> str:
    """Approximate text for truncated Stage-1 body (decode body[:-off])."""
    if off == 0:
        return full_text
    body = list(body_ids)
    if len(body) <= off:
        return ""
    return tok.decode(body[:-off], skip_special_tokens=True)


def build_text_features(
    texts: list[str], train_mask: np.ndarray
) -> tuple[np.ndarray, TfidfVectorizer]:
    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=5000,
    )
    # Fit word TF-IDF on train only; also char ngrams via second vectorizer concat
    vec_c = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=1,
        max_features=5000,
    )
    tr_texts = [texts[i] for i in range(len(texts)) if train_mask[i]]
    vec.fit(tr_texts)
    vec_c.fit(tr_texts)
    w = vec.transform(texts).toarray()
    c = vec_c.transform(texts).toarray()
    return np.hstack([w, c]).astype(np.float64), (vec, vec_c)


def decoder_features_at_position(
    rows_cont: list[dict],
    tok,
    model_logits_fn,
    position: str,
) -> np.ndarray:
    """B2 features: [seq_lp_margin, first_div_margin, constrained_pred].

    model_logits_fn(ids)-> logits np array [V] at last position.
    Computed offline via teacher-forcing with a small local model OR
    precomputed fields. Here we use stored constrained outcome +
    approximate margins from constrained_steps if present; otherwise
    recompute with provided fn.
    """
    # Prefer recomputation through fn when available; for discovery analysis
    # we recompute using transformers locally if GPU unavailable — skip heavy
    # path and use features from continuations when possible.
    feats = []
    off = {"end0": 0, "end2": 2, "end4": 4}[position]
    for r in rows_cont:
        # Sequence logprob margin proxy: use final constrained step logit gap if any
        steps = r.get("constrained_steps") or []
        if steps:
            last = steps[-1]
            # chosen vs other
            scores = last.get("candidate_scores") or last.get("scores")
            if scores and len(scores) == 2:
                margin = float(scores[0] - scores[1])
            else:
                margin = 0.0
            first_div = float(steps[0].get("logit_margin", margin)) if steps else margin
        else:
            margin = 0.0
            first_div = 0.0
        pred = 1.0 if r.get("chosen_state") == r.get("alternate_state") else 0.0
        # Adjust for truncation position by scaling body length (descriptive proxy)
        body_n = int(r.get("stage1_body_n_tokens", 0))
        scale = max(body_n - off, 1) / max(body_n, 1)
        feats.append([margin * scale, first_div * scale, pred])
    return np.asarray(feats, dtype=np.float64)


def gen_stats_features(rows_cont: list[dict], position: str) -> np.ndarray:
    off = {"end0": 0, "end2": 2, "end4": 4}[position]
    feats = []
    for r in rows_cont:
        lps = list(r.get("stage1_token_logprobs") or [])
        ents = list(r.get("stage1_token_entropies") or [])
        # Align to body length used for Response when possible
        n_body = int(r.get("stage1_body_n_tokens") or len(lps))
        n = max(n_body - off, 1)
        # Use first n of generation stats (proxy if body truncated differently)
        use_lp = lps[:n] if lps else [0.0]
        use_ent = ents[:n] if ents else [0.0]
        feats.append(
            [
                float(n),
                float(np.mean(use_lp)),
                float(use_lp[-1]),
                float(np.mean(use_ent)),
                float(use_ent[-1]),
            ]
        )
    return np.asarray(feats, dtype=np.float64)


def _oof_baseline(
    x: np.ndarray, y: np.ndarray, families: list[str]
) -> dict[str, Any]:
    return _pooled_oof_auroc(x, y, families)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-run-dir", required=True)
    ap.add_argument(
        "--population-summary",
        default=str(REPO_ROOT / "artifacts/phase20a1_population/population_summary.json"),
    )
    ap.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "artifacts/phase20a2_discovery"),
    )
    args = ap.parse_args()

    extract_dir = Path(args.extract_run_dir)
    tens = load_file(str(extract_dir / "discovery_activations.safetensors"))
    meta = json.loads((extract_dir / "discovery_meta.json").read_text("utf-8"))
    pop = json.loads(Path(args.population_summary).read_text("utf-8"))
    gen_run = pop["run_id"]
    conts = {
        json.loads(x)["continuation_id"]: json.loads(x)
        for x in (REPO_ROOT / "artifacts/runs" / gen_run / "continuations.jsonl")
        .read_text("utf-8")
        .splitlines()
        if x.strip()
    }

    y = np.asarray(tens["labels_alternate"], dtype=int)
    families = list(meta["families"])
    cont_ids = list(meta["continuation_ids"])
    rows_cont = [conts[cid] for cid in cont_ids]
    layers = [int(x) for x in tens["layers"]]
    layer_index = {b: i for i, b in enumerate(layers)}

    # k0 AUROC must be ~0.50
    k0_aurocs = {}
    for b in BLOCKS:
        j = layer_index[b]
        x = tens["activations_k0"][:, j, :]
        res = _pooled_oof_auroc(x, y, families)
        k0_aurocs[b] = res["pooled_auroc"]
    k0_ok = all(
        abs(v - 0.5) <= K0_AUROC_TOL for v in k0_aurocs.values() if not math.isnan(v)
    )
    if not meta.get("k0_identity_ok"):
        raise SystemExit("k0 identity failed in extract meta")
    if not k0_ok:
        raise SystemExit(f"k0 AUROC not chance: {k0_aurocs}")

    # Texts for B1
    texts_by_pos = {
        pos: [
            r["consideration_text"]
            if pos == "end0"
            else (
                # decode truncated body
                ""  # filled below
            )
            for r in rows_cont
        ]
        for pos in POSITIONS
    }
    # Fill truncated texts via body ids decode without tokenizer dependency:
    # use consideration_text for all positions as conservative surface baseline
    # (same text across positions is OK descriptively; B3 still position-aware).
    for pos in POSITIONS:
        texts_by_pos[pos] = [r["consideration_text"] for r in rows_cont]

    discovery_grid = []
    best_by_pos: dict[str, dict] = {}

    for pos in POSITIONS:
        x_b3 = gen_stats_features(rows_cont, pos)
        x_b2 = decoder_features_at_position(rows_cont, None, None, pos)
        # B1 via LOFO-aware TF-IDF: fit inside each fold
        b1_oof = np.full(len(y), np.nan)
        for fold in lofo_folds():
            held = fold["held_out_family"]
            tr = np.array([f != held for f in families])
            te = np.array([f == held for f in families])
            x_b1, _ = build_text_features(texts_by_pos[pos], tr)
            if len(np.unique(y[tr])) < 2:
                continue
            probe = _fit_lr(x_b1[tr], y[tr])
            b1_oof[te] = probe.decision_scores(x_b1[te])
        b1_auroc = _auroc(y[np.isfinite(b1_oof)], b1_oof[np.isfinite(b1_oof)])

        b2_res = _oof_baseline(x_b2, y, families)
        b3_res = _oof_baseline(x_b3, y, families)

        # B4: concat B1 features (refit foldwise) with B2+B3
        b4_oof = np.full(len(y), np.nan)
        for fold in lofo_folds():
            held = fold["held_out_family"]
            tr = np.array([f != held for f in families])
            te = np.array([f == held for f in families])
            x_b1, _ = build_text_features(texts_by_pos[pos], tr)
            x_tr = np.hstack([x_b1[tr], x_b2[tr], x_b3[tr]])
            x_te = np.hstack([x_b1[te], x_b2[te], x_b3[te]])
            if len(np.unique(y[tr])) < 2:
                continue
            probe = _fit_lr(x_tr, y[tr])
            b4_oof[te] = probe.decision_scores(x_te)
        b4_auroc = _auroc(y[np.isfinite(b4_oof)], b4_oof[np.isfinite(b4_oof)])

        baseline_table = {
            "B1": b1_auroc,
            "B2": b2_res["pooled_auroc"],
            "B3": b3_res["pooled_auroc"],
            "B4": b4_auroc,
        }
        best_type = max(baseline_table, key=lambda k: baseline_table[k])
        best_auroc = baseline_table[best_type]
        best_by_pos[pos] = {
            "best_baseline_type": best_type,
            "best_baseline_auroc": best_auroc,
            "baselines": baseline_table,
            "b4_oof": b4_oof,
            "b1_oof": b1_oof,
            "b2_oof": b2_res["oof_scores"],
            "b3_oof": b3_res["oof_scores"],
        }

        for b in BLOCKS:
            j = layer_index[b]
            x = tens[f"activations_{pos}"][:, j, :]
            act_res = _pooled_oof_auroc(x, y, families)
            discovery_grid.append(
                {
                    "block": b,
                    "position": pos,
                    "act_auroc": act_res["pooled_auroc"],
                    "act_per_family": act_res["per_family_auroc"],
                    "best_baseline_type": best_type,
                    "best_baseline_auroc": best_auroc,
                    "delta": act_res["pooled_auroc"] - best_auroc,
                    "baselines": baseline_table,
                    "oof_act": act_res["oof_scores"],
                }
            )

    chosen = select_discovery_candidate(
        [
            {
                "block": g["block"],
                "position": g["position"],
                "act_auroc": g["act_auroc"],
                "best_baseline_auroc": g["best_baseline_auroc"],
                "best_baseline_type": g["best_baseline_type"],
            }
            for g in discovery_grid
            if not math.isnan(g["act_auroc"])
        ]
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if chosen is None:
        status = STATUS_DISC_HOLD
        summary = {
            "created_at": utc_now_iso(),
            "status": status,
            "k0_aurocs": k0_aurocs,
            "k0_ok": k0_ok,
            "discovery_grid": [
                {k: v for k, v in g.items() if k != "oof_act"} for g in discovery_grid
            ],
            "selected": None,
            "guarantee": GUARANTEE,
        }
        write_json(out / "discovery_summary.json", summary)
        print(json.dumps({"status": status, "n_grid": len(discovery_grid)}, indent=2))
        return 0

    # Recover OOF for chosen
    g_chosen = next(
        g
        for g in discovery_grid
        if g["block"] == chosen["block"] and g["position"] == chosen["position"]
    )
    pos = chosen["position"]
    base_type = chosen["best_baseline_type"]
    oof_act = g_chosen["oof_act"]
    oof_base = best_by_pos[pos][
        {"B1": "b1_oof", "B2": "b2_oof", "B3": "b3_oof", "B4": "b4_oof"}[base_type]
    ]

    # Combiner on discovery OOF only
    valid = np.isfinite(oof_act) & np.isfinite(oof_base)
    x_comb = np.stack([oof_act[valid], oof_base[valid]], axis=1)
    combiner = LogisticRegression(
        C=PROBE_C, max_iter=PROBE_MAX_ITER, random_state=PROBE_SEED
    )
    combiner.fit(x_comb, y[valid])
    combiner_export = {
        "coef": combiner.coef_.ravel().tolist(),
        "intercept": float(combiner.intercept_[0]),
        "C": PROBE_C,
    }

    # Refit activation probe + matched baseline on ALL training prompts
    j = layer_index[chosen["block"]]
    x_act_all = tens[f"activations_{pos}"][:, j, :]
    act_probe = _fit_lr(x_act_all, y)
    act_export = {k: v.tolist() for k, v in act_probe.export_npz_arrays().items()}

    # Refit baseline on all train
    tr_all = np.ones(len(y), dtype=bool)
    if base_type == "B1":
        x_b1, vecs = build_text_features(texts_by_pos[pos], tr_all)
        base_probe = _fit_lr(x_b1, y)
        base_export = {
            "type": "B1",
            **{k: v.tolist() for k, v in base_probe.export_npz_arrays().items()},
        }
    elif base_type == "B2":
        x = decoder_features_at_position(rows_cont, None, None, pos)
        base_probe = _fit_lr(x, y)
        base_export = {
            "type": "B2",
            **{k: v.tolist() for k, v in base_probe.export_npz_arrays().items()},
        }
    elif base_type == "B3":
        x = gen_stats_features(rows_cont, pos)
        base_probe = _fit_lr(x, y)
        base_export = {
            "type": "B3",
            **{k: v.tolist() for k, v in base_probe.export_npz_arrays().items()},
        }
    else:
        x_b1, _ = build_text_features(texts_by_pos[pos], tr_all)
        x_b2 = decoder_features_at_position(rows_cont, None, None, pos)
        x_b3 = gen_stats_features(rows_cont, pos)
        x = np.hstack([x_b1, x_b2, x_b3])
        base_probe = _fit_lr(x, y)
        base_export = {
            "type": "B4",
            **{k: v.tolist() for k, v in base_probe.export_npz_arrays().items()},
        }

    status = STATUS_DISC_PASS
    frozen = {
        "block": chosen["block"],
        "position": chosen["position"],
        "act_auroc_discovery": chosen["act_auroc"],
        "best_baseline_type": base_type,
        "best_baseline_auroc": chosen["best_baseline_auroc"],
        "delta": chosen["act_auroc"] - chosen["best_baseline_auroc"],
        "tie_break_path": chosen["tie_break_path"],
        "activation_probe": act_export,
        "baseline": base_export,
        "combiner": combiner_export,
        "activation_probe_sha256": hashlib.sha256(
            json.dumps(act_export, sort_keys=True).encode()
        ).hexdigest(),
        "baseline_sha256": hashlib.sha256(
            json.dumps(base_export, sort_keys=True).encode()
        ).hexdigest(),
        "combiner_sha256": hashlib.sha256(
            json.dumps(combiner_export, sort_keys=True).encode()
        ).hexdigest(),
    }
    summary = {
        "created_at": utc_now_iso(),
        "status": status,
        "extract_run_id": Path(args.extract_run_dir).name,
        "k0_aurocs": k0_aurocs,
        "k0_ok": k0_ok,
        "discovery_grid": [
            {k: v for k, v in g.items() if k != "oof_act"} for g in discovery_grid
        ],
        "baseline_by_position": {
            p: {
                "best_baseline_type": best_by_pos[p]["best_baseline_type"],
                "best_baseline_auroc": best_by_pos[p]["best_baseline_auroc"],
                "baselines": best_by_pos[p]["baselines"],
            }
            for p in POSITIONS
        },
        "selected": frozen,
        "guarantee": GUARANTEE,
    }
    write_json(out / "discovery_summary.json", summary)
    write_json(out / "frozen_candidate.json", frozen)
    print(
        json.dumps(
            {
                "status": status,
                "selected_block": frozen["block"],
                "selected_position": frozen["position"],
                "act_auroc": frozen["act_auroc_discovery"],
                "baseline": frozen["best_baseline_type"],
                "delta": frozen["delta"],
                "activation_probe_sha256": frozen["activation_probe_sha256"],
                "baseline_sha256": frozen["baseline_sha256"],
                "combiner_sha256": frozen["combiner_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
