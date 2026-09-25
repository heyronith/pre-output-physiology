#!/usr/bin/env python3
"""Local Phase 3B1 analysis on train/validation activations only.

Does not load locked-test activations or evaluate locked-test performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.baselines import (  # noqa: E402
    MajorityBaseline,
    RandomDirectionControl,
    make_text_baseline_pipeline,
    prefix_structural_features,
    shuffled_labels,
    visible_prompt_plus_prefix_text,
)
from pre_output_physiology.metrics import (  # noqa: E402
    bootstrap_auroc_ci_by_group,
    evaluate_binary_classifier,
    paired_delta_auroc_ci_by_group,
)
from pre_output_physiology.probes import MeanLinearProbe, ProbeConfig  # noqa: E402
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402
from pre_output_physiology.trajectory import (  # noqa: E402
    COARSE_TRANSFORMER_BLOCKS,
    PREFIX_LENGTHS_K,
    PRIMARY_REGIME_A,
    PRIMARY_REGIME_B,
    analyze_prompt_response_boundary,
    eligible_for_k,
)

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _sha_ids(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--data-dir",
        default=str(REPO_ROOT / "data/processed/phase3_roleplay"),
    )
    parser.add_argument(
        "--freeze",
        default=str(
            REPO_ROOT / "artifacts/phase3b_dev/surface_baseline_freeze.json"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "artifacts/phase3b_dev"),
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))
    if freeze.get("locked_test_used") is not False:
        raise SystemExit("surface freeze used locked test")
    selected_c = {int(k): float(v) for k, v in freeze["selected_C_by_k"].items()}

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("locked_test_present"):
        raise SystemExit("run includes locked test")

    from safetensors.numpy import load_file

    k0 = load_file(str(run_dir / "k0_prompt_groups.safetensors"))
    k0_acts = k0["activations"].astype(np.float32)
    k0_logits = k0["logit_summaries"].astype(np.float32)
    group_map = json.loads((run_dir / "k0_group_map.json").read_text(encoding="utf-8"))
    group_index = {g["prompt_sha256"]: g["group_index"] for g in group_map}

    # Load trajectory shards
    shards = []
    for sm in manifest["trajectory_shards"]:
        meta = json.loads((run_dir / sm["meta"]).read_text(encoding="utf-8"))
        tens = load_file(str(run_dir / sm["shard"]))
        if "locked_test" in meta.get("splits", []):
            raise SystemExit("locked_test in shard")
        shards.append((meta, tens))

    example_ids = []
    splits = []
    prompt_hashes = []
    labels = []
    resp_lens = []
    acts_list = []
    valid_list = []
    logit_list = []
    for meta, tens in shards:
        example_ids.extend(meta["example_ids"])
        splits.extend(meta["splits"])
        prompt_hashes.extend(meta["prompt_sha256"])
        labels.append(tens["labels"])
        resp_lens.append(tens["response_n_tokens"])
        acts_list.append(tens["activations"].astype(np.float32))
        valid_list.append(tens["valid_mask"].astype(bool))
        logit_list.append(tens["logit_summaries"].astype(np.float32))

    labels = np.concatenate(labels)
    resp_lens = np.concatenate(resp_lens)
    acts = np.concatenate(acts_list, axis=0)  # [N,9,6,H]
    valid = np.concatenate(valid_list, axis=0)
    logits = np.concatenate(logit_list, axis=0)
    n = len(example_ids)
    layers = COARSE_TRANSFORMER_BLOCKS
    ks = PREFIX_LENGTHS_K
    layer_index = {li: i for i, li in enumerate(layers)}
    k_index = {k: i for i, k in enumerate(ks)}

    # Broadcast k=0 from groups
    for i, ph in enumerate(prompt_hashes):
        gi = group_index[ph]
        acts[i, :, k_index[0], :] = k0_acts[gi]
        logits[i, k_index[0], :] = k0_logits[gi]
        valid[i, k_index[0]] = True

    # Metadata from source (text for baselines)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, use_fast=True
    )
    by_id = {}
    for _split_name, fname in (
        ("phase3_train", "phase3_train_metadata.jsonl"),
        ("phase3_validation", "phase3_validation_metadata.jsonl"),
    ):
        for row in _load_jsonl(data_dir / fname):
            if row["example_id"] in by_id:
                continue
            tok = analyze_prompt_response_boundary(
                row["input_formatted"], row["model_outputs"], tokenizer
            )
            row = dict(row)
            row["_tok"] = tok
            row["canonical_response_n_tokens"] = len(tok.response_suffix_ids)
            by_id[row["example_id"]] = row

    # Align rows
    rows = []
    for i, eid in enumerate(example_ids):
        src = by_id[eid]
        assert src["prompt_sha256"] == prompt_hashes[i]
        assert int(src["eventual_deception"]) == int(labels[i])
        rows.append(src)

    train_idx = np.asarray([i for i, s in enumerate(splits) if s == "phase3_train"])
    val_idx = np.asarray([i for i, s in enumerate(splits) if s == "phase3_validation"])
    if len(train_idx) + len(val_idx) != n:
        raise SystemExit("unexpected splits")

    # k=0 consistency checks on validation multi-response groups
    groups_val: dict[str, list[int]] = defaultdict(list)
    for i in val_idx.tolist():
        groups_val[prompt_hashes[i]].append(i)
    k0_score_diffs = []
    for _ph, idxs in groups_val.items():
        if len(idxs) < 2:
            continue
        vecs = acts[idxs, layer_index[12], k_index[0]]
        k0_score_diffs.append(float(np.max(np.abs(vecs - vecs[0]))))
    k0_act_max_diff = max(k0_score_diffs) if k0_score_diffs else 0.0

    cell_results = []
    surface_by_k = {}
    logit_by_k = {}
    structural_by_k = {}
    majority_by_k = {}
    neg_controls = []

    for k in ks:
        ki = k_index[k]
        tr_elig = [
            i
            for i in train_idx.tolist()
            if eligible_for_k(int(resp_lens[i]), k) and valid[i, ki]
        ]
        va_elig = [
            i
            for i in val_idx.tolist()
            if eligible_for_k(int(resp_lens[i]), k) and valid[i, ki]
        ]
        if not tr_elig or not va_elig:
            continue
        y_tr = labels[tr_elig]
        y_va = labels[va_elig]
        g_va = [prompt_hashes[i] for i in va_elig]

        # Surface texts
        x_tr_txt = [
            visible_prompt_plus_prefix_text(
                rows[i]["input_formatted"],
                rows[i]["model_outputs"],
                k,
                rows[i]["_tok"].response_offsets_in_model_outputs,
            )
            for i in tr_elig
        ]
        x_va_txt = [
            visible_prompt_plus_prefix_text(
                rows[i]["input_formatted"],
                rows[i]["model_outputs"],
                k,
                rows[i]["_tok"].response_offsets_in_model_outputs,
            )
            for i in va_elig
        ]
        c = selected_c[k]
        pipe = make_text_baseline_pipeline(C=c, random_state=42)
        pipe.fit(x_tr_txt, y_tr)
        surf_scores = pipe.predict_proba(x_va_txt)[:, 1]
        surf_ci = bootstrap_auroc_ci_by_group(
            y_va, surf_scores, np.asarray(g_va), n_bootstrap=args.n_bootstrap, seed=0
        )
        surface_by_k[str(k)] = {
            "C": c,
            "n_train": len(tr_elig),
            "n_val": len(va_elig),
            "auroc": float(roc_auc_score(y_va, surf_scores)),
            **{kk: surf_ci[kk] for kk in ("auroc_ci_low", "auroc_ci_high")},
            "eligible_ids_sha256": {
                "train": _sha_ids([example_ids[i] for i in tr_elig]),
                "val": _sha_ids([example_ids[i] for i in va_elig]),
            },
        }

        # Prompt-only uses C at k=0
        if k == 0:
            prompt_only_scores = surf_scores
        else:
            c0 = selected_c[0]
            x_tr_p = [rows[i]["input_formatted"] for i in tr_elig]
            x_va_p = [rows[i]["input_formatted"] for i in va_elig]
            pipe_p = make_text_baseline_pipeline(C=c0, random_state=42)
            pipe_p.fit(x_tr_p, y_tr)
            prompt_only_scores = pipe_p.predict_proba(x_va_p)[:, 1]
        surface_by_k[str(k)]["prompt_only_auroc"] = float(
            roc_auc_score(y_va, prompt_only_scores)
        )

        # Majority
        maj = MajorityBaseline().fit(y_tr)
        majority_by_k[str(k)] = {
            "prevalence": maj.prevalence_,
            "val_auroc": 0.5,
        }

        # Structural
        def struct_mat(idxs, k_local: int):
            feats = []
            for i in idxs:
                prefix = visible_prompt_plus_prefix_text(
                    rows[i]["input_formatted"],
                    rows[i]["model_outputs"],
                    k_local,
                    rows[i]["_tok"].response_offsets_in_model_outputs,
                )
                resp_part = prefix[len(rows[i]["input_formatted"]) :]
                d = prefix_structural_features(resp_part, k_local)
                feats.append([d[kk] for kk in sorted(d)])
            return np.asarray(feats, dtype=np.float64)

        st = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(C=1.0, max_iter=1000, random_state=42),
                ),
            ]
        )
        st.fit(struct_mat(tr_elig, k), y_tr)
        st_scores = st.predict_proba(struct_mat(va_elig, k))[:, 1]
        structural_by_k[str(k)] = {
            "auroc": float(roc_auc_score(y_va, st_scores)),
        }

        # Logit summary baseline
        lt = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(C=1.0, max_iter=1000, random_state=42),
                ),
            ]
        )
        lt.fit(logits[tr_elig, ki], y_tr)
        lt_scores = lt.predict_proba(logits[va_elig, ki])[:, 1]
        logit_by_k[str(k)] = {"auroc": float(roc_auc_score(y_va, lt_scores))}

        for layer in layers:
            lj = layer_index[layer]
            x_tr = acts[tr_elig, lj, ki]
            x_va = acts[va_elig, lj, ki]
            probe = MeanLinearProbe(ProbeConfig())
            probe.fit(x_tr, y_tr)
            scores = probe.predict_proba(x_va)
            metrics = evaluate_binary_classifier(
                y_va, scores, n_bootstrap=min(200, args.n_bootstrap), bootstrap_seed=0
            )
            auroc_ci = bootstrap_auroc_ci_by_group(
                y_va, scores, np.asarray(g_va), n_bootstrap=args.n_bootstrap, seed=0
            )
            delta = paired_delta_auroc_ci_by_group(
                y_va,
                scores,
                surf_scores,
                np.asarray(g_va),
                n_bootstrap=args.n_bootstrap,
                seed=0,
            )
            cell = {
                "layer": layer,
                "k": k,
                "n_train": len(tr_elig),
                "n_val": len(va_elig),
                "n_prompt_groups_val": len(set(g_va)),
                "class1_prevalence_val": float(np.mean(y_va)),
                "auroc": float(roc_auc_score(y_va, scores)),
                "auroc_ci_low": auroc_ci["auroc_ci_low"],
                "auroc_ci_high": auroc_ci["auroc_ci_high"],
                "auprc": metrics["auprc"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "surface_auroc": surface_by_k[str(k)]["auroc"],
                "delta_auroc": delta["delta_auroc"],
                "delta_auroc_ci_low": delta["delta_auroc_ci_low"],
                "delta_auroc_ci_high": delta["delta_auroc_ci_high"],
                "prompt_only_auroc": surface_by_k[str(k)]["prompt_only_auroc"],
                "logit_summary_auroc": logit_by_k[str(k)]["auroc"],
                "structural_auroc": structural_by_k[str(k)]["auroc"],
                "is_primary_regime_a": layer == PRIMARY_REGIME_A["layer"]
                and k == PRIMARY_REGIME_A["k"],
                "is_primary_regime_b": layer == PRIMARY_REGIME_B["layer"]
                and k == PRIMARY_REGIME_B["k"],
            }
            cell_results.append(cell)

            # Negative controls once per primary-ish cells and all cells briefly
            y_shuf = shuffled_labels(y_tr, seed=42)
            probe_s = MeanLinearProbe(ProbeConfig())
            probe_s.fit(x_tr, y_shuf)
            shuf_scores = probe_s.predict_proba(x_va)
            rnd = RandomDirectionControl(dim=x_tr.shape[1], seed=1000 + layer * 100 + k)
            rnd.fit()
            rnd_tr = rnd.scores(x_tr)
            rnd_va = rnd.scores(x_va)
            # fit LR on random projection scores
            lr = LogisticRegression(C=0.01, max_iter=500, random_state=42)
            lr.fit(rnd_tr.reshape(-1, 1), y_tr)
            rnd_scores = lr.predict_proba(rnd_va.reshape(-1, 1))[:, 1]
            neg_controls.append(
                {
                    "layer": layer,
                    "k": k,
                    "shuffled_label_auroc": float(roc_auc_score(y_va, shuf_scores)),
                    "random_direction_auroc": float(roc_auc_score(y_va, rnd_scores)),
                }
            )

    # k0 within-group surface score identity (prompt-only)
    k0_cells = [c for c in cell_results if c["k"] == 0 and c["layer"] == 12]
    summary = {
        "created_at": utc_now_iso(),
        "run_id": manifest["run_id"],
        "run_dir": str(run_dir),
        "locked_test_used": False,
        "regime_c_run": False,
        "causal_interventions": False,
        "n_examples": n,
        "n_train": int(len(train_idx)),
        "n_validation": int(len(val_idx)),
        "n_prompt_groups_k0": len(group_map),
        "k0_within_group_activation_max_abs_diff_layer12": k0_act_max_diff,
        "surface_by_k": surface_by_k,
        "logit_by_k": logit_by_k,
        "structural_by_k": structural_by_k,
        "majority_by_k": majority_by_k,
        "cells": cell_results,
        "negative_controls": neg_controls,
        "primary_regime_a": next(
            c for c in cell_results if c["is_primary_regime_a"]
        ),
        "primary_regime_b": next(
            c for c in cell_results if c["is_primary_regime_b"]
        ),
        "manifest_hashes": {
            "k0_sha256": manifest["k0_artifact"]["sha256"],
            "shards": [
                {"shard": s["shard"], "sha256": s["sha256"]}
                for s in manifest["trajectory_shards"]
            ],
            "total_artifact_bytes": manifest["total_artifact_bytes"],
            "estimated_cost_usd": manifest["estimated_cost_usd"],
            "wall_seconds": manifest["wall_seconds"],
        },
        "engineering_gate": {
            "all_54_cells_present": len(cell_results) == 54,
            "n_cells": len(cell_results),
        },
    }
    write_json(out_dir / "phase3b_dev_metrics.json", summary)
    # compact table csv-like json
    write_json(
        out_dir / "phase3b_dev_cells.json",
        {"cells": cell_results, "negative_controls": neg_controls},
    )
    print(
        json.dumps(
            {
                "out": str(out_dir / "phase3b_dev_metrics.json"),
                "n_cells": len(cell_results),
                "regime_a_auroc": summary["primary_regime_a"]["auroc"],
                "regime_a_delta": summary["primary_regime_a"]["delta_auroc"],
                "regime_b_auroc": summary["primary_regime_b"]["auroc"],
                "regime_b_delta": summary["primary_regime_b"]["delta_auroc"],
                "k0_act_max_diff": k0_act_max_diff,
            },
            indent=2,
        )
    )
    _ = k0_cells
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
