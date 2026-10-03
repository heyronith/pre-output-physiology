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
    # Match freeze_phase3b_surface_baselines._ids_hash (sorted).
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


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
    parser.add_argument(
        "--compare-metrics",
        default=str(
            REPO_ROOT / "artifacts/phase3b_dev/original_dev_run_f19e058f_metrics.json"
        ),
        help="Original development metrics JSON for cell-wise comparison",
    )
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
    if manifest.get("compute_dtype") != "bfloat16":
        raise SystemExit(f"compute_dtype={manifest.get('compute_dtype')}")
    if manifest.get("activation_storage_dtype") != "float32":
        raise SystemExit(
            f"activation_storage_dtype={manifest.get('activation_storage_dtype')}"
        )

    from safetensors.numpy import load_file

    k0 = load_file(str(run_dir / "k0_prompt_groups.safetensors"))
    if k0["activations"].dtype != np.float32:
        raise SystemExit(f"k0 activations dtype {k0['activations'].dtype} != float32")
    k0_acts = np.asarray(k0["activations"], dtype=np.float32)
    k0_logits = np.asarray(k0["logit_summaries"], dtype=np.float32)
    group_map = json.loads((run_dir / "k0_group_map.json").read_text(encoding="utf-8"))
    group_index = {g["prompt_sha256"]: g["group_index"] for g in group_map}

    # Load trajectory shards + integrity audit
    shards = []
    act_min = float("inf")
    act_max = float("-inf")
    act_max_abs = 0.0
    finite_ok = True
    for sm in manifest["trajectory_shards"]:
        meta = json.loads((run_dir / sm["meta"]).read_text(encoding="utf-8"))
        tens = load_file(str(run_dir / sm["shard"]))
        if "locked_test" in meta.get("splits", []):
            raise SystemExit("locked_test in shard")
        raw_acts = tens["activations"]
        if raw_acts.dtype != np.float32:
            raise SystemExit(f"shard {sm['shard']} dtype {raw_acts.dtype} != float32")
        blob = (run_dir / sm["shard"]).read_bytes()
        got_sha = hashlib.sha256(blob).hexdigest()
        if got_sha != sm["sha256"]:
            raise SystemExit(f"shard sha mismatch {sm['shard']}")
        if not np.isfinite(raw_acts).all():
            finite_ok = False
            raise SystemExit(f"NaN/Inf in shard {sm['shard']}")
        act_min = min(act_min, float(np.min(raw_acts)))
        act_max = max(act_max, float(np.max(raw_acts)))
        act_max_abs = max(act_max_abs, float(np.max(np.abs(raw_acts))))
        expected_shape_tail = (
            len(COARSE_TRANSFORMER_BLOCKS),
            len(PREFIX_LENGTHS_K),
            int(manifest["hidden_size"]),
        )
        if raw_acts.shape[1:] != expected_shape_tail:
            raise SystemExit(f"shape mismatch {sm['shard']}: {raw_acts.shape}")
        shards.append((meta, tens))

    if not np.isfinite(k0_acts).all():
        raise SystemExit("NaN/Inf in k0 activations")
    act_min = min(act_min, float(np.min(k0_acts)))
    act_max = max(act_max, float(np.max(k0_acts)))
    act_max_abs = max(act_max_abs, float(np.max(np.abs(k0_acts))))
    k0_blob = (run_dir / "k0_prompt_groups.safetensors").read_bytes()
    if hashlib.sha256(k0_blob).hexdigest() != manifest["k0_artifact"]["sha256"]:
        raise SystemExit("k0 artifact sha mismatch")

    integrity = {
        "activation_storage_dtype": "float32",
        "compute_dtype": "bfloat16",
        "all_finite": finite_ok,
        "activation_min": act_min,
        "activation_max": act_max,
        "activation_max_abs": act_max_abs,
        "n_shards_checked": len(shards),
        "k0_dtype": str(k0["activations"].dtype),
    }
    write_json(out_dir / "activation_integrity_audit.json", integrity)

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
        acts_list.append(np.asarray(tens["activations"], dtype=np.float32))
        valid_list.append(tens["valid_mask"].astype(bool))
        logit_list.append(np.asarray(tens["logit_summaries"], dtype=np.float32))

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

    # Eligibility IDs must match freeze (metadata file order, not shard order)
    for split_name, fname, freeze_key in (
        ("phase3_train", "phase3_train_metadata.jsonl", "train"),
        ("phase3_validation", "phase3_validation_metadata.jsonl", "validation"),
    ):
        meta_rows = _load_jsonl(data_dir / fname)
        for k in ks:
            elig_ids = []
            for r in meta_rows:
                tok = by_id[r["example_id"]]["_tok"]
                if eligible_for_k(len(tok.response_suffix_ids), k):
                    elig_ids.append(r["example_id"])
            got = _sha_ids(elig_ids)
            expect = freeze["eligibility"][str(k)][freeze_key]["eligible_ids_sha256"]
            if got != expect:
                raise SystemExit(
                    f"eligibility hash mismatch {split_name} k={k}: {got} != {expect}"
                )
            n_expect = freeze["eligibility"][str(k)][freeze_key]["n_eligible"]
            if len(elig_ids) != n_expect:
                raise SystemExit(
                    f"eligibility count mismatch {split_name} k={k}: "
                    f"{len(elig_ids)} != {n_expect}"
                )

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
    k0_surface_scores_by_idx: dict[int, float] = {}

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
                "train": freeze["eligibility"][str(k)]["train"]["eligible_ids_sha256"],
                "val": freeze["eligibility"][str(k)]["validation"][
                    "eligible_ids_sha256"
                ],
            },
        }

        # Prompt-only uses C at k=0
        if k == 0:
            prompt_only_scores = surf_scores
            # Store prompt-only scores for all val rows (identity safeguard)
            x_all_va = [rows[i]["input_formatted"] for i in val_idx.tolist()]
            all_scores = pipe.predict_proba(x_all_va)[:, 1]
            for ii, sc in zip(val_idx.tolist(), all_scores, strict=True):
                k0_surface_scores_by_idx[int(ii)] = float(sc)
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
                    "fixed_random_projection_diagnostic_auroc": float(
                        roc_auc_score(y_va, rnd_scores)
                    ),
                    # Alias retained for older readers; not a chance/null control.
                    "random_direction_auroc": float(roc_auc_score(y_va, rnd_scores)),
                }
            )

    # k0 within-group surface score identity (prompt-only)
    k0_surf_diffs = []
    for _ph, idxs in groups_val.items():
        if len(idxs) < 2:
            continue
        sc = np.asarray([k0_surface_scores_by_idx[i] for i in idxs], dtype=np.float64)
        k0_surf_diffs.append(float(np.max(np.abs(sc - sc[0]))))
    k0_surf_max_diff = max(k0_surf_diffs) if k0_surf_diffs else 0.0

    # Compare to original development run (float16-storage)
    compare_path = Path(args.compare_metrics)
    comparison: dict = {}
    if compare_path.is_file():
        orig = json.loads(compare_path.read_text(encoding="utf-8"))
        orig_by = {(c["layer"], c["k"]): c for c in orig["cells"]}
        deltas = []
        for c in cell_results:
            o = orig_by[(c["layer"], c["k"])]
            d = float(c["auroc"] - o["auroc"])
            deltas.append(
                {
                    "layer": c["layer"],
                    "k": c["k"],
                    "auroc_canonical": c["auroc"],
                    "auroc_original": o["auroc"],
                    "delta_auroc_rerun_minus_original": d,
                    "delta_delta_auroc": float(c["delta_auroc"] - o["delta_auroc"]),
                }
            )
        max_abs = max(abs(x["delta_auroc_rerun_minus_original"]) for x in deltas)
        a12 = next(x for x in deltas if x["layer"] == 12 and x["k"] == 0)
        b12 = next(x for x in deltas if x["layer"] == 12 and x["k"] == 1)
        comparison = {
            "original_run_id": orig.get("run_id"),
            "compare_metrics_path": str(compare_path),
            "n_cells_compared": len(deltas),
            "max_abs_auroc_delta_across_54": max_abs,
            "regime_a_block12_k0": a12,
            "regime_b_block12_k1": b12,
            "cells": deltas,
            "unexpectedly_large": max_abs > 0.05,
        }
        write_json(out_dir / "canonical_vs_original_comparison.json", comparison)
        if comparison["unexpectedly_large"]:
            print(
                "WARNING: max |ΔAUROC| vs original > 0.05 — "
                "report and STOP before Phase 3B2"
            )

    summary = {
        "created_at": utc_now_iso(),
        "run_id": manifest["run_id"],
        "run_dir": str(run_dir),
        "canonical_run": True,
        "preserves_original_dev_run_id": "phase3b1_extract_20260925T151054Z_f19e058f",
        "git_commit": manifest.get("git_commit"),
        "canonical_run_code_sha": manifest.get("canonical_run_code_sha"),
        "working_tree_clean": manifest.get("working_tree_clean"),
        "extractor_sha256": manifest.get("extractor_sha256"),
        "analysis_script_sha256": manifest.get("analysis_script_sha256"),
        "compute_dtype": "bfloat16",
        "activation_storage_dtype": "float32",
        "extraction_mode": manifest.get("extraction_mode"),
        "batch_size": manifest.get("batch_size"),
        "future_response_tokens_present": False,
        "full_sequence_teacher_forced_primary": False,
        "original_preflight_gate_relaxed": False,
        "locked_test_used": False,
        "regime_c_run": False,
        "causal_interventions": False,
        "n_examples": n,
        "n_train": int(len(train_idx)),
        "n_validation": int(len(val_idx)),
        "n_prompt_groups_k0": len(group_map),
        "k0_within_group_activation_max_abs_diff_layer12": k0_act_max_diff,
        "k0_within_group_surface_score_max_abs_diff": k0_surf_max_diff,
        "activation_integrity": integrity,
        "surface_by_k": surface_by_k,
        "logit_by_k": logit_by_k,
        "structural_by_k": structural_by_k,
        "majority_by_k": majority_by_k,
        "cells": cell_results,
        "negative_controls": neg_controls,
        "negative_control_notes": {
            "shuffled_label_probe": "chance/null control (seed 42)",
            "fixed_random_projection_diagnostic": (
                "NOT a chance/null control; a fixed random projection of a "
                "representation with distributed class structure can retain signal"
            ),
        },
        "primary_regime_a": next(
            c for c in cell_results if c["is_primary_regime_a"]
        ),
        "primary_regime_b": next(
            c for c in cell_results if c["is_primary_regime_b"]
        ),
        "comparison_vs_original": (
            {k: comparison[k] for k in comparison if k != "cells"}
            if comparison
            else None
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
        "surface_baseline_freeze_sha256": hashlib.sha256(
            Path(args.freeze).read_bytes()
        ).hexdigest(),
        "engineering_gate": {
            "all_54_cells_present": len(cell_results) == 54,
            "n_cells": len(cell_results),
            "k0_activation_identity_ok": k0_act_max_diff == 0.0,
            "k0_surface_score_identity_ok": k0_surf_max_diff == 0.0,
            "activation_all_finite": integrity["all_finite"],
        },
    }
    write_json(out_dir / "phase3b_dev_metrics.json", summary)
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
                "k0_surf_max_diff": k0_surf_max_diff,
                "activation_max_abs": integrity["activation_max_abs"],
                "max_abs_auroc_delta_vs_original": comparison.get(
                    "max_abs_auroc_delta_across_54"
                )
                if comparison
                else None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
