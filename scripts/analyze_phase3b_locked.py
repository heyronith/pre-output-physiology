#!/usr/bin/env python3
"""Phase 3B2 one-shot locked-test analysis.

Fits frozen models on combined Phase 3 train+validation (3000 rows), then
evaluates once on the untouched locked test. Does not tune on test.

Integrity checks must pass before labels enter predictive evaluation.
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
SURFACE_FREEZE_SHA = (
    "b25c5095c1cba440103343f90408c4caac0b59785bdc28b57a46abea405bcd00"
)
CANONICAL_DEV_RUN_ID = "phase3b1_extract_20260925T154005Z_39505f42"


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _sha_ids(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_activation_bundle(run_dir: Path, *, expect_locked: bool) -> dict:
    from safetensors.numpy import load_file

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if expect_locked:
        if manifest.get("locked_test_present") is not True:
            raise SystemExit("locked run missing locked_test_present=true")
        if set(manifest.get("splits_present", [])) != {"locked_test"}:
            raise SystemExit(f"unexpected splits: {manifest.get('splits_present')}")
    else:
        if manifest.get("locked_test_present"):
            raise SystemExit("dev run includes locked test")
        if "locked_test" in manifest.get("splits_present", []):
            raise SystemExit("dev run lists locked_test split")

    if manifest.get("compute_dtype") != "bfloat16":
        raise SystemExit(f"compute_dtype={manifest.get('compute_dtype')}")
    if manifest.get("activation_storage_dtype") != "float32":
        raise SystemExit(
            f"activation_storage_dtype={manifest.get('activation_storage_dtype')}"
        )
    if manifest.get("extraction_mode") != "truncated_prefix_single_example":
        raise SystemExit("extraction_mode mismatch")
    if int(manifest.get("batch_size", -1)) != 1:
        raise SystemExit("batch_size != 1")
    if manifest.get("model_revision") != MODEL_REVISION:
        raise SystemExit("model revision mismatch")

    k0 = load_file(str(run_dir / "k0_prompt_groups.safetensors"))
    if k0["activations"].dtype != np.float32:
        raise SystemExit(f"k0 dtype {k0['activations'].dtype}")
    k0_acts = np.asarray(k0["activations"], dtype=np.float32)
    k0_logits = np.asarray(k0["logit_summaries"], dtype=np.float32)
    if not np.isfinite(k0_acts).all():
        raise SystemExit("NaN/Inf in k0 activations")
    group_map = json.loads((run_dir / "k0_group_map.json").read_text(encoding="utf-8"))
    group_index = {g["prompt_sha256"]: g["group_index"] for g in group_map}

    act_min = float(np.min(k0_acts))
    act_max = float(np.max(k0_acts))
    act_max_abs = float(np.max(np.abs(k0_acts)))

    shards = []
    for sm in manifest["trajectory_shards"]:
        meta = json.loads((run_dir / sm["meta"]).read_text(encoding="utf-8"))
        tens = load_file(str(run_dir / sm["shard"]))
        splits = set(meta.get("splits", []))
        if expect_locked:
            if splits != {"locked_test"}:
                raise SystemExit(f"non-locked split in shard {sm['shard']}: {splits}")
        else:
            if "locked_test" in splits:
                raise SystemExit(f"locked_test in dev shard {sm['shard']}")
        raw = tens["activations"]
        if raw.dtype != np.float32:
            raise SystemExit(f"shard dtype {raw.dtype}")
        got = _sha_file(run_dir / sm["shard"])
        if got != sm["sha256"]:
            raise SystemExit(f"shard sha mismatch {sm['shard']}")
        if not np.isfinite(raw).all():
            raise SystemExit(f"NaN/Inf in {sm['shard']}")
        expected_tail = (
            len(COARSE_TRANSFORMER_BLOCKS),
            len(PREFIX_LENGTHS_K),
            int(manifest["hidden_size"]),
        )
        if raw.shape[1:] != expected_tail:
            raise SystemExit(f"shape mismatch {sm['shard']}: {raw.shape}")
        act_min = min(act_min, float(np.min(raw)))
        act_max = max(act_max, float(np.max(raw)))
        act_max_abs = max(act_max_abs, float(np.max(np.abs(raw))))
        shards.append((meta, tens))

    k0_blob = (run_dir / "k0_prompt_groups.safetensors").read_bytes()
    if hashlib.sha256(k0_blob).hexdigest() != manifest["k0_artifact"]["sha256"]:
        raise SystemExit("k0 sha mismatch")

    example_ids: list[str] = []
    splits_list: list[str] = []
    prompt_hashes: list[str] = []
    labels_parts = []
    resp_parts = []
    acts_parts = []
    valid_parts = []
    logit_parts = []
    for meta, tens in shards:
        example_ids.extend(meta["example_ids"])
        splits_list.extend(meta["splits"])
        prompt_hashes.extend(meta["prompt_sha256"])
        labels_parts.append(tens["labels"])
        resp_parts.append(tens["response_n_tokens"])
        acts_parts.append(np.asarray(tens["activations"], dtype=np.float32))
        valid_parts.append(tens["valid_mask"].astype(bool))
        logit_parts.append(np.asarray(tens["logit_summaries"], dtype=np.float32))

    labels = np.concatenate(labels_parts)
    resp_lens = np.concatenate(resp_parts)
    acts = np.concatenate(acts_parts, axis=0)
    valid = np.concatenate(valid_parts, axis=0)
    logits = np.concatenate(logit_parts, axis=0)
    k_index = {k: i for i, k in enumerate(PREFIX_LENGTHS_K)}
    for i, ph in enumerate(prompt_hashes):
        gi = group_index[ph]
        acts[i, :, k_index[0], :] = k0_acts[gi]
        logits[i, k_index[0], :] = k0_logits[gi]
        valid[i, k_index[0]] = True

    integrity = {
        "activation_storage_dtype": "float32",
        "compute_dtype": "bfloat16",
        "all_finite": True,
        "activation_min": act_min,
        "activation_max": act_max,
        "activation_max_abs": act_max_abs,
        "n_shards_checked": len(shards),
        "n_rows": len(example_ids),
        "n_prompt_groups_k0": len(group_map),
    }
    return {
        "manifest": manifest,
        "example_ids": example_ids,
        "splits": splits_list,
        "prompt_hashes": prompt_hashes,
        "labels": labels,
        "resp_lens": resp_lens,
        "acts": acts,
        "valid": valid,
        "logits": logits,
        "group_map": group_map,
        "integrity": integrity,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dev-run-dir",
        default=str(
            REPO_ROOT / "artifacts/runs" / CANONICAL_DEV_RUN_ID
        ),
    )
    parser.add_argument("--locked-run-dir", required=True)
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
        "--eligibility-freeze",
        default=str(
            REPO_ROOT / "artifacts/phase3b_locked/locked_eligibility_freeze.json"
        ),
    )
    parser.add_argument(
        "--dev-metrics",
        default=str(REPO_ROOT / "artifacts/phase3b_dev/phase3b_dev_metrics.json"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "artifacts/phase3b_locked"),
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))
    if freeze.get("locked_test_used") is not False:
        raise SystemExit("surface freeze used locked test")
    freeze_sha = _sha_file(Path(args.freeze))
    if freeze_sha != SURFACE_FREEZE_SHA:
        raise SystemExit(f"surface freeze hash drift {freeze_sha}")
    selected_c = {int(k): float(v) for k, v in freeze["selected_C_by_k"].items()}
    if any(v != 10.0 for v in selected_c.values()):
        raise SystemExit(f"surface C drift: {selected_c}")

    elig_freeze = json.loads(
        Path(args.eligibility_freeze).read_text(encoding="utf-8")
    )
    if elig_freeze.get("predictive_metrics_computed"):
        raise SystemExit("eligibility freeze already contains predictive metrics")
    elig_sha = _sha_file(Path(args.eligibility_freeze))

    print("Loading development activations (fit only)...")
    dev = _load_activation_bundle(Path(args.dev_run_dir), expect_locked=False)
    print("Loading locked-test activations (integrity before labels)...")
    locked = _load_activation_bundle(Path(args.locked_run_dir), expect_locked=True)

    # Integrity gate: eligibility hashes / counts vs freeze (IDs only)
    locked_ids = locked["example_ids"]
    if len(locked_ids) != 500:
        raise SystemExit(f"expected 500 locked rows, got {len(locked_ids)}")
    if len(locked["group_map"]) != 53:
        raise SystemExit(f"expected 53 locked prompt groups, got {len(locked['group_map'])}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, use_fast=True
    )

    def annotate(path: Path) -> dict[str, dict]:
        by_id = {}
        for row in _load_jsonl(path):
            tok = analyze_prompt_response_boundary(
                row["input_formatted"], row["model_outputs"], tokenizer
            )
            row = dict(row)
            row["_tok"] = tok
            row["canonical_response_n_tokens"] = len(tok.response_suffix_ids)
            by_id[row["example_id"]] = row
        return by_id

    # Metadata for feature construction (dev includes labels for fitting)
    by_id_dev = {}
    by_id_dev.update(annotate(data_dir / "phase3_train_metadata.jsonl"))
    by_id_dev.update(annotate(data_dir / "phase3_validation_metadata.jsonl"))

    # Locked metadata for integrity first WITHOUT using labels for decisions
    locked_meta_rows = _load_jsonl(data_dir / "locked_test_metadata.jsonl")
    if len(locked_meta_rows) != 500:
        raise SystemExit("locked metadata size mismatch")
    by_id_locked_text = {}
    for row in locked_meta_rows:
        tok = analyze_prompt_response_boundary(
            row["input_formatted"], row["model_outputs"], tokenizer
        )
        by_id_locked_text[row["example_id"]] = {
            "example_id": row["example_id"],
            "prompt_sha256": row["prompt_sha256"],
            "input_formatted": row["input_formatted"],
            "model_outputs": row["model_outputs"],
            "_tok": tok,
            "canonical_response_n_tokens": len(tok.response_suffix_ids),
            # labels loaded only after integrity gate
            "eventual_deception": None,
        }

    layers = COARSE_TRANSFORMER_BLOCKS
    ks = PREFIX_LENGTHS_K
    layer_index = {li: i for i, li in enumerate(layers)}
    k_index = {k: i for i, k in enumerate(ks)}

        # Verify locked eligibility against freeze (text lengths only)
    for k in ks:
        elig_ids_ordered = [
            r["example_id"]
            for r in locked_meta_rows
            if eligible_for_k(
                by_id_locked_text[r["example_id"]]["canonical_response_n_tokens"], k
            )
        ]
        got = _sha_ids(elig_ids_ordered)
        expect = elig_freeze["eligibility"][str(k)]["eligible_ids_sha256"]
        if got != expect:
            raise SystemExit(f"locked eligibility hash mismatch k={k}")
        if len(elig_ids_ordered) != elig_freeze["eligibility"][str(k)]["n_eligible"]:
            raise SystemExit(f"locked eligibility count mismatch k={k}")

    write_json(out_dir / "locked_activation_integrity_audit.json", locked["integrity"])
    print("Integrity gate PASSED — loading locked-test labels for one-shot evaluation")

    # ---- labels enter predictive evaluation only here ----
    for row in locked_meta_rows:
        by_id_locked_text[row["example_id"]]["eventual_deception"] = int(
            row["eventual_deception"]
        )

    # Align locked activation rows
    locked_rows = []
    for i, eid in enumerate(locked["example_ids"]):
        src = by_id_locked_text[eid]
        assert src["prompt_sha256"] == locked["prompt_hashes"][i]
        # Artifact may store placeholder labels; prefer metadata labels
        locked_rows.append(src)
    y_locked = np.asarray(
        [int(r["eventual_deception"]) for r in locked_rows], dtype=np.int64
    )

    # Align development rows
    dev_rows = []
    for i, eid in enumerate(dev["example_ids"]):
        src = by_id_dev[eid]
        assert src["prompt_sha256"] == dev["prompt_hashes"][i]
        assert int(src["eventual_deception"]) == int(dev["labels"][i])
        dev_rows.append(src)
    y_dev = np.asarray(dev["labels"], dtype=np.int64)

    # k0 within-group identity on locked test
    groups_locked: dict[str, list[int]] = defaultdict(list)
    for i, ph in enumerate(locked["prompt_hashes"]):
        groups_locked[ph].append(i)
    k0_act_diffs = []
    for _ph, idxs in groups_locked.items():
        if len(idxs) < 2:
            continue
        vecs = locked["acts"][idxs, layer_index[12], k_index[0]]
        k0_act_diffs.append(float(np.max(np.abs(vecs - vecs[0]))))
    k0_act_max_diff = max(k0_act_diffs) if k0_act_diffs else 0.0

    cell_results = []
    surface_by_k = {}
    logit_by_k = {}
    structural_by_k = {}
    majority_by_k = {}
    neg_controls = []
    k0_surface_by_idx: dict[int, float] = {}

    for k in ks:
        ki = k_index[k]
        # Combined development eligible
        dev_elig = [
            i
            for i in range(len(dev_rows))
            if eligible_for_k(int(dev["resp_lens"][i]), k) and dev["valid"][i, ki]
        ]
        te_elig = [
            i
            for i in range(len(locked_rows))
            if eligible_for_k(int(locked["resp_lens"][i]), k) and locked["valid"][i, ki]
        ]
        if not dev_elig or not te_elig:
            continue
        # Match freeze eligible ID set for locked
        te_ids = [locked["example_ids"][i] for i in te_elig]
        if _sha_ids(te_ids) != elig_freeze["eligibility"][str(k)]["eligible_ids_sha256"]:
            raise SystemExit(f"locked eligible set drift at analysis k={k}")

        y_tr = y_dev[dev_elig]
        y_te = y_locked[te_elig]
        g_te = [locked["prompt_hashes"][i] for i in te_elig]

        x_tr_txt = [
            visible_prompt_plus_prefix_text(
                dev_rows[i]["input_formatted"],
                dev_rows[i]["model_outputs"],
                k,
                dev_rows[i]["_tok"].response_offsets_in_model_outputs,
            )
            for i in dev_elig
        ]
        x_te_txt = [
            visible_prompt_plus_prefix_text(
                locked_rows[i]["input_formatted"],
                locked_rows[i]["model_outputs"],
                k,
                locked_rows[i]["_tok"].response_offsets_in_model_outputs,
            )
            for i in te_elig
        ]
        c = selected_c[k]
        pipe = make_text_baseline_pipeline(C=c, random_state=42)
        pipe.fit(x_tr_txt, y_tr)
        surf_scores = pipe.predict_proba(x_te_txt)[:, 1]
        surf_ci = bootstrap_auroc_ci_by_group(
            y_te, surf_scores, np.asarray(g_te), n_bootstrap=args.n_bootstrap, seed=0
        )
        surface_by_k[str(k)] = {
            "C": c,
            "n_dev": len(dev_elig),
            "n_test": len(te_elig),
            "auroc": float(roc_auc_score(y_te, surf_scores)),
            **{kk: surf_ci[kk] for kk in ("auroc_ci_low", "auroc_ci_high")},
            "eligible_ids_sha256": {
                "locked_test": elig_freeze["eligibility"][str(k)][
                    "eligible_ids_sha256"
                ],
            },
        }

        if k == 0:
            prompt_only_scores = surf_scores
            x_all = [locked_rows[i]["input_formatted"] for i in range(len(locked_rows))]
            all_sc = pipe.predict_proba(x_all)[:, 1]
            for ii, sc in enumerate(all_sc):
                k0_surface_by_idx[ii] = float(sc)
        else:
            c0 = selected_c[0]
            x_tr_p = [dev_rows[i]["input_formatted"] for i in dev_elig]
            x_te_p = [locked_rows[i]["input_formatted"] for i in te_elig]
            pipe_p = make_text_baseline_pipeline(C=c0, random_state=42)
            pipe_p.fit(x_tr_p, y_tr)
            prompt_only_scores = pipe_p.predict_proba(x_te_p)[:, 1]
        surface_by_k[str(k)]["prompt_only_auroc"] = float(
            roc_auc_score(y_te, prompt_only_scores)
        )

        maj = MajorityBaseline().fit(y_tr)
        majority_by_k[str(k)] = {
            "prevalence_dev": maj.prevalence_,
            "val_auroc": 0.5,
        }

        def struct_mat(rows_local, idxs, k_local: int):
            feats = []
            for i in idxs:
                prefix = visible_prompt_plus_prefix_text(
                    rows_local[i]["input_formatted"],
                    rows_local[i]["model_outputs"],
                    k_local,
                    rows_local[i]["_tok"].response_offsets_in_model_outputs,
                )
                resp_part = prefix[len(rows_local[i]["input_formatted"]) :]
                d = prefix_structural_features(resp_part, k_local)
                feats.append([d[kk] for kk in sorted(d)])
            return np.asarray(feats, dtype=np.float64)

        st = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
            ]
        )
        st.fit(struct_mat(dev_rows, dev_elig, k), y_tr)
        st_scores = st.predict_proba(struct_mat(locked_rows, te_elig, k))[:, 1]
        structural_by_k[str(k)] = {"auroc": float(roc_auc_score(y_te, st_scores))}

        lt = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
            ]
        )
        lt.fit(dev["logits"][dev_elig, ki], y_tr)
        lt_scores = lt.predict_proba(locked["logits"][te_elig, ki])[:, 1]
        logit_by_k[str(k)] = {"auroc": float(roc_auc_score(y_te, lt_scores))}

        for layer in layers:
            lj = layer_index[layer]
            x_tr = dev["acts"][dev_elig, lj, ki]
            x_te = locked["acts"][te_elig, lj, ki]
            probe = MeanLinearProbe(ProbeConfig())
            probe.fit(x_tr, y_tr)
            scores = probe.predict_proba(x_te)
            metrics = evaluate_binary_classifier(
                y_te, scores, n_bootstrap=min(200, args.n_bootstrap), bootstrap_seed=0
            )
            auroc_ci = bootstrap_auroc_ci_by_group(
                y_te, scores, np.asarray(g_te), n_bootstrap=args.n_bootstrap, seed=0
            )
            delta = paired_delta_auroc_ci_by_group(
                y_te,
                scores,
                surf_scores,
                np.asarray(g_te),
                n_bootstrap=args.n_bootstrap,
                seed=0,
            )
            cell = {
                "layer": layer,
                "k": k,
                "n_dev": len(dev_elig),
                "n_test": len(te_elig),
                "n_prompt_groups_test": len(set(g_te)),
                "class1_prevalence_test": float(np.mean(y_te)),
                "auroc": float(roc_auc_score(y_te, scores)),
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
                "secondary_exploratory": not (
                    (layer == PRIMARY_REGIME_A["layer"] and k == PRIMARY_REGIME_A["k"])
                    or (layer == PRIMARY_REGIME_B["layer"] and k == PRIMARY_REGIME_B["k"])
                ),
            }
            cell_results.append(cell)

            y_shuf = shuffled_labels(y_tr, seed=42)
            probe_s = MeanLinearProbe(ProbeConfig())
            probe_s.fit(x_tr, y_shuf)
            shuf_scores = probe_s.predict_proba(x_te)
            rnd = RandomDirectionControl(dim=x_tr.shape[1], seed=1000 + layer * 100 + k)
            rnd.fit()
            rnd_tr = rnd.scores(x_tr)
            rnd_te = rnd.scores(x_te)
            lr = LogisticRegression(C=0.01, max_iter=500, random_state=42)
            lr.fit(rnd_tr.reshape(-1, 1), y_tr)
            rnd_scores = lr.predict_proba(rnd_te.reshape(-1, 1))[:, 1]
            neg_controls.append(
                {
                    "layer": layer,
                    "k": k,
                    "shuffled_label_auroc": float(roc_auc_score(y_te, shuf_scores)),
                    "fixed_random_projection_diagnostic_auroc": float(
                        roc_auc_score(y_te, rnd_scores)
                    ),
                }
            )

    k0_surf_diffs = []
    for _ph, idxs in groups_locked.items():
        if len(idxs) < 2:
            continue
        sc = np.asarray([k0_surface_by_idx[i] for i in idxs], dtype=np.float64)
        k0_surf_diffs.append(float(np.max(np.abs(sc - sc[0]))))
    k0_surf_max_diff = max(k0_surf_diffs) if k0_surf_diffs else 0.0

    primary_a = next(c for c in cell_results if c["is_primary_regime_a"])
    primary_b = next(c for c in cell_results if c["is_primary_regime_b"])

    # Dev vs locked comparison for primary endpoints
    dev_metrics = json.loads(Path(args.dev_metrics).read_text(encoding="utf-8"))
    dev_a = dev_metrics["primary_regime_a"]
    dev_b = dev_metrics["primary_regime_b"]
    comparison = {
        "regime_a": {
            "dev_auroc": dev_a["auroc"],
            "locked_auroc": primary_a["auroc"],
            "delta_locked_minus_dev_auroc": primary_a["auroc"] - dev_a["auroc"],
            "dev_delta_auroc": dev_a["delta_auroc"],
            "locked_delta_auroc": primary_a["delta_auroc"],
            "delta_locked_minus_dev_delta_auroc": (
                primary_a["delta_auroc"] - dev_a["delta_auroc"]
            ),
            "dev_surface_auroc": dev_a["surface_auroc"],
            "locked_surface_auroc": primary_a["surface_auroc"],
        },
        "regime_b": {
            "dev_auroc": dev_b["auroc"],
            "locked_auroc": primary_b["auroc"],
            "delta_locked_minus_dev_auroc": primary_b["auroc"] - dev_b["auroc"],
            "dev_delta_auroc": dev_b["delta_auroc"],
            "locked_delta_auroc": primary_b["delta_auroc"],
            "delta_locked_minus_dev_delta_auroc": (
                primary_b["delta_auroc"] - dev_b["delta_auroc"]
            ),
            "dev_surface_auroc": dev_b["surface_auroc"],
            "locked_surface_auroc": primary_b["surface_auroc"],
        },
    }

    locked_man = locked["manifest"]
    summary = {
        "created_at": utc_now_iso(),
        "phase": "phase3b2_locked_test",
        "one_shot_locked_test_evaluation": True,
        "locked_test_hyperparameter_tuning": False,
        "run_id": locked_man["run_id"],
        "locked_run_dir": str(Path(args.locked_run_dir)),
        "dev_run_id": CANONICAL_DEV_RUN_ID,
        "dev_run_dir": str(Path(args.dev_run_dir)),
        "fit_on": "phase3_train_plus_validation_combined",
        "n_dev_rows": len(dev_rows),
        "n_locked_rows": len(locked_rows),
        "n_prompt_groups_locked_k0": len(locked["group_map"]),
        "git_commit": locked_man.get("git_commit"),
        "working_tree_clean": locked_man.get("working_tree_clean"),
        "extractor_sha256": locked_man.get("extractor_sha256"),
        "locked_analysis_script_sha256": _sha_file(Path(__file__)),
        "dev_analysis_script_sha256": _sha_file(
            REPO_ROOT / "scripts/analyze_phase3b_dev.py"
        ),
        "surface_baseline_freeze_sha256": freeze_sha,
        "locked_eligibility_freeze_sha256": elig_sha,
        "compute_dtype": "bfloat16",
        "activation_storage_dtype": "float32",
        "extraction_mode": "truncated_prefix_single_example",
        "batch_size": 1,
        "future_response_tokens_present": False,
        "locked_test_used": True,
        "regime_c_run": False,
        "causal_interventions": False,
        "activation_probe": {
            "C": 0.01,
            "max_iter": 500,
            "random_state": 42,
            "normalize": True,
        },
        "surface_C_by_k": {str(k): 10.0 for k in ks},
        "k0_within_group_activation_max_abs_diff_layer12": k0_act_max_diff,
        "k0_within_group_surface_score_max_abs_diff": k0_surf_max_diff,
        "activation_integrity": locked["integrity"],
        "surface_by_k": surface_by_k,
        "logit_by_k": logit_by_k,
        "structural_by_k": structural_by_k,
        "majority_by_k": majority_by_k,
        "cells": cell_results,
        "negative_controls": neg_controls,
        "negative_control_notes": {
            "shuffled_label_probe": "chance/null control (seed 42)",
            "fixed_random_projection_diagnostic": (
                "NOT a chance/null control; may retain distributed class signal"
            ),
        },
        "primary_regime_a": primary_a,
        "primary_regime_b": primary_b,
        "comparison_dev_vs_locked": comparison,
        "manifest_hashes": {
            "k0_sha256": locked_man["k0_artifact"]["sha256"],
            "shards": [
                {"shard": s["shard"], "sha256": s["sha256"]}
                for s in locked_man["trajectory_shards"]
            ],
            "total_artifact_bytes": locked_man["total_artifact_bytes"],
            "estimated_cost_usd": locked_man["estimated_cost_usd"],
            "wall_seconds": locked_man["wall_seconds"],
            "n_scientific_forwards": locked_man["n_scientific_forwards"],
        },
        "engineering_gate": {
            "all_54_cells_present": len(cell_results) == 54,
            "n_cells": len(cell_results),
            "k0_activation_identity_ok": k0_act_max_diff == 0.0,
            "k0_surface_score_identity_ok": k0_surf_max_diff == 0.0,
            "activation_all_finite": locked["integrity"]["all_finite"],
            "primary_endpoints_frozen": True,
        },
    }
    write_json(out_dir / "phase3b_locked_metrics.json", summary)
    write_json(
        out_dir / "phase3b_locked_cells.json",
        {"cells": cell_results, "negative_controls": neg_controls},
    )
    print(
        json.dumps(
            {
                "out": str(out_dir / "phase3b_locked_metrics.json"),
                "n_cells": len(cell_results),
                "regime_a_auroc": primary_a["auroc"],
                "regime_a_delta": primary_a["delta_auroc"],
                "regime_b_auroc": primary_b["auroc"],
                "regime_b_delta": primary_b["delta_auroc"],
                "regime_b_delta_ci": [
                    primary_b["delta_auroc_ci_low"],
                    primary_b["delta_auroc_ci_high"],
                ],
                "k0_act_max_diff": k0_act_max_diff,
                "k0_surf_max_diff": k0_surf_max_diff,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
