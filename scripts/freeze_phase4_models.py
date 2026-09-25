#!/usr/bin/env python3
"""Freeze Phase 3 L12/k0 and L12/k1 probes + surface comparators for Phase 4.

Fits only on the 3000-row Phase 3 development set (train+validation).
Never uses locked-test rows or Phase 4 labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from safetensors.numpy import load_file
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.baselines import (  # noqa: E402
    make_text_baseline_pipeline,
    visible_prompt_plus_prefix_text,
)
from pre_output_physiology.probes import MeanLinearProbe, ProbeConfig  # noqa: E402
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402
from pre_output_physiology.trajectory import (  # noqa: E402
    COARSE_TRANSFORMER_BLOCKS,
    PREFIX_LENGTHS_K,
    analyze_prompt_response_boundary,
    eligible_for_k,
)

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
CANONICAL_DEV_RUN = "phase3b1_extract_20260925T154005Z_39505f42"
SURFACE_FREEZE_SHA = (
    "b25c5095c1cba440103343f90408c4caac0b59785bdc28b57a46abea405bcd00"
)


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _sha_ids(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dev-run-dir",
        default=str(REPO_ROOT / "artifacts/runs" / CANONICAL_DEV_RUN),
    )
    parser.add_argument(
        "--data-dir",
        default=str(REPO_ROOT / "data/processed/phase3_roleplay"),
    )
    parser.add_argument(
        "--artifact-dir",
        default=str(REPO_ROOT / "artifacts/phase4_models"),
    )
    parser.add_argument(
        "--summary-dir",
        default=str(REPO_ROOT / "artifacts/phase4a_summaries"),
    )
    args = parser.parse_args()

    run_dir = Path(args.dev_run_dir)
    data_dir = Path(args.data_dir)
    art_dir = Path(args.artifact_dir)
    sum_dir = Path(args.summary_dir)
    art_dir.mkdir(parents=True, exist_ok=True)
    sum_dir.mkdir(parents=True, exist_ok=True)

    freeze = json.loads(
        (REPO_ROOT / "artifacts/phase3b_dev/surface_baseline_freeze.json").read_text(
            encoding="utf-8"
        )
    )
    if _sha_file(REPO_ROOT / "artifacts/phase3b_dev/surface_baseline_freeze.json") != (
        SURFACE_FREEZE_SHA
    ):
        raise SystemExit("surface freeze hash drift")
    if freeze.get("locked_test_used") is not False:
        raise SystemExit("surface freeze used locked test")

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("locked_test_present"):
        raise SystemExit("dev run includes locked test")
    if manifest.get("run_id") != CANONICAL_DEV_RUN:
        raise SystemExit(f"unexpected run_id {manifest.get('run_id')}")

    # Load activations
    k0 = load_file(str(run_dir / "k0_prompt_groups.safetensors"))
    k0_acts = np.asarray(k0["activations"], dtype=np.float32)
    group_map = json.loads((run_dir / "k0_group_map.json").read_text(encoding="utf-8"))
    group_index = {g["prompt_sha256"]: g["group_index"] for g in group_map}

    example_ids: list[str] = []
    splits: list[str] = []
    prompt_hashes: list[str] = []
    labels_parts = []
    resp_parts = []
    acts_parts = []
    valid_parts = []
    for sm in manifest["trajectory_shards"]:
        meta = json.loads((run_dir / sm["meta"]).read_text(encoding="utf-8"))
        if "locked_test" in meta.get("splits", []):
            raise SystemExit("locked_test in development shard")
        tens = load_file(str(run_dir / sm["shard"]))
        example_ids.extend(meta["example_ids"])
        splits.extend(meta["splits"])
        prompt_hashes.extend(meta["prompt_sha256"])
        labels_parts.append(tens["labels"])
        resp_parts.append(tens["response_n_tokens"])
        acts_parts.append(np.asarray(tens["activations"], dtype=np.float32))
        valid_parts.append(tens["valid_mask"].astype(bool))

    labels = np.concatenate(labels_parts)
    resp_lens = np.concatenate(resp_parts)
    acts = np.concatenate(acts_parts, axis=0)
    valid = np.concatenate(valid_parts, axis=0)
    if len(example_ids) != 3000:
        raise SystemExit(f"expected 3000 development rows, got {len(example_ids)}")
    if any(s == "locked_test" for s in splits):
        raise SystemExit("locked_test split present")

    layer_index = {li: i for i, li in enumerate(COARSE_TRANSFORMER_BLOCKS)}
    k_index = {k: i for i, k in enumerate(PREFIX_LENGTHS_K)}
    for i, ph in enumerate(prompt_hashes):
        gi = group_index[ph]
        acts[i, :, k_index[0], :] = k0_acts[gi]
        valid[i, k_index[0]] = True

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, use_fast=True
    )
    by_id: dict[str, dict] = {}
    for fname in ("phase3_train_metadata.jsonl", "phase3_validation_metadata.jsonl"):
        for row in _load_jsonl(data_dir / fname):
            tok = analyze_prompt_response_boundary(
                row["input_formatted"], row["model_outputs"], tokenizer
            )
            row = dict(row)
            row["_tok"] = tok
            by_id[row["example_id"]] = row
    # Ensure locked test never enters fit set
    locked_ids = {
        r["example_id"] for r in _load_jsonl(data_dir / "locked_test_metadata.jsonl")
    }
    if locked_ids & set(example_ids):
        raise SystemExit("locked-test IDs leaked into development fit set")

    rows = [by_id[eid] for eid in example_ids]
    fit_ids_sha = _sha_ids(example_ids)

    probe_manifest_entries = []
    for layer, k, name in (
        (12, 1, "probe_l12_k1"),
        (12, 0, "probe_l12_k0"),
    ):
        lj = layer_index[layer]
        ki = k_index[k]
        elig = [
            i
            for i in range(len(example_ids))
            if eligible_for_k(int(resp_lens[i]), k) and valid[i, ki]
        ]
        x = acts[elig, lj, ki]
        y = labels[elig]
        probe = MeanLinearProbe(ProbeConfig())
        probe.fit(x, y)
        mean = np.asarray(probe.transformation_mean, dtype=np.float64).ravel()
        std = np.asarray(probe.transformation_std, dtype=np.float64).ravel()
        coef = np.asarray(probe.classifier.coef_, dtype=np.float64).ravel()
        intercept = float(np.asarray(probe.classifier.intercept_).ravel()[0])
        payload = {
            "mean": mean,
            "std": std,
            "coef": coef,
            "intercept": np.asarray([intercept], dtype=np.float64),
            "C": np.asarray([0.01], dtype=np.float64),
            "layer": np.asarray([layer], dtype=np.int64),
            "k": np.asarray([k], dtype=np.int64),
            "n_fit": np.asarray([len(elig)], dtype=np.int64),
            "hidden_dim": np.asarray([x.shape[1]], dtype=np.int64),
        }
        out_path = art_dir / f"{name}.npz"
        # store as npz (gitignored alongside safetensors policy via *.npz? add to gitignore)
        np.savez_compressed(out_path, **payload)
        # also joblib of sklearn state for convenience
        joblib_path = art_dir / f"{name}_sklearn.joblib"
        joblib.dump(
            {
                "probe_config": {
                    "C": 0.01,
                    "fit_intercept": True,
                    "normalize": True,
                    "max_iter": 500,
                    "random_state": 42,
                },
                "mean": mean,
                "std": std,
                "coef": coef,
                "intercept": intercept,
                "sklearn_coef_": probe.classifier.coef_,
                "sklearn_intercept_": probe.classifier.intercept_,
            },
            joblib_path,
        )
        entry = {
            "name": name,
            "layer": layer,
            "k": k,
            "C": 0.01,
            "random_state": 42,
            "normalize": True,
            "max_iter": 500,
            "fit_intercept": True,
            "n_fit_examples": len(elig),
            "hidden_dim": int(x.shape[1]),
            "fit_example_ids_sha256": _sha_ids([example_ids[i] for i in elig]),
            "artifact_npz": str(out_path.relative_to(REPO_ROOT)),
            "artifact_npz_sha256": _sha_file(out_path),
            "artifact_joblib": str(joblib_path.relative_to(REPO_ROOT)),
            "artifact_joblib_sha256": _sha_file(joblib_path),
            "canonical_b1_run_id": CANONICAL_DEV_RUN,
            "model_revision": MODEL_REVISION,
            "locked_test_used": False,
            "phase4_labels_used": False,
        }
        probe_manifest_entries.append(entry)

    probe_manifest = {
        "created_at": utc_now_iso(),
        "purpose": "Phase 4 frozen Phase-3 deception probes (immutable)",
        "fit_on": "phase3_train_plus_validation_combined",
        "n_development_rows": 3000,
        "all_fit_example_ids_sha256": fit_ids_sha,
        "canonical_b1_run_id": CANONICAL_DEV_RUN,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "locked_test_used": False,
        "phase4_labels_used": False,
        "probes": {e["name"]: e for e in probe_manifest_entries},
        "primary_probe": "probe_l12_k1",
        "secondary_probe": "probe_l12_k0",
    }
    write_json(sum_dir / "frozen_probe_manifest.json", probe_manifest)

    # Surface comparators k0 / k1
    surface_entries = {}
    for k, name in ((0, "surface_k0_prompt_only"), (1, "surface_k1_prompt_plus_1")):
        elig = [
            i
            for i in range(len(example_ids))
            if eligible_for_k(int(resp_lens[i]), k) and valid[i, k_index[k]]
        ]
        texts = [
            visible_prompt_plus_prefix_text(
                rows[i]["input_formatted"],
                rows[i]["model_outputs"],
                k,
                rows[i]["_tok"].response_offsets_in_model_outputs,
            )
            for i in elig
        ]
        y = labels[elig]
        pipe = make_text_baseline_pipeline(C=10.0, random_state=42)
        pipe.fit(texts, y)
        path = art_dir / f"{name}.joblib"
        joblib.dump(pipe, path)
        surface_entries[name] = {
            "name": name,
            "k": k,
            "C": 10.0,
            "random_state": 42,
            "n_fit_examples": len(elig),
            "fit_example_ids_sha256": _sha_ids([example_ids[i] for i in elig]),
            "artifact_joblib": str(path.relative_to(REPO_ROOT)),
            "artifact_joblib_sha256": _sha_file(path),
            "word_tfidf_ngram_range": [1, 2],
            "char_tfidf_ngram_range": [3, 5],
            "locked_test_used": False,
            "phase4_labels_used": False,
        }

    surface_manifest = {
        "created_at": utc_now_iso(),
        "purpose": "Phase 4 frozen Phase-3 surface comparators",
        "fit_on": "phase3_train_plus_validation_combined",
        "n_development_rows": 3000,
        "all_fit_example_ids_sha256": fit_ids_sha,
        "canonical_b1_run_id": CANONICAL_DEV_RUN,
        "surface_baseline_freeze_sha256": SURFACE_FREEZE_SHA,
        "locked_test_used": False,
        "phase4_labels_used": False,
        "models": surface_entries,
    }
    write_json(sum_dir / "frozen_surface_manifest.json", surface_manifest)

    print(
        json.dumps(
            {
                "probe_manifest": str(sum_dir / "frozen_probe_manifest.json"),
                "surface_manifest": str(sum_dir / "frozen_surface_manifest.json"),
                "probes": {
                    e["name"]: e["artifact_npz_sha256"]
                    for e in probe_manifest_entries
                },
                "surfaces": {
                    k: v["artifact_joblib_sha256"] for k, v in surface_entries.items()
                },
                "locked_test_used": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
