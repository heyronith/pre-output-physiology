#!/usr/bin/env python3
"""Local Phase 2 probe fitting and metrics (no GPU)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from safetensors.numpy import load_file

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.metrics import evaluate_binary_classifier  # noqa: E402
from pre_output_physiology.probes import MeanLinearProbe, ProbeConfig  # noqa: E402
from pre_output_physiology.provenance import (  # noqa: E402
    build_run_manifest,
    git_commit,
    sha256_file,
    software_versions,
    utc_now_iso,
    write_json,
)


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_metadata(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="artifacts/runs/<run_id>")
    parser.add_argument(
        "--experiment-config",
        default=str(REPO_ROOT / "configs/experiments/phase2_positive_control.yaml"),
    )
    parser.add_argument(
        "--dataset-config",
        default=str(REPO_ROOT / "configs/datasets/roleplay_deception.yaml"),
    )
    parser.add_argument(
        "--model-config",
        default=str(REPO_ROOT / "configs/models/mistral_7b_instruct_v02.yaml"),
    )
    parser.add_argument(
        "--prepare-manifest",
        default=str(
            REPO_ROOT
            / "data/processed/phase2_roleplay/manifests/prepare_manifest.json"
        ),
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    exp = load_yaml(Path(args.experiment_config))
    ds = load_yaml(Path(args.dataset_config))
    model_cfg = load_yaml(Path(args.model_config))
    prepare = json.loads(Path(args.prepare_manifest).read_text(encoding="utf-8"))
    gpu_result = json.loads((run_dir / "gpu_result.json").read_text(encoding="utf-8"))

    tensors = load_file(str(run_dir / "activations.safetensors"))
    activations = np.asarray(tensors["activations"], dtype=np.float64)
    labels_from_file = np.asarray(tensors["labels"], dtype=np.int64)
    meta = load_metadata(run_dir / "metadata.jsonl")

    if len(meta) != activations.shape[0]:
        raise SystemExit("metadata/activation length mismatch")
    if not np.array_equal(labels_from_file, np.asarray([r["binary_label"] for r in meta])):
        raise SystemExit("label mismatch between safetensors and metadata")

    train_idx = [i for i, r in enumerate(meta) if r["split"] == "train"]
    test_idx = [i for i, r in enumerate(meta) if r["split"] == "test"]
    if not train_idx or not test_idx:
        raise SystemExit(
            "Need both train and test rows in metadata; "
            f"got train={len(train_idx)} test={len(test_idx)}"
        )

    x_train = activations[train_idx]
    y_train = labels_from_file[train_idx]
    x_test = activations[test_idx]
    y_test = labels_from_file[test_idx]

    probe_cfg = ProbeConfig(
        C=float(exp["probe"]["C"]),
        fit_intercept=bool(exp["probe"]["fit_intercept"]),
        normalize=bool(exp["probe"]["normalize"]),
        max_iter=int(exp["probe"]["max_iter"]),
        random_state=int(exp["probe"]["random_state"]),
    )
    probe = MeanLinearProbe(probe_cfg).fit(x_train, y_train)
    scores = probe.predict_proba(x_test)
    preds = probe.predict(x_test)

    metrics = evaluate_binary_classifier(
        y_test,
        scores,
        preds,
        n_bootstrap=int(exp["analysis"]["bootstrap_resamples"]),
        bootstrap_seed=int(exp["analysis"]["bootstrap_seed"]),
    )
    metrics["probe"] = probe.export_state()
    metrics["train_n"] = int(len(train_idx))
    metrics["test_n"] = int(len(test_idx))
    metrics["train_pos"] = int(y_train.sum())
    metrics["train_neg"] = int(len(y_train) - y_train.sum())

    auroc = metrics["auroc"]
    gate = exp["acceptance_gate"]
    if auroc >= gate["pass_min_auroc"] and metrics["auroc_ci_low"] > 0.5:
        verdict = "PASS"
    elif auroc >= gate["investigate_min_auroc"]:
        verdict = "INVESTIGATE"
    else:
        verdict = "FAIL"
    metrics["phase2_verdict"] = verdict

    write_json(run_dir / "metrics.json", metrics)

    act_hash = sha256_file(run_dir / "activations.safetensors")
    meta_hash = sha256_file(run_dir / "metadata.jsonl")
    manifest = build_run_manifest(
        run_id=gpu_result["run_id"],
        timestamp=utc_now_iso(),
        git_commit=git_commit(REPO_ROOT),
        model_id=model_cfg["model_id"],
        model_revision=model_cfg["revision"],
        dataset_revision=ds["lasr_hf_revision"],
        config_file=str(Path(args.experiment_config).resolve()),
        random_seed=probe_cfg.random_state,
        hardware="modal",
        gpu_type=gpu_result.get("gpu_type", "L40S"),
        software_versions=software_versions(),
        precision=model_cfg["dtype"],
        generation_parameters={
            "mode": "teacher_forced_no_generation",
            "add_special_tokens": False,
            "text_construction": "input_formatted+model_outputs",
        },
        activation_locations_collected={
            "transformer_block_index": exp["instrumentation"]["transformer_block_index"],
            "hook_module_path": exp["instrumentation"]["hook_module_path"],
            "aggregation": exp["instrumentation"]["aggregation"],
        },
        output_artifact_hashes={
            "activations.safetensors": act_hash,
            "metadata.jsonl": meta_hash,
            "metrics.json": sha256_file(run_dir / "metrics.json"),
        },
        # Phase 2 extras
        apollo_revision=ds["apollo_revision"],
        lasr_code_revision=ds["lasr_code_revision"],
        lasr_hf_repo=ds["lasr_hf_repo"],
        train_jsonl=ds["train_jsonl"],
        test_jsonl=ds["test_jsonl"],
        transformer_block_index=exp["instrumentation"]["transformer_block_index"],
        hook_module_path=exp["instrumentation"]["hook_module_path"],
        aggregation_method=exp["instrumentation"]["aggregation"],
        batch_size=gpu_result.get("batch_size"),
        model_dtype=gpu_result.get("dtype"),
        tokenizer_revision=model_cfg["revision"],
        modal_gpu=gpu_result.get("gpu_type"),
        execution_duration_seconds={
            "model_load": gpu_result.get("model_load_seconds"),
            "extract": gpu_result.get("extract_seconds"),
            "wall": gpu_result.get("wall_seconds"),
        },
        estimated_cost_usd=gpu_result.get("estimated_cost_usd"),
        input_artifact_hashes={
            "train_jsonl": prepare["files"]["train"]["sha256"],
            "test_jsonl": prepare["files"]["test"]["sha256"],
        },
        phase2_verdict=verdict,
        quantization=None,
    )
    write_json(run_dir / "run_manifest.json", manifest)

    # Small commit-friendly summary (no activations).
    summary_dir = REPO_ROOT / "artifacts" / "phase2_summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        summary_dir / f"{gpu_result['run_id']}_metrics_summary.json",
        {
            "run_id": gpu_result["run_id"],
            "verdict": verdict,
            "auroc": metrics["auroc"],
            "auroc_ci": [metrics["auroc_ci_low"], metrics["auroc_ci_high"]],
            "auprc": metrics["auprc"],
            "accuracy": metrics["accuracy"],
            "tpr_at_or_below_1pct_fpr": metrics["tpr_at_or_below_1pct_fpr"],
            "actual_fpr_at_tpr_metric": metrics["actual_fpr_at_tpr_metric"],
            "n_train": metrics["train_n"],
            "n_test": metrics["test_n"],
            "estimated_cost_usd": gpu_result.get("estimated_cost_usd"),
            "l40s_usd_per_hour_assumed": gpu_result.get("l40s_usd_per_hour_assumed"),
            "activation_shape": gpu_result.get("activation_shape"),
            "activations_sha256": act_hash,
        },
    )

    print(json.dumps({"verdict": verdict, "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
