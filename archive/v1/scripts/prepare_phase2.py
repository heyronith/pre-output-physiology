#!/usr/bin/env python3
"""Download only Phase 2 RoleplayDeception JSONL text/label files and write local manifests."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.data import (  # noqa: E402
    cross_split_leakage_report,
    parse_roleplay_jsonl,
    summarize_split,
)
from pre_output_physiology.provenance import (  # noqa: E402
    assert_not_main_revision,
    sha256_file,
    utc_now_iso,
    write_json,
)


def load_dataset_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    assert_not_main_revision("lasr_hf_revision", cfg["lasr_hf_revision"])
    assert_not_main_revision("apollo_revision", cfg["apollo_revision"])
    assert_not_main_revision("lasr_code_revision", cfg["lasr_code_revision"])
    return cfg


def download_jsonl(cfg: dict, filename: str, dest_dir: Path) -> Path:
    local = hf_hub_download(
        repo_id=cfg["lasr_hf_repo"],
        repo_type="dataset",
        revision=cfg["lasr_hf_revision"],
        filename=filename,
    )
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / Path(filename).name
    shutil.copy2(local, out)
    return out


def maybe_copy_apollo_yaml(cfg: dict, dest_dir: Path) -> Path | None:
    candidates = [
        Path("/tmp/deception-detection") / cfg["apollo_source_file"],
        REPO_ROOT / "data" / "external" / "apollo_roleplaying_dataset.yaml",
    ]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "apollo_roleplaying_dataset.yaml"
    for cand in candidates:
        if cand.is_file():
            shutil.copy2(cand, dest)
            digest = sha256_file(dest)
            expected = cfg["apollo_dataset_yaml_sha256"]
            if digest != expected:
                raise ValueError(
                    f"Apollo dataset.yaml SHA256 mismatch: got {digest}, expected {expected}"
                )
            return dest
    return None


def select_preflight(examples: list, n_pos: int = 50, n_neg: int = 50) -> list:
    pos = [ex for ex in examples if ex.binary_label == 1][:n_pos]
    neg = [ex for ex in examples if ex.binary_label == 0][:n_neg]
    return pos + neg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-config",
        default=str(REPO_ROOT / "configs/datasets/roleplay_deception.yaml"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "data" / "processed" / "phase2_roleplay"),
    )
    args = parser.parse_args()

    cfg = load_dataset_config(Path(args.dataset_config))
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw_jsonl"
    meta_dir = out_dir / "manifests"
    meta_dir.mkdir(parents=True, exist_ok=True)

    train_path = download_jsonl(cfg, cfg["train_jsonl"], raw_dir)
    test_path = download_jsonl(cfg, cfg["test_jsonl"], raw_dir)
    apollo_path = maybe_copy_apollo_yaml(cfg, out_dir / "apollo")

    train = parse_roleplay_jsonl(
        train_path, split="train", source_filename=cfg["train_jsonl"]
    )
    test = parse_roleplay_jsonl(
        test_path, split="test", source_filename=cfg["test_jsonl"]
    )
    preflight = select_preflight(train, 50, 50)

    train_summary = summarize_split(train)
    test_summary = summarize_split(test)
    leakage = cross_split_leakage_report(train, test)

    index = {
        "created_at": utc_now_iso(),
        "dataset_config": str(Path(args.dataset_config).resolve()),
        "lasr_hf_repo": cfg["lasr_hf_repo"],
        "lasr_hf_revision": cfg["lasr_hf_revision"],
        "apollo_revision": cfg["apollo_revision"],
        "apollo_source_file": cfg["apollo_source_file"],
        "apollo_dataset_yaml_sha256": cfg["apollo_dataset_yaml_sha256"],
        "apollo_local_copy": str(apollo_path) if apollo_path else None,
        "lasr_code_revision": cfg["lasr_code_revision"],
        "files": {
            "train": {
                "path": str(train_path),
                "sha256": sha256_file(train_path),
                "hub_filename": cfg["train_jsonl"],
            },
            "test": {
                "path": str(test_path),
                "sha256": sha256_file(test_path),
                "hub_filename": cfg["test_jsonl"],
            },
        },
        "train_summary": train_summary,
        "test_summary": test_summary,
        "leakage": leakage,
        "label_semantics": cfg["label_mapping"],
        "teacher_forced_text": cfg["teacher_forced_text"],
        "preflight_n": len(preflight),
        "preflight_n_pos": sum(ex.binary_label == 1 for ex in preflight),
        "preflight_n_neg": sum(ex.binary_label == 0 for ex in preflight),
    }
    write_json(meta_dir / "prepare_manifest.json", index)

    for name, rows in ("train", train), ("test", test), ("preflight", preflight):
        payload = [
            {
                "example_id": ex.example_id,
                "split": ex.split if name != "preflight" else "preflight",
                "source_filename": ex.source_filename,
                "teacher_forced_text": ex.teacher_forced_text,
                "text_sha256": ex.text_sha256,
                "binary_label": ex.binary_label,
                "scale_label": ex.scale_label,
                "labels_field": ex.labels_field,
            }
            for ex in rows
        ]
        out = out_dir / f"{name}_examples.jsonl"
        with out.open("w", encoding="utf-8") as handle:
            for row in payload:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"Wrote {out} ({len(payload)} rows)")

    print(
        json.dumps(
            {
                "train": train_summary,
                "test": test_summary,
                "leakage_n_id_overlap": leakage["n_id_overlap"],
                "leakage_n_text_overlap": leakage["n_exact_text_overlap"],
                "leakage_n_scenario_overlap": leakage["n_scenario_prefix_overlap"],
                "apollo_yaml_present": apollo_path is not None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
