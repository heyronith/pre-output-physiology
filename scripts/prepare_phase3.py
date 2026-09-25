#!/usr/bin/env python3
"""Phase 3A local preparation: grouping, splits, onset candidates, audit index.

No Modal. No new activations. No locked-test predictive fitting.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.baselines import contract_as_dict  # noqa: E402
from pre_output_physiology.data import binary_label_from_scale  # noqa: E402
from pre_output_physiology.onset import (  # noqa: E402
    annotate_onset,
    onset_coverage_summary,
)
from pre_output_physiology.provenance import (  # noqa: E402
    sha256_file,
    sha256_text,
    utc_now_iso,
    write_json,
    write_jsonl,
)
from pre_output_physiology.trajectory import (  # noqa: E402
    COARSE_TRANSFORMER_BLOCKS,
    PREFIX_LENGTHS_K,
    PRIMARY_ANCHOR_LAYER,
    compare_grouping_keys,
    id_prefix,
)

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
PRESERVE_FIELDS = [
    "ids",
    "input",
    "input_formatted",
    "model_outputs",
    "scale_labels",
    "labels",
    "scale_label_confidence",
    "scale_label_model",
    "scale_label_explanation",
    "label_explanation",
    "inputs",
]


def load_raw_jsonl(path: Path, upstream_split: str, source_filename: str) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            prompt = raw["input_formatted"]
            output = raw["model_outputs"]
            scale = float(raw["scale_labels"])
            row = {
                "example_id": str(raw["ids"]),
                "upstream_split": upstream_split,
                "source_filename": source_filename,
                "prompt_sha256": sha256_text(prompt),
                "id_prefix": id_prefix(str(raw["ids"])),
                "teacher_forced_text_sha256": sha256_text(prompt + output),
                "eventual_deception": binary_label_from_scale(scale),
            }
            for field in PRESERVE_FIELDS:
                if field in raw:
                    row[field] = raw[field]
            rows.append(row)
    return rows


def split_summary(rows: list[dict]) -> dict:
    labels = Counter(r["eventual_deception"] for r in rows)
    groups = {r["prompt_sha256"] for r in rows}
    return {
        "n_rows": len(rows),
        "n_prompt_groups": len(groups),
        "n_class1": labels.get(1, 0),
        "n_class0": labels.get(0, 0),
        "class1_prevalence": (labels.get(1, 0) / len(rows)) if rows else None,
    }


def assert_no_group_overlap(a: list[dict], b: list[dict], name_a: str, name_b: str) -> None:
    ga = {r["prompt_sha256"] for r in a}
    gb = {r["prompt_sha256"] for r in b}
    overlap = ga & gb
    if overlap:
        raise SystemExit(
            f"Prompt-group overlap between {name_a} and {name_b}: {len(overlap)} groups"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-jsonl",
        default=str(
            REPO_ROOT
            / "data/processed/phase2_roleplay/raw_jsonl/mistral_7b_incentivised_train.jsonl"
        ),
    )
    parser.add_argument(
        "--test-jsonl",
        default=str(
            REPO_ROOT
            / "data/processed/phase2_roleplay/raw_jsonl/mistral_7b_incentivised_test.jsonl"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "data/processed/phase3_roleplay"),
    )
    parser.add_argument("--skip-tokenizer", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = out_dir / "manifests"
    meta_dir.mkdir(parents=True, exist_ok=True)
    audit_raw_dir = out_dir / "onset_audit_raw"
    audit_raw_dir.mkdir(parents=True, exist_ok=True)
    audit_index_dir = REPO_ROOT / "artifacts" / "phase3a_summaries"
    audit_index_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path(args.train_jsonl)
    test_path = Path(args.test_jsonl)
    if not train_path.is_file() or not test_path.is_file():
        raise SystemExit(
            "Phase 2 raw JSONL missing. Run scripts/prepare_phase2.py first "
            f"(looked for {train_path} and {test_path})."
        )

    upstream_train = load_raw_jsonl(
        train_path, "upstream_train", "roleplaying/mistral_7b_incentivised_train.jsonl"
    )
    locked_test = load_raw_jsonl(
        test_path, "locked_test", "roleplaying/mistral_7b_incentivised_test.jsonl"
    )

    if len(upstream_train) != 3000 or len(locked_test) != 500:
        raise SystemExit(
            f"Unexpected sizes train={len(upstream_train)} test={len(locked_test)}"
        )

    grouping_report = compare_grouping_keys(
        [r["example_id"] for r in upstream_train + locked_test],
        [r["prompt_sha256"] for r in upstream_train + locked_test],
    )

    # GroupShuffleSplit on upstream train only
    groups = [r["prompt_sha256"] for r in upstream_train]
    y = [r["eventual_deception"] for r in upstream_train]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, val_idx = next(gss.split(upstream_train, y, groups))
    phase3_train = [upstream_train[i] for i in train_idx]
    phase3_val = [upstream_train[i] for i in val_idx]

    assert_no_group_overlap(phase3_train, phase3_val, "train", "val")
    assert_no_group_overlap(phase3_train, locked_test, "train", "locked_test")
    assert_no_group_overlap(phase3_val, locked_test, "val", "locked_test")

    split_packs = (
        ("train", phase3_train),
        ("validation", phase3_val),
        ("locked_test", locked_test),
    )
    for name, rows in split_packs:
        summ = split_summary(rows)
        if summ["n_class0"] == 0 or summ["n_class1"] == 0:
            raise SystemExit(f"Pathological split {name}: missing a class: {summ}")

    # Scenario deception rates (descriptive)
    group_labels: dict[str, list[int]] = defaultdict(list)
    for r in upstream_train + locked_test:
        group_labels[r["prompt_sha256"]].append(r["eventual_deception"])
    scenario_rates = {
        g: float(sum(ls) / len(ls)) for g, ls in group_labels.items()
    }

    tokenizer = None
    if not args.skip_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, use_fast=True
        )

    def annotate_split(rows: list[dict], split_name: str) -> list:
        anns = []
        for r in rows:
            ann = annotate_onset(
                example_id=r["example_id"],
                split=split_name,
                prompt_sha256=r["prompt_sha256"],
                model_output=r["model_outputs"],
                binary_label=r["eventual_deception"],
                scale_label_explanation=r.get("scale_label_explanation"),
                label_explanation=r.get("label_explanation") or None,
                tokenizer=tokenizer,
            )
            anns.append(ann)
            r["onset_status"] = ann.onset_status
            r["conservative_deceptive_sentence_onset_char"] = (
                ann.conservative_deceptive_sentence_onset_char
            )
            r["conservative_deceptive_sentence_onset_token"] = (
                ann.conservative_deceptive_sentence_onset_token
            )
            r["evidence_span_sha256"] = ann.evidence_span_sha256
        return anns

    train_anns = annotate_split(phase3_train, "phase3_train")
    val_anns = annotate_split(phase3_val, "phase3_validation")
    test_anns = annotate_split(locked_test, "locked_test")

    coverage = {
        "phase3_train": onset_coverage_summary(train_anns),
        "phase3_validation": onset_coverage_summary(val_anns),
        "locked_test": onset_coverage_summary(test_anns),
    }

    # Write processed metadata (no model weights; includes text for local audit)
    for name, rows in (
        ("phase3_train", phase3_train),
        ("phase3_validation", phase3_val),
        ("locked_test", locked_test),
    ):
        # Strip bulky nested inputs from commit-facing slim index later; full local copy here
        write_jsonl(out_dir / f"{name}_metadata.jsonl", rows)

    write_jsonl(
        out_dir / "onset_annotations.jsonl",
        [a.to_dict() for a in train_anns + val_anns + test_anns],
    )

    # Audit sample (seed 42): raw local artifact + commit-safe index
    import numpy as np

    rng = np.random.default_rng(42)

    def sample_anns(pool, n):
        pool = list(pool)
        if len(pool) <= n:
            return pool
        idx = rng.choice(len(pool), size=n, replace=False)
        return [pool[i] for i in sorted(idx.tolist())]

    hi_tv = [
        a
        for a in train_anns + val_anns
        if a.onset_status == "high_confidence"
    ]
    hi_te = [a for a in test_anns if a.onset_status == "high_confidence"]
    unresolved = [
        a
        for a in train_anns + val_anns + test_anns
        if a.onset_status == "unresolved"
    ]
    audit_anns = (
        sample_anns(hi_tv, 30)
        + sample_anns(hi_te, 20)
        + sample_anns(unresolved, 20)
    )
    id_to_row = {
        r["example_id"]: r for r in phase3_train + phase3_val + locked_test
    }
    audit_raw = []
    audit_index = []
    for a in audit_anns:
        row = id_to_row[a.example_id]
        audit_raw.append(
            {
                "example_id": a.example_id,
                "split": a.split,
                "prompt_sha256": a.prompt_sha256,
                "id_prefix": row["id_prefix"],
                "model_outputs": row["model_outputs"],
                "scale_label_explanation": row.get("scale_label_explanation"),
                "evidence_span": a.evidence_span,
                "conservative_deceptive_sentence_onset_char": (
                    a.conservative_deceptive_sentence_onset_char
                ),
                "conservative_deceptive_sentence_onset_token": (
                    a.conservative_deceptive_sentence_onset_token
                ),
                "onset_status": a.onset_status,
            }
        )
        audit_index.append(
            {
                "example_id": a.example_id,
                "split": a.split,
                "prompt_sha256": a.prompt_sha256,
                "onset_status": a.onset_status,
                "evidence_span_sha256": a.evidence_span_sha256,
                "onset_char": a.conservative_deceptive_sentence_onset_char,
                "onset_token": a.conservative_deceptive_sentence_onset_token,
                "model_output_sha256": sha256_text(row["model_outputs"]),
                "explanation_sha256": sha256_text(
                    str(row.get("scale_label_explanation") or "")
                ),
            }
        )

    audit_raw_path = audit_raw_dir / "onset_audit_sample.jsonl"
    write_jsonl(audit_raw_path, audit_raw)
    write_jsonl(audit_index_dir / "onset_audit_index.jsonl", audit_index)

    preserved_present = sorted(
        {
            f
            for r in upstream_train[:1]
            for f in PRESERVE_FIELDS
            if f in r
        }
    )

    manifest = {
        "created_at": utc_now_iso(),
        "phase": "phase3a",
        "gpu_authorized": False,
        "activations_collected": False,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "source_files": {
            "train": {
                "path": str(train_path),
                "sha256": sha256_file(train_path),
                "hub_filename": "roleplaying/mistral_7b_incentivised_train.jsonl",
            },
            "test": {
                "path": str(test_path),
                "sha256": sha256_file(test_path),
                "hub_filename": "roleplaying/mistral_7b_incentivised_test.jsonl",
            },
        },
        "grouping": grouping_report,
        "splits": {
            "phase3_train": split_summary(phase3_train),
            "phase3_validation": split_summary(phase3_val),
            "locked_test": split_summary(locked_test),
        },
        "group_overlap": {
            "train_val": 0,
            "train_test": 0,
            "val_test": 0,
        },
        "coarse_transformer_blocks": COARSE_TRANSFORMER_BLOCKS,
        "prefix_lengths_k": PREFIX_LENGTHS_K,
        "primary_anchor_layer": PRIMARY_ANCHOR_LAYER,
        "surface_baseline_contract": contract_as_dict(),
        "onset_coverage": coverage,
        "preserved_fields_present": preserved_present,
        "judge_explanation_fields_present": [
            f
            for f in ("scale_label_explanation", "label_explanation")
            if f in preserved_present
        ],
        "scenario_deception_rate_summary": {
            "n_groups": len(scenario_rates),
            "mean_rate": float(sum(scenario_rates.values()) / len(scenario_rates)),
            "n_groups_rate_0": sum(1 for v in scenario_rates.values() if v == 0.0),
            "n_groups_rate_1": sum(1 for v in scenario_rates.values() if v == 1.0),
            "n_groups_mixed": sum(1 for v in scenario_rates.values() if 0.0 < v < 1.0),
        },
        "onset_audit_raw_path": str(audit_raw_path),
        "onset_audit_index_path": str(audit_index_dir / "onset_audit_index.jsonl"),
        "onset_audit_note": (
            "Raw audit JSONL with model outputs/explanations is gitignored under "
            "data/processed/. Commit-safe index stores IDs/hashes only."
        ),
        "split_method": "GroupShuffleSplit(test_size=0.20, random_state=42)",
    }
    write_json(meta_dir / "phase3a_prepare_manifest.json", manifest)
    write_json(audit_index_dir / "phase3a_prepare_summary.json", {
        "splits": manifest["splits"],
        "grouping_agree": grouping_report["agree"],
        "grouping": {
            k: grouping_report[k]
            for k in (
                "n_id_prefixes",
                "n_prompt_hashes",
                "n_prefixes_with_multiple_hashes",
                "n_hashes_with_multiple_prefixes",
                "scientific_key",
                "agree",
            )
        },
        "onset_coverage": coverage,
        "coarse_transformer_blocks": COARSE_TRANSFORMER_BLOCKS,
        "prefix_lengths_k": PREFIX_LENGTHS_K,
        "primary_anchor_layer": PRIMARY_ANCHOR_LAYER,
        "gpu_authorized": False,
        "activations_collected": False,
    })

    # Also dump experiment config snapshot confirmation
    exp = yaml.safe_load(
        (REPO_ROOT / "configs/experiments/phase3_preoutput_scan.yaml").read_text(
            encoding="utf-8"
        )
    )
    print(json.dumps({
        "status": exp.get("status"),
        "splits": manifest["splits"],
        "grouping_agree": grouping_report["agree"],
        "onset_coverage": coverage,
        "audit_raw": str(audit_raw_path),
        "audit_index": str(audit_index_dir / "onset_audit_index.jsonl"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
