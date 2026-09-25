"""RoleplayDeception / LASR JSONL ingestion for Phase 2."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from pre_output_physiology.provenance import sha256_text

SplitName = Literal["train", "test", "preflight"]


@dataclass(frozen=True)
class RoleplayExample:
    example_id: str
    split: str
    source_filename: str
    input_text: str
    input_formatted: str
    model_output: str
    teacher_forced_text: str
    messages: list[dict[str, Any]]
    scale_label: float
    binary_label: int
    labels_field: str
    text_sha256: str
    scale_label_confidence: float | None = None
    scale_label_model: str | None = None

    def to_metadata_row(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "split": self.split,
            "source_filename": self.source_filename,
            "text_sha256": self.text_sha256,
            "binary_label": self.binary_label,
            "scale_label": self.scale_label,
            "labels_field": self.labels_field,
            "token_count": None,
        }


def binary_label_from_scale(scale_label: float, positive_class_max_inclusive: float = 5) -> int:
    """Map honesty-grade scale_labels to binary deception label (LASR convention).

    FACT FROM SOURCE (activation_handlers._load_labels_from_local_jsonl):
      scale_labels <= 5 -> 1
      scale_labels > 5  -> 0

    Class 1 semantics: deceptive / dishonest (low honesty grade; labels=="positive").
    Class 0 semantics: honest (high honesty grade; labels=="negative").
    """
    return 1 if float(scale_label) <= positive_class_max_inclusive else 0


def build_teacher_forced_text(input_formatted: str, model_output: str) -> str:
    """Exact upstream formatted prompt + fixed response (no chat-template rebuild)."""
    return f"{input_formatted}{model_output}"


def parse_roleplay_jsonl(
    path: Path | str,
    *,
    split: str,
    source_filename: str,
    positive_class_max_inclusive: float = 5,
) -> list[RoleplayExample]:
    path = Path(path)
    examples: list[RoleplayExample] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            messages = row["inputs"]
            if isinstance(messages, str):
                messages = json.loads(messages)
            input_formatted = row["input_formatted"]
            model_output = row["model_outputs"]
            teacher = build_teacher_forced_text(input_formatted, model_output)
            scale = float(row["scale_labels"])
            examples.append(
                RoleplayExample(
                    example_id=str(row["ids"]),
                    split=split,
                    source_filename=source_filename,
                    input_text=str(row.get("input", "")),
                    input_formatted=input_formatted,
                    model_output=model_output,
                    teacher_forced_text=teacher,
                    messages=list(messages),
                    scale_label=scale,
                    binary_label=binary_label_from_scale(
                        scale, positive_class_max_inclusive
                    ),
                    labels_field=str(row.get("labels", "")),
                    text_sha256=sha256_text(teacher),
                    scale_label_confidence=(
                        float(row["scale_label_confidence"])
                        if row.get("scale_label_confidence") is not None
                        else None
                    ),
                    scale_label_model=row.get("scale_label_model"),
                )
            )
    return examples


def scenario_prefix(example_id: str) -> str:
    return str(example_id).split("_", 1)[0]


def summarize_split(examples: Iterable[RoleplayExample]) -> dict[str, Any]:
    examples = list(examples)
    texts = [ex.teacher_forced_text for ex in examples]
    ids = [ex.example_id for ex in examples]
    text_counts = Counter(texts)
    id_counts = Counter(ids)
    duplicate_texts = sorted([t for t, c in text_counts.items() if c > 1], key=len)
    duplicate_ids = sorted([i for i, c in id_counts.items() if c > 1])
    binary = Counter(ex.binary_label for ex in examples)
    scenarios = Counter(scenario_prefix(ex.example_id) for ex in examples)
    return {
        "n": len(examples),
        "n_positive_class1": int(binary.get(1, 0)),
        "n_negative_class0": int(binary.get(0, 0)),
        "class1_prevalence": (binary.get(1, 0) / len(examples)) if examples else None,
        "n_unique_texts": len(text_counts),
        "n_duplicate_text_groups": sum(1 for c in text_counts.values() if c > 1),
        "n_rows_in_duplicate_text_groups": sum(c for c in text_counts.values() if c > 1),
        "n_extra_duplicate_text_rows": len(examples) - len(text_counts),
        "duplicate_text_sha256s": [sha256_text(t) for t in duplicate_texts[:50]],
        "n_duplicate_ids": len(duplicate_ids),
        "duplicate_ids": duplicate_ids[:50],
        "n_unique_scenario_prefixes": len(scenarios),
        "labels_field_counts": dict(Counter(ex.labels_field for ex in examples)),
        "scale_label_counts": {
            str(k): v for k, v in Counter(ex.scale_label for ex in examples).items()
        },
    }


def cross_split_leakage_report(
    train: list[RoleplayExample], test: list[RoleplayExample]
) -> dict[str, Any]:
    train_ids = {ex.example_id for ex in train}
    test_ids = {ex.example_id for ex in test}
    train_texts = {ex.teacher_forced_text for ex in train}
    test_texts = {ex.teacher_forced_text for ex in test}
    train_sc = {scenario_prefix(ex.example_id) for ex in train}
    test_sc = {scenario_prefix(ex.example_id) for ex in test}
    return {
        "id_overlap": sorted(train_ids & test_ids),
        "n_id_overlap": len(train_ids & test_ids),
        "exact_text_overlap_sha256s": sorted(
            sha256_text(t) for t in (train_texts & test_texts)
        ),
        "n_exact_text_overlap": len(train_texts & test_texts),
        "scenario_prefix_overlap": sorted(train_sc & test_sc),
        "n_scenario_prefix_overlap": len(train_sc & test_sc),
    }


def examples_to_jsonable(examples: list[RoleplayExample]) -> list[dict[str, Any]]:
    return [asdict(ex) for ex in examples]
