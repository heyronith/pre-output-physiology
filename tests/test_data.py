"""Dataset/label mapping tests — no GPU / downloads."""

from __future__ import annotations

from pre_output_physiology.data import (
    binary_label_from_scale,
    build_teacher_forced_text,
    cross_split_leakage_report,
    parse_roleplay_jsonl,
    summarize_split,
)
from pre_output_physiology.provenance import sha256_text


def test_binary_label_mapping_lasr_convention() -> None:
    assert binary_label_from_scale(1) == 1
    assert binary_label_from_scale(5) == 1
    assert binary_label_from_scale(5.0) == 1
    assert binary_label_from_scale(6) == 0
    assert binary_label_from_scale(10) == 0


def test_teacher_forced_text_concatenation() -> None:
    text = build_teacher_forced_text("<s> [INST] hi [/INST] Role:", " secret")
    assert text == "<s> [INST] hi [/INST] Role: secret"
    assert sha256_text(text) == sha256_text(text)


def test_parse_roleplay_jsonl(tmp_path) -> None:
    path = tmp_path / "toy.jsonl"
    row = {
        "ids": "1_0",
        "input": "Q?",
        "input_formatted": "<s> [INST] sys\n\nQ? [/INST] A:",
        "model_outputs": " no",
        "inputs": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Q?"},
            {"role": "assistant", "content": "A:"},
            {"role": "assistant", "content": " no"},
        ],
        "scale_labels": 1,
        "labels": "positive",
        "scale_label_confidence": 10,
        "scale_label_model": "gpt-4o-mini",
    }
    path.write_text(__import__("json").dumps(row) + "\n", encoding="utf-8")
    examples = parse_roleplay_jsonl(path, split="train", source_filename="toy.jsonl")
    assert len(examples) == 1
    assert examples[0].binary_label == 1
    assert examples[0].teacher_forced_text.endswith(" no")
    summary = summarize_split(examples)
    assert summary["n"] == 1
    assert summary["n_positive_class1"] == 1


def test_cross_split_leakage_detects_overlap() -> None:
    from pre_output_physiology.data import RoleplayExample

    def ex(eid: str, text: str, split: str) -> RoleplayExample:
        return RoleplayExample(
            example_id=eid,
            split=split,
            source_filename="x",
            input_text="",
            input_formatted=text,
            model_output="",
            teacher_forced_text=text,
            messages=[],
            scale_label=1,
            binary_label=1,
            labels_field="positive",
            text_sha256=sha256_text(text),
        )

    train = [ex("1_0", "aaa", "train")]
    test = [ex("1_1", "aaa", "test"), ex("2_0", "bbb", "test")]
    report = cross_split_leakage_report(train, test)
    assert report["n_exact_text_overlap"] == 1
    assert report["n_scenario_prefix_overlap"] == 1
