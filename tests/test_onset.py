"""Conservative deceptive-onset annotation tests (activation-independent)."""

from __future__ import annotations

from pre_output_physiology.onset import (
    annotate_onset,
    char_index_to_generated_token_index,
    conservative_sentence_onset_char,
    find_exact_evidence_span,
    sentence_start_index,
)


class _FakeTok:
    """Minimal tokenizer stub with offset mapping."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        # One "token" per whitespace-separated word for testing.
        tokens = text.split(" ") if text else []
        ids = list(range(len(tokens)))
        offsets = []
        pos = 0
        for i, tok in enumerate(tokens):
            if i > 0:
                pos += 1  # space
            start = pos
            end = pos + len(tok)
            offsets.append((start, end))
            pos = end
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(f"t{i}" for i in ids)


def test_exact_quoted_evidence_preferred() -> None:
    output = "Hello. The vault is empty. Goodbye."
    expl = 'The model lied when it said "The vault is empty."'
    span = find_exact_evidence_span(output, expl)
    assert span is not None
    assert "vault is empty" in span.lower()


def test_unresolved_without_exact_span() -> None:
    ann = annotate_onset(
        example_id="x",
        split="train",
        prompt_sha256="h",
        model_output="I refuse to answer.",
        binary_label=1,
        scale_label_explanation="The response is evasive and deceptive overall.",
    )
    assert ann.onset_status == "unresolved"


def test_honest_not_applicable() -> None:
    ann = annotate_onset(
        example_id="x",
        split="train",
        prompt_sha256="h",
        model_output="The vault is empty.",
        binary_label=0,
        scale_label_explanation='Said "The vault is empty."',
    )
    assert ann.onset_status == "not_applicable_honest"


def test_conservative_onset_moves_to_sentence_start() -> None:
    text = "Intro sentence. Later the vault is empty forever. End."
    evidence = "vault is empty"
    onset = conservative_sentence_onset_char(text, evidence)
    assert onset == text.find("Later the vault")


def test_sentence_start_index() -> None:
    text = "A. B sentence here."
    idx = text.find("sentence")
    assert sentence_start_index(text, idx) == text.find("B")


def test_char_to_token_alignment() -> None:
    text = "alpha beta gamma"
    tok = _FakeTok()
    # start of "beta"
    char_i = text.find("beta")
    assert char_index_to_generated_token_index(text, char_i, tok) == 1
    # start of string
    assert char_index_to_generated_token_index(text, 0, tok) == 0


def test_high_confidence_annotation_has_token_index() -> None:
    output = "First. The safe code is 0000 tonight."
    expl = 'Deceptive claim: "The safe code is 0000 tonight."'
    ann = annotate_onset(
        example_id="e1",
        split="phase3_train",
        prompt_sha256="abc",
        model_output=output,
        binary_label=1,
        scale_label_explanation=expl,
        tokenizer=_FakeTok(),
    )
    assert ann.onset_status == "high_confidence"
    assert ann.evidence_span is not None
    assert ann.conservative_deceptive_sentence_onset_char is not None
    assert ann.conservative_deceptive_sentence_onset_token is not None
    assert ann.explanation_source_field == "scale_label_explanation"
