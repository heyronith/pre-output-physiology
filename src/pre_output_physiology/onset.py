"""Activation-independent conservative deceptive-onset candidates."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any

from pre_output_physiology.provenance import sha256_text

_QUOTE_RE = re.compile(r"[\"“”]([^\"“”]{5,})[\"“”]|'([^']{5,})'")


def normalize_for_match(text: str) -> str:
    """Normalize quotes/whitespace only (no fuzzy semantic matching)."""
    text = (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )
    return re.sub(r"\s+", " ", text).strip()


def extract_quoted_spans(explanation: str) -> list[str]:
    spans: list[str] = []
    for match in _QUOTE_RE.finditer(explanation):
        span = match.group(1) or match.group(2) or ""
        span = normalize_for_match(span)
        if len(span) >= 8:
            spans.append(span)
    return spans


def _locate_span_in_original(original: str, normalized_span: str) -> tuple[str, int] | None:
    """Map a whitespace-normalized span back to a substring of ``original``."""
    if not normalized_span:
        return None
    idx = original.find(normalized_span)
    if idx >= 0:
        return original[idx : idx + len(normalized_span)], idx
    idx = original.lower().find(normalized_span.lower())
    if idx >= 0:
        return original[idx : idx + len(normalized_span)], idx

    # Collapse whitespace / curly quotes while tracking indices into original.
    norm_chars: list[str] = []
    norm_to_orig: list[int] = []
    prev_space = False
    for i, ch in enumerate(original):
        if ch.isspace():
            if not prev_space and norm_chars:
                norm_chars.append(" ")
                norm_to_orig.append(i)
                prev_space = True
            continue
        mapped = (
            '"'
            if ch in "\u201c\u201d"
            else "'"
            if ch in "\u2018\u2019"
            else ch
        )
        norm_chars.append(mapped)
        norm_to_orig.append(i)
        prev_space = False

    start_map = 0
    while start_map < len(norm_chars) and norm_chars[start_map].isspace():
        start_map += 1
    end_map = len(norm_chars)
    while end_map > start_map and norm_chars[end_map - 1].isspace():
        end_map -= 1
    collapsed = "".join(norm_chars[start_map:end_map])
    target = normalize_for_match(normalized_span)
    pos = collapsed.lower().find(target.lower())
    if pos < 0:
        return None
    a = start_map + pos
    b = a + len(target) - 1
    if a >= len(norm_to_orig) or b >= len(norm_to_orig):
        return None
    orig_start = norm_to_orig[a]
    orig_end = norm_to_orig[b] + 1
    return original[orig_start:orig_end], orig_start


def find_exact_evidence_span(
    model_output: str,
    explanation: str,
    *,
    min_chars: int = 20,
    min_words: int = 6,
) -> str | None:
    """Return a high-confidence evidence substring of model_output found in explanation."""
    out_n = normalize_for_match(model_output)
    expl_n = normalize_for_match(explanation)
    if not out_n or not expl_n:
        return None
    out_lower = out_n.lower()
    expl_lower = expl_n.lower()

    best: str | None = None
    for quote in extract_quoted_spans(explanation):
        q = normalize_for_match(quote)
        if len(q) < 8 or q.lower() not in out_lower:
            continue
        located = _locate_span_in_original(model_output, q)
        if located is None:
            continue
        span, _ = located
        if best is None or len(span) > len(best):
            best = span
    if best is not None:
        return best

    words = out_n.split()
    max_w = min(len(words), 16)
    for wlen in range(max_w, min_words - 1, -1):
        for i in range(0, len(words) - wlen + 1):
            span_n = " ".join(words[i : i + wlen])
            if len(span_n) < min_chars:
                continue
            if span_n.lower() not in expl_lower:
                continue
            located = _locate_span_in_original(model_output, span_n)
            if located is not None:
                return located[0]
    return None


def sentence_start_index(text: str, char_index: int) -> int:
    """Move backward to the start of the sentence containing char_index."""
    if char_index <= 0:
        return 0
    char_index = min(char_index, max(0, len(text) - 1))
    i = char_index
    while i > 0:
        ch = text[i - 1]
        if ch in ".?!":
            j = i
            while j < len(text) and text[j].isspace():
                j += 1
            return j
        i -= 1
    return 0


def conservative_sentence_onset_char(model_output: str, evidence_span: str) -> int | None:
    located = _locate_span_in_original(model_output, evidence_span)
    if located is None:
        return None
    _, idx = located
    return sentence_start_index(model_output, idx)


@dataclass
class OnsetAnnotation:
    example_id: str
    split: str
    prompt_sha256: str
    onset_status: str
    evidence_span: str | None
    evidence_span_sha256: str | None
    explanation_source_field: str | None
    conservative_deceptive_sentence_onset_char: int | None
    conservative_deceptive_sentence_onset_token: int | None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def char_index_to_generated_token_index(
    model_output: str,
    char_index: int,
    tokenizer,
    *,
    input_formatted: str | None = None,
) -> int:
    """Map character index in model_output to assistant token index.

    When ``input_formatted`` is provided, uses the canonical Phase 3
    full-sequence response suffix offsets. Otherwise falls back to standalone
    tokenization of ``model_output`` (legacy / unit-test path).
    """
    char_index = max(0, min(char_index, len(model_output)))
    if input_formatted is not None:
        from pre_output_physiology.trajectory import (
            analyze_prompt_response_boundary,
            char_index_to_canonical_response_token,
        )

        tok = analyze_prompt_response_boundary(
            input_formatted, model_output, tokenizer
        )
        return char_index_to_canonical_response_token(
            char_index, tok.response_offsets_in_model_outputs
        )

    encoded = tokenizer(
        model_output,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded.get("offset_mapping")
    if not offsets:
        ids = encoded["input_ids"]
        for i in range(len(ids)):
            prefix = tokenizer.decode(ids[: i + 1], skip_special_tokens=False)
            if len(prefix) >= char_index:
                return i
        return max(0, len(ids) - 1)

    for i, (start, end) in enumerate(offsets):
        if start <= char_index < end:
            return i
        if char_index < start:
            return max(0, i - 1)
    return max(0, len(offsets) - 1)


def annotate_onset(
    *,
    example_id: str,
    split: str,
    prompt_sha256: str,
    model_output: str,
    binary_label: int,
    scale_label_explanation: str | None,
    label_explanation: str | None = None,
    tokenizer=None,
    input_formatted: str | None = None,
) -> OnsetAnnotation:
    if binary_label != 1:
        return OnsetAnnotation(
            example_id=example_id,
            split=split,
            prompt_sha256=prompt_sha256,
            onset_status="not_applicable_honest",
            evidence_span=None,
            evidence_span_sha256=None,
            explanation_source_field=None,
            conservative_deceptive_sentence_onset_char=None,
            conservative_deceptive_sentence_onset_token=None,
        )

    source_field = None
    explanation = ""
    if scale_label_explanation and str(scale_label_explanation).strip():
        explanation = str(scale_label_explanation)
        source_field = "scale_label_explanation"
    elif label_explanation and str(label_explanation).strip():
        explanation = str(label_explanation)
        source_field = "label_explanation"

    evidence = find_exact_evidence_span(model_output, explanation) if explanation else None
    if evidence is None:
        return OnsetAnnotation(
            example_id=example_id,
            split=split,
            prompt_sha256=prompt_sha256,
            onset_status="unresolved",
            evidence_span=None,
            evidence_span_sha256=None,
            explanation_source_field=source_field,
            conservative_deceptive_sentence_onset_char=None,
            conservative_deceptive_sentence_onset_token=None,
            notes="no_exact_evidence_span_in_judge_explanation",
        )

    onset_char = conservative_sentence_onset_char(model_output, evidence)
    onset_tok = None
    if onset_char is not None and tokenizer is not None:
        onset_tok = char_index_to_generated_token_index(
            model_output,
            onset_char,
            tokenizer,
            input_formatted=input_formatted,
        )

    return OnsetAnnotation(
        example_id=example_id,
        split=split,
        prompt_sha256=prompt_sha256,
        onset_status="high_confidence",
        evidence_span=evidence,
        evidence_span_sha256=sha256_text(evidence),
        explanation_source_field=source_field,
        conservative_deceptive_sentence_onset_char=onset_char,
        conservative_deceptive_sentence_onset_token=onset_tok,
    )


def onset_coverage_summary(annotations: list[OnsetAnnotation]) -> dict[str, Any]:
    deceptive = [a for a in annotations if a.onset_status != "not_applicable_honest"]
    hi = [a for a in deceptive if a.onset_status == "high_confidence"]
    unresolved = [a for a in deceptive if a.onset_status == "unresolved"]
    toks = sorted(
        a.conservative_deceptive_sentence_onset_token
        for a in hi
        if a.conservative_deceptive_sentence_onset_token is not None
    )

    def frac(pred) -> float | None:
        if not toks:
            return None
        return sum(1 for t in toks if pred(t)) / len(toks)

    def percentile(p: float) -> float | None:
        if not toks:
            return None
        if len(toks) == 1:
            return float(toks[0])
        idx = (len(toks) - 1) * p
        lo = math.floor(idx)
        hi_i = math.ceil(idx)
        if lo == hi_i:
            return float(toks[lo])
        return float(toks[lo] * (hi_i - idx) + toks[hi_i] * (idx - lo))

    return {
        "n_total": len(annotations),
        "n_deceptive": len(deceptive),
        "n_high_confidence_onset": len(hi),
        "n_unresolved": len(unresolved),
        "coverage_among_deceptive": (len(hi) / len(deceptive)) if deceptive else None,
        "n_unique_prompt_groups_high_confidence": len({a.prompt_sha256 for a in hi}),
        "onset_token_n": len(toks),
        "onset_token_median": percentile(0.5),
        "onset_token_q1": percentile(0.25),
        "onset_token_q3": percentile(0.75),
        "prop_onset_eq_0": frac(lambda t: t == 0),
        "prop_onset_le_1": frac(lambda t: t <= 1),
        "prop_onset_le_2": frac(lambda t: t <= 2),
        "prop_onset_le_4": frac(lambda t: t <= 4),
        "prop_onset_le_8": frac(lambda t: t <= 8),
        "prop_onset_gt_8": frac(lambda t: t > 8),
        "prop_onset_gt_16": frac(lambda t: t > 16),
    }
