"""Stage-23D true token-index error regression tests (no GPU / no Gemma)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pre_output_physiology.phase23_open_grader import (
    ONSET_ACCEPTANCE,
    ONSET_TOKENIZER_HF_ID,
    ONSET_TOKENIZER_REVISION,
    absolute_token_index_error,
    char_diff_over_four_proxy,
    load_frozen_onset_tokenizer,
    response_char_to_onset_token_index,
)

REPO = Path(__file__).resolve().parents[1]
SAMPLE_MANIFEST = REPO / "artifacts/phase23_onset_sample/manifest.json"
SAMPLE_JSONL = REPO / "artifacts/phase23_onset_sample/onset_validation.jsonl"
EXPECTED_SAMPLE_SHA = "5a439094f23cbb22ec121a7f2c074466a112a77bac737f36ce7dc6e7ddd357ca"


@pytest.fixture(scope="module")
def tok():
    return load_frozen_onset_tokenizer()


def test_frozen_onset_sample_sha() -> None:
    from pre_output_physiology.phase23_open_grader import _sha_json

    man = json.loads(SAMPLE_MANIFEST.read_text(encoding="utf-8"))
    assert man["onset_sample_sha256"] == EXPECTED_SAMPLE_SHA
    assert man["n"] == 80
    assert len(man["continuation_ids"]) == 80
    assert _sha_json(man["continuation_ids"]) == EXPECTED_SAMPLE_SHA
    rows = [
        json.loads(x)
        for x in SAMPLE_JSONL.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    assert len(rows) == 80
    assert {r["continuation_id"] for r in rows} == set(man["continuation_ids"])


def test_tokenizer_identity_pinned() -> None:
    assert ONSET_TOKENIZER_HF_ID == "mistralai/Mistral-7B-Instruct-v0.2"
    assert ONSET_TOKENIZER_REVISION == "63a8b081895390a26e140280378bc85ec8bce07a"


def test_onset_acceptance_thresholds_unchanged() -> None:
    assert ONSET_ACCEPTANCE == {
        "min_explicit_span_agreement": 0.90,
        "min_exact_quote_match_rate": 0.80,
        "max_median_abs_token_error": 2.0,
        "char_offset_tolerance": 8,
    }


def test_reproduces_frozen_reference_token_indices(tok) -> None:
    rows = [
        json.loads(x)
        for x in SAMPLE_JSONL.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    checked = 0
    for r in rows:
        if r.get("onset_token_index") is None:
            continue
        got = response_char_to_onset_token_index(
            tok,
            answer_prefix=r["answer_prefix"],
            full_response=r["full_response"],
            char_start=int(r["onset_char_start"]),
        )
        assert got == r["onset_token_index"], r["continuation_id"]
        checked += 1
    assert checked == 60


def test_onset_at_token_boundary(tok) -> None:
    prefix = "Coach:"
    # Craft text where char_start lands exactly at a token start.
    full = prefix + " Yes. No."
    # Find first token boundary after prefix via encode walk
    ids = tok.encode(full, add_special_tokens=False)
    # Decode progressively to find a boundary char
    boundary = None
    for i in range(1, len(ids)):
        text_i = tok.decode(ids[:i], skip_special_tokens=False)
        # Prefer a boundary inside the completion
        if len(text_i) > len(prefix):
            boundary = len(text_i)
            break
    assert boundary is not None
    idx = response_char_to_onset_token_index(
        tok, answer_prefix=prefix, full_response=full, char_start=boundary
    )
    assert idx is not None
    # Same boundary remapped deterministically
    idx2 = response_char_to_onset_token_index(
        tok, answer_prefix=prefix, full_response=full, char_start=boundary
    )
    assert idx == idx2


def test_onset_mid_token_same_index(tok) -> None:
    prefix = "A:"
    full = prefix + " unbelievable"
    # Map several char offsets inside the completion; adjacent chars in the
    # same token must share an index (token error 0).
    start = len(prefix) + 1
    end = len(prefix) + 5
    idxs = [
        response_char_to_onset_token_index(
            tok, answer_prefix=prefix, full_response=full, char_start=c
        )
        for c in range(start, end)
    ]
    assert all(i is not None for i in idxs)
    # At least one pair within the window shares a token index
    assert any(idxs[i] == idxs[j] for i in range(len(idxs)) for j in range(i + 1, len(idxs)))


def test_leading_whitespace(tok) -> None:
    prefix = "Bot:"
    full = prefix + "  hello world"
    # onset at first non-space after prefix spaces
    c = full.index("h")
    idx = response_char_to_onset_token_index(
        tok, answer_prefix=prefix, full_response=full, char_start=c
    )
    assert idx is not None and idx >= 0


def test_punctuation_adjacent(tok) -> None:
    prefix = "X:"
    full = prefix + "Wait—really?"
    c = full.index("r")
    idx = response_char_to_onset_token_index(
        tok, answer_prefix=prefix, full_response=full, char_start=c
    )
    assert idx is not None


def test_unicode_multibyte(tok) -> None:
    prefix = "Y:"
    full = prefix + " café naïve 日本語"
    c = full.index("日")
    idx = response_char_to_onset_token_index(
        tok, answer_prefix=prefix, full_response=full, char_start=c
    )
    assert idx is not None


def test_same_token_zero_error(tok) -> None:
    prefix = "Z:"
    full = prefix + " misunderstanding"
    c0 = len(prefix) + 2
    c1 = c0 + 1
    i0 = response_char_to_onset_token_index(
        tok, answer_prefix=prefix, full_response=full, char_start=c0
    )
    i1 = response_char_to_onset_token_index(
        tok, answer_prefix=prefix, full_response=full, char_start=c1
    )
    if i0 == i1:
        assert absolute_token_index_error(i0, i1) == 0


def test_crossing_token_boundary_error_one(tok) -> None:
    prefix = "Q:"
    full = prefix + " aaa bbb"
    # Walk chars until token index increments
    prev = None
    found = False
    for c in range(len(prefix), len(full)):
        cur = response_char_to_onset_token_index(
            tok, answer_prefix=prefix, full_response=full, char_start=c
        )
        if prev is not None and cur is not None and cur == prev + 1:
            assert absolute_token_index_error(prev, cur) == 1
            found = True
            break
        prev = cur
    assert found


def test_deterministic_repeated_tokenization(tok) -> None:
    prefix = "P:"
    full = prefix + " The package was delayed by weather."
    c = full.index("d")
    a = [
        response_char_to_onset_token_index(
            tok, answer_prefix=prefix, full_response=full, char_start=c
        )
        for _ in range(5)
    ]
    assert len(set(a)) == 1


def test_proxy_char_over_four_differs_from_true_token_error(tok) -> None:
    """Regression: dense BPE packing makes /4 overestimate true token distance."""
    prefix = "R:"
    full = (
        prefix
        + " The international organization established comprehensive "
        "regulations for environmental protection and sustainability "
        "initiatives worldwide."
    )
    c_ref = len(prefix)
    c_pred = c_ref + 60
    i_ref = response_char_to_onset_token_index(
        tok, answer_prefix=prefix, full_response=full, char_start=c_ref
    )
    i_pred = response_char_to_onset_token_index(
        tok, answer_prefix=prefix, full_response=full, char_start=c_pred
    )
    assert i_ref is not None and i_pred is not None
    true_err = absolute_token_index_error(i_ref, i_pred)
    proxy = char_diff_over_four_proxy(c_ref, c_pred)
    assert true_err is not None
    assert proxy >= 15.0
    assert true_err < proxy
    assert isinstance(true_err, int)


def test_analyze_script_rejects_char_over_four_proxy() -> None:
    src = (REPO / "scripts/analyze_phase23d_onset.py").read_text(encoding="utf-8")
    assert "/ 4.0" not in src
    assert "/4.0" not in src
    assert "char_diff_over_four_proxy" not in src or "deprecated" in src.lower()
    assert "response_char_to_onset_token_index" in src
    assert "absolute_token_index_error" in src
    assert "open_grader_for_k60_authorized\": False" in src or (
        "open_grader_for_k60_authorized: false" in src
    )
    # Must never auto-authorize K60 on PASS
    compact = "".join(src.split())
    assert '"open_grader_for_k60_authorized":False' in compact
    assert "status==STATUS_23D_PASS" not in compact
