"""Temporal alignment / prompt-boundary / canonical tokenization tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from pre_output_physiology.baselines import visible_prompt_plus_prefix_text
from pre_output_physiology.trajectory import (
    COARSE_TRANSFORMER_BLOCKS,
    PREFIX_LENGTHS_K,
    PRIMARY_ANCHOR_LAYER,
    PromptResponseTokenization,
    TemporalPoint,
    activation_generated_index,
    analyze_prompt_response_boundary,
    assert_future_tokens_excluded,
    char_index_to_canonical_response_token,
    compare_grouping_keys,
    prompt_boundary_activations_must_match,
)


@dataclass
class _FakeEncoding:
    input_ids: list[int]
    offset_mapping: list[tuple[int, int]] | None = None

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


class _BoundaryTokenizer:
    """Tokenizer stub with controllable full-sequence vs prompt-only IDs."""

    def __init__(
        self,
        *,
        prompt_ids: list[int],
        response_ids: list[int],
        full_ids: list[int] | None = None,
        full_offsets: list[tuple[int, int]] | None = None,
        standalone_response_ids: list[int] | None = None,
    ) -> None:
        self.prompt_ids = prompt_ids
        self.response_ids = response_ids
        self.full_ids = full_ids if full_ids is not None else prompt_ids + response_ids
        self.standalone_response_ids = (
            standalone_response_ids
            if standalone_response_ids is not None
            else response_ids
        )
        self._full_offsets = full_offsets
        self._last_prompt: str | None = None
        self._last_response: str | None = None

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        # Identify which call by comparing to last prompt/response set externally
        if self._last_prompt is not None and text == self._last_prompt:
            return _FakeEncoding(list(self.prompt_ids))
        if self._last_response is not None and text == self._last_response:
            return _FakeEncoding(list(self.standalone_response_ids))
        if (
            self._last_prompt is not None
            and self._last_response is not None
            and text == self._last_prompt + self._last_response
        ):
            offsets = self._full_offsets
            if offsets is None:
                # Default: one char per token for prompt then response
                offsets = []
                pos = 0
                for _ in self.prompt_ids:
                    offsets.append((pos, pos + 1))
                    pos += 1
                # pad remaining prompt chars into last prompt token end
                boundary = len(self._last_prompt)
                if offsets and offsets[-1][1] < boundary:
                    s, _ = offsets[-1]
                    offsets[-1] = (s, boundary)
                    pos = boundary
                for _ in self.response_ids:
                    end = min(pos + 1, len(text))
                    offsets.append((pos, end if end > pos else pos))
                    pos = end
                while len(offsets) < len(self.full_ids):
                    offsets.append((pos, pos))
            if return_offsets_mapping:
                return _FakeEncoding(list(self.full_ids), list(offsets))
            return _FakeEncoding(list(self.full_ids))
        # Fallback: treat as prompt-only for first call patterns
        return _FakeEncoding(list(self.prompt_ids))


def test_frozen_coarse_grid() -> None:
    assert COARSE_TRANSFORMER_BLOCKS == [0, 4, 8, 12, 16, 20, 24, 28, 31]
    assert PREFIX_LENGTHS_K == [0, 1, 2, 4, 8, 16]
    assert PRIMARY_ANCHOR_LAYER == 12
    assert 12 in COARSE_TRANSFORMER_BLOCKS


def test_k0_is_prompt_boundary() -> None:
    tp = TemporalPoint(0)
    assert tp.regime == "prompt_boundary_propensity"
    assert tp.generated_token_index is None
    assert activation_generated_index(0) is None


@pytest.mark.parametrize(
    ("k", "expected_idx"),
    [(1, 0), (2, 1), (4, 3), (8, 7), (16, 15)],
)
def test_activation_index_is_k_minus_one(k: int, expected_idx: int) -> None:
    """Off-by-one guard: state at generated index k-1 before predicting next."""
    assert activation_generated_index(k) == expected_idx
    assert TemporalPoint(k).generated_token_index == expected_idx
    assert TemporalPoint(k).regime == "early_trajectory_prediction"


def test_future_tokens_excluded_contract() -> None:
    assert_future_tokens_excluded(4, total_generated_tokens=10)
    with pytest.raises(ValueError):
        assert_future_tokens_excluded(11, total_generated_tokens=10)


def test_prompt_boundary_identical_prompts_must_match() -> None:
    acts = np.array([[1.0, 2.0], [1.0, 2.0], [0.0, 1.0]], dtype=float)
    groups = ["a", "a", "b"]
    out = prompt_boundary_activations_must_match(acts, groups)
    assert out["max_abs_within_group"] == 0.0


def test_prompt_boundary_mismatch_raises() -> None:
    acts = np.array([[1.0, 2.0], [1.0, 2.1]], dtype=float)
    with pytest.raises(AssertionError):
        prompt_boundary_activations_must_match(acts, ["g", "g"], atol=1e-5)


def test_grouping_key_agreement_detection() -> None:
    report = compare_grouping_keys(["p1_0", "p1_1", "p2_0"], ["h1", "h1", "h2"])
    assert report["agree"] is True
    assert report["scientific_key"] == "prompt_sha256"

    report2 = compare_grouping_keys(["p1_0", "p1_1"], ["h1", "h2"])
    assert report2["agree"] is False
    assert report2["n_prefixes_with_multiple_hashes"] == 1


def test_exact_prompt_prefix_preserved() -> None:
    prompt = "ABCD"
    response = "EF"
    tok = _BoundaryTokenizer(prompt_ids=[10, 11, 12, 13], response_ids=[20, 21])
    tok._last_prompt = prompt
    tok._last_response = response
    # Offsets: one char per token
    tok._full_offsets = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    result = analyze_prompt_response_boundary(prompt, response, tok)
    assert result.prompt_prefix_exact is True
    assert result.boundary_token_straddle is False
    assert result.response_suffix_ids == [20, 21]
    assert result.response_start_token_index == 4
    assert result.full_sequence_index_for_k(0) == 3  # final prompt token
    assert result.full_sequence_index_for_k(1) == 4  # first response token


def test_detect_synthetic_prefix_mismatch() -> None:
    prompt = "ABCD"
    response = "EF"
    tok = _BoundaryTokenizer(
        prompt_ids=[10, 11, 12, 13],
        response_ids=[20, 21],
        full_ids=[10, 11, 99, 13, 20, 21],  # third prompt id corrupted in full
    )
    tok._last_prompt = prompt
    tok._last_response = response
    tok._full_offsets = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    result = analyze_prompt_response_boundary(prompt, response, tok)
    assert result.prompt_prefix_exact is False


def test_detect_token_straddling_boundary() -> None:
    prompt = "ABCD"
    response = "EF"
    tok = _BoundaryTokenizer(
        prompt_ids=[10, 11],
        response_ids=[20],
        full_ids=[10, 11, 20],
    )
    tok._last_prompt = prompt
    tok._last_response = response
    # Second token straddles boundary at char 4
    tok._full_offsets = [(0, 2), (2, 5), (5, 6)]
    result = analyze_prompt_response_boundary(prompt, response, tok)
    assert result.boundary_token_straddle is True
    assert result.straddling_token_index == 1


def test_k_maps_to_prompt_and_first_response_tokens() -> None:
    tok_info = PromptResponseTokenization(
        prompt_ids=[1, 2, 3],
        full_ids=[1, 2, 3, 4, 5],
        response_suffix_ids=[4, 5],
        response_start_token_index=3,
        full_offsets=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
        response_offsets_in_model_outputs=[(0, 1), (1, 2)],
        prompt_prefix_exact=True,
        boundary_token_straddle=False,
        straddling_token_index=None,
        standalone_response_ids=[4, 5],
        standalone_equals_suffix=True,
        boundary_char=3,
    )
    assert tok_info.full_sequence_index_for_k(0) == 2
    assert tok_info.full_sequence_index_for_k(1) == 3
    assert activation_generated_index(1) == 0


def test_prefix_surface_text_excludes_future_response() -> None:
    prompt = "PROMPT|"
    response = "ABCDEFGH"
    offsets = [(0, 2), (2, 4), (4, 6), (6, 8)]
    visible0 = visible_prompt_plus_prefix_text(prompt, response, 0, offsets)
    assert visible0 == prompt
    assert "A" not in visible0[len(prompt) :]
    visible2 = visible_prompt_plus_prefix_text(prompt, response, 2, offsets)
    assert visible2 == prompt + "ABCD"
    assert "EFGH" not in visible2
    assert not visible2.endswith(response)


def test_char_to_canonical_response_token() -> None:
    offsets = [(0, 3), (3, 7), (7, 10)]
    assert char_index_to_canonical_response_token(0, offsets) == 0
    assert char_index_to_canonical_response_token(3, offsets) == 1
    assert char_index_to_canonical_response_token(9, offsets) == 2


def test_standalone_mismatch_does_not_change_canonical_suffix() -> None:
    prompt = "ABCD"
    response = "EF"
    tok = _BoundaryTokenizer(
        prompt_ids=[10, 11, 12, 13],
        response_ids=[20, 21],
        standalone_response_ids=[99, 100],  # different standalone
    )
    tok._last_prompt = prompt
    tok._last_response = response
    tok._full_offsets = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    result = analyze_prompt_response_boundary(prompt, response, tok)
    assert result.standalone_equals_suffix is False
    assert result.response_suffix_ids == [20, 21]
    assert result.standalone_response_ids == [99, 100]
