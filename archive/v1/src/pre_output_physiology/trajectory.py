"""Temporal alignment and prompt-group contracts for Phase 3 trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

# Pre-registered Phase 3 coarse scan (frozen before activations).
COARSE_TRANSFORMER_BLOCKS: list[int] = [0, 4, 8, 12, 16, 20, 24, 28, 31]
PREFIX_LENGTHS_K: list[int] = [0, 1, 2, 4, 8, 16]
PRIMARY_ANCHOR_LAYER: int = 12

# Pre-registered Phase 3B1 primary endpoints (frozen before activations).
PRIMARY_REGIME_A: dict[str, int] = {"layer": 12, "k": 0}
PRIMARY_REGIME_B: dict[str, int] = {"layer": 12, "k": 1}


def eligible_for_k(canonical_response_token_length: int, k: int) -> bool:
    """Eligibility for fixed-k trajectory analysis.

    k=0: all rows eligible.
    k>0: require response length *strictly greater than* k so at least one
    future response token remains unseen.
    """
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if k == 0:
        return True
    return int(canonical_response_token_length) > int(k)


def eligibility_mask(
    response_lengths: list[int] | np.ndarray,
    k: int,
) -> np.ndarray:
    lengths = np.asarray(response_lengths, dtype=int)
    if k == 0:
        return np.ones(len(lengths), dtype=bool)
    return lengths > k


@dataclass(frozen=True)
class TemporalPoint:
    """One analysis locus along a teacher-forced assistant trajectory.

    k = number of assistant/generated tokens already visible.
    For k == 0: prompt-boundary; activation at final prompt token.
    For k > 0: activation at generated-token index (k - 1), i.e. the state
    used immediately before predicting the next token. Tokens at indices
    >= k must not inform this representation.

    Response-token indices are defined from the suffix of the canonical
    tokenization of ``input_formatted + model_outputs`` (see
    :func:`analyze_prompt_response_boundary`).
    """

    k: int

    def __post_init__(self) -> None:
        if self.k < 0:
            raise ValueError(f"k must be >= 0, got {self.k}")

    @property
    def regime(self) -> str:
        if self.k == 0:
            return "prompt_boundary_propensity"
        return "early_trajectory_prediction"

    @property
    def generated_token_index(self) -> int | None:
        """Index into assistant tokens of the activation token; None at k=0."""
        if self.k == 0:
            return None
        return self.k - 1

    @property
    def visible_generated_token_count(self) -> int:
        return self.k

    @property
    def activation_full_sequence_index(self) -> int | None:
        """Index into full (prompt+response) sequence; None until boundary known."""
        return None


@dataclass(frozen=True)
class PromptResponseTokenization:
    """Canonical Phase 3 prompt/response token alignment for one example."""

    prompt_ids: list[int]
    full_ids: list[int]
    response_suffix_ids: list[int]
    response_start_token_index: int
    full_offsets: list[tuple[int, int]]
    response_offsets_in_model_outputs: list[tuple[int, int]]
    prompt_prefix_exact: bool
    boundary_token_straddle: bool
    straddling_token_index: int | None
    standalone_response_ids: list[int]
    standalone_equals_suffix: bool
    boundary_char: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def full_sequence_index_for_k(self, k: int) -> int:
        """Full-sequence token index of the activation at visible prefix length k."""
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        if k == 0:
            if not self.prompt_ids:
                raise ValueError("empty prompt_ids")
            return len(self.prompt_ids) - 1
        if k > len(self.response_suffix_ids):
            raise ValueError(
                f"k={k} exceeds response length {len(self.response_suffix_ids)}"
            )
        return self.response_start_token_index + (k - 1)

    def visible_response_char_end(self, k: int) -> int:
        """Exclusive end char offset into model_outputs for the first k response tokens."""
        if k < 0:
            raise ValueError("k < 0")
        if k == 0:
            return 0
        if k > len(self.response_offsets_in_model_outputs):
            raise ValueError("k exceeds canonical response tokens")
        return self.response_offsets_in_model_outputs[k - 1][1]


def analyze_prompt_response_boundary(
    input_formatted: str,
    model_outputs: str,
    tokenizer,
) -> PromptResponseTokenization:
    """Canonical Phase 3 tokenization of prompt + response.

    Phase 3B must use this helper (or an equivalent calling the same contract)
    rather than reimplementing token alignment independently.

    Checks:
      1. ``full_ids[:len(prompt_ids)] == prompt_ids``
      2. No nonempty token straddles ``len(input_formatted)``
    Canonical response tokens are the full-sequence suffix, even if standalone
    response tokenization differs.
    """
    prompt = input_formatted
    response = model_outputs
    full_text = prompt + response
    boundary_char = len(prompt)

    prompt_ids = list(
        tokenizer(prompt, add_special_tokens=False)["input_ids"]
    )
    standalone_response_ids = list(
        tokenizer(response, add_special_tokens=False)["input_ids"]
    )
    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    full_ids = list(encoded["input_ids"])
    offsets_raw = encoded.get("offset_mapping")
    if not offsets_raw:
        raise ValueError(
            "Tokenizer must return offset_mapping for Phase 3 boundary audit "
            "(use a fast tokenizer)."
        )
    full_offsets = [(int(s), int(e)) for s, e in offsets_raw]

    n_prompt = len(prompt_ids)
    prompt_prefix_exact = full_ids[:n_prompt] == prompt_ids

    straddling_token_index: int | None = None
    for i, (start, end) in enumerate(full_offsets):
        if start == end:
            continue  # empty special/padding-like span
        if start < boundary_char < end:
            straddling_token_index = i
            break
    boundary_token_straddle = straddling_token_index is not None

    response_suffix_ids = full_ids[n_prompt:]
    response_offsets: list[tuple[int, int]] = []
    for start, end in full_offsets[n_prompt:]:
        response_offsets.append((start - boundary_char, end - boundary_char))

    return PromptResponseTokenization(
        prompt_ids=prompt_ids,
        full_ids=full_ids,
        response_suffix_ids=response_suffix_ids,
        response_start_token_index=n_prompt,
        full_offsets=full_offsets,
        response_offsets_in_model_outputs=response_offsets,
        prompt_prefix_exact=prompt_prefix_exact,
        boundary_token_straddle=boundary_token_straddle,
        straddling_token_index=straddling_token_index,
        standalone_response_ids=standalone_response_ids,
        standalone_equals_suffix=standalone_response_ids == response_suffix_ids,
        boundary_char=boundary_char,
    )


def char_index_to_canonical_response_token(
    char_index: int,
    response_offsets_in_model_outputs: list[tuple[int, int]],
) -> int:
    """Map a character index in model_outputs to a canonical response-token index."""
    if not response_offsets_in_model_outputs:
        return 0
    char_index = max(0, char_index)
    for i, (start, end) in enumerate(response_offsets_in_model_outputs):
        if start <= char_index < end:
            return i
        if char_index < start:
            return max(0, i - 1)
    return len(response_offsets_in_model_outputs) - 1


def activation_generated_index(k: int) -> int | None:
    """Map visible prefix length k to assistant-token activation index."""
    return TemporalPoint(k).generated_token_index


def assert_future_tokens_excluded(k: int, total_generated_tokens: int) -> None:
    """Future tokens (indices >= k) are not part of the visible prefix."""
    if k < 0:
        raise ValueError("k < 0")
    if k > total_generated_tokens:
        raise ValueError(
            f"k={k} exceeds total_generated_tokens={total_generated_tokens}"
        )


def prompt_boundary_activations_must_match(
    activations: np.ndarray,
    prompt_group_ids: list[str] | np.ndarray,
    *,
    atol: float = 1e-5,
) -> dict[str, float]:
    """Identical prompts must share identical prompt-boundary activations.

    activations: [N, H] prompt-boundary vectors aligned with prompt_group_ids.
    Returns max absolute difference within any group (0 if all singletons).
    """
    activations = np.asarray(activations)
    prompt_group_ids = np.asarray(prompt_group_ids)
    if len(activations) != len(prompt_group_ids):
        raise ValueError("activations and prompt_group_ids length mismatch")
    max_abs = 0.0
    for gid in np.unique(prompt_group_ids):
        rows = activations[prompt_group_ids == gid]
        if len(rows) < 2:
            continue
        diffs = np.max(np.abs(rows - rows[0]), axis=1)
        max_abs = max(max_abs, float(np.max(diffs)))
    if max_abs > atol:
        raise AssertionError(
            f"Prompt-boundary activations disagree within a prompt group: "
            f"max_abs_diff={max_abs} > atol={atol}"
        )
    return {"max_abs_within_group": max_abs, "atol": atol}


def id_prefix(example_id: str) -> str:
    return str(example_id).split("_", 1)[0]


def compare_grouping_keys(
    example_ids: list[str],
    prompt_hashes: list[str],
) -> dict[str, object]:
    """Compare ID-prefix groups vs exact prompt-hash groups."""
    if len(example_ids) != len(prompt_hashes):
        raise ValueError("length mismatch")
    prefix_to_hashes: dict[str, set[str]] = {}
    hash_to_prefixes: dict[str, set[str]] = {}
    for eid, ph in zip(example_ids, prompt_hashes, strict=True):
        pref = id_prefix(eid)
        prefix_to_hashes.setdefault(pref, set()).add(ph)
        hash_to_prefixes.setdefault(ph, set()).add(pref)
    prefixes_with_multiple_hashes = {
        p: sorted(hs) for p, hs in prefix_to_hashes.items() if len(hs) > 1
    }
    hashes_with_multiple_prefixes = {
        h: sorted(ps) for h, ps in hash_to_prefixes.items() if len(ps) > 1
    }
    agree = (
        len(prefixes_with_multiple_hashes) == 0
        and len(hashes_with_multiple_prefixes) == 0
    )
    return {
        "agree": agree,
        "n_id_prefixes": len(prefix_to_hashes),
        "n_prompt_hashes": len(hash_to_prefixes),
        "n_prefixes_with_multiple_hashes": len(prefixes_with_multiple_hashes),
        "n_hashes_with_multiple_prefixes": len(hashes_with_multiple_prefixes),
        "prefixes_with_multiple_hashes_sample": dict(
            list(prefixes_with_multiple_hashes.items())[:20]
        ),
        "hashes_with_multiple_prefixes_sample": dict(
            list(hashes_with_multiple_prefixes.items())[:20]
        ),
        "scientific_key": "prompt_sha256",
    }
