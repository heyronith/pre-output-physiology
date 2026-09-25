"""Temporal alignment and prompt-group contracts for Phase 3 trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Pre-registered Phase 3 coarse scan (frozen before activations).
COARSE_TRANSFORMER_BLOCKS: list[int] = [0, 4, 8, 12, 16, 20, 24, 28, 31]
PREFIX_LENGTHS_K: list[int] = [0, 1, 2, 4, 8, 16]
PRIMARY_ANCHOR_LAYER: int = 12


@dataclass(frozen=True)
class TemporalPoint:
    """One analysis locus along a teacher-forced assistant trajectory.

    k = number of assistant/generated tokens already visible.
    For k == 0: prompt-boundary; activation at final prompt token.
    For k > 0: activation at generated-token index (k - 1), i.e. the state
    used immediately before predicting the next token. Tokens at indices
    >= k must not inform this representation.
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
