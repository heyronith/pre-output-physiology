"""Phase 5B discovery train/validation family split (deterministic)."""

from __future__ import annotations

import hashlib
from typing import Any

from pre_output_physiology.phase5_conditions import DISCOVERY_FAMILIES, LOCKED_FAMILIES

FAMILY_SPLIT_VERSION = "phase5_family_split_v1"
FAMILY_SPLIT_PREFIX = f"{FAMILY_SPLIT_VERSION}|"

# Future probe procedure (recorded; not executed in Phase 5B).
FUTURE_PROBE_PROCEDURE: dict[str, Any] = {
    "primary_endpoint": "controlled_prefix_k1_after_token_12107_Response",
    "secondary_endpoint": "k0_prompt_boundary",
    "layers": [0, 4, 8, 12, 16, 20, 24, 28, 31],
    "probe": {
        "kind": "standardized_logistic_regression",
        "C": 0.01,
        "fit_intercept": True,
        "max_iter": 500,
        "seed": 42,
        "per_layer_per_endpoint_hyperparameter_tuning": False,
    },
    "train_population": "discovery_train_families_only",
    "layer_selection_population": "discovery_validation_families_only",
    "locked_families_unavailable_until_probe_frozen": True,
}


def family_split_digest(family: str) -> str:
    payload = f"{FAMILY_SPLIT_PREFIX}{family}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_discovery_family_split(
    discovery_families: tuple[str, ...] = DISCOVERY_FAMILIES,
) -> dict[str, Any]:
    if len(discovery_families) != 6:
        raise ValueError(f"expected 6 discovery families, got {len(discovery_families)}")
    if set(discovery_families) & set(LOCKED_FAMILIES):
        raise ValueError("discovery families overlap locked families")
    ranked = sorted(
        (
            {
                "family": fam,
                "digest_sha256": family_split_digest(fam),
            }
            for fam in discovery_families
        ),
        key=lambda row: row["digest_sha256"],
    )
    train = [row["family"] for row in ranked[:4]]
    validation = [row["family"] for row in ranked[4:]]
    return {
        "split_version": FAMILY_SPLIT_VERSION,
        "method": (
            "SHA256(phase5_family_split_v1|<family>); sort ascending by digest; "
            "first 4 = train; last 2 = validation"
        ),
        "outcome_independent": True,
        "ranked_by_digest": ranked,
        "train_families": train,
        "validation_families": validation,
        "n_train_families": len(train),
        "n_validation_families": len(validation),
        "n_train_base_scenarios_expected": 640,
        "n_validation_base_scenarios_expected": 320,
        "n_train_prompts_expected": 1280,
        "n_validation_prompts_expected": 640,
        "locked_families_excluded": list(LOCKED_FAMILIES),
    }


def sha_sorted_ids(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def sha_prompt_texts(rows: list[dict[str, Any]]) -> str:
    ordered = sorted(rows, key=lambda r: r["example_id"])
    payload = "\n".join(f"{r['example_id']}\t{r['prompt_text']}" for r in ordered)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
