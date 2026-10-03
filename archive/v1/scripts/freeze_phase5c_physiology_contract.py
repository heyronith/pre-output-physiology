#!/usr/bin/env python3
"""Freeze Phase 5C physiology contracts before any activation extraction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase5_physiology import (  # noqa: E402
    EXPECTED_TRAIN_ALL_PAIR_SHA256,
    EXPECTED_VAL_ALL_PAIR_SHA256,
    SEMANTIC_EMBEDDING_MODEL_ID,
    SEMANTIC_EMBEDDING_REVISION,
    physiology_contract_dict,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402


def main() -> int:
    out = REPO_ROOT / "artifacts/phase5c_physiology_freeze"
    out.mkdir(parents=True, exist_ok=True)
    primary = json.loads(
        (
            REPO_ROOT
            / "artifacts/phase5b_discovery_split/primary_all_pair_populations.json"
        ).read_text(encoding="utf-8")
    )
    if primary["train"]["pair_ids_sha256"] != EXPECTED_TRAIN_ALL_PAIR_SHA256:
        raise SystemExit("train all-pair hash drift")
    if primary["validation"]["pair_ids_sha256"] != EXPECTED_VAL_ALL_PAIR_SHA256:
        raise SystemExit("validation all-pair hash drift")

    contract = physiology_contract_dict()
    payload = {
        "created_at": utc_now_iso(),
        "phase": "phase5c",
        "pre_extraction_freeze": True,
        "executed": False,
        "contract": contract,
        "train_all_pair_sha256": EXPECTED_TRAIN_ALL_PAIR_SHA256,
        "validation_all_pair_sha256": EXPECTED_VAL_ALL_PAIR_SHA256,
        "semantic_embedding_model_id": SEMANTIC_EMBEDDING_MODEL_ID,
        "semantic_embedding_revision": SEMANTIC_EMBEDDING_REVISION,
        "activations_collected": False,
        "probes_fit": False,
        "locked_families_run": False,
        "causal_interventions": False,
    }
    write_json(out / "physiology_contract.json", payload)
    write_json(
        out / "output_logit_feature_formula.json",
        {
            "created_at": utc_now_iso(),
            **contract["output_logit_feature_formula"],
        },
    )
    write_json(
        out / "candidate_selection_rule.json",
        {
            "created_at": utc_now_iso(),
            **contract["candidate_selection"],
            "locked_test_gates": contract["locked_test_gates"],
        },
    )
    write_json(
        out / "semantic_embedding_pin.json",
        {
            "created_at": utc_now_iso(),
            "model_id": SEMANTIC_EMBEDDING_MODEL_ID,
            "revision": SEMANTIC_EMBEDDING_REVISION,
            "pooling": "mean_pool_attention_mask",
            "frozen_before_labels_or_results": True,
        },
    )
    print(json.dumps({"out": str(out.relative_to(REPO_ROOT)), "ok": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
