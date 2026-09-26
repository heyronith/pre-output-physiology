"""Phase 6B frozen corpus/probe pins and verification (prospective amendment D083)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/experiments/phase6_intent_specificity.yaml"
FINAL_PROMPTS_PATH = REPO_ROOT / "data/processed/phase6_design/final_candidate_prompts.jsonl"
FINAL_SCENARIOS_PATH = REPO_ROOT / "data/processed/phase6_design/final_base_scenarios.jsonl"
PROBE_PATH = REPO_ROOT / "artifacts/phase5c_discovery_physiology/selected_probe_k1.npz"

PHASE6A_HOLD_HEAD = "0f1673e7c2eae9b0a90ab6e96f111cb2684ce032"
PHASE6A_OUTCOME = "phase6a_specificity_pilot_hold_operational_format_failure"

EXPECTED_PROMPT_TEXT_SHA256 = "fa4ad2629e911e0226620dd10edbf9ff9d517051837be818e55a56b2f0124721"
EXPECTED_SCENARIO_TEXT_SHA256 = "839b931be2d1982b3d08393f2c1004dc450e52f17aba90974e2f4e06ec7742f1"
EXPECTED_SCENARIO_IDS_SHA256 = "6bb09c18ad206d39c354e8520dd1ad1f9eb7b855f07f3fda64e25622f3881034"
EXPECTED_PROBE_SHA256 = "fa725af194eb1ca227301e6519029c818e942dab4bc130de060aa0754f4709c8"

N_BASE = 720
N_PROMPTS = 2880
LAYER = 12
CONTROLLED_PREFIX_TOKEN_ID = 12107

STATUS_AUTHORIZED = "phase6b_factorial_frozen_probe_authorized"
STATUS_PRIMARY_COMPLETE = "phase6b_factorial_primary_complete_awaiting_audit"


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_ids(ids: list[str]) -> str:
    return sha_bytes("\n".join(sorted(ids)).encode())


def sha_prompt_texts(rows: list[dict[str, Any]]) -> str:
    ordered = sorted(rows, key=lambda r: r["example_id"])
    return sha_bytes("\n".join(f"{r['example_id']}\t{r['prompt_text']}" for r in ordered).encode())


def sha_scenarios(rows: list[dict[str, Any]]) -> str:
    ordered = sorted(rows, key=lambda r: r["base_scenario_id"])
    payload = "\n".join(
        "\t".join(
            [
                r["base_scenario_id"],
                r["topic_sentence"],
                r["user_question"],
                r["record_state"],
                r["alternate_state"],
            ]
        )
        for r in ordered
    )
    return sha_bytes(payload.encode())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def load_verified_corpus() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    """Return (prompt_rows sorted by example_id, scenario_rows, verified hashes) or raise."""
    prompts = _read_jsonl(FINAL_PROMPTS_PATH)
    scenarios = _read_jsonl(FINAL_SCENARIOS_PATH)
    got = {
        "final_prompt_text_sha256": sha_prompt_texts(prompts),
        "final_scenario_text_sha256": sha_scenarios(scenarios),
        "final_base_scenario_ids_sha256": sha_ids([r["base_scenario_id"] for r in scenarios]),
    }
    want = {
        "final_prompt_text_sha256": EXPECTED_PROMPT_TEXT_SHA256,
        "final_scenario_text_sha256": EXPECTED_SCENARIO_TEXT_SHA256,
        "final_base_scenario_ids_sha256": EXPECTED_SCENARIO_IDS_SHA256,
    }
    for k, v in want.items():
        if got[k] != v:
            raise SystemExit(f"STOP: {k} mismatch: {got[k]} != {v}")
    if len(prompts) != N_PROMPTS or len(scenarios) != N_BASE:
        raise SystemExit("STOP: corpus size mismatch")
    base_ids = {r["base_scenario_id"] for r in scenarios}
    counts: dict[str, int] = {}
    for r in prompts:
        if r["base_scenario_id"] not in base_ids:
            raise SystemExit("STOP: prompt references unknown scenario")
        counts[r["base_scenario_id"]] = counts.get(r["base_scenario_id"], 0) + 1
    if set(counts.values()) != {4} or len(counts) != N_BASE:
        raise SystemExit("STOP: not exactly 4 conditions per scenario")
    return sorted(prompts, key=lambda r: r["example_id"]), scenarios, got


def verify_probe_sha() -> str:
    sha = sha_bytes(PROBE_PATH.read_bytes())
    if sha != EXPECTED_PROBE_SHA256:
        raise SystemExit(f"STOP: probe sha mismatch {sha}")
    return sha


def label_free_payload(prompts_sorted: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """GPU payload: opaque row index + unchanged prompt text only."""
    return [{"row_index": i, "prompt_text": r["prompt_text"]} for i, r in enumerate(prompts_sorted)]
