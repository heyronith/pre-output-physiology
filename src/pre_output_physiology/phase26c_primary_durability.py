"""Phase 26C primary — durable checkpoint / resume helpers (CPU-testable; no GPU)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pre_output_physiology.phase26c_primary import (
    EXPECTED_PROMPT_BANK_SHA256,
    MODEL_ID,
    MODEL_REVISION,
    N_PLANNED_GENERATIONS,
    PROTOCOL_VERSION,
    TOKENIZER_REVISION,
    validate_raw_against_manifest,
)

CHECKPOINT_BATCH_SIZE = 16
RESULTS_VOLUME_NAME = "phase26c-primary-results"
RESULTS_MOUNT = "/phase26c_results"

FROZEN_INFERENCE_MANIFEST_SHA256 = (
    "b8f0ca6d5617694009767b9acb05122f011b95e385154ffcb745217f73695f0e"
)
FROZEN_SEED_MANIFEST_SHA256 = (
    "5291eca3ba0e8ec1e9122fb810dde95f42c9d7547c47d40682c436d0ce71f7db"
)
FROZEN_SELECTION_MANIFEST_SHA256 = (
    "67e6385a9c1cfa8c534cafc2e9eb581e00ed6d30152cd3581e4a93bc3848eb11"
)

RAW_MATCH_FIELDS = (
    "scenario_id",
    "condition",
    "prompt_id",
    "prompt_sha256",
    "rollout_type",
    "rollout_index",
    "seed",
)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def run_key_from_manifest_sha(manifest_sha256: str) -> str:
    if len(manifest_sha256) < 12:
        raise ValueError("manifest sha too short for run key")
    return f"{PROTOCOL_VERSION}_{manifest_sha256[:12]}"


def run_dir(results_root: Path, run_key: str) -> Path:
    return Path(results_root) / run_key


def atomic_write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    tmp.replace(path)


def validate_checkpoint_row_against_job(
    row: dict[str, Any], job: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    if row.get("job_id") != job.get("job_id"):
        errors.append(
            f"job_id mismatch row={row.get('job_id')!r} job={job.get('job_id')!r}"
        )
    for field in RAW_MATCH_FIELDS:
        if row.get(field) != job.get(field):
            errors.append(
                f"{row.get('job_id')}: {field} row={row.get(field)!r} manifest={job.get(field)!r}"
            )
    if row.get("state_id") == "SAFE" or job.get("state_id") == "SAFE":
        errors.append(f"{row.get('job_id')}: SAFE not allowed")
    if row.get("state_id") is not None and row.get("state_id") != job.get("state_id"):
        errors.append(f"{row.get('job_id')}: state_id mismatch")
    return errors


def load_checkpoints(
    *,
    checkpoint_dir: Path,
    jobs_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Load immutable checkpoints; HARD STOP on conflicts/duplicates."""
    completed: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    if not checkpoint_dir.exists():
        return completed, errors
    files = sorted(checkpoint_dir.glob("checkpoint_*.jsonl"))
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            jid = row.get("job_id")
            if jid is None:
                errors.append(f"{path.name}: missing job_id")
                continue
            if jid not in jobs_by_id:
                errors.append(f"{path.name}: unknown job_id {jid}")
                continue
            if jid in completed:
                errors.append(f"duplicate completed job_id {jid}")
                continue
            row_errs = validate_checkpoint_row_against_job(row, jobs_by_id[jid])
            if row_errs:
                errors.extend(row_errs)
                continue
            completed[jid] = row
    return completed, errors


def write_checkpoint_batch(
    *,
    checkpoint_dir: Path,
    batch_index: int,
    rows: list[dict[str, Any]],
) -> Path:
    if not rows:
        raise ValueError("empty checkpoint batch")
    if len(rows) > CHECKPOINT_BATCH_SIZE:
        raise ValueError(f"batch size {len(rows)} > {CHECKPOINT_BATCH_SIZE}")
    path = checkpoint_dir / f"checkpoint_{batch_index:05d}.jsonl"
    if path.exists():
        raise ValueError(f"refusing to overwrite checkpoint {path.name}")
    atomic_write_jsonl(path, rows)
    return path


def reconstruct_ordered_rows(
    *,
    manifest_jobs: list[dict[str, Any]],
    completed: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(completed) != N_PLANNED_GENERATIONS:
        raise ValueError(
            f"cannot complete: completed={len(completed)} != {N_PLANNED_GENERATIONS}"
        )
    ordered = []
    for job in manifest_jobs:
        jid = job["job_id"]
        if jid not in completed:
            raise ValueError(f"missing completed row for {jid}")
        ordered.append(completed[jid])
    validate_raw_against_manifest(ordered, manifest_jobs)
    return ordered


def write_progress(
    path: Path,
    *,
    protocol_version: str,
    run_key: str,
    authorization_commit: str,
    function_call_id: str | None,
    manifest_sha256: str,
    started_utc: str,
    last_checkpoint_utc: str | None,
    completed_jobs: int,
    total_jobs: int,
    technical_failures: int,
    last_completed_job_id: str | None,
    state: str,
) -> None:
    payload = {
        "protocol_version": protocol_version,
        "run_key": run_key,
        "authorization_commit": authorization_commit,
        "function_call_id": function_call_id,
        "manifest_sha256": manifest_sha256,
        "started_utc": started_utc,
        "last_checkpoint_utc": last_checkpoint_utc,
        "completed_jobs": completed_jobs,
        "total_jobs": total_jobs,
        "remaining_jobs": total_jobs - completed_jobs,
        "technical_failures": technical_failures,
        "last_completed_job_id": last_completed_job_id,
        "state": state,
        "checkpoint_batch_size": CHECKPOINT_BATCH_SIZE,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "activation_capture": False,
        "output_hidden_states": False,
        "logit_save": False,
        "physiology": False,
        "expected_prompt_bank_sha256": EXPECTED_PROMPT_BANK_SHA256,
    }
    atomic_write_json(path, payload)


def write_heartbeat(path: Path, *, utc: str, state: str, completed_jobs: int) -> None:
    atomic_write_json(
        path,
        {
            "utc": utc,
            "state": state,
            "completed_jobs": completed_jobs,
            "total_jobs": N_PLANNED_GENERATIONS,
        },
    )


def unfinished_jobs(
    manifest_jobs: list[dict[str, Any]], completed_ids: set[str]
) -> list[dict[str, Any]]:
    return [j for j in manifest_jobs if j["job_id"] not in completed_ids]


def assert_frozen_hashes(
    *,
    inference_sha: str,
    seed_sha: str,
    selection_sha: str,
    bank_sha: str,
) -> None:
    if inference_sha != FROZEN_INFERENCE_MANIFEST_SHA256:
        raise ValueError(f"inference manifest sha mismatch: {inference_sha}")
    if seed_sha != FROZEN_SEED_MANIFEST_SHA256:
        raise ValueError(f"seed manifest sha mismatch: {seed_sha}")
    if selection_sha != FROZEN_SELECTION_MANIFEST_SHA256:
        raise ValueError(f"selection manifest sha mismatch: {selection_sha}")
    if bank_sha != EXPECTED_PROMPT_BANK_SHA256:
        raise ValueError(f"prompt bank sha mismatch: {bank_sha}")
