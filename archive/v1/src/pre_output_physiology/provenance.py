"""Hashing and run-manifest helpers for reproducible artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path | str) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(repo_root: Path | str | None = None) -> str | None:
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    try:
        out = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def write_json(path: Path | str, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path | str, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def assert_not_main_revision(name: str, revision: str) -> None:
    if str(revision).strip().lower() == "main":
        raise ValueError(f"{name} revision must not be 'main' (got {revision!r})")


def software_versions() -> dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for pkg in ("numpy", "pandas", "sklearn", "torch", "transformers", "yaml", "pydantic"):
        try:
            mod = __import__(pkg if pkg != "sklearn" else "sklearn")
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not_imported"
    return versions


def build_run_manifest(**fields: Any) -> dict[str, Any]:
    """Assemble a Phase-1-compatible run manifest plus Phase-2 fields."""
    required = [
        "run_id",
        "timestamp",
        "git_commit",
        "model_id",
        "model_revision",
        "dataset_revision",
        "config_file",
        "random_seed",
        "hardware",
        "gpu_type",
        "software_versions",
        "precision",
        "generation_parameters",
        "activation_locations_collected",
        "output_artifact_hashes",
    ]
    missing = [k for k in required if k not in fields]
    if missing:
        raise ValueError(f"Run manifest missing required fields: {missing}")
    for key in ("model_revision", "dataset_revision"):
        assert_not_main_revision(key, str(fields[key]))
    return dict(fields)
