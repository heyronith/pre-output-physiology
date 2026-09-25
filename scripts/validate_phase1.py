#!/usr/bin/env python3
"""Inexpensive Phase 1 integrity checks. Does not download models or launch Modal."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

REQUIRED_FILES = [
    "README.md",
    "pyproject.toml",
    ".gitignore",
    ".env.example",
    "docs/research_brief.md",
    "docs/research_protocol.md",
    "docs/literature_basis.md",
    "docs/methodology_constraints.md",
    "docs/decision_log.md",
    "docs/reproducibility.md",
    "configs/models/mistral_7b_instruct_v02.yaml",
    "configs/experiments/phase2_positive_control.yaml",
    "configs/experiments/phase3_preoutput_scan.yaml",
    "src/pre_output_physiology/__init__.py",
    "src/pre_output_physiology/config.py",
    "src/pre_output_physiology/schemas.py",
    "scripts/validate_phase1.py",
    "tests/test_configs.py",
    "tests/test_protocol_contract.py",
    "data/README.md",
    "artifacts/README.md",
]

REQUIRED_MODEL_FIELDS = [
    "model_id",
    "revision",
    "dtype",
    "quantization",
    "trust_remote_code",
    "device_policy",
    "seed",
]

HYPOTHESIS_PATTERNS = [
    r"\bH1\b",
    r"\bH2\b",
    r"\bH3\b",
    r"\bH4\b",
    r"\bH5\b",
]

CONTROL_PATTERNS = [
    r"KNOWN-TRUTH\s*/\s*HONEST",
    r"KNOWN-TRUTH\s*/\s*DECEPTIVE",
    r"\bUNCERTAINTY\b",
    r"FALSE-BELIEF",
    r"STRATEGIC-NONDECEPTIVE",
    r"EXPLICIT-DECEPTION\s*/\s*ROLEPLAY",
]

REPRO_FIELDS = [
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

SECRET_NAME_PATTERNS = [
    re.compile(r"(^|/)\.env$"),
    re.compile(r"(^|/)credentials.*\.json$", re.I),
    re.compile(r"(^|/)secrets(/|$)", re.I),
    re.compile(r"\.(pem|key)$", re.I),
]


class CheckResult:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.passes: list[str] = []

    def ok(self, message: str) -> None:
        self.passes.append(message)
        print(f"PASS  {message}")

    def fail(self, message: str) -> None:
        self.failures.append(message)
        print(f"FAIL  {message}")


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"{path} root must be a mapping")
    return data


def tracked_paths() -> list[str]:
    git_dir = REPO_ROOT / ".git"
    if not git_dir.exists():
        return []
    import subprocess

    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in completed.stdout.splitlines() if line]


def check_required_files(result: CheckResult) -> None:
    for rel in REQUIRED_FILES:
        path = REPO_ROOT / rel
        if path.is_file():
            result.ok(f"required file exists: {rel}")
        else:
            result.fail(f"missing required file: {rel}")


def check_model_config(result: CheckResult) -> None:
    path = REPO_ROOT / "configs/models/mistral_7b_instruct_v02.yaml"
    try:
        cfg = load_yaml(path)
    except Exception as exc:  # noqa: BLE001 — surface parse errors to the validator
        result.fail(f"model config failed to parse: {exc}")
        return

    for field in REQUIRED_MODEL_FIELDS:
        if field in cfg:
            result.ok(f"model config field present: {field}")
        else:
            result.fail(f"model config missing field: {field}")

    if cfg.get("model_id") == PRIMARY_MODEL_ID:
        result.ok("primary model_id matches frozen checkpoint")
    else:
        result.fail(
            f"primary model_id must be {PRIMARY_MODEL_ID!r}, got {cfg.get('model_id')!r}"
        )

    if cfg.get("quantization") is None:
        result.ok("primary config quantization is null")
    else:
        result.fail(f"primary config must not enable quantization; got {cfg.get('quantization')!r}")

    revision = cfg.get("revision")
    if isinstance(revision, str) and revision.strip().lower() == "main":
        result.fail("model revision must not silently be 'main'")
    elif revision == "TO_BE_PINNED_BEFORE_PHASE2" or (
        isinstance(revision, str) and len(revision) >= 7
    ):
        result.ok(f"model revision acceptable for Phase 1: {revision}")
    else:
        result.fail(f"unexpected model revision value: {revision!r}")


def check_experiment_configs(result: CheckResult) -> None:
    for rel in (
        "configs/experiments/phase2_positive_control.yaml",
        "configs/experiments/phase3_preoutput_scan.yaml",
    ):
        path = REPO_ROOT / rel
        try:
            cfg = load_yaml(path)
            result.ok(f"experiment config parses: {rel}")
        except Exception as exc:  # noqa: BLE001
            result.fail(f"experiment config failed to parse ({rel}): {exc}")
            continue
        for field in ("experiment_id", "phase", "status", "model_config_path"):
            if field not in cfg:
                result.fail(f"{rel} missing field: {field}")


def check_protocol(result: CheckResult) -> None:
    protocol = (REPO_ROOT / "docs/research_protocol.md").read_text(encoding="utf-8")
    for pattern in HYPOTHESIS_PATTERNS:
        if re.search(pattern, protocol):
            result.ok(f"protocol contains hypothesis marker: {pattern}")
        else:
            result.fail(f"protocol missing hypothesis marker: {pattern}")

    if re.search(r"null\s*/\s*failure|failure conditions", protocol, re.I):
        result.ok("protocol contains null/failure criteria section")
    else:
        result.fail("protocol missing null/failure criteria")

    failure_bullets = [
        "no better than appropriate prompt/text/logit baselines",
        "held-out scenarios",
        "explicit deception wording",
        "uncertainty or truth",
        "distribution shift",
        "sham/random",
        "refusal",
    ]
    missing_failure = [item for item in failure_bullets if item.lower() not in protocol.lower()]
    if not missing_failure:
        result.ok("protocol null/failure criteria cover required themes")
    else:
        result.fail(f"protocol null/failure criteria missing themes: {missing_failure}")

    for pattern in CONTROL_PATTERNS:
        if re.search(pattern, protocol, re.I):
            result.ok(f"protocol contains control condition: {pattern}")
        else:
            result.fail(f"protocol missing control condition: {pattern}")


def check_reproducibility(result: CheckResult) -> None:
    text = (REPO_ROOT / "docs/reproducibility.md").read_text(encoding="utf-8")
    for field in REPRO_FIELDS:
        if field in text:
            result.ok(f"reproducibility field documented: {field}")
        else:
            result.fail(f"reproducibility field missing: {field}")


def check_no_tracked_secrets(result: CheckResult) -> None:
    paths = tracked_paths()
    if not paths:
        # Pre-commit / no git yet: scan working tree filenames cautiously.
        candidates = [str(p.relative_to(REPO_ROOT)) for p in REPO_ROOT.rglob("*") if p.is_file()]
        paths = candidates

    offenders = []
    for rel in paths:
        if rel == ".env.example":
            continue
        if any(pat.search(rel) for pat in SECRET_NAME_PATTERNS):
            offenders.append(rel)

    if offenders:
        result.fail(f"possible secret files present/tracked: {offenders}")
    else:
        result.ok("no obvious secret files tracked")


def main() -> int:
    print(f"Phase 1 validation — repo root: {REPO_ROOT}")
    result = CheckResult()
    check_required_files(result)
    check_model_config(result)
    check_experiment_configs(result)
    check_protocol(result)
    check_reproducibility(result)
    check_no_tracked_secrets(result)

    print()
    print(f"Passed: {len(result.passes)}")
    print(f"Failed: {len(result.failures)}")
    if result.failures:
        print("Phase 1 validation FAILED")
        return 1
    print("Phase 1 validation PASSED")
    print("NO PRIMARY EXPERIMENTAL DATA WAS COLLECTED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
