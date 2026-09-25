"""Protocol contract tests — documentation must encode frozen scientific requirements."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (REPO_ROOT / "docs/research_protocol.md").read_text(encoding="utf-8")
REPRO = (REPO_ROOT / "docs/reproducibility.md").read_text(encoding="utf-8")
LITERATURE = (REPO_ROOT / "docs/literature_basis.md").read_text(encoding="utf-8")
DECISIONS = (REPO_ROOT / "docs/decision_log.md").read_text(encoding="utf-8")


def test_hypotheses_h1_through_h5_present() -> None:
    for hyp in ("H1", "H2", "H3", "H4", "H5"):
        assert re.search(rf"\b{hyp}\b", PROTOCOL), f"missing {hyp}"


def test_h5_explicitly_separated_from_phase1() -> None:
    assert re.search(r"H5.*not tested in Phase 1|not tested in Phase 1.*H5", PROTOCOL, re.I | re.S)


def test_null_failure_conditions_present() -> None:
    assert re.search(r"null\s*/\s*failure|failure conditions", PROTOCOL, re.I)
    for theme in (
        "baselines",
        "held-out",
        "wording",
        "uncertainty",
        "distribution shift",
        "sham",
        "refusal",
    ):
        assert theme.lower() in PROTOCOL.lower(), f"missing failure theme: {theme}"


def test_required_control_conditions_present() -> None:
    patterns = [
        r"KNOWN-TRUTH\s*/\s*HONEST",
        r"KNOWN-TRUTH\s*/\s*DECEPTIVE",
        r"\bUNCERTAINTY\b",
        r"FALSE-BELIEF",
        r"STRATEGIC-NONDECEPTIVE",
        r"EXPLICIT-DECEPTION\s*/\s*ROLEPLAY",
    ]
    for pattern in patterns:
        assert re.search(pattern, PROTOCOL, re.I), f"missing control: {pattern}"


def test_pre_first_token_intent_constraint_prominent() -> None:
    assert "pre-first-token" in PROTOCOL.lower() or "pre-first-token" in PROTOCOL
    assert "propensity" in PROTOCOL.lower()
    assert "deceptive intent" in PROTOCOL.lower()


def test_measurement_regimes_pre_registered() -> None:
    assert "Prompt-boundary" in PROTOCOL or "prompt-boundary" in PROTOCOL.lower()
    assert "Pre-deceptive-output" in PROTOCOL or "pre-deceptive-output" in PROTOCOL.lower()


def test_reproducibility_fields_documented() -> None:
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
    for field in required:
        assert field in REPRO, f"missing reproducibility field: {field}"


def test_literature_records_lasr_sha_and_mixtral_limitation() -> None:
    assert "f4c6ad69b10a5436a2e819c69009431802a0f5f7" in LITERATURE
    assert "2026.findings-acl.1139" in LITERATURE
    assert "Mixtral" in LITERATURE
    assert "Do NOT state that the ACL work validated Mistral-7B on InsiderTrading" in LITERATURE or (
        "Do not" in LITERATURE and "InsiderTrading" in LITERATURE and "Mixtral" in LITERATURE
    )


def test_decision_log_contains_d001_through_d006() -> None:
    for code in ("D001", "D002", "D003", "D004", "D005", "D006"):
        assert code in DECISIONS
