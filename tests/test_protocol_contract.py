"""Contract test: frozen protocol + gates phrases must remain present."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

REQUIRED_PHRASES = (
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "last prompt token",
    "120",
    "60",
    "paired CI excludes 0",
    "random direction",
    "text baseline",
    "no layer search on locked data",
)


def test_protocol_and_gates_contain_frozen_phrases() -> None:
    protocol = (REPO / "docs/protocol.md").read_text(encoding="utf-8")
    gates = (REPO / "docs/GATES.md").read_text(encoding="utf-8")
    combined = protocol + "\n" + gates
    missing = [p for p in REQUIRED_PHRASES if p not in combined]
    assert not missing, f"missing required phrases: {missing}"
