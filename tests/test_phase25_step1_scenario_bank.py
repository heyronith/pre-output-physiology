from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_committed_bank_validates():
    mod = load_module(REPO / "scripts/validate_phase25_step1_scenario_bank.py", "p25_validator")
    result = mod.validate(REPO)
    assert result["status"] == "PASS", result["errors"]


def test_generator_is_deterministic_in_memory():
    mod = load_module(REPO / "scripts/generate_phase25_step1_scenario_bank.py", "p25_generator")
    b1, p1, g1, a1 = mod.generate()
    b2, p2, g2, a2 = mod.generate()
    assert b1 == b2
    assert p1 == p2
    assert g1 == g2
    assert a1 == a2
    report = mod.validate(b1, p1, g1, a1)
    assert report["status"] == "PASS", report["errors"]

    frozen_b = [json.loads(x) for x in (REPO / "data/phase25/scenario_bank/base_scenarios.jsonl").read_text().splitlines()]
    frozen_p = [json.loads(x) for x in (REPO / "data/phase25/scenario_bank/prompts.jsonl").read_text().splitlines()]
    frozen_g = [json.loads(x) for x in (REPO / "data/phase25/scenario_bank/gold_answers.jsonl").read_text().splitlines()]
    frozen_a = [json.loads(x) for x in (REPO / "data/phase25/scenario_bank/content_audit.jsonl").read_text().splitlines()]
    assert b1 == frozen_b
    assert p1 == frozen_p
    assert g1 == frozen_g
    assert a1 == frozen_a


def test_primary_design_counts_and_balance():
    bases = [json.loads(x) for x in (REPO / "data/phase25/scenario_bank/base_scenarios.jsonl").read_text().splitlines()]
    prompts = [json.loads(x) for x in (REPO / "data/phase25/scenario_bank/prompts.jsonl").read_text().splitlines()]
    assert len(bases) == 180
    assert len(prompts) == 720
    for domain in sorted({b["domain"] for b in bases}):
        rows = [b for b in bases if b["domain"] == domain]
        assert len(rows) == 30
        assert sum(b["gold_decision"] == "YES" for b in rows) == 15
        assert sum(b["gold_decision"] == "NO" for b in rows) == 15
