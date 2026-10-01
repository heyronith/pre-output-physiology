import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/phase25/scenario_bank_v2"


def loadj(name):
    return [json.loads(x) for x in (DATA / name).read_text().splitlines() if x.strip()]


def test_counts_and_family_balance():
    bases = loadj("base_scenarios.jsonl")
    assert len(bases) == 240
    fam = {}
    for b in bases:
        fam.setdefault(b["family_id"], []).append(b["gold_decision"])
    assert len(fam) == 60
    assert all(sorted(v) == ["NO", "NO", "YES", "YES"] for v in fam.values())


def test_variant_counterbalance_within_domain():
    bases = loadj("base_scenarios.jsonl")
    from collections import Counter, defaultdict

    x = defaultdict(Counter)
    for b in bases:
        x[(b["domain"], b["variant_index"])][b["gold_decision"]] += 1
    assert all(c == {"YES": 5, "NO": 5} for c in x.values())


def test_family_split_unit_and_logic_coverage():
    schema = json.loads((DATA / "schema.json").read_text())
    assert schema["independent_split_unit"] == "family_id"
    meta = loadj("family_metadata.jsonl")
    from collections import defaultdict

    d = defaultdict(set)
    for x in meta:
        d[x["domain"]].add(x["logic_structure"])
    assert len(meta) == 60
    assert all(len(v) >= 4 for v in d.values())


def test_kax_and_d_structure():
    bases = {x["base_id"]: x for x in loadj("base_scenarios.jsonl")}
    prompts = loadj("prompts.jsonl")
    pm = {(p["base_id"], p["form"]): p for p in prompts}
    assert len(prompts) == 960
    for bid, b in bases.items():
        k, a, x, d = [pm[(bid, f)] for f in "KAXD"]
        for key in (
            "context_text",
            "rule_text",
            "evidence_items",
            "question",
            "gold_decision",
            "family_id",
        ):
            assert k[key] == a[key] == x[key]
        assert len(d["evidence_items"]) == len(b["evidence_items"]) - 1
        assert b["decisive_evidence_id"] not in d["available_evidence_ids"]


def test_validator_passes():
    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/validate_phase25_step1_v2.py"),
            "--repo-root",
            str(ROOT),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr


def test_generator_reproduces_byte_exact(tmp_path):
    # Copy only frozen config/scripts needed for generation.
    (tmp_path / "configs").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "configs/phase25_step1_v2_families.json").write_bytes(
        (ROOT / "configs/phase25_step1_v2_families.json").read_bytes()
    )
    (tmp_path / "scripts/generate_phase25_step1_v2.py").write_bytes(
        (ROOT / "scripts/generate_phase25_step1_v2.py").read_bytes()
    )
    r = subprocess.run(
        [
            sys.executable,
            str(tmp_path / "scripts/generate_phase25_step1_v2.py"),
            "--repo-root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    for name in (
        "base_scenarios.jsonl",
        "prompts.jsonl",
        "gold_answers.jsonl",
        "family_metadata.jsonl",
        "schema.json",
    ):
        assert (tmp_path / "data/phase25/scenario_bank_v2" / name).read_bytes() == (
            DATA / name
        ).read_bytes()
