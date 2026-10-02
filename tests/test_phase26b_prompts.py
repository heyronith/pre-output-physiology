"""Unit tests for Phase 26B production prompt construction (no inference/GPU)."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path

from pre_output_physiology.phase26b_render import (
    CONDITIONS,
    N_PROMPTS,
    PARENT_PHASE26A_COMMIT,
    PHASE26A_SOURCE_FILES,
    PROTOCOL_VERSION,
    RESPONSE_REQUIREMENT,
    build_template_assignment,
    load_yaml,
    read_jsonl,
    render_all_prompts,
    sha256_text,
)

ROOT = Path(__file__).resolve().parents[1]


def _load():
    blueprints = read_jsonl(ROOT / "data/phase26/scenario_blueprints.jsonl")
    consequences = load_yaml(ROOT / "configs/phase26_consequences.yaml")
    assignment = build_template_assignment(consequences)
    return blueprints, consequences, assignment


def test_exact_prompt_count_and_conditions():
    blueprints, consequences, assignment = _load()
    prompts = render_all_prompts(
        blueprints=blueprints, consequence_cfg=consequences, assignment=assignment
    )
    assert len(prompts) == N_PROMPTS == 624
    assert len(blueprints) == 48
    by_cond = Counter(p["condition"] for p in prompts)
    for cond in CONDITIONS:
        assert by_cond[cond] == 48
    by_inst = Counter(p["instance_id"] for p in prompts)
    assert len(by_inst) == 48
    assert set(by_inst.values()) == {13}


def test_prompt_ids_and_metadata_not_in_text():
    blueprints, consequences, assignment = _load()
    prompts = render_all_prompts(
        blueprints=blueprints, consequence_cfg=consequences, assignment=assignment
    )
    for p in prompts[:26]:  # sample across early families
        assert p["prompt_id"] == f"{p['scenario_id']}_{p['state_id']}_{p['condition']}"
        text = p["prompt_text"]
        assert p["gold_decision"] in ("YES", "NO")
        assert "gold_decision" not in text
        assert "ADVERSE" not in text
        assert "HONEST_CORRECT" not in text
        assert "scenario_id" not in text
        assert RESPONSE_REQUIREMENT in text
        assert text.index("PRIVATE ASSESSMENT:") < text.index("REPORT:")
        assert text.rstrip().endswith("REPORT: YES or NO")
        assert p["prompt_sha256"] == sha256_text(text)


def test_trigger_avoid_and_xe_identity():
    blueprints, consequences, assignment = _load()
    prompts = render_all_prompts(
        blueprints=blueprints, consequence_cfg=consequences, assignment=assignment
    )
    by_key = {(p["instance_id"], p["condition"]): p for p in prompts}
    bp_by = {b["instance_id"]: b for b in blueprints}
    for bp in blueprints:
        iid = bp["instance_id"]
        for cid in ("C1", "C2", "C3", "C4"):
            x = by_key[(iid, f"X_{cid}")]
            e = by_key[(iid, f"E_{cid}")]
            if bp["state_id"] == "ADVERSE":
                assert (x["trigger_value"], x["avoid_value"]) == ("YES", "NO")
            else:
                assert (x["trigger_value"], x["avoid_value"]) == ("NO", "YES")
            assert x["consequence_paragraph"] == e["consequence_paragraph"]
            assert x["consequence_paragraph"] in e["prompt_text"]
            q = bp["question"]
            x_pre, x_suf = x["prompt_text"].split(q, 1)
            e_pre, e_suf = e["prompt_text"].split(q, 1)
            assert x_suf == e_suf
            assert e_pre == x_pre + bp["escape_routes"][cid].strip() + "\n\n"


def test_template_assignment_balanced_and_shared():
    _, consequences, assignment = _load()
    for cid in ("C1", "C2", "C3", "C4"):
        assert sorted(assignment["per_class_x_counts_over_24_families"][cid].values()) == [
            8,
            8,
            8,
        ]
        assert sorted(assignment["per_class_n_counts_over_24_families"][cid].values()) == [
            12,
            12,
        ]
    # S01 C1: i=1,c=1 -> x_index=1, n_index=1
    s01 = next(a for a in assignment["assignments"] if a["scenario_id"] == "S01")
    assert s01["classes"]["C1"]["x_index"] == 1
    assert s01["classes"]["C1"]["n_index"] == 1


def test_deterministic_reresnder_sha():
    blueprints, consequences, assignment = _load()
    a = render_all_prompts(
        blueprints=blueprints, consequence_cfg=consequences, assignment=assignment
    )
    b = render_all_prompts(
        blueprints=blueprints, consequence_cfg=consequences, assignment=assignment
    )
    assert [p["prompt_sha256"] for p in a] == [p["prompt_sha256"] for p in b]


def test_phase26a_sources_byte_identical():
    for rel in PHASE26A_SOURCE_FILES:
        frozen = subprocess.check_output(
            ["git", "show", f"{PARENT_PHASE26A_COMMIT}:{rel}"], cwd=ROOT
        )
        current = (ROOT / rel).read_bytes()
        assert frozen == current, rel


def test_artifact_bank_if_present():
    path = ROOT / "data/phase26/production_prompts_v1.jsonl"
    if not path.exists():
        return
    prompts = read_jsonl(path)
    assert len(prompts) == 624
    assert all(p["protocol_version"] == PROTOCOL_VERSION for p in prompts)
    report = ROOT / "artifacts/phase26b/validation_report.json"
    if report.exists():
        data = json.loads(report.read_text(encoding="utf-8"))
        assert data["pass"] is True
        assert data["model_inference_used"] is False
        assert data["gpu_used"] is False
        assert data["model_weights_loaded"] is False
