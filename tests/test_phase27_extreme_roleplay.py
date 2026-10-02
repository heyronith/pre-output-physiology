"""Phase 27 extreme roleplay pre-GPU tests (no model load / no GPU)."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path

import pytest
import yaml

from pre_output_physiology.phase27_extreme_roleplay import (
    CHECKPOINT_BATCH_SIZE,
    CONSEQUENCE_CLASSES,
    MODEL_ID,
    MODEL_REVISION,
    N_PLANNED_GENERATIONS,
    N_ROLLOUTS,
    N_SCENARIOS,
    PARENT_PHASE26C_RESULTS_COMMIT,
    PHASE26C_FROZEN_RESULT_FILES,
    PROTOCOL_VERSION,
    STAY_IN_CHARACTER,
    TOKENIZER_REVISION,
    build_inference_manifest,
    build_prompt_bank,
    build_scenario_rows,
    find_banned_hits,
    load_checkpoint_job_ids,
    mixed_eligible,
    render_chat_plaintext,
    rollout_seed,
    run_key_from_manifest_sha,
    score_to_primary,
    sha256_file,
    unfinished_jobs,
    validate_scenario_record,
)

ROOT = Path(__file__).resolve().parents[1]


def test_counts_and_balance():
    rows = build_scenario_rows()
    assert len(rows) == N_SCENARIOS == 12
    by_class = Counter(r["consequence_class"] for r in rows)
    for cid in CONSEQUENCE_CLASSES:
        assert by_class[cid] == 3
    assert sum(1 for r in rows if r["ground_truth"] == "YES") == 6
    assert sum(1 for r in rows if r["ground_truth"] == "NO") == 6
    assert sum(1 for r in rows if r["advantageous_false"] == "YES") == 6
    assert sum(1 for r in rows if r["advantageous_false"] == "NO") == 6


def test_scenario_ids_exact():
    ids = [r["scenario_id"] for r in build_scenario_rows()]
    assert ids == [f"S27-{i:02d}" for i in range(1, 13)]


def test_manifest_288_and_24_per_scenario():
    prompts = build_prompt_bank(build_scenario_rows())
    jobs = build_inference_manifest(prompts)
    assert len(jobs) == N_PLANNED_GENERATIONS == 288
    assert len({j["job_id"] for j in jobs}) == 288
    assert set(Counter(j["scenario_id"] for j in jobs).values()) == {N_ROLLOUTS}
    assert len({j["seed"] for j in jobs}) == 288


def test_frozen_seed_vector():
    assert rollout_seed(prompt_sha256="abc", rollout_index=0) == 3239543168
    assert rollout_seed(prompt_sha256="abc", rollout_index=1) != 3239543168
    a = rollout_seed(prompt_sha256="abc", rollout_index=0)
    b = rollout_seed(prompt_sha256="abc", rollout_index=0)
    assert a == b
    assert 0 <= a <= 0xFFFFFFFF


def test_prompt_hashes_stable():
    a = build_prompt_bank(build_scenario_rows())
    b = build_prompt_bank(build_scenario_rows())
    assert [p["prompt_sha256"] for p in a] == [p["prompt_sha256"] for p in b]
    for p in a:
        rendered = render_chat_plaintext(
            system_scenario=p["system_scenario"],
            user_question=p["user_question"],
            assistant_prefix=p["assistant_prefix"],
        )
        assert rendered == p["rendered_prompt_text"]
        assert rendered.endswith(p["assistant_prefix"])


def test_banned_language_and_no_report_format():
    for s in build_scenario_rows():
        visible = "\n".join(
            [s["system_content"], s["user_question"], s["assistant_prefix"]]
        )
        assert not find_banned_hits(visible)
        assert "PRIVATE ASSESSMENT" not in visible
        assert "REPORT:" not in visible
        assert "YES or NO" not in visible
        assert STAY_IN_CHARACTER in s["system_content"]
        assert validate_scenario_record(s) == []


def test_model_and_generation_contract():
    cfg = yaml.safe_load(
        (ROOT / "configs/phase27_extreme_roleplay_feasibility.yaml").read_text()
    )
    assert cfg["model"]["model_id"] == MODEL_ID
    assert cfg["model"]["revision"] == MODEL_REVISION == TOKENIZER_REVISION
    assert cfg["model"]["quantization"] is None
    assert cfg["generation"]["temperature"] == 1.0
    assert cfg["generation"]["max_new_tokens"] == 120
    assert cfg["generation"]["top_p"] is None
    assert cfg["generation"]["top_k"] is None
    assert cfg["generation"]["greedy"] is False
    assert cfg["status"] == "phase27_extreme_roleplay_preflight_ready"
    assert cfg["authorizations"]["modal_gpu_behavior_authorized"] is False
    for k in (
        "activation_capture_authorized",
        "output_hidden_states_authorized",
        "logit_save_authorized",
        "physiology_collection_authorized",
        "grader_execution_authorized",
    ):
        assert cfg["authorizations"][k] is False


def test_mixed_eligible_and_score_mapping():
    assert mixed_eligible(n_honest=10, n_deceptive=10, n_ambiguous=4)  # 20 labelable
    assert mixed_eligible(n_honest=4, n_deceptive=16, n_ambiguous=4)
    assert not mixed_eligible(n_honest=3, n_deceptive=17, n_ambiguous=4)  # honest < 4
    assert not mixed_eligible(n_honest=10, n_deceptive=9, n_ambiguous=5)  # labelable 19
    assert score_to_primary(1) == "HONEST"
    assert score_to_primary(4) == "AMBIGUOUS"
    assert score_to_primary(7) == "DECEPTIVE"


def test_checkpoint_resume_logic(tmp_path: Path):
    jobs = build_inference_manifest(build_prompt_bank(build_scenario_rows()))[:20]
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir()
    # write 12 completed
    lines = [
        json.dumps({"job_id": j["job_id"], "seed": j["seed"]}) for j in jobs[:12]
    ]
    (ckpt / "checkpoint_00000.jsonl").write_text("\n".join(lines) + "\n")
    completed, errors = load_checkpoint_job_ids(ckpt)
    assert not errors
    assert len(completed) == 12
    rem = unfinished_jobs(jobs, completed)
    assert len(rem) == 8
    assert rem[0]["job_id"] == jobs[12]["job_id"]
    # duplicate rejection
    (ckpt / "checkpoint_00001.jsonl").write_text(lines[0] + "\n")
    _, errors2 = load_checkpoint_job_ids(ckpt)
    assert any("duplicate" in e for e in errors2)


def test_run_key_and_checkpoint_batch():
    assert CHECKPOINT_BATCH_SIZE == 12
    key = run_key_from_manifest_sha("6968ebcd535e35639d15ce9fc7e4e3a55f950c718dc751f139087e8cfe64b84a")
    assert key == "phase27_extreme_roleplay_v1_6968ebcd535e"


def test_runner_refuses_preflight_status():
    text = (ROOT / "modal/phase27_extreme_roleplay_feasibility.py").read_text()
    assert "phase27_extreme_roleplay_generation_authorized" in text
    assert "modal_gpu_behavior_authorized" in text
    assert "activation_capture" in text
    cfg = yaml.safe_load(
        (ROOT / "configs/phase27_extreme_roleplay_feasibility.yaml").read_text()
    )
    assert cfg["status"] != "phase27_extreme_roleplay_generation_authorized"


def test_phase26c_artifacts_byte_identical():
    for rel in PHASE26C_FROZEN_RESULT_FILES:
        frozen = subprocess.check_output(
            ["git", "show", f"{PARENT_PHASE26C_RESULTS_COMMIT}:{rel}"], cwd=ROOT
        )
        assert frozen == (ROOT / rel).read_bytes(), rel


def test_artifacts_if_present():
    prompts = ROOT / "data/phase27/extreme_roleplay_prompts_v1.jsonl"
    if not prompts.exists():
        return
    assert sha256_file(prompts) == (
        "d8678c7da5c901d0c1d276e6f322dc4134f7531394aaf9b1910b2206fc4cc4d7"
    )
    inf = ROOT / "artifacts/phase27/inference_manifest.jsonl"
    assert sha256_file(inf) == (
        "6968ebcd535e35639d15ce9fc7e4e3a55f950c718dc751f139087e8cfe64b84a"
    )
    seeds = ROOT / "artifacts/phase27/seed_manifest.jsonl"
    assert sha256_file(seeds) == (
        "891e882c51cebeab3bee5e4181ae16f9e52a4fc4ea7c7616cc038e15f45e3298"
    )
    grading = ROOT / "docs/phase27_extreme_roleplay_grading_spec.md"
    assert sha256_file(grading) == (
        "bc5c6d4e174dffa1bea6a16d847386113a36cff3d1b87b55756940f809d3ab5e"
    )
    jobs = [
        json.loads(l) for l in inf.read_text().splitlines() if l.strip()
    ]
    assert len(jobs) == 288
    assert PROTOCOL_VERSION == "phase27_extreme_roleplay_v1"
