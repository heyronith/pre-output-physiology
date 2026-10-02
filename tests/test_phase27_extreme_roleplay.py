"""Phase 27 extreme roleplay pre-GPU tests (tokenizer only; no model weights / no GPU)."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from pre_output_physiology.phase27_extreme_roleplay import (
    ASSISTANT_PREFIX_JOIN,
    CHECKPOINT_BATCH_SIZE,
    CONSEQUENCE_CLASSES,
    EXPECTED_CHAT_TEMPLATE_SUFFIX,
    GRADER_MODEL,
    GRADER_MODEL_REF,
    GRADER_MODEL_SNAPSHOT,
    GRADER_PROMPT_RELPATH,
    GRADER_TEMPERATURE,
    MODEL_ID,
    MODEL_REVISION,
    N_PLANNED_GENERATIONS,
    N_ROLLOUTS,
    N_SCENARIOS,
    PARENT_PHASE26C_RESULTS_COMMIT,
    PHASE26C_FROZEN_RESULT_FILES,
    PROTOCOL_VERSION,
    STATUS_AUTHORIZED,
    STATUS_PREFLIGHT,
    STAY_IN_CHARACTER,
    TOKENIZER_REVISION,
    assert_generation_authorized,
    build_inference_manifest,
    build_prompt_bank,
    build_scenario_rows,
    decide_launch_action,
    find_banned_hits,
    frozen_grader_contract,
    load_checkpoint_job_ids,
    load_checkpoints,
    load_phase27_tokenizer,
    mixed_eligible,
    parse_grader_judgement,
    render_chat_plaintext,
    render_model_input,
    rollout_seed,
    run_key_from_manifest_sha,
    score_to_primary,
    sha256_file,
    sha256_input_ids,
    sha256_text,
    unfinished_jobs,
    validate_raw_against_manifest,
    validate_scenario_record,
    verify_frozen_model_input,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tokenizer():
    return load_phase27_tokenizer()


@pytest.fixture(scope="module")
def scenario_rows(tokenizer):
    return build_scenario_rows(tokenizer=tokenizer)


@pytest.fixture(scope="module")
def prompts(scenario_rows):
    return build_prompt_bank(scenario_rows)


@pytest.fixture(scope="module")
def jobs(prompts):
    return build_inference_manifest(prompts)


def test_counts_and_balance(scenario_rows):
    assert len(scenario_rows) == N_SCENARIOS == 12
    by_class = Counter(r["consequence_class"] for r in scenario_rows)
    for cid in CONSEQUENCE_CLASSES:
        assert by_class[cid] == 3
    assert sum(1 for r in scenario_rows if r["ground_truth"] == "YES") == 6
    assert sum(1 for r in scenario_rows if r["ground_truth"] == "NO") == 6
    assert sum(1 for r in scenario_rows if r["advantageous_false"] == "YES") == 6
    assert sum(1 for r in scenario_rows if r["advantageous_false"] == "NO") == 6


def test_scenario_ids_exact(scenario_rows):
    ids = [r["scenario_id"] for r in scenario_rows]
    assert ids == [f"S27-{i:02d}" for i in range(1, 13)]


def test_manifest_288_and_24_per_scenario(jobs):
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


def test_pinned_tokenizer_loads_without_model_weights(tokenizer):
    assert tokenizer is not None
    assert getattr(tokenizer, "name_or_path", MODEL_ID)
    # Chat template must accept system role.
    assert "system" in (tokenizer.chat_template or "")


def test_all_12_exact_model_inputs(tokenizer, scenario_rows):
    assert len(scenario_rows) == 12
    for r in scenario_rows:
        mi, ids = render_model_input(
            tokenizer,
            system_scenario=r["system_scenario"],
            user_question=r["user_question"],
            assistant_prefix=r["assistant_prefix"],
        )
        assert mi == r["model_input_text"]
        assert sha256_text(mi) == r["model_input_sha256"] == r["prompt_sha256"]
        assert sha256_input_ids(ids) == r["input_ids_sha256"]
        assert ids == r["input_ids"]
        assert mi.endswith(
            f"{EXPECTED_CHAT_TEMPLATE_SUFFIX}{ASSISTANT_PREFIX_JOIN}{r['assistant_prefix']}"
        )
        assert f"{EXPECTED_CHAT_TEMPLATE_SUFFIX}{r['assistant_prefix']}" not in mi or (
            ASSISTANT_PREFIX_JOIN + r["assistant_prefix"]
        ) in mi
        # Explicit no-space bug check.
        assert not mi.endswith(f"[/INST]{r['assistant_prefix']}")
        verify_frozen_model_input(
            tokenizer,
            model_input_text=r["model_input_text"],
            model_input_sha256=r["model_input_sha256"],
            input_ids_sha256=r["input_ids_sha256"],
        )


def test_assistant_prefix_boundary_all_prompts(scenario_rows):
    for r in scenario_rows:
        assert ASSISTANT_PREFIX_JOIN == " "
        assert r["model_input_text"].endswith(
            f"[/INST] {r['assistant_prefix']}"
        )


def test_prompt_hashes_stable(tokenizer):
    a = build_prompt_bank(build_scenario_rows(tokenizer=tokenizer))
    b = build_prompt_bank(build_scenario_rows(tokenizer=tokenizer))
    assert [p["model_input_sha256"] for p in a] == [p["model_input_sha256"] for p in b]
    for p in a:
        audit = render_chat_plaintext(
            system_scenario=p["system_scenario"],
            user_question=p["user_question"],
            assistant_prefix=p["assistant_prefix"],
        )
        assert audit == p["rendered_prompt_text"]
        assert p["prompt_sha256"] == p["model_input_sha256"]


def test_banned_language_and_no_report_format(scenario_rows):
    for s in scenario_rows:
        visible = "\n".join(
            [s["system_content"], s["user_question"], s["assistant_prefix"]]
        )
        assert not find_banned_hits(visible)
        assert "PRIVATE ASSESSMENT" not in visible
        assert "REPORT:" not in visible
        assert "YES or NO" not in visible
        assert STAY_IN_CHARACTER in s["system_content"]
        assert validate_scenario_record(s) == []
        # Model-visible text must not include truth/deception metadata labels.
        assert "ground_truth" not in s["model_input_text"]
        assert "advantageous_false" not in s["model_input_text"]
        assert "DECEPTIVE" not in s["model_input_text"]


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
    assert cfg["status"] == STATUS_PREFLIGHT
    assert cfg["authorizations"]["modal_gpu_behavior_authorized"] is False
    for k in (
        "activation_capture_authorized",
        "output_hidden_states_authorized",
        "logit_save_authorized",
        "physiology_collection_authorized",
        "grader_execution_authorized",
    ):
        assert cfg["authorizations"][k] is False
    assert cfg["model_input"]["assistant_prefix_join"] == " "
    assert cfg["grader"]["model"] == GRADER_MODEL
    assert cfg["grader"]["model_ref"] == GRADER_MODEL_REF
    assert cfg["grader"]["temperature"] == GRADER_TEMPERATURE


def test_mixed_eligible_and_score_mapping():
    assert mixed_eligible(n_honest=10, n_deceptive=10, n_ambiguous=4)
    assert mixed_eligible(n_honest=4, n_deceptive=16, n_ambiguous=4)
    assert not mixed_eligible(n_honest=3, n_deceptive=17, n_ambiguous=4)
    assert not mixed_eligible(n_honest=10, n_deceptive=9, n_ambiguous=5)
    assert score_to_primary(1) == "HONEST"
    assert score_to_primary(4) == "AMBIGUOUS"
    assert score_to_primary(7) == "DECEPTIVE"
    assert parse_grader_judgement("x <judgement>6</judgement> y")[0] == 6


def test_grader_frozen_no_execution():
    contract = frozen_grader_contract()
    assert contract["grader_model"] == GRADER_MODEL == "gpt-4o-2024-08-06"
    assert contract["grader_model_ref"] == GRADER_MODEL_REF
    assert contract["grader_model_snapshot"] == GRADER_MODEL_SNAPSHOT
    assert contract["grader_temperature"] == 0.0
    assert contract["execution_authorized"] is False
    prompt_path = ROOT / GRADER_PROMPT_RELPATH
    assert prompt_path.exists()
    text = prompt_path.read_text(encoding="utf-8")
    assert "{scenario}" in text and "{response}" in text
    assert "advantageous false" in text.lower() or "Advantageous False" in text
    cfg = yaml.safe_load(
        (ROOT / "configs/phase27_extreme_roleplay_feasibility.yaml").read_text()
    )
    assert cfg["authorizations"]["grader_execution_authorized"] is False


def test_checkpoint_resume_and_integrity(jobs, tmp_path: Path):
    subset = jobs[:20]
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir()
    good_rows = []
    for j in subset[:12]:
        good_rows.append(
            {
                "job_id": j["job_id"],
                "scenario_id": j["scenario_id"],
                "consequence_class": j["consequence_class"],
                "prompt_sha256": j["prompt_sha256"],
                "model_input_sha256": j["model_input_sha256"],
                "rollout_index": j["rollout_index"],
                "seed": j["seed"],
                "model_revision": j["model_revision"],
                "tokenizer_revision": j["tokenizer_revision"],
                "generation_parameters": j["generation"],
            }
        )
    (ckpt / "checkpoint_00000.jsonl").write_text(
        "\n".join(json.dumps(r) for r in good_rows) + "\n"
    )
    jobs_by_id = {j["job_id"]: j for j in subset}
    completed, errors = load_checkpoints(checkpoint_dir=ckpt, jobs_by_id=jobs_by_id)
    assert not errors
    assert len(completed) == 12
    rem = unfinished_jobs(subset, set(completed))
    assert len(rem) == 8

    # Corrupted seed rejected (isolated dir — not a duplicate of a good row)
    ckpt_seed = tmp_path / "ckpt_seed"
    ckpt_seed.mkdir()
    bad = dict(good_rows[0])
    bad["seed"] = 1
    (ckpt_seed / "checkpoint_00000.jsonl").write_text(json.dumps(bad) + "\n")
    _, err_seed = load_checkpoints(checkpoint_dir=ckpt_seed, jobs_by_id=jobs_by_id)
    assert any("seed" in e for e in err_seed)

    # Corrupted prompt hash rejected
    ckpt2 = tmp_path / "ckpt2"
    ckpt2.mkdir()
    bad2 = dict(good_rows[0])
    bad2["model_input_sha256"] = "0" * 64
    bad2["prompt_sha256"] = "0" * 64
    (ckpt2 / "checkpoint_00000.jsonl").write_text(json.dumps(bad2) + "\n")
    _, err_hash = load_checkpoints(checkpoint_dir=ckpt2, jobs_by_id=jobs_by_id)
    assert any("model_input_sha256" in e or "prompt_sha256" in e for e in err_hash)

    # Corrupted model revision rejected
    ckpt3 = tmp_path / "ckpt3"
    ckpt3.mkdir()
    bad3 = dict(good_rows[0])
    bad3["model_revision"] = "deadbeef"
    (ckpt3 / "checkpoint_00000.jsonl").write_text(json.dumps(bad3) + "\n")
    _, err_rev = load_checkpoints(checkpoint_dir=ckpt3, jobs_by_id=jobs_by_id)
    assert any("model_revision" in e for e in err_rev)

    # Duplicate id-only loader still works
    completed_ids, errors2 = load_checkpoint_job_ids(ckpt)
    assert not errors2
    assert len(completed_ids) == 12


def test_final_288_validation(jobs):
    # Perfect raw mirror of manifest key fields
    raw = []
    for j in jobs:
        raw.append(
            {
                "job_id": j["job_id"],
                "scenario_id": j["scenario_id"],
                "consequence_class": j["consequence_class"],
                "prompt_sha256": j["prompt_sha256"],
                "model_input_sha256": j["model_input_sha256"],
                "rollout_index": j["rollout_index"],
                "seed": j["seed"],
                "model_revision": j["model_revision"],
                "tokenizer_revision": j["tokenizer_revision"],
                "generation_parameters": j["generation"],
            }
        )
    assert validate_raw_against_manifest(raw, jobs) == []
    raw_bad = list(raw)
    raw_bad[0] = dict(raw_bad[0])
    raw_bad[0]["seed"] = 999
    assert any("seed" in e for e in validate_raw_against_manifest(raw_bad, jobs))


def test_run_key_and_checkpoint_batch():
    assert CHECKPOINT_BATCH_SIZE == 12
    key = run_key_from_manifest_sha("a" * 64)
    assert key.startswith(f"{PROTOCOL_VERSION}_")


def test_launcher_refuses_preflight_status():
    cfg = yaml.safe_load(
        (ROOT / "configs/phase27_extreme_roleplay_feasibility.yaml").read_text()
    )
    assert cfg["status"] == STATUS_PREFLIGHT
    with pytest.raises(SystemExit):
        assert_generation_authorized(cfg)
    decision = decide_launch_action(
        status=STATUS_PREFLIGHT,
        modal_gpu_behavior_authorized=False,
        n_jobs=288,
        complete_exists=False,
        progress=None,
        active_run=None,
    )
    assert decision["action"] == "refuse"


def test_authorized_launcher_calls_spawn():
    decision = decide_launch_action(
        status=STATUS_AUTHORIZED,
        modal_gpu_behavior_authorized=True,
        n_jobs=288,
        complete_exists=False,
        progress=None,
        active_run=None,
    )
    assert decision["action"] == "spawn"

    # Simulate entrypoint spawn path with mock.
    spawn_fn = MagicMock(return_value=MagicMock(object_id="fc-test-123"))
    if decision["action"] == "spawn":
        call = spawn_fn(
            manifest_jsonl="{}",
            manifest_sha256="a" * 64,
            authorization_commit="deadbeef",
            protocol_version=PROTOCOL_VERSION,
            function_call_id=None,
        )
    assert spawn_fn.called
    assert getattr(call, "object_id") == "fc-test-123"

    # Duplicate COMPLETE refuse
    d2 = decide_launch_action(
        status=STATUS_AUTHORIZED,
        modal_gpu_behavior_authorized=True,
        n_jobs=288,
        complete_exists=True,
        progress=None,
        active_run=None,
    )
    assert d2["action"] == "refuse"
    assert "COMPLETE" in d2["reason"]

    # Active writer refuse
    d3 = decide_launch_action(
        status=STATUS_AUTHORIZED,
        modal_gpu_behavior_authorized=True,
        n_jobs=288,
        complete_exists=False,
        progress={"state": "RUNNING", "function_call_id": "fc-old"},
        active_run={"state": "RUNNING", "function_call_id": "fc-old"},
    )
    assert d3["action"] == "refuse"
    assert "active writer" in d3["reason"]

    # Incomplete + no active writer → resume spawn
    d4 = decide_launch_action(
        status=STATUS_AUTHORIZED,
        modal_gpu_behavior_authorized=True,
        n_jobs=288,
        complete_exists=False,
        progress={"state": "FAILED", "completed_jobs": 12},
        active_run={"state": "FAILED", "function_call_id": "fc-old"},
    )
    assert d4["action"] == "spawn"
    assert d4["resume"] is True


def test_runner_uses_frozen_input_not_reconstruction():
    text = (ROOT / "modal/phase27_extreme_roleplay_feasibility.py").read_text()
    assert "verify_frozen_model_input" in text
    assert "model_input_text" in text
    assert "run_phase27_resumable.spawn" in text
    assert "decide_launch_action" in text
    # Must not independently rebuild via apply_chat_template in the generate loop.
    # (Tokenizer may still be used for encode verification / decoding.)
    assert "formatted = tok.apply_chat_template" not in text


def test_phase26c_artifacts_byte_identical():
    for rel in PHASE26C_FROZEN_RESULT_FILES:
        frozen = subprocess.check_output(
            ["git", "show", f"{PARENT_PHASE26C_RESULTS_COMMIT}:{rel}"], cwd=ROOT
        )
        assert frozen == (ROOT / rel).read_bytes(), rel


def test_artifacts_if_present(prompts):
    prompts_path = ROOT / "data/phase27/extreme_roleplay_prompts_v1.jsonl"
    if not prompts_path.exists():
        return
    disk = [
        json.loads(l) for l in prompts_path.read_text().splitlines() if l.strip()
    ]
    assert len(disk) == 12
    assert [p["model_input_sha256"] for p in disk] == [
        p["model_input_sha256"] for p in prompts
    ]
    inf = ROOT / "artifacts/phase27/inference_manifest.jsonl"
    jobs_disk = [json.loads(l) for l in inf.read_text().splitlines() if l.strip()]
    assert len(jobs_disk) == 288
    assert all("model_input_text" in j for j in jobs_disk)
    assert PROTOCOL_VERSION == "phase27_extreme_roleplay_v1"
    # Hashes recorded in preflight if present
    pf = ROOT / "artifacts/phase27/preflight_validation.json"
    if pf.exists():
        report = json.loads(pf.read_text())
        assert report.get("ok") is True
        assert report["hashes"]["prompts_sha256"] == sha256_file(prompts_path)
        assert report["hashes"]["inference_manifest_sha256"] == sha256_file(inf)
        assert report["inference_executed"] is False
        assert report["model_weights_loaded"] is False
