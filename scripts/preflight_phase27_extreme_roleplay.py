#!/usr/bin/env python3
"""Phase 27 preflight validation (CPU only; no GPU / no model load)."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

import yaml

from pre_output_physiology.phase27_extreme_roleplay import (
    CONSEQUENCE_CLASSES,
    MODEL_ID,
    MODEL_REVISION,
    N_PLANNED_GENERATIONS,
    N_ROLLOUTS,
    N_SCENARIOS,
    PARENT_PHASE26C_RESULTS_COMMIT,
    PHASE26C_FROZEN_RESULT_FILES,
    PROTOCOL_VERSION,
    TOKENIZER_REVISION,
    build_inference_manifest,
    build_prompt_bank,
    build_scenario_rows,
    find_banned_hits,
    read_jsonl,
    rollout_seed,
    sha256_file,
    validate_scenario_record,
)


def git_show_bytes(root: Path, commit: str, rel: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{rel}"], cwd=root)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()
    root = Path(args.repo_root)
    cfg_path = root / "configs/phase27_extreme_roleplay_feasibility.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    if cfg.get("status") != "phase27_extreme_roleplay_preflight_ready":
        errors.append(f"unexpected status {cfg.get('status')}")
    auth = cfg["authorizations"]
    if auth.get("modal_gpu_behavior_authorized") is not False:
        errors.append("modal_gpu_behavior_authorized must be false")
    for key in (
        "activation_capture_authorized",
        "output_hidden_states_authorized",
        "logit_save_authorized",
        "probe_fitting_authorized",
        "sae_analysis_authorized",
        "causal_intervention_authorized",
        "physiology_collection_authorized",
        "grader_execution_authorized",
    ):
        if auth.get(key) is not False:
            errors.append(f"{key} must be false")

    # Phase 26C immutability
    for rel in PHASE26C_FROZEN_RESULT_FILES:
        path = root / rel
        if not path.exists():
            errors.append(f"missing Phase 26C artifact {rel}")
            continue
        frozen = git_show_bytes(root, PARENT_PHASE26C_RESULTS_COMMIT, rel)
        if path.read_bytes() != frozen:
            errors.append(f"Phase 26C artifact modified vs {PARENT_PHASE26C_RESULTS_COMMIT}: {rel}")

    scenarios = read_jsonl(root / cfg["paths"]["scenarios"])
    prompts = read_jsonl(root / cfg["paths"]["prompts"])
    jobs = read_jsonl(root / cfg["paths"]["inference_manifest"])
    seeds = read_jsonl(root / cfg["paths"]["seed_manifest"])

    if len(scenarios) != N_SCENARIOS:
        errors.append(f"scenarios {len(scenarios)} != {N_SCENARIOS}")
    if len(prompts) != N_SCENARIOS:
        errors.append(f"prompts {len(prompts)} != {N_SCENARIOS}")
    if len(jobs) != N_PLANNED_GENERATIONS:
        errors.append(f"jobs {len(jobs)} != {N_PLANNED_GENERATIONS}")
    if len(seeds) != N_PLANNED_GENERATIONS:
        errors.append(f"seeds {len(seeds)} != {N_PLANNED_GENERATIONS}")

    by_class = Counter(s["consequence_class"] for s in scenarios)
    for cid in CONSEQUENCE_CLASSES:
        if by_class[cid] != 3:
            errors.append(f"{cid} count {by_class[cid]} != 3")
    n_yes = sum(1 for s in scenarios if s["ground_truth"] == "YES")
    n_no = sum(1 for s in scenarios if s["ground_truth"] == "NO")
    if n_yes != 6 or n_no != 6:
        errors.append(f"truth balance YES={n_yes} NO={n_no}")

    for s in scenarios:
        errs = validate_scenario_record(s)
        if errs:
            errors.append(f"{s['scenario_id']}: {errs}")
        hits = find_banned_hits(
            "\n".join(
                [s["system_content"], s["user_question"], s["assistant_prefix"]]
            )
        )
        if hits:
            errors.append(f"{s['scenario_id']} banned: {hits}")

    # Rebuild and compare
    rebuilt_sc = build_scenario_rows()
    rebuilt_pr = build_prompt_bank(rebuilt_sc)
    rebuilt_jobs = build_inference_manifest(rebuilt_pr)
    if [p["prompt_sha256"] for p in prompts] != [p["prompt_sha256"] for p in rebuilt_pr]:
        errors.append("prompt bank not reproducible")
    if [j["seed"] for j in jobs] != [j["seed"] for j in rebuilt_jobs]:
        errors.append("seeds not reproducible")

    by_sid = Counter(j["scenario_id"] for j in jobs)
    if set(by_sid.values()) != {N_ROLLOUTS}:
        errors.append(f"rollouts/scenario not {N_ROLLOUTS}: {dict(by_sid)}")

    # Frozen seed vector for first prompt
    p0 = sorted(prompts, key=lambda x: x["scenario_id"])[0]
    expected0 = rollout_seed(prompt_sha256=p0["prompt_sha256"], rollout_index=0)
    got0 = next(
        j["seed"]
        for j in jobs
        if j["scenario_id"] == p0["scenario_id"] and j["rollout_index"] == 0
    )
    if got0 != expected0:
        errors.append("seed formula mismatch on first job")

    if rollout_seed(prompt_sha256="abc", rollout_index=0) != 3239543168:
        errors.append(
            f"frozen seed vector mismatch: "
            f"{rollout_seed(prompt_sha256='abc', rollout_index=0)}"
        )

    if cfg["model"]["revision"] != MODEL_REVISION:
        errors.append("model revision mismatch")
    if cfg["model"]["model_id"] != MODEL_ID:
        errors.append("model id mismatch")
    if cfg["model"]["tokenizer_revision"] != TOKENIZER_REVISION:
        errors.append("tokenizer revision mismatch")
    if cfg["model"].get("quantization") is not None:
        errors.append("quantization must be null")

    hashes = {
        "scenarios_sha256": sha256_file(root / cfg["paths"]["scenarios"]),
        "prompts_sha256": sha256_file(root / cfg["paths"]["prompts"]),
        "inference_manifest_sha256": sha256_file(root / cfg["paths"]["inference_manifest"]),
        "seed_manifest_sha256": sha256_file(root / cfg["paths"]["seed_manifest"]),
        "grading_spec_sha256": sha256_file(root / cfg["paths"]["grading_spec"]),
        "config_sha256": sha256_file(cfg_path),
        "protocol_doc_sha256": sha256_file(root / cfg["paths"]["protocol_doc"]),
    }

    config_hash = {
        "protocol_version": PROTOCOL_VERSION,
        "parent_phase26c_results_commit": PARENT_PHASE26C_RESULTS_COMMIT,
        "status": cfg.get("status"),
        "modal_gpu_behavior_authorized": False,
        "physiology_collection_authorized": False,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "n_scenarios": len(scenarios),
        "n_planned_generations": len(jobs),
        "hashes": hashes,
        "inference_executed": False,
        "model_weights_loaded": False,
    }
    out_hash = root / cfg["paths"]["config_hash_manifest"]
    out_hash.parent.mkdir(parents=True, exist_ok=True)
    out_hash.write_text(json.dumps(config_hash, indent=2, sort_keys=True) + "\n")

    report = {
        "ok": not errors,
        "errors": errors,
        "protocol_version": PROTOCOL_VERSION,
        "parent_phase26c_results_commit": PARENT_PHASE26C_RESULTS_COMMIT,
        "phase26c_artifacts_byte_identical": not any("Phase 26C" in e for e in errors),
        "n_scenarios": len(scenarios),
        "n_prompts": len(prompts),
        "n_jobs": len(jobs),
        "counts_by_class": dict(by_class),
        "ground_truth_yes": n_yes,
        "ground_truth_no": n_no,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "hashes": hashes,
        "gpu_authorized": False,
        "inference_executed": False,
        "model_weights_loaded": False,
        "physiology_disabled": True,
    }
    out = root / cfg["paths"]["preflight"]
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        print("FAIL")
        for e in errors[:40]:
            print(f"  - {e}")
        return 1
    print("PASS — Phase 27 preflight ready (no inference).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
