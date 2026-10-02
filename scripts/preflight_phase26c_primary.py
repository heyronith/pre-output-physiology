#!/usr/bin/env python3
"""Build Phase 26C primary frozen manifests and run preflight (CPU only; no inference)."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

import yaml

from pre_output_physiology.phase26c_primary import (
    ATTN_IMPLEMENTATION,
    EXPECTED_PROMPT_BANK_SHA256,
    MODEL_ID,
    MODEL_REVISION,
    N_ACTIVE_PROMPTS,
    N_E_GENERATIONS,
    N_FAMILIES,
    N_K_GENERATIONS,
    N_N_GENERATIONS,
    N_PLANNED_GENERATIONS,
    N_X_GENERATIONS,
    PARENT_PHASE26A_COMMIT,
    PARENT_PHASE26B_COMMIT,
    PHASE26A_SOURCE_FILES,
    PHASE26B_FROZEN_FILES,
    PROTOCOL_VERSION,
    TOKENIZER_REVISION,
    build_inference_manifest,
    build_seed_manifest,
    build_selection_manifest,
    load_prompt_bank,
    scan_prompt_truth_leaks,
    select_adverse_prompts,
    sha256_bytes,
    validate_template_assignment,
    write_jsonl,
)


def git_show_bytes(root: Path, commit: str, rel: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{rel}"], cwd=root)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()
    root = Path(args.repo_root)
    cfg_path = root / "configs/phase26c_primary_behavioral_feasibility.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    # --- Phase 26A/B immutability ---
    for rel in PHASE26A_SOURCE_FILES:
        if (root / rel).read_bytes() != git_show_bytes(root, PARENT_PHASE26A_COMMIT, rel):
            errors.append(f"Phase 26A modified vs {PARENT_PHASE26A_COMMIT}: {rel}")
    for rel in PHASE26B_FROZEN_FILES:
        if (root / rel).read_bytes() != git_show_bytes(root, PARENT_PHASE26B_COMMIT, rel):
            errors.append(f"Phase 26B modified vs {PARENT_PHASE26B_COMMIT}: {rel}")

    bank_path = root / cfg["paths"]["prompt_bank"]
    bank_sha = sha256_bytes(bank_path.read_bytes())
    if bank_sha != EXPECTED_PROMPT_BANK_SHA256:
        errors.append(f"prompt bank sha256 mismatch: {bank_sha}")

    try:
        bank = load_prompt_bank(bank_path)
        selected = select_adverse_prompts(bank)
    except ValueError as exc:
        errors.append(str(exc))
        selected = []

    # Hash identity with bank for each selected prompt
    bank_by_id = {p["prompt_id"]: p for p in bank} if selected else {}
    for p in selected:
        src = bank_by_id[p["prompt_id"]]
        if p["prompt_text"] != src["prompt_text"]:
            errors.append(f"{p['prompt_id']}: prompt_text mutated")
        if p["prompt_sha256"] != src["prompt_sha256"]:
            errors.append(f"{p['prompt_id']}: prompt_sha256 mismatch")
        from pre_output_physiology.phase26c_primary import sha256_text

        if sha256_text(p["prompt_text"]) != p["prompt_sha256"]:
            errors.append(f"{p['prompt_id']}: text hash != recorded sha256")

    if any(p["state_id"] != "ADVERSE" for p in selected):
        errors.append("SAFE present in selection")
    if any("SAFE" in p["prompt_id"] for p in selected):
        errors.append("SAFE prompt_id in selection")

    assignment = json.loads(
        (root / cfg["paths"]["template_assignment"]).read_text(encoding="utf-8")
    )
    tmpl = (
        validate_template_assignment(selected, assignment)
        if selected
        else {"ok": False, "errors": ["no selection"], "per_class_x_counts_over_24_families": {}, "per_class_n_counts_over_24_families": {}}
    )
    if not tmpl["ok"]:
        errors.extend(tmpl["errors"])

    leaks = scan_prompt_truth_leaks(selected) if selected else []
    # Filter SAFE operational wording: only hard-fail on non-authorized hits
    hard_leaks = []
    for finding in leaks:
        bad = [h for h in finding["hits"] if h != "meta:SAFE"]
        # Also ignore if only hits are from operational prose already frozen —
        # ADVERSE must never appear; gold/truth leaks are hard fails.
        if bad:
            hard_leaks.append({**finding, "hits": bad})
    if hard_leaks:
        errors.append(f"truth/metadata leaks in {len(hard_leaks)} prompts: {hard_leaks[:5]}")

    selection_rows = build_selection_manifest(selected) if selected else []
    jobs = build_inference_manifest(selected) if selected else []
    seeds = build_seed_manifest(jobs) if jobs else []

    if jobs:
        by_rt = Counter(j["rollout_type"] for j in jobs)
        by_cond_family = Counter()
        # generation totals by condition family
        k_gens = sum(1 for j in jobs if j["condition"] == "K")
        n_gens = sum(1 for j in jobs if j["condition"].startswith("N_"))
        x_gens = sum(1 for j in jobs if j["condition"].startswith("X_"))
        e_gens = sum(1 for j in jobs if j["condition"].startswith("E_"))
        if k_gens != N_K_GENERATIONS:
            errors.append(f"K gens {k_gens} != {N_K_GENERATIONS}")
        if n_gens != N_N_GENERATIONS:
            errors.append(f"N gens {n_gens} != {N_N_GENERATIONS}")
        if x_gens != N_X_GENERATIONS:
            errors.append(f"X gens {x_gens} != {N_X_GENERATIONS}")
        if e_gens != N_E_GENERATIONS:
            errors.append(f"E gens {e_gens} != {N_E_GENERATIONS}")
        if len(jobs) != N_PLANNED_GENERATIONS:
            errors.append(f"total jobs {len(jobs)} != {N_PLANNED_GENERATIONS}")
        if any(j["state_id"] != "ADVERSE" for j in jobs):
            errors.append("SAFE in inference manifest")
        if by_rt.get("greedy") != N_FAMILIES:
            errors.append(f"greedy count {by_rt.get('greedy')} != {N_FAMILIES}")

    # Write frozen artifacts
    write_jsonl(root / cfg["paths"]["selection_manifest"], selection_rows)
    write_jsonl(root / cfg["paths"]["inference_manifest"], jobs)
    write_jsonl(root / cfg["paths"]["seed_manifest"], seeds)

    # Config hash manifest
    hash_targets = [
        "configs/phase26c_primary_behavioral_feasibility.yaml",
        "src/pre_output_physiology/phase26c_primary.py",
        cfg["paths"]["prompt_bank"],
        cfg["paths"]["template_assignment"],
        cfg["paths"]["selection_manifest"],
        cfg["paths"]["inference_manifest"],
        cfg["paths"]["seed_manifest"],
    ]
    config_hash = {
        "protocol_version": PROTOCOL_VERSION,
        "parent_phase26a_commit": PARENT_PHASE26A_COMMIT,
        "parent_phase26b_commit": PARENT_PHASE26B_COMMIT,
        "hashes": {
            rel: sha256_bytes((root / rel).read_bytes())
            for rel in hash_targets
            if (root / rel).exists()
        },
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "attn_implementation": ATTN_IMPLEMENTATION,
        "activation_capture_authorized": False,
        "physiology_collection_authorized": False,
        "modal_gpu_authorized": bool(
            cfg["authorizations"].get("modal_gpu_mistral_screening_authorized")
        ),
        "inference_executed": False,
    }
    out_hash = root / cfg["paths"]["config_hash_manifest"]
    out_hash.parent.mkdir(parents=True, exist_ok=True)
    out_hash.write_text(json.dumps(config_hash, indent=2, sort_keys=True) + "\n")

    # Authorization hard asserts
    auth = cfg["authorizations"]
    for key in (
        "activation_capture_authorized",
        "output_hidden_states_authorized",
        "logit_save_authorized",
        "probe_fitting_authorized",
        "sae_analysis_authorized",
        "grader_model_authorized",
        "physiology_collection_authorized",
        "safe_robustness_run_authorized",
    ):
        if auth.get(key) is not False:
            errors.append(f"{key} must be false")

    if cfg.get("status") != "phase26c_primary_preflight_ready":
        errors.append(f"unexpected status {cfg.get('status')}")

    report = {
        "ok": not errors,
        "errors": errors,
        "protocol_version": PROTOCOL_VERSION,
        "parent_phase26a_commit": PARENT_PHASE26A_COMMIT,
        "parent_phase26b_commit": PARENT_PHASE26B_COMMIT,
        "prompt_bank_sha256": bank_sha,
        "phase26a_byte_identical": not any("Phase 26A" in e for e in errors),
        "phase26b_byte_identical": not any("Phase 26B" in e for e in errors),
        "n_families": N_FAMILIES if selected else 0,
        "n_active_prompts": len(selected),
        "n_safe_in_selection": sum(1 for p in selected if p["state_id"] == "SAFE"),
        "prompt_counts": dict(Counter(p["condition"] for p in selected)),
        "generation_counts": {
            "K": N_K_GENERATIONS,
            "N": N_N_GENERATIONS,
            "X": N_X_GENERATIONS,
            "E": N_E_GENERATIONS,
            "total": N_PLANNED_GENERATIONS,
        },
        "n_inference_jobs": len(jobs),
        "n_stochastic_seeds": len(seeds),
        "template_validation": tmpl,
        "truth_leak_findings_hard": hard_leaks,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "attn_implementation": ATTN_IMPLEMENTATION,
        "activations_disabled": True,
        "physiology_disabled": True,
        "gpu_authorized": False,
        "inference_executed": False,
        "selection_manifest_sha256": sha256_bytes(
            (root / cfg["paths"]["selection_manifest"]).read_bytes()
        )
        if selection_rows
        else None,
        "inference_manifest_sha256": sha256_bytes(
            (root / cfg["paths"]["inference_manifest"]).read_bytes()
        )
        if jobs
        else None,
        "seed_manifest_sha256": sha256_bytes(
            (root / cfg["paths"]["seed_manifest"]).read_bytes()
        )
        if seeds
        else None,
    }
    out = root / cfg["paths"]["preflight"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in report if k != "truth_leak_findings_hard"}, indent=2))
    if hard_leaks:
        print("HARD LEAKS:", json.dumps(hard_leaks[:10], indent=2))
    if errors:
        print("FAIL")
        for e in errors[:30]:
            print(f"  - {e}")
        return 1
    print("PASS — Phase 26C primary preflight ready (no inference).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
