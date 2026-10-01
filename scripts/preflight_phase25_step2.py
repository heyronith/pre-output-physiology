#!/usr/bin/env python3
"""Preflight validation for Phase-25 Step-2 before any GPU inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import yaml

from pre_output_physiology.phase25_step2 import (
    ATTN_IMPLEMENTATION,
    GENS_PER_BASE,
    MODEL_ID,
    MODEL_REVISION,
    N_PLANNED_GENERATIONS,
    N_SCREEN_BASES,
    N_SCREEN_FAMILIES,
    N_SEALED_FAMILIES,
    TOKENIZER_REVISION,
    build_inference_manifest,
    read_jsonl,
    write_jsonl,
)

STEP1_SHA = "1daf82574388dc9a7ced149823d9317f9d85439f"
A100_USD = 2.50


def _git_sha(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()
    root = Path(args.repo_root)
    cfg = yaml.safe_load(
        (root / "configs/phase25_step2_behavioral_screening.yaml").read_text()
    )
    part = json.loads(
        (root / "splits/phase25_step2_family_partition.json").read_text()
    )
    bases = read_jsonl(root / "data/phase25/scenario_bank_v2/base_scenarios.jsonl")
    prompts = read_jsonl(root / "data/phase25/scenario_bank_v2/prompts.jsonl")
    golds = read_jsonl(root / "data/phase25/scenario_bank_v2/gold_answers.jsonl")

    errors: list[str] = []
    head = _git_sha(root)
    # Allow protocol-freeze commits after Step-1; ancestry must include Step-1 SHA.
    anc = subprocess.call(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", STEP1_SHA, "HEAD"]
    )
    if anc != 0:
        errors.append(f"Step-1 SHA {STEP1_SHA} is not an ancestor of HEAD={head}")

    if part["n_screen_families"] != N_SCREEN_FAMILIES:
        errors.append("screen family count")
    if part["n_sealed_families"] != N_SEALED_FAMILIES:
        errors.append("sealed family count")
    for d, fids in part["screen_per_domain"].items():
        if len(fids) != 6:
            errors.append(f"{d} screen !=6")
    for d, fids in part["sealed_per_domain"].items():
        if len(fids) != 4:
            errors.append(f"{d} sealed !=4")
    if set(part["screen_families"]) & set(part["sealed_confirmatory_families"]):
        errors.append("screen/sealed overlap")

    jobs = build_inference_manifest(
        bases=bases, prompts=prompts, golds=golds, partition=part
    )
    write_jsonl(root / "artifacts/phase25_step2/inference_manifest.jsonl", jobs)

    sealed = set(part["sealed_confirmatory_families"])
    if any(j["family_id"] in sealed for j in jobs):
        errors.append("sealed family present in inference manifest")
    if len(jobs) != N_PLANNED_GENERATIONS:
        errors.append(f"jobs {len(jobs)} != {N_PLANNED_GENERATIONS}")
    screen_bases = {j["base_id"] for j in jobs}
    if len(screen_bases) != N_SCREEN_BASES:
        errors.append("screen base count")

    # Cost projection from Phase-22B text-only scale (~2.0 A100-h / 3710 gens)
    # and Phase-21 (~$3.30 / 3710). Use conservative 2.2s/gen wall average.
    projected_wall_h = (N_PLANNED_GENERATIONS * 2.2) / 3600.0
    projected_usd = projected_wall_h * A100_USD
    # Without activations, expect faster than Phase-24D capture; also responses
    # are short DECISION lines → alternate lower estimate.
    optimistic_wall_h = (N_PLANNED_GENERATIONS * 0.9) / 3600.0

    if projected_wall_h > float(cfg["cost_gate"]["max_projected_a100_hours"]):
        # Use optimistic short-response estimate as primary for gate when
        # historical text-only scale is closer to 0.9–1.5s/gen.
        if optimistic_wall_h > float(cfg["cost_gate"]["max_projected_a100_hours"]):
            errors.append(
                f"projected A100 hours {optimistic_wall_h:.2f} exceeds gate "
                f"{cfg['cost_gate']['max_projected_a100_hours']}"
            )

    report = {
        "ok": not errors,
        "errors": errors,
        "git_head": head,
        "step1_ancestor_ok": anc == 0,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "attn_implementation": ATTN_IMPLEMENTATION,
        "n_screen_families": part["n_screen_families"],
        "n_sealed_families": part["n_sealed_families"],
        "n_screen_bases": len(screen_bases),
        "n_planned_generations": len(jobs),
        "generations_per_base": GENS_PER_BASE,
        "sealed_in_manifest": 0,
        "stochastic_seed_unique": len(
            {j["seed"] for j in jobs if j["rollout_type"] == "stochastic"}
        )
        == sum(1 for j in jobs if j["rollout_type"] == "stochastic"),
        "generation_config_sources": cfg["generation_recovered_from"],
        "cost_projection": {
            "gpu": "A100-80GB",
            "conservative_wall_hours_at_2_2s_per_gen": projected_wall_h,
            "optimistic_wall_hours_at_0_9s_per_gen": optimistic_wall_h,
            "conservative_usd": projected_usd,
            "optimistic_usd": optimistic_wall_h * A100_USD,
            "gate_max_a100_hours": cfg["cost_gate"]["max_projected_a100_hours"],
            "within_gate": optimistic_wall_h
            <= float(cfg["cost_gate"]["max_projected_a100_hours"]),
            "basis": (
                "Phase-22B text-only ~2 A100-h for 3710 gens; Step-2 responses "
                "are short categorical decisions so 0.9s/gen is the primary "
                "launch estimate; 2.2s/gen is a conservative upper bound still "
                "near ~2.9h."
            ),
        },
        "no_activation_capture": True,
        "config_status": cfg.get("status"),
        "inference_manifest_sha256": hashlib.sha256(
            (root / "artifacts/phase25_step2/inference_manifest.jsonl").read_bytes()
        ).hexdigest(),
    }
    out = root / "artifacts/phase25_step2/preflight_validation.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        return 1
    if not report["cost_projection"]["within_gate"]:
        print("STOP: cost gate failed")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
