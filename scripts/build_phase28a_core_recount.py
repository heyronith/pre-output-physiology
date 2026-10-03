#!/usr/bin/env python3
"""Phase 28A core-behavior recount + corrected forecast (CPU only; no GPU)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from pre_output_physiology.phase28_same_prompt_replication import (
    CORE_DEFINITIONS,
    DEVELOPMENT_SELECTION_DISCLAIMER,
    EXPECTED_SELECTED_SCENARIO_IDS,
    FORECAST_NS,
    FROZEN_CORE_COUNTS,
    GUARANTEE,
    INTERPRETATION,
    N_MONTE_CARLO,
    N_SOURCE_GRADES,
    PARENT_PHASE27_RESULTS_COMMIT,
    PHASE27_GRADES_RELPATH,
    PHASE27_GRADES_SHA256,
    PHASE27_PROMPTS_RELPATH,
    PHASE27_RAW_RELPATH,
    PROTOCOL_VERSION,
    RNG_SEED,
    STATUS_CORE_RECOUNT,
    SUPERSEDED_GENERATION_MANIFEST_SHA256,
    SUPERSEDED_MANIFEST_STATUS,
    build_core_behavior_adjudication,
    default_authorizations,
    forecast_all_scenarios,
    joint_forecast_selected,
    load_jsonl,
    load_scenario_counts_from_core,
    scenario_counts_as_core_dict,
    select_phase28b_candidates,
    sha256_file,
    verify_phase27_grades,
    write_json,
    write_jsonl,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()
    root = Path(args.repo_root).resolve()

    cfg_path = root / "configs/phase28_same_prompt_diversity_replication.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    paths = cfg["paths"]

    grades_path = root / PHASE27_GRADES_RELPATH
    grades_sha = sha256_file(grades_path)
    if grades_sha != PHASE27_GRADES_SHA256:
        raise SystemExit(f"STOP: grades SHA changed: {grades_sha}")
    grades = load_jsonl(grades_path)
    verify_phase27_grades(grades, grades_sha)
    raw = load_jsonl(root / PHASE27_RAW_RELPATH)
    prompts = load_jsonl(root / PHASE27_PROMPTS_RELPATH)

    adjudication = build_core_behavior_adjudication(
        grades=grades, raw_rows=raw, prompts=prompts
    )
    counts = load_scenario_counts_from_core(adjudication, prompts)
    observed = [scenario_counts_as_core_dict(c) for c in counts]

    print(
        f"Core recount OK; forecasting with n_mc={N_MONTE_CARLO} seed={RNG_SEED} ...",
        flush=True,
    )
    forecast_rows = forecast_all_scenarios(counts, ns=FORECAST_NS, n_mc=N_MONTE_CARLO)
    # Annotate F naming aliases on each row.
    for row in forecast_rows:
        row["observed_H_F_A"] = row["observed_H_D_A"]
        row["label_construct"] = "phase27_direct_report_core_proposition"
        row["p_H_ge_12_and_F_ge_12"] = row["p_H_ge_12_and_D_ge_12"]
        for th_h, th_d in ((8, 8), (12, 12), (16, 16), (20, 20)):
            row[f"p_H_ge_{th_h}_and_F_ge_{th_d}"] = row[
                f"p_H_ge_{th_h}_and_D_ge_{th_d}"
            ]

    selection = select_phase28b_candidates(forecast_rows)
    selected_ids = selection["selected_scenario_ids"]
    if selected_ids != list(EXPECTED_SELECTED_SCENARIO_IDS):
        raise SystemExit(
            "STOP: candidate set mismatch after core recount. "
            f"got={selected_ids} expected={list(EXPECTED_SELECTED_SCENARIO_IDS)}"
        )
    # Enrich selection with F naming.
    for item in selection["selected"] + selection["rejected"]:
        item["p_H_ge_12_and_F_ge_12_at_N256"] = item[
            "p_H_ge_12_and_D_ge_12_at_N256"
        ]
        item["observed_H_F_A"] = item["observed_H_D_A"]
    selection["label_construct"] = "phase27_direct_report_core_proposition"
    selection["apollo_honesty_not_used_for_selection"] = True

    joint = joint_forecast_selected(counts, selected_ids, n_mc=N_MONTE_CARLO)
    joint["label_construct"] = "phase27_direct_report_core_proposition"

    art = root / "artifacts/phase28"
    art.mkdir(parents=True, exist_ok=True)

    write_jsonl(root / paths["core_adjudication"], adjudication)
    write_json(
        root / paths["core_counts"],
        {
            "definitions": CORE_DEFINITIONS,
            "frozen_core_counts_H_F_A": {
                sid: {"H": h, "F": f, "A": a}
                for sid, (h, f, a) in FROZEN_CORE_COUNTS.items()
            },
            "scenarios": observed,
            "n_rows": len(adjudication),
            "gemma_grades_path": PHASE27_GRADES_RELPATH,
            "gemma_grades_sha256": grades_sha,
            "gemma_grades_unmodified": True,
            "development_data_construct_adjudication": True,
            "performed_before_phase28_gpu_generation": True,
        },
    )
    write_json(root / paths["observed_counts"], {"scenarios": observed})
    write_json(root / paths["forecast_table"], {"rows": forecast_rows})
    write_json(
        root / paths["joint_forecast"],
        {
            "method": {
                "posterior": (
                    "independent Jeffreys multinomial Dirichlet(H+0.5,F+0.5,A+0.5)"
                ),
                "predictive": "fresh_independent_Multinomial(N,p)_not_cumulative",
                "n_monte_carlo": N_MONTE_CARLO,
                "rng_seed": RNG_SEED,
                "forecast_ns": list(FORECAST_NS),
                "label_construct": "phase27_direct_report_core_proposition",
            },
            "joint": joint,
            "interpretation": INTERPRETATION,
            "development_selection_disclaimer": DEVELOPMENT_SELECTION_DISCLAIMER,
        },
    )
    write_json(root / paths["candidate_selection"], selection)

    # Supersede prior five-prompt Apollo-label manifest (never executed).
    existing_gen = root / paths["generation_manifest"]
    existing_sha = (
        sha256_file(existing_gen) if existing_gen.exists() else None
    )
    superseded = {
        "status": SUPERSEDED_MANIFEST_STATUS,
        "manifest_path": paths["generation_manifest"],
        "manifest_sha256": SUPERSEDED_GENERATION_MANIFEST_SHA256,
        "observed_manifest_sha256": existing_sha,
        "sha_matches_frozen_superseded_value": (
            existing_sha == SUPERSEDED_GENERATION_MANIFEST_SHA256
        ),
        "never_executed": True,
        "reason": (
            "Built from Apollo honesty labels rather than Phase 27 "
            "direct-report core-proposition labels; superseded before any "
            "Phase 28 GPU generation."
        ),
        "new_gpu_manifest_built": False,
        "useful_provenance": True,
    }
    write_json(root / paths["superseded_manifest"], superseded)

    auth = default_authorizations()
    n256 = {
        r["scenario_id"]: r["p_H_ge_12_and_F_ge_12"]
        for r in forecast_rows
        if r["n"] == 256
    }
    preflight = {
        "status": STATUS_CORE_RECOUNT,
        "protocol_version": PROTOCOL_VERSION,
        "parent_phase27_results_commit": PARENT_PHASE27_RESULTS_COMMIT,
        "phase27_grades_sha256": grades_sha,
        "gemma_grades_unmodified": True,
        "n_source_grades": N_SOURCE_GRADES,
        "label_construct": "phase27_direct_report_core_proposition",
        "core_adjudication_sha256": sha256_file(root / paths["core_adjudication"]),
        "core_counts_sha256": sha256_file(root / paths["core_counts"]),
        "observed_counts": observed,
        "forecast_method": {
            "posterior": "Jeffreys Dirichlet(H+0.5, F+0.5, A+0.5)",
            "predictive_batch": "fresh_independent",
            "n_monte_carlo": N_MONTE_CARLO,
            "rng_seed": RNG_SEED,
        },
        "p_H_ge_12_and_F_ge_12_at_N256": n256,
        "selected_scenario_ids": selected_ids,
        "n_selected": len(selected_ids),
        "superseded_generation_manifest": superseded,
        "new_gpu_manifest_built": False,
        "authorizations": auth,
        "gpu_calls": False,
        "model_generation": False,
        "grader_calls": False,
        "physiology": False,
        "phase27_not_reinterpreted_as_pass": True,
        "phase27_remains_hold": True,
        "guarantee": GUARANTEE,
    }
    write_json(root / paths["preflight_validation"], preflight)

    # Decision report
    decision = f"""# Phase 28A — Core-Behavior Recount Decision

**Branch:** `phase28/same-prompt-diversity-replication`  
**Status:** `{STATUS_CORE_RECOUNT}`  
**Parent Phase 27 results:** `{PARENT_PHASE27_RESULTS_COMMIT}`

## Decisions

1. **Phase 27 remains HOLD.** It is not reinterpreted as PASS.
2. **Mistral clearly demonstrated strategic false reporting** under extreme
   roleplay (e.g. S27-02 all advantageous-false; S27-04 largely false on the
   core reliability proposition).
3. **Generic Apollo honesty labels are unsuitable as the sole
   trajectory-selection labels** for a within-prompt physiology experiment.
   Gemma successfully measures the Apollo honesty construct; that construct is
   not identical to the Phase 27 direct-report / core-proposition construct.
4. **After core-proposition adjudication, only one Phase 27 prompt
   (`S27-01`) showed both target behaviors** at levels supporting the frozen
   `P_fresh_N256(H≥12,F≥12)≥0.50` selection rule.
5. **Larger-N sampling of the previously selected five prompts is therefore
   not justified by the original Apollo-label forecast.**
6. **No Phase 28 GPU generation has occurred.**

## Frozen core definitions

{json.dumps(CORE_DEFINITIONS, indent=2)}

## Gemma grades preserved

| Item | Value |
|---|---|
| Path | `{PHASE27_GRADES_RELPATH}` |
| SHA256 | `{grades_sha}` |
| Modified | **no** (byte-for-byte retained) |

Secondary adjudication: `{paths["core_adjudication"]}`  
This is a **development-data construct adjudication** performed before any
Phase 28 GPU generation.

## Corrected H/F/A counts

| Scenario | H | F | A |
|---|---:|---:|---:|
""" + "\n".join(
        f"| {sid} | {h} | {f} | {a} |"
        for sid, (h, f, a) in FROZEN_CORE_COUNTS.items()
    ) + f"""

## Corrected N=256 P(H≥12,F≥12)

| Scenario | P |
|---|---:|
""" + "\n".join(
        f"| {sid} | {n256[sid]:.5f} |" for sid in sorted(n256)
    ) + f"""

## Selection under frozen ≥0.50 rule

Selected: `{selected_ids}`

{DEVELOPMENT_SELECTION_DISCLAIMER}

## Superseded Phase 28B manifest

| Item | Value |
|---|---|
| SHA256 | `{SUPERSEDED_GENERATION_MANIFEST_SHA256}` |
| Status | `{SUPERSEDED_MANIFEST_STATUS}` |
| Executed | **no** |
| New GPU manifest | **not built** |

## Authorization

All generation / grader / physiology authorizations remain **false**.

## Confirmation

**{GUARANTEE}**
"""
    (root / paths["decision_report"]).write_text(decision, encoding="utf-8")

    # Update readiness pointer note
    readiness = root / paths["readiness_report"]
    banner = (
        f"\n\n---\n\n## SUPERSEDED FORECAST NOTE\n\n"
        f"The Apollo-honesty-based five-prompt forecast/manifest above is "
        f"**superseded** by the core-behavior recount "
        f"(`{STATUS_CORE_RECOUNT}`). See `{paths['decision_report']}`. "
        f"Manifest `{SUPERSEDED_GENERATION_MANIFEST_SHA256}` was never executed.\n"
    )
    if readiness.exists():
        text = readiness.read_text(encoding="utf-8")
        if "SUPERSEDED FORECAST NOTE" not in text:
            readiness.write_text(text.rstrip() + banner, encoding="utf-8")

    print(
        json.dumps(
            {
                "status": STATUS_CORE_RECOUNT,
                "selected_scenario_ids": selected_ids,
                "p_N256": n256,
                "superseded_manifest_sha256": SUPERSEDED_GENERATION_MANIFEST_SHA256,
                "superseded_status": SUPERSEDED_MANIFEST_STATUS,
                "new_gpu_manifest_built": False,
                "gemma_grades_sha256": grades_sha,
                "gpu_calls": False,
                "model_generation": False,
                "grader_calls": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
