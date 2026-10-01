#!/usr/bin/env python3
"""Aggregate Phase 24G-R2 Modal-500 results after LOPO=28 and nested=500."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pre_output_physiology.phase24c_design import sha256_file  # noqa: E402
from pre_output_physiology.phase24g_r2_modal500 import (  # noqa: E402
    AMENDMENT,
    GUARANTEE,
    LOPO_N,
    NESTED_REPS,
    PROXY_COMPARE_NESTED,
    STARTING_SHA,
    STATUS,
    assert_phase24f_immutable,
    category_change_statement,
    classify_diagnostic_category,
    compare_proxy_vs_full,
    config_hash,
    config_payload,
    load_all_lopo_rows,
    load_all_nested_rows,
    lopo_path,
    nested_path,
    rep_id,
    summarize_lopo,
    summarize_nested,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

OUT = REPO / "artifacts/phase24g_r2_modal500"
RESULTS = OUT / "results"  # local mirror of volume results/
PRIOR_24G = REPO / "artifacts/phase24g_diagnostics"
REPORT = REPO / "reports/phase24g_r2_modal500_diagnostic_correction.md"
AUDIT = REPO / "reports/phase24g_r2_modal500_code_audit_manifest.md"


def main() -> int:
    root = RESULTS
    if not (root / "nested").exists():
        raise SystemExit(f"STOP: missing {root}/nested — download from Modal first")

    prompts = sorted(
        {
            json.loads(p.read_text())["left_out_prompt"]
            for p in (root / "lopo" / "full").glob("lopo_*.json")
        }
    )
    if len(prompts) != LOPO_N:
        # fall back to split manifest
        split = json.loads(
            (
                REPO / "artifacts/phase24c_design/split_manifest.json"
            ).read_text(encoding="utf-8")
        )
        prompts = sorted(split["train_prompt_ids"] + split["validation_prompt_ids"])

    lopo_rows = load_all_lopo_rows(root, prompts, proxy=False)
    nested_rows = load_all_nested_rows(root)
    lopo_sum = summarize_lopo(lopo_rows)
    # Adapt nested summary keys: summarize_nested expects discovery_delta etc.
    nested_sum = summarize_nested(nested_rows)
    nested_sum["n_reps_required"] = NESTED_REPS

    # Proxy comparison
    proxy_lopo = load_all_lopo_rows(root, prompts, proxy=True)
    proxy_nested = []
    pdir = root / "nested_proxy"
    for rep in range(PROXY_COMPARE_NESTED):
        path = pdir / f"rep_{rep_id(rep)}.json"
        proxy_nested.append(json.loads(path.read_text(encoding="utf-8")))
    proxy_vs = {
        "lopo": compare_proxy_vs_full(proxy_lopo, lopo_rows),
        "nested_first_50": compare_proxy_vs_full(
            proxy_nested, nested_rows[:PROXY_COMPARE_NESTED]
        ),
    }

    imm = assert_phase24f_immutable(REPO)
    old_cat = json.loads(
        (PRIOR_24G / "diagnostic_category.json").read_text(encoding="utf-8")
    )["category"]
    prior_topo = json.loads(
        (PRIOR_24G / "val_test_topology.json").read_text(encoding="utf-8")
    )
    prior_same = json.loads(
        (PRIOR_24G / "same_prompt_robustness.json").read_text(encoding="utf-8")
    )
    prior_probe = json.loads(
        (PRIOR_24G / "probe_direction_stability.json").read_text(encoding="utf-8")
    )
    p24f = json.loads(
        (
            REPO / "artifacts/phase24f_confirmation/primary_metrics.json"
        ).read_text(encoding="utf-8")
    )
    evidence = {
        "topology_pearson": prior_topo["overall"]["pearson"],
        "lopo_exact_frac": lopo_sum["frac_exact_t1_l20"],
        "lopo_neighborhood_frac": lopo_sum["frac_neighborhood"],
        "optimism_gap": nested_sum["optimism_gap_mean"],
        "test_same_prompt_frac_gt0": prior_same.get(
            "test_ge2_combined_summary", {}
        ).get("fraction_gt_0", float("nan")),
        "probe_mean_cosine": prior_probe["coordinates"]["frozen_candidate"][
            "stability"
        ]["mean_pairwise_cosine"],
        "test_delta_auroc": p24f["delta_auroc"],
        "test_ci_width": float(
            p24f["delta_auroc_bootstrap"]["ci95"][1]
            - p24f["delta_auroc_bootstrap"]["ci95"][0]
        ),
    }
    new_cat = classify_diagnostic_category(evidence)
    change = category_change_statement(old_cat, new_cat)

    write_json(OUT / "lopo_full_pipeline.json", {"summary": lopo_sum, "rows": lopo_rows})
    write_json(
        OUT / "nested_full_pipeline_500.json",
        {"summary": nested_sum, "rows": nested_rows},
    )
    write_json(OUT / "proxy_vs_full_comparison.json", proxy_vs)
    write_json(
        OUT / "corrected_diagnostic_category.json",
        {
            "previous_category_phase24g": old_cat,
            "corrected_category": new_cat,
            "category_C_status": change,
            "evidence": evidence,
            "amendment": AMENDMENT,
            "nested_reps": NESTED_REPS,
            "phase24f_remains_fail": True,
        },
    )
    write_json(OUT / "phase24f_immutability.json", imm)

    ending = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO)
        .decode()
        .strip()
    )
    cfg = config_payload()
    manifest = {
        "created_at": utc_now_iso(),
        "starting_sha": STARTING_SHA,
        "ending_sha": ending,
        "status": STATUS,
        "nested_reps": NESTED_REPS,
        "lopo_n": LOPO_N,
        "amendment": AMENDMENT,
        "guarantee": GUARANTEE,
        "config_hash": config_hash(cfg),
        "category": new_cat,
        "category_C_status": change,
        "lopo_summary": lopo_sum,
        "nested_summary": {
            k: nested_sum[k]
            for k in nested_sum
            if k
            in (
                "n_reps_completed",
                "frac_candidate_exists",
                "expected_discovery_delta_mean",
                "expected_discovery_delta_median",
                "expected_heldout_delta_mean",
                "expected_heldout_delta_median",
                "optimism_gap_mean",
                "optimism_gap_median",
                "optimism_gap_empirical_percentile_025_975",
                "frac_exact_t1_l20",
                "time_hist",
                "layer_hist",
            )
        },
        "proxy_vs_full": proxy_vs,
    }
    write_json(OUT / "correction_manifest.json", manifest)

    REPORT.write_text(
        f"""# Phase 24G-R2 — Modal 500 corrected diagnostics

**Status:** `{STATUS}`

## Prospective amendment

{AMENDMENT}

{GUARANTEE}

## Results

LOPO exact t=1/L20: **{lopo_sum['frac_exact_t1_l20']:.4f}**
Nested candidate frequency: **{nested_sum['frac_candidate_exists']:.4f}**
Discovery ΔAUROC mean/median: **{nested_sum['expected_discovery_delta_mean']:.4f}** /
**{nested_sum['expected_discovery_delta_median']:.4f}**
Held-out ΔAUROC mean/median: **{nested_sum['expected_heldout_delta_mean']:.4f}** /
**{nested_sum['expected_heldout_delta_median']:.4f}**
Optimism gap mean/median: **{nested_sum['optimism_gap_mean']:.4f}** /
**{nested_sum['optimism_gap_median']:.4f}**
Optimism empirical 2.5–97.5% interval (of 500 gaps; not a CI on the mean):
**{nested_sum['optimism_gap_empirical_percentile_025_975']}**
Nested t=1/L20 frequency: **{nested_sum['frac_exact_t1_l20']:.4f}**

Corrected category: **{new_cat}** (Category C: **{change}**)
Phase-24F FAIL unchanged.
""",
        encoding="utf-8",
    )

    hashes = {
        p.name: sha256_file(str(p))
        for p in OUT.iterdir()
        if p.is_file() and p.name != "artifact_hashes.json"
    }
    write_json(OUT / "artifact_hashes.json", hashes)
    print(json.dumps({"status": STATUS, "category": new_cat, "change": change}, indent=2))
    return 0


if __name__ == "__main__":
    # silence unused import warnings for helpers referenced in docs
    _ = (lopo_path, nested_path)
    raise SystemExit(main())
