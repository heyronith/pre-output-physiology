#!/usr/bin/env python3
"""Phase 22A — sampling forecast from Phase-21 onset data (zero model calls)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase22_forecast import (  # noqa: E402
    CREDIBLE_P_BOTH,
    FORECAST_TOTALS,
    GEN_RUN_ID,
    GRADE_RUN_ID,
    GUARANTEE,
    N_MONTE_CARLO,
    N_OBSERVED,
    ONSET_RUN_ID,
    RNG_SEED,
    STRONG_P_BOTH,
    _sha_json,
    decide_support,
    forecast_additional_sampling,
    load_prompt_counts,
    summarize_stability,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

ANNOTATED = (
    REPO_ROOT
    / "artifacts/runs"
    / ONSET_RUN_ID
    / "annotated.jsonl"
)
OUT_DIR = REPO_ROOT / "artifacts/phase22a_forecast"
REPORT = REPO_ROOT / "reports/phase22a_sampling_forecast.md"


def main() -> int:
    rows = [
        json.loads(x)
        for x in ANNOTATED.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    counts = load_prompt_counts(rows)
    stability = summarize_stability(counts)

    per_prompt = [
        {
            "prompt_id": c.prompt_id,
            "split": c.split,
            "n_honest": c.n_honest,
            "n_deceptive_explicit": c.n_deceptive_explicit,
            "n_ambiguous": c.n_ambiguous,
            "n_exclude": c.n_exclude,
            "n_deceptive_no_explicit": c.n_deceptive_no_explicit,
            "p_hat_honest": c.n_honest / N_OBSERVED,
            "p_hat_deceptive_explicit": c.n_deceptive_explicit / N_OBSERVED,
            "qualifies_at_10": c.qualifies_observed,
        }
        for c in counts
    ]

    forecasts = []
    forecasts_struct_zero = []
    for k in FORECAST_TOTALS:
        print(f"forecasting K={k} ...", flush=True)
        forecasts.append(
            forecast_additional_sampling(
                counts, total_rollouts=k, n_mc=N_MONTE_CARLO, seed=RNG_SEED
            )
        )
        forecasts_struct_zero.append(
            forecast_additional_sampling(
                counts,
                total_rollouts=k,
                n_mc=N_MONTE_CARLO,
                seed=RNG_SEED,
                structural_zeros=True,
            )
        )

    decision = decide_support(forecasts)
    decision_struct = decide_support(forecasts_struct_zero)
    summary = {
        "created_at": utc_now_iso(),
        "phase": "phase22",
        "status": decision["status"],
        "decision": decision,
        "sources": {
            "generation_run_id": GEN_RUN_ID,
            "grade_run_id": GRADE_RUN_ID,
            "onset_run_id": ONSET_RUN_ID,
            "annotated_path": str(ANNOTATED.relative_to(REPO_ROOT)),
        },
        "method": {
            "model": "Dirichlet-Multinomial posterior predictive",
            "categories": ["honest", "deceptive_explicit", "other"],
            "prior": "Jeffreys Dirichlet(0.5,0.5,0.5)",
            "conditioning": "keep observed 10; draw K-10 additional",
            "n_monte_carlo": N_MONTE_CARLO,
            "rng_seed": RNG_SEED,
            "gates": {
                "min_honest": 2,
                "min_deceptive_explicit": 2,
                "min_train_qualifying": 25,
                "min_test_qualifying": 8,
            },
            "no_mistral_calls": True,
            "no_openai_calls": True,
            "no_activations": True,
            "no_physiology": True,
        },
        "observed_at_10": stability,
        "forecasts": forecasts,
        "sensitivity_structural_zeros": {
            "note": (
                "Categories never observed in the 10 rollouts are fixed at p=0 "
                "(no Jeffreys rescue of one-sided prompts). Descriptive only; "
                "primary decision uses Jeffreys Dirichlet-Multinomial."
            ),
            "forecasts": forecasts_struct_zero,
            "decision": decision_struct,
        },
        "guarantee": GUARANTEE,
    }
    summary["summary_sha256"] = _sha_json(
        {k: summary[k] for k in ("sources", "method", "observed_at_10", "forecasts", "decision")}
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUT_DIR / "forecast_summary.json", summary)
    write_json(OUT_DIR / "per_prompt_counts.json", {"prompts": per_prompt})

    # Markdown report
    lines = [
        "# Phase 22A — sampling forecast (zero model calls)",
        "",
        f"**Status:** `{decision['status']}`",
        "",
        f"**Decision:** {decision['decision_text']}",
        "",
        f"Best forecast: K={decision['best_total_rollouts']} with "
        f"P(both gates)={decision['best_p_both_gates']:.4f} "
        f"(credible≥{CREDIBLE_P_BOTH}, strong≥{STRONG_P_BOTH}).",
        "",
        "## Method",
        "",
        "- Data: Phase-21 onset annotations only "
        f"(`{ONSET_RUN_ID}`).",
        "- Per prompt: 3-way counts (honest / deceptive-with-valid-explicit-onset / other).",
        "- Prior: Jeffreys Dirichlet(0.5, 0.5, 0.5).",
        "- Predictive: keep the observed 10 rollouts; draw K−10 additional from the "
        "posterior predictive Multinomial.",
        f"- Monte Carlo replicates: {N_MONTE_CARLO} (seed {RNG_SEED}).",
        "- Gates unchanged: ≥2 honest & ≥2 explicit-onset deceptive; "
        "TRAIN≥25, TEST≥8 qualifying prompts.",
        "- No Mistral / OpenAI / activations / physiology.",
        "",
        "## Observed stability at K=10",
        "",
        f"- TRAIN qualifying: {stability['n_train_qualifying_at_10']}",
        f"- TEST qualifying: {stability['n_test_qualifying_at_10']}",
        "",
        "| bucket (n_honest / n_dec_explicit) | n prompts |",
        "|---|---:|",
    ]
    for k, v in sorted(stability["buckets_by_honest_vs_dec_explicit_counts"].items()):
        lines.append(f"| `{k}` | {v} |")
    lines += [
        "",
        "## Forecast table",
        "",
        "| K | E[TRAIN qual] | E[TEST qual] | P(TRAIN≥25) | P(TEST≥8) | P(both) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for f in forecasts:
        lines.append(
            f"| {f['total_rollouts_per_prompt']} | "
            f"{f['expected_train_qualifying']:.2f} | "
            f"{f['expected_test_qualifying']:.2f} | "
            f"{f['p_train_ge_25']:.4f} | "
            f"{f['p_test_ge_8']:.4f} | "
            f"{f['p_both_gates']:.4f} |"
        )
    lines += [
        "",
        "## Sensitivity: structural zeros (no prior rescue)",
        "",
        "If a class was never seen in the 10 rollouts, its future probability is fixed at 0.",
        "",
        f"Sensitivity verdict: **{decision_struct['decision_text']}** "
        f"(best P(both)={decision_struct['best_p_both_gates']:.4f} at "
        f"K={decision_struct['best_total_rollouts']}).",
        "",
        "| K | E[TRAIN] | E[TEST] | P(TRAIN≥25) | P(TEST≥8) | P(both) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for f in forecasts_struct_zero:
        lines.append(
            f"| {f['total_rollouts_per_prompt']} | "
            f"{f['expected_train_qualifying']:.2f} | "
            f"{f['expected_test_qualifying']:.2f} | "
            f"{f['p_train_ge_25']:.4f} | "
            f"{f['p_test_ge_8']:.4f} | "
            f"{f['p_both_gates']:.4f} |"
        )
    lines += ["", "## Guarantee", "", GUARANTEE, ""]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"status": decision["status"], **decision}, indent=2))
    print("primary forecasts:")
    for f in forecasts:
        print(
            f"  K={f['total_rollouts_per_prompt']}: "
            f"E[train]={f['expected_train_qualifying']:.2f} "
            f"E[test]={f['expected_test_qualifying']:.2f} "
            f"P(both)={f['p_both_gates']:.4f}"
        )
    print("structural-zero sensitivity:")
    for f in forecasts_struct_zero:
        print(
            f"  K={f['total_rollouts_per_prompt']}: "
            f"E[train]={f['expected_train_qualifying']:.2f} "
            f"E[test]={f['expected_test_qualifying']:.2f} "
            f"P(both)={f['p_both_gates']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
