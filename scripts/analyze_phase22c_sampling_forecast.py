#!/usr/bin/env python3
"""Phase 22C — updated K=20 sampling forecast (zero model calls)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase22c_forecast import (  # noqa: E402
    FORECAST_TOTALS,
    GUARANTEE,
    INTERPRETATION,
    N_MONTE_CARLO,
    N_OBSERVED,
    PHASE21_ONSET_RUN,
    PHASE22B_ONSET_RUN,
    RNG_SEED,
    _sha_json,
    decide_support,
    empirical_weak_prior,
    forecast_model,
    load_prompt_counts,
    reproduce_k20_gates,
    summarize_observed,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

OUT_DIR = REPO_ROOT / "artifacts/phase22c_forecast"
REPORT = REPO_ROOT / "reports/phase22c_updated_sampling_forecast.md"
MODELS = (
    "jeffreys",
    "empirical_weak",
    "structural_zeros",
    "conservative_rare_event",
)


def _load(run_id: str) -> list[dict]:
    path = REPO_ROOT / "artifacts/runs" / run_id / "annotated.jsonl"
    return [
        json.loads(x)
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def _fmt_row(f: dict) -> str:
    return (
        f"| {f['total_rollouts_per_prompt']} | "
        f"{f['expected_train_qualifying']:.2f} "
        f"[{f['train_qualifying_ci95'][0]:.0f}, {f['train_qualifying_ci95'][1]:.0f}] | "
        f"{f['expected_test_qualifying']:.2f} "
        f"[{f['test_qualifying_ci95'][0]:.0f}, {f['test_qualifying_ci95'][1]:.0f}] | "
        f"{f['p_train_ge_25']:.4f} | "
        f"{f['p_test_ge_8']:.4f} | "
        f"{f['p_both_gates']:.4f} |"
    )


def main() -> int:
    rows21 = _load(PHASE21_ONSET_RUN)
    rows22 = _load(PHASE22B_ONSET_RUN)
    combined = rows21 + rows22
    counts = load_prompt_counts(combined)
    k20 = reproduce_k20_gates(counts)
    if not k20["reproduced"]:
        raise SystemExit(
            f"STOP: pipeline error — K=20 gates did not reproduce "
            f"(train={k20['n_train_qualifying']}, test={k20['n_test_qualifying']}; "
            f"expected {k20['expected_train']}/{k20['expected_test']})"
        )

    observed = summarize_observed(counts)
    emp_prior = empirical_weak_prior(counts)
    print(
        f"K=20 reproduced: train={k20['n_train_qualifying']} "
        f"test={k20['n_test_qualifying']}; emp_prior={emp_prior}",
        flush=True,
    )

    results_by_model: dict[str, list[dict]] = {}
    for model in MODELS:
        print(f"=== model={model} ===", flush=True)
        results_by_model[model] = []
        for k in FORECAST_TOTALS:
            print(f"  forecasting K={k} ...", flush=True)
            results_by_model[model].append(
                forecast_model(
                    counts,
                    total_rollouts=k,
                    model=model,  # type: ignore[arg-type]
                    n_mc=N_MONTE_CARLO,
                    seed=RNG_SEED,
                    empirical_prior=emp_prior if model == "empirical_weak" else None,
                )
            )

    decision = decide_support(results_by_model)

    per_prompt = [
        {
            "prompt_id": c.prompt_id,
            "split": c.split,
            "n_honest": c.n_honest,
            "n_deceptive_explicit": c.n_deceptive_explicit,
            "n_other": c.n_other,
            "n_total": c.n_total,
            "qualifies_at_20": c.qualifies_observed,
        }
        for c in counts
    ]

    model_defs = {
        m: results_by_model[m][0]["model_meta"] for m in MODELS
    }

    summary = {
        "created_at": utc_now_iso(),
        "phase": "phase22c",
        "status": decision["status"],
        "decision": decision,
        "interpretation": INTERPRETATION,
        "sources": {
            "phase21_onset_run_id": PHASE21_ONSET_RUN,
            "phase22b_onset_run_id": PHASE22B_ONSET_RUN,
            "n_combined_rows": len(combined),
            "n_observed_per_prompt": N_OBSERVED,
        },
        "k20_reproduction": k20,
        "method": {
            "models": list(MODELS),
            "model_definitions": model_defs,
            "forecast_totals": list(FORECAST_TOTALS),
            "conditioning": "keep observed 20; draw K-20 additional",
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
            "no_k_gt_20_generation": True,
        },
        "observed_at_20": observed,
        "forecasts_by_model": results_by_model,
        "guarantee": GUARANTEE,
        "authorizations": {
            "model_calls_authorized": False,
            "mistral_calls_authorized": False,
            "openai_calls_authorized": False,
            "activation_extraction_authorized": False,
            "probe_fitting_authorized": False,
            "physiology_authorized": False,
            "k_gt_20_generation_authorized": False,
            "adaptive_resampling_authorized": False,
            "prompt_changes_authorized": False,
            "threshold_changes_authorized": False,
        },
    }
    summary["summary_sha256"] = _sha_json(
        {
            k: summary[k]
            for k in (
                "sources",
                "method",
                "k20_reproduction",
                "observed_at_20",
                "forecasts_by_model",
                "decision",
            )
        }
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUT_DIR / "forecast_summary.json", summary)
    write_json(OUT_DIR / "per_prompt_counts.json", {"prompts": per_prompt})

    lines = [
        "# Phase 22C — updated sampling forecast from observed K=20",
        "",
        f"**Status:** `{decision['status']}`",
        "",
        f"**Decision:** {decision['decision_text']}",
        "",
        f"**Smallest supported K:** {decision['smallest_supported_k']}",
        "",
        f"**Interpretation:** {INTERPRETATION}",
        "",
        "## Sources",
        "",
        f"- Phase-21 onset: `{PHASE21_ONSET_RUN}`",
        f"- Phase-22B onset: `{PHASE22B_ONSET_RUN}`",
        f"- Combined: {N_OBSERVED} rollouts × 371 prompts",
        "",
        "## K=20 reproduction (required)",
        "",
        f"- TRAIN qualifying: **{k20['n_train_qualifying']}** "
        f"(expected {k20['expected_train']})",
        f"- TEST qualifying: **{k20['n_test_qualifying']}** "
        f"(expected {k20['expected_test']})",
        f"- Reproduced: **{k20['reproduced']}** → HOLD",
        "",
        "## Method",
        "",
        "- 3-way multinomial: honest / deceptive-explicit / other.",
        "- Condition on observed 20; draw K−20 additional.",
        f"- Monte Carlo: {N_MONTE_CARLO} (seed {RNG_SEED}).",
        "- Gates unchanged: ≥2/2; TRAIN≥25 / TEST≥8.",
        "- No Mistral / OpenAI / activations / physiology / new generation.",
        "",
        "### Model definitions",
        "",
    ]
    for m in MODELS:
        meta = model_defs[m]
        lines.append(f"**`{m}`:** {meta.get('formulation')}")
        lines.append("")

    lines += [
        "## Observed at K=20",
        "",
        f"- Qualifying: TRAIN {observed['n_train_qualifying_at_20']} / "
        f"TEST {observed['n_test_qualifying_at_20']}",
        f"- Prompts with zero honest: {observed['n_prompts_zero_honest']}",
        f"- Prompts with zero deceptive-explicit: "
        f"{observed['n_prompts_zero_deceptive_explicit']}",
        f"- Global proportions (h, d, other): "
        f"{[round(x, 4) for x in observed['global_proportions']]}",
        "",
        "### Phase-22B switching (descriptive)",
        "",
        "| bucket | acquired / n |",
        "|---|---:|",
        "| honest>0, dec=0 → acquire dec | 11/134 |",
        "| dec>0, honest=0 → acquire honest | 8/171 |",
        "| exact 10/0 → acquire dec | 3/63 |",
        "| exact 0/10 → acquire honest | 2/106 |",
        "",
        "## Forecast tables",
        "",
    ]
    for m in MODELS:
        lines += [
            f"### `{m}`",
            "",
            "| K | E[TRAIN] [95% PI] | E[TEST] [95% PI] | "
            "P(TRAIN≥25) | P(TEST≥8) | P(both) |",
            "|---:|---|---|---:|---:|---:|",
        ]
        for f in results_by_model[m]:
            lines.append(_fmt_row(f))
        lines.append("")

    lines += [
        "## Decision rule application",
        "",
        "| K | Jeffreys P(both) | Empirical P(both) | "
        "Conservative P(both) | Structural P(both) | Meets support? |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for k in FORECAST_TOTALS:
        row = decision["per_k"][str(k)]
        lines.append(
            f"| {k} | {row['jeffreys_p_both']:.4f} | "
            f"{row['empirical_weak_p_both']:.4f} | "
            f"{row['conservative_p_both']:.4f} | "
            f"{row['structural_zeros_p_both']:.4f} | "
            f"{'yes' if row['meets_support_rule'] else 'no'} |"
        )
    lines += [
        "",
        "Support requires ≥2 non-structural-zero models with P(BOTH)≥0.8 "
        "at the same K **and** conservative rare-event P(BOTH)≥0.5.",
        "",
        "## Authorizations (all false)",
        "",
        "- model / Mistral / OpenAI calls",
        "- activation extraction / probe fitting / physiology",
        "- K>20 generation / adaptive resampling",
        "- prompt / threshold changes",
        "",
        "## Guarantee",
        "",
        GUARANTEE,
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"status": decision["status"], **decision}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
