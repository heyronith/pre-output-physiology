#!/usr/bin/env python3
"""Phase 16A: same-prompt sampling-scaling design forecast (no model calls)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase15_onset import (  # noqa: E402
    PHASE14_RAW_CONTINUATIONS_SHA256,
    PHASE14_RUN_ID,
    RULE_HASH,
    RULE_VERSION,
)
from pre_output_physiology.phase16_forecast import (  # noqa: E402
    DESIGN_HASH,
    GUARANTEE,
    MONTE_CARLO_SEED,
    N_GRID,
    N_REPLICATES,
    TEMPERATURE_GRID,
    design_manifest,
    empirical_prompt_counts,
    heterogeneity_diagnostic,
    heterogeneity_penalty,
    interpret_forecast,
    run_optimistic_forecast,
    run_primary_forecast,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

PHASE15_RULE_HASH = RULE_HASH
PHASE15_RULE_VERSION = RULE_VERSION

P15_SUMMARY = REPO_ROOT / "artifacts/phase15a_onset_reanalysis/reanalysis_summary.json"
P15_RULES = REPO_ROOT / "artifacts/phase15a_onset_reanalysis/annotation_rules.json"
RAW_PATH = (
    REPO_ROOT
    / "artifacts/runs"
    / PHASE14_RUN_ID
    / "calibration_continuations.jsonl"
)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return float(obj)


def _fmt_ci(d: dict[str, float], digits: int = 1) -> str:
    return (
        f"{d['median']:.{digits}f} "
        f"[{d['ci95_low']:.{digits}f}, {d['ci95_high']:.{digits}f}]"
    )


def write_report(summary: dict, path: Path) -> None:
    het = summary["heterogeneity"]
    prim = summary["primary_forecast"]
    opt = summary["optimistic_forecast"]
    pen = summary["heterogeneity_penalty"]
    interp = summary["interpretation"]
    lines = [
        "# Phase 16A — same-prompt sampling-scaling design forecast",
        "",
        f"**Source run:** `{summary['phase14_run_id']}`  ",
        f"**Raw SHA256:** `{summary['raw_continuations_sha256']}`  ",
        f"**Phase-15 rule:** `{summary['phase15_rule_version']}` / "
        f"`{summary['phase15_rule_hash']}`  ",
        f"**Design hash:** `{summary['design_hash']}`  ",
        f"**Monte Carlo:** seed `{summary['monte_carlo_seed']}`, "
        f"{summary['n_replicates']} replicates  ",
        f"**Status:** `{summary['status']}`  ",
        f"**Outcome:** `{interp['outcome']}`  ",
        "",
        "Phase 14A and Phase 15A remain HOLDs. No language-model calls. "
        "Primary forecast preserves prompt heterogeneity via bootstrap + "
        "Jeffreys posteriors. Optimistic pooled forecast is descriptive only.",
        "",
        "## Empirical prompt heterogeneity",
        "",
        "| T | Zero-alt | 1–2 alt | ≥3 alt | Alt-rate median | IQR | Min | Max |",
        "| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for t in TEMPERATURE_GRID:
        h = het[str(t)]
        iqr = h["empirical_alternate_rate_iqr"]
        lines.append(
            f"| {t} | {h['zero_alt_groups']} | {h['one_to_two_alt_groups']} | "
            f"{h['ge3_alt_groups']} | {h['empirical_alternate_rate_median']:.3f} | "
            f"[{iqr[0]:.3f}, {iqr[1]:.3f}] | "
            f"{h['empirical_alternate_rate_min']:.3f} | "
            f"{h['empirical_alternate_rate_max']:.3f} |"
        )
    lines += [
        "",
        "## Primary T×N forecast (median [95% CI])",
        "",
        "| T | N | ≥1/class | ≥3/class | ≥4/class | ≥6/class | ≥8/class | "
        "Valid rec | Valid alt | Valid frac | Alt frac |",
        "| ---: | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for t in TEMPERATURE_GRID:
        for n in N_GRID:
            c = prim[str(t)][str(n)]
            lines.append(
                f"| {t} | {n} | {_fmt_ci(c['groups_ge1_each_class'])} | "
                f"{_fmt_ci(c['groups_ge3_each_class'])} | "
                f"{_fmt_ci(c['groups_ge4_each_class'])} | "
                f"{_fmt_ci(c['groups_ge6_each_class'])} | "
                f"{_fmt_ci(c['groups_ge8_each_class'])} | "
                f"{_fmt_ci(c['total_valid_record'], 0)} | "
                f"{_fmt_ci(c['total_valid_alternate'], 0)} | "
                f"{_fmt_ci(c['valid_fraction'], 3)} | "
                f"{_fmt_ci(c['alternate_fraction_among_valid'], 3)} |"
            )
    lines += [
        "",
        "## ≥4/class probabilities",
        "",
        "| T | N | P(≥16/32) | P(≥20/32) |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for t in TEMPERATURE_GRID:
        for n in N_GRID:
            c = prim[str(t)][str(n)]
            lines.append(
                f"| {t} | {n} | {c['prob_groups_ge4_ge16']:.4f} | "
                f"{c['prob_groups_ge4_ge20']:.4f} |"
            )
    lines += [
        "",
        "## ≥6/class probabilities",
        "",
        "| T | N | P(≥16/32) | P(≥20/32) |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for t in TEMPERATURE_GRID:
        for n in N_GRID:
            c = prim[str(t)][str(n)]
            lines.append(
                f"| {t} | {n} | {c['prob_groups_ge6_ge16']:.4f} | "
                f"{c['prob_groups_ge6_ge20']:.4f} |"
            )
    lines += [
        "",
        "## Optimistic pooled forecast (not primary)",
        "",
        "| T | N | ≥4/class median | P(≥20/32 ge4) | Alt frac median |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for t in TEMPERATURE_GRID:
        for n in N_GRID:
            c = opt[str(t)][str(n)]
            lines.append(
                f"| {t} | {n} | {c['groups_ge4_each_class']['median']:.1f} | "
                f"{c['prob_groups_ge4_ge20']:.4f} | "
                f"{c['alternate_fraction_among_valid']['median']:.3f} |"
            )
    lines += [
        "",
        "## Heterogeneity penalty (optimistic − primary)",
        "",
        "| T | N | Δ P(ge4≥20) | Δ median ge4 groups |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for t in TEMPERATURE_GRID:
        for n in N_GRID:
            p = pen[str(t)][str(n)]
            lines.append(
                f"| {t} | {n} | {p['delta_prob_ge4_ge20']:+.4f} | "
                f"{p['delta_median_groups_ge4']:+.1f} |"
            )
    rec = interp.get("recommended_prospective_t_n")
    lines += [
        "",
        f"**Frozen interpretation:** `{interp['outcome']}`  ",
        f"**Recommended prospective T/N:** "
        f"{'none (not plausible)' if rec is None else rec}  ",
        "",
        GUARANTEE,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "artifacts/phase16a_scaling_forecast"))
    ap.add_argument("--report", default=str(REPO_ROOT / "reports/phase16a_scaling_forecast.md"))
    args = ap.parse_args()

    raw_sha = hashlib.sha256(RAW_PATH.read_bytes()).hexdigest()
    if raw_sha != PHASE14_RAW_CONTINUATIONS_SHA256:
        raise SystemExit(f"raw SHA mismatch: {raw_sha}")
    p15 = json.loads(P15_SUMMARY.read_text("utf-8"))
    rules = json.loads(P15_RULES.read_text("utf-8"))
    if p15["rule_hash"] != PHASE15_RULE_HASH or rules["rule_hash"] != PHASE15_RULE_HASH:
        raise SystemExit("Phase-15 rule hash mismatch")
    if p15["rule_version"] != PHASE15_RULE_VERSION:
        raise SystemExit("Phase-15 rule version mismatch")
    if p15["raw_continuations_sha256"] != raw_sha:
        raise SystemExit("Phase-15 summary raw SHA mismatch")
    if len(p15["annotated_rows"]) != 1536:
        raise SystemExit("expected 1536 annotated rows")

    # Freeze design before computing results.
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = design_manifest()
    if manifest["design_hash"] != DESIGN_HASH:
        raise SystemExit("design hash drift")
    write_json(out / "design_manifest.json", manifest)

    counts = empirical_prompt_counts(p15["annotated_rows"])
    write_json(out / "empirical_prompt_counts.json", _jsonable(counts))
    het = heterogeneity_diagnostic(counts)

    primary = run_primary_forecast(counts)
    optimistic = run_optimistic_forecast(counts)
    interp = interpret_forecast(primary)
    penalty = heterogeneity_penalty(primary, optimistic)

    summary = {
        "created_at": utc_now_iso(),
        "phase": "phase16a",
        "phase14_run_id": PHASE14_RUN_ID,
        "raw_continuations_sha256": raw_sha,
        "phase15_rule_version": PHASE15_RULE_VERSION,
        "phase15_rule_hash": PHASE15_RULE_HASH,
        "phase14_status_unchanged": "phase14a_same_prompt_trajectory_calibration_hold",
        "phase15_status_unchanged": "phase15a_same_prompt_onset_reanalysis_hold",
        "design_hash": DESIGN_HASH,
        "monte_carlo_seed": MONTE_CARLO_SEED,
        "n_replicates": N_REPLICATES,
        "n_grid": list(N_GRID),
        "temperature_grid": list(TEMPERATURE_GRID),
        "heterogeneity": het,
        "primary_forecast": _jsonable(primary),
        "optimistic_forecast": _jsonable(optimistic),
        "heterogeneity_penalty": _jsonable(penalty),
        "interpretation": interp,
        "status": interp["status"],
        "model_calls": 0,
        "final_model_calls": 0,
        "locked_model_calls": 0,
        "activations_collected": False,
        "resampling_from_model": False,
        "guarantee": GUARANTEE,
    }
    write_json(out / "forecast_summary.json", summary)
    write_json(
        out / "forecast_manifest.json",
        {
            "design_hash": DESIGN_HASH,
            "status": interp["status"],
            "outcome": interp["outcome"],
            "monte_carlo_seed": MONTE_CARLO_SEED,
            "n_replicates": N_REPLICATES,
            "model_calls": 0,
        },
    )
    write_report(summary, Path(args.report))
    print(
        json.dumps(
            {
                "status": interp["status"],
                "outcome": interp["outcome"],
                "design_hash": DESIGN_HASH,
                "recommended_prospective_t_n": interp["recommended_prospective_t_n"],
                "prob_ge4_ge20_at_N128": {
                    str(t): primary[str(t)][128]["prob_groups_ge4_ge20"]
                    for t in TEMPERATURE_GRID
                },
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
