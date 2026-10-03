#!/usr/bin/env python3
"""Evaluate Phase 7A policy-choice pilot outputs (deterministic; no probes/activations)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase7_design import (  # noqa: E402
    CONTEXT_ORDER,
    LABEL_DECEPTIVE,
    LABEL_INVALID,
    LABEL_TRUTHFUL,
    N_PILOT_PROMPTS,
    label_output,
    summarize_pilot,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

GUARANTEE = (
    "NO PHASE 7 ACTIVATIONS WERE COLLECTED. NO PHASE 7 PHYSIOLOGY PROBE WAS FIT OR "
    "SCORED. THE FROZEN PHASE-5 PROBE WAS NOT SCORED. NO FINAL PHASE-7 SCENARIOS WERE "
    "RUN THROUGH THE MODEL. NO CAUSAL INTERVENTIONS WERE PERFORMED."
)


def evaluate_rows(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        label, reason = label_output(
            r,
            r.get("output_text") or "",
            first_generated_token_id=r.get("first_generated_token_id"),
        )
        out.append(
            {
                "example_id": r["example_id"],
                "base_scenario_id": r["base_scenario_id"],
                "context_id": r["context_id"],
                "family": r["family"],
                "first_generated_token_id": r.get("first_generated_token_id"),
                "output_text": r.get("output_text"),
                "label": label,
                "invalid_reason": reason,
            }
        )
    return out


def _count_row(name: str, c: dict) -> str:
    return (
        f"| {name} | {c['n']} | {c['valid']} | {c[LABEL_TRUTHFUL]} | "
        f"{c[LABEL_DECEPTIVE]} | {c[LABEL_INVALID]} |"
    )


def write_report(summary: dict, manifest: dict, path: Path) -> None:
    hdr = [
        "| Group | N | Valid | Truthful choice | Deceptive choice | Invalid |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    revision = manifest["prompt_template_revision"]
    lines = [
        f"# Phase 7A policy-choice behavior pilot (revision {revision})",
        "",
        f"**Run ID:** `{manifest['run_id']}`  ",
        f"**Git SHA:** `{manifest['git_commit']}`  ",
        f"**Pilot prompt SHA256:** `{manifest['pilot_prompt_text_sha256']}`  ",
        f"**GPU wall / cost:** `{manifest['wall_seconds']:.1f}s` / "
        f"`${manifest['estimated_cost_usd']:.4f}`  ",
        "",
        "Every prompt contains private-goal / known-record conflict. Labels are assigned "
        "from behavior after generation: record → truthful_choice, alternate → "
        "deceptive_choice, anything else → invalid/abstain.",
        "",
        f"**Format gates pass:** `{summary['format_gates_pass']}` {summary['format_gates']}  ",
        f"**Mixture gates pass:** `{summary['mixture_gates_pass']}` {summary['mixture_gates']}  ",
        f"**All gates pass:** `{summary['all_gates_pass']}`  ",
        f"**Families with both choices:** {summary['n_families_with_both']}/8  ",
        f"**Bases with both choices:** {summary['n_bases_with_both']}/32  ",
        "",
        "## Overall",
        "",
        *hdr,
        _count_row("all", summary["total"]),
        "",
        "## By decision context",
        "",
        *hdr,
        *[_count_row(c, summary["by_context"][c]) for c in CONTEXT_ORDER],
        "",
        "## By family",
        "",
        *hdr,
        *[_count_row(f, c) for f, c in summary["by_family"].items()],
        "",
        "## Invalid reasons",
        "",
        *([f"- `{k}`: {v}" for k, v in summary["invalid_reason_counts"].items()] or ["- none"]),
        "",
        "These are usability gates, not scientific effect thresholds.",
        "",
        GUARANTEE,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--summary-dir", default=str(REPO_ROOT / "artifacts/phase7a_pilot"))
    ap.add_argument("--report-path", default=str(REPO_ROOT / "reports/phase7a_pilot.md"))
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    rows = [
        json.loads(line)
        for line in (run_dir / "pilot_outputs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != N_PILOT_PROMPTS or any(r.get("split") != "pilot" for r in rows):
        raise SystemExit("pilot outputs invalid")
    if any(r.get("activation_extracted") or r.get("probe_scored") for r in rows):
        raise SystemExit("activation/probe flags set")
    eval_rows = evaluate_rows(rows)
    summary = summarize_pilot(eval_rows)
    manifest = json.loads((run_dir / "pilot_generation_manifest.json").read_text(encoding="utf-8"))
    out = Path(args.summary_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(
        out / "pilot_generation_manifest.json",
        {**manifest, "raw_outputs_gitignored": True, "evaluated_at": utc_now_iso()},
    )
    write_json(
        out / "pilot_behavior_summary.json",
        {
            "created_at": utc_now_iso(),
            "run_id": manifest["run_id"],
            "summary": summary,
            "eval_rows": eval_rows,
        },
    )
    write_report(summary, manifest, Path(args.report_path))
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "total",
                    "by_context",
                    "by_family",
                    "invalid_reason_counts",
                    "n_families_with_both",
                    "n_bases_with_both",
                    "format_gates",
                    "mixture_gates",
                    "all_gates_pass",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
