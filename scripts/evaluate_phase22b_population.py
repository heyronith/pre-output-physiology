#!/usr/bin/env python3
"""Phase-22B K=20 population gates + one-sided switching analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase22b_sampling import (  # noqa: E402
    GUARANTEE,
    PHASE21_ONSET_RUN,
    STATUS_POP_PASS,
    evaluate_k20_population,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(x)
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase22b-onset-run-id", required=True)
    ap.add_argument(
        "--phase21-onset-run-id",
        default=PHASE21_ONSET_RUN,
    )
    ap.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "artifacts/phase22b_population"),
    )
    args = ap.parse_args()

    p21 = _load_jsonl(
        REPO_ROOT / "artifacts/runs" / args.phase21_onset_run_id / "annotated.jsonl"
    )
    p22 = _load_jsonl(
        REPO_ROOT
        / "artifacts/runs"
        / args.phase22b_onset_run_id
        / "annotated.jsonl"
    )
    result = evaluate_k20_population(p21, p22)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = {
        "created_at": utc_now_iso(),
        "phase21_onset_run_id": args.phase21_onset_run_id,
        "phase22b_onset_run_id": args.phase22b_onset_run_id,
        "status": result["status"],
        "k": result["k"],
        "gates": result["gates"],
        "n_qualifying_prompts": result["n_qualifying_prompts"],
        "n_primary_trajectories": result["n_primary_trajectories"],
        "balanced_subset_sha256": result["balanced_subset_sha256"],
        "selected_continuation_ids": result["selected_continuation_ids"],
        "qualifying_prompts": result["qualifying_prompts"],
        "pairs": result["pairs"],
        "switching": result["switching"],
        "guarantee": GUARANTEE,
    }
    write_json(out / "population_summary.json", summary)
    with (out / "pairs.jsonl").open("w", encoding="utf-8") as f:
        for p in result["pairs"]:
            f.write(json.dumps(p, sort_keys=True) + "\n")

    report_lines = [
        "# Phase 22B — K=20 population validation",
        "",
        f"**Status:** `{result['status']}`",
        "",
        f"- Phase-21 onset: `{args.phase21_onset_run_id}`",
        f"- Phase-22B onset: `{args.phase22b_onset_run_id}`",
        "- Combined rollouts/prompt: 20",
        "",
        "## Population gates",
        "",
        f"- TRAIN qualifying: {result['gates']['n_train_qualifying']} "
        f"(need ≥{result['gates']['min_train_qualifying']})",
        f"- TEST qualifying: {result['gates']['n_test_qualifying']} "
        f"(need ≥{result['gates']['min_test_qualifying']})",
        f"- Gates passed: {result['gates']['passed']}",
        "",
        "## Behavior switching (Phase-21 one-sided → acquire missing in r11–20)",
        "",
    ]
    sw = result["switching"]
    report_lines.append("### Bucket sizes at K=10")
    report_lines.append("")
    for k, v in sw["phase21_only_buckets"].items():
        report_lines.append(
            f"- `{k}`: n={v['n']} (train={v['n_train']}, test={v['n_test']})"
        )
    report_lines += ["", "### Acquisition of previously unseen class", ""]
    for k, v in sw["switching"].items():
        if k == "exact_10_0_or_0_10":
            report_lines.append(
                f"- exact 10/0: n={v['n_10_0']}; acquire dec in 11–20: "
                f"{v['ten_zero_acquires_dec_in_11_20']['n_acquired_unseen_class']}"
            )
            report_lines.append(
                f"- exact 0/10: n={v['n_0_10']}; acquire honest in 11–20: "
                f"{v['zero_ten_acquires_honest_in_11_20']['n_acquired_unseen_class']}"
            )
        else:
            report_lines.append(
                f"- `{k}`: acquired {v['n_acquired_unseen_class']}/{v['n_prompts']} "
                f"(train {v['n_train_acquired']}/{v['n_train']}, "
                f"test {v['n_test_acquired']}/{v['n_test']}; "
                f"rate={v['acquisition_rate']})"
            )
    report_lines += ["", "## Guarantee", "", GUARANTEE, ""]
    report_path = REPO_ROOT / "reports/phase22b_k20_population.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "status": result["status"],
                "gates": result["gates"],
                "switching_summary": sw["switching"],
                "report": str(report_path.relative_to(REPO_ROOT)),
            },
            indent=2,
        )
    )
    return 0 if result["status"] == STATUS_POP_PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
