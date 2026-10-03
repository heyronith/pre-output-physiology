#!/usr/bin/env python3
"""Evaluate Phase 14A trajectory calibration and select T* (no model calls)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase14_design import (  # noqa: E402
    LABEL_ALTERNATE,
    LABEL_INVALID,
    LABEL_LEAK,
    LABEL_RECORD,
    N_CALIBRATION_CONTINUATIONS,
    TEMPERATURE_GRID,
    label_continuation,
    rates_by_temperature,
    select_temperature,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

GUARANTEE = (
    "PHASE 14A CALIBRATED STOCHASTIC SAME-PROMPT TRAJECTORIES ONLY. EXACT-PROMPT K0 "
    "ACTIVATIONS ARE DEFINED AS A FUTURE NEGATIVE CONTROL AND CANNOT PREDICT RANDOM "
    "CONTINUATION IDENTITY. NO ACTIVATIONS WERE COLLECTED. NO PHYSIOLOGY PROBE WAS FIT "
    "OR SCORED. NO FINAL OR LOCKED PHASE-14 PROMPTS WERE RUN THROUGH THE MODEL. THE "
    "SAMPLING GRID WAS NOT EXTENDED. NO POST-RESULT PROMPT TUNING OR CAUSAL "
    "INTERVENTIONS WERE PERFORMED."
)


def label_row(r: dict) -> dict:
    if r["response_token_id"] != 12107:
        raise SystemExit(f"Response token not 12107: {r['continuation_id']}")
    if r["constrained_generated_ids"] != r["candidate_ids_visible_order"][
        r["candidate_states_visible_order"].index(r["chosen_state"])
    ]:
        # chosen_state must match one of the visible candidates' generated ids
        vis = r["candidate_states_visible_order"]
        ids = r["candidate_ids_visible_order"]
        by = dict(zip(vis, ids, strict=True))
        if r["constrained_generated_ids"] != by[r["chosen_state"]]:
            raise SystemExit(f"completion integrity failed: {r['continuation_id']}")
    stage1_ok = bool(r["consideration_text"]) and bool(r["stage1_n_tokens"])
    lab = label_continuation(
        consideration_text=r["consideration_text"],
        chosen_state=r["chosen_state"],
        record=r["record_state"],
        alternate=r["alternate_state"],
        stage1_ok_format=stage1_ok,
    )
    return {
        "continuation_id": r["continuation_id"],
        "prompt_group_id": r["prompt_group_id"],
        "family": r["family"],
        "temperature": r["temperature"],
        "sample_index": r["sample_index"],
        "sample_seed": r["sample_seed"],
        "consideration_text": r["consideration_text"],
        "chosen_state": r["chosen_state"],
        "label": lab["label"],
        "leak": lab["leak"],
        "valid_nonleaking": lab["valid_nonleaking"],
        "stage1_reasons": lab["stage1"]["reasons"],
        "stage1_n_tokens": r["stage1_n_tokens"],
        "stage1_hit_stop": r["stage1_hit_stop"],
    }


def write_report(summary: dict, path: Path) -> None:
    rates = summary["rates_by_temperature"]
    sel = summary["selection"]
    lines = [
        "# Phase 14A — same-prompt stochastic trajectory calibration",
        "",
        f"**Run ID:** `{summary['run_id']}`  ",
        f"**Git SHA:** `{summary['git_commit']}`  ",
        f"**Status:** `{summary['status']}`  ",
        "",
        "Phase-10 single-objective payoff at K=10. Stage 1: stochastic `Consideration:` "
        "(stop at `.`/newline, max 32). Stage 2: forced `\\n Response` (12107) + "
        "constrained greedy choice. k0 is a future negative control only.",
        "",
        "| T | n | Valid frac | Leak rate | Record frac | Alternate frac | "
        "Groups ≥3/class | Median valid/prompt | Eligible |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    elig = set(sel["eligible_temperatures"])
    for t in TEMPERATURE_GRID:
        r = rates[str(t)]
        lines.append(
            f"| {t} | {r['n']} | {r['valid_fraction']:.3f} | {r['leak_rate']:.3f} | "
            f"{r['record_fraction']:.3f} | {r['alternate_fraction']:.3f} | "
            f"{r['n_prompt_groups_ge3_each_class']}/32 | "
            f"{r['median_valid_continuations_per_prompt']:.1f} | "
            f"{t in elig} |"
        )
    lines += [
        "",
        f"**Eligible temperatures:** {sel['eligible_temperatures']}  ",
        f"**Selected T\\*:** {sel['t_star']}  ",
        f"**Tie-break path:** {sel['tie_break_path']}  ",
        f"**Final model calls:** {summary['final_model_calls']}  ",
        f"**Locked model calls:** {summary['locked_model_calls']}  ",
        "",
        GUARANTEE,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "artifacts/phase14a_calibration"))
    ap.add_argument("--report", default=str(REPO_ROOT / "reports/phase14a_calibration.md"))
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    rows = [
        json.loads(x)
        for x in (run_dir / "calibration_continuations.jsonl")
        .read_text("utf-8")
        .splitlines()
        if x.strip()
    ]
    if len(rows) != N_CALIBRATION_CONTINUATIONS:
        raise SystemExit("continuation count invalid")
    if any(not r["prompt_group_id"].startswith("calib_") for r in rows):
        raise SystemExit("non-calibration prompt in run")
    manifest = json.loads((run_dir / "calibration_manifest.json").read_text("utf-8"))
    labeled = [label_row(r) for r in rows]
    rates = rates_by_temperature(labeled)
    sel = select_temperature(rates)
    status = (
        "phase14a_same_prompt_trajectory_calibration_pass_awaiting_audit"
        if sel["t_star"] is not None
        else "phase14a_same_prompt_trajectory_calibration_hold"
    )
    label_counts = {
        str(t): {
            lab: sum(
                1
                for r in labeled
                if float(r["temperature"]) == float(t) and r["label"] == lab
            )
            for lab in (LABEL_RECORD, LABEL_ALTERNATE, LABEL_LEAK, LABEL_INVALID)
        }
        for t in TEMPERATURE_GRID
    }
    summary = {
        "created_at": utc_now_iso(),
        "run_id": manifest["run_id"],
        "git_commit": manifest["git_commit"],
        "temperature_grid": list(TEMPERATURE_GRID),
        "n_continuations": len(labeled),
        "label_counts_by_temperature": label_counts,
        "rates_by_temperature": {str(k): v for k, v in rates.items()},
        "selection": sel,
        "status": status,
        "final_model_calls": 0,
        "locked_model_calls": 0,
        "response_token_id_verified": all(r["response_token_id"] == 12107 for r in rows),
        "k0_theorem": (
            "Same model + exact same prompt/context = identical k0 activations. "
            "k0 cannot predict which random continuation will subsequently be sampled."
        ),
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "calibration_manifest.json", manifest)
    write_json(out / "calibration_summary.json", {**summary, "labeled_rows": labeled})
    write_report(summary, Path(args.report))
    print(
        json.dumps(
            {
                "label_counts_by_temperature": label_counts,
                "rates": {
                    str(t): {
                        k: rates[t][k]
                        for k in (
                            "valid_fraction",
                            "leak_rate",
                            "record_fraction",
                            "alternate_fraction",
                            "n_prompt_groups_ge3_each_class",
                            "median_valid_continuations_per_prompt",
                        )
                    }
                    for t in TEMPERATURE_GRID
                },
                "eligible_temperatures": sel["eligible_temperatures"],
                "t_star": sel["t_star"],
                "tie_break_path": sel["tie_break_path"],
                "status": status,
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
