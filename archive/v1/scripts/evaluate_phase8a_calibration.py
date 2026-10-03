#!/usr/bin/env python3
"""Evaluate Phase 8A calibration choices, apply the frozen W* rule, instantiate final prompts.

No model calls. Final prompts are written (gitignored) and hashed only if W* exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from prepare_phase8_design import final_prompts, sha_prompt_texts, sha_scenarios  # noqa: E402

from pre_output_physiology.phase8_design import (  # noqa: E402
    CALIBRATION_FAMILIES,
    LABEL_GOAL,
    LABEL_TRUTHFUL,
    N_CALIBRATION_EVALS,
    N_FINAL_BASES,
    W_GRID,
    candidate_order,
    rates_by_w,
    select_w_star,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

DATA = REPO_ROOT / "data/processed/phase8_design"
GUARANTEE = (
    "PHASE 8A USED CALIBRATION-ONLY SCENARIOS TO SELECT A SINGLE GLOBAL TRADEOFF WEIGHT BY "
    "A RULE FROZEN BEFORE MODEL CALLS. NO FINAL PHASE-8 SCENARIO WAS RUN THROUGH THE MODEL. "
    "NO ACTIVATIONS WERE COLLECTED. NO PHYSIOLOGY PROBE WAS FIT OR SCORED. THE PHASE-5 PROBE "
    "WAS NOT SCORED. THE CALIBRATION GRID WAS NOT EXTENDED AFTER RESULTS. NO CAUSAL "
    "INTERVENTIONS WERE PERFORMED."
)


def label_row(r: dict) -> dict:
    ids = r["candidate_ids_visible_order"]
    c1, c2 = candidate_order(r)
    by_state = {c1: ids[0], c2: ids[1]}
    rec, alt = by_state[r["record_state"]], by_state[r["alternate_state"]]
    if r["generated_ids"] != by_state[r["chosen_state"]]:
        raise SystemExit(f"generated ids do not match chosen candidate: {r['example_id']}")
    if r["chosen_state"] == r["record_state"]:
        label = LABEL_TRUTHFUL
    elif r["chosen_state"] == r["alternate_state"]:
        label = LABEL_GOAL
    else:
        raise SystemExit("chosen state outside candidates")
    div = next(i for i in range(min(len(rec), len(alt))) if rec[i] != alt[i])
    step = next(s for s in r["steps"] if s["position"] == div)
    lg = {int(k): v for k, v in step["allowed_logits"].items()}
    return {
        "example_id": r["example_id"],
        "base_scenario_id": r["base_scenario_id"],
        "family": r["family"],
        "w": r["w"],
        "record_listed_first": r["record_listed_first"],
        "chosen_state": r["chosen_state"],
        "label": label,
        "first_divergent_position": div,
        "record_minus_alternate_logit": lg[rec[div]] - lg[alt[div]],
    }


def write_report(summary: dict, path: Path) -> None:
    sel = summary["selection"]
    rates = summary["rates_by_w"]
    fams = list(CALIBRATION_FAMILIES)
    lines = [
        "# Phase 8A — policy-frontier calibration (constrained forced choice)",
        "",
        f"**Run ID:** `{summary['run_id']}`  ",
        f"**Git SHA:** `{summary['git_commit']}`  ",
        f"**Status:** `{summary['status']}`  ",
        "",
        "Forced-choice policy assay: formatted prompt + `Response` (12107), then greedy decoding "
        "restricted to the two candidate state strings. Not unrestricted generation. "
        "Goal-favored rate = fraction choosing the alternate state.",
        "",
        "| W | Overall goal-favored | "
        + " | ".join(fams)
        + " | Families in [0.2, 0.8] | Eligible |",
        "| ---: | ---: | " + " | ".join("---:" for _ in fams) + " | ---: | --- |",
    ]
    for t in sel["table"]:
        w = t["w"]
        cells = [f"{rates[str(w)]['by_family'][f]['goal_favored_rate']:.3f}" for f in fams]
        lines.append(
            f"| {w} | {t['overall_rate']:.3f} | "
            + " | ".join(cells)
            + f" | {t['n_families_in_bounds']} | {t['eligible']} |"
        )
    lines += [
        "",
        f"**Eligible W:** {sel['eligible_w']}  ",
        f"**Selected W\\*:** {sel['w_star']}  ",
        f"**Tie-break path:** {sel['tie_break_path']}  ",
    ]
    if summary.get("final_prompt_text_sha256"):
        lines.append(
            "**Final prompt SHA256 (960 prompts at W\\*):** "
            f"`{summary['final_prompt_text_sha256']}`"
        )
    lines += ["", GUARANTEE, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "artifacts/phase8a_calibration"))
    ap.add_argument("--report", default=str(REPO_ROOT / "reports/phase8a_calibration.md"))
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    rows = [
        json.loads(x)
        for x in (run_dir / "calibration_choices.jsonl").read_text("utf-8").splitlines()
        if x.strip()
    ]
    if len(rows) != N_CALIBRATION_EVALS or any(
        not r["example_id"].startswith("calib_") for r in rows
    ):
        raise SystemExit("calibration rows invalid")
    manifest = json.loads((run_dir / "calibration_manifest.json").read_text("utf-8"))
    labeled = [label_row(r) for r in rows]
    rates = rates_by_w(labeled)
    sel = select_w_star(rates)
    summary = {
        "created_at": utc_now_iso(),
        "run_id": manifest["run_id"],
        "git_commit": manifest["git_commit"],
        "w_grid": list(W_GRID),
        "rates_by_w": {str(k): v for k, v in rates.items()},
        "selection": sel,
        "status": "phase8a_policy_frontier_frozen_awaiting_audit"
        if sel["w_star"] is not None
        else "phase8a_policy_frontier_hold",
        "final_model_calls": 0,
    }
    if sel["w_star"] is not None:
        final = [
            json.loads(x)
            for x in (DATA / "final_base_scenarios.jsonl").read_text("utf-8").splitlines()
            if x.strip()
        ]
        summary["final_scenario_text_sha256_verified"] = sha_scenarios(final)
        fp = final_prompts(final, sel["w_star"])
        if len(fp) != N_FINAL_BASES:
            raise SystemExit("final prompt count drift")
        with (DATA / f"final_policy_prompts_w{sel['w_star']}.jsonl").open(
            "w", encoding="utf-8"
        ) as fh:
            for p in fp:
                fh.write(json.dumps(p, sort_keys=True) + "\n")
        summary["final_prompt_text_sha256"] = sha_prompt_texts(fp)
        summary["n_final_prompts"] = len(fp)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "calibration_manifest.json", manifest)
    write_json(out / "calibration_summary.json", {**summary, "labeled_rows": labeled})
    write_report(summary, Path(args.report))
    print(
        json.dumps(
            {
                "overall": {w: round(v["goal_favored_rate"], 4) for w, v in rates.items()},
                "by_family": {
                    w: {f: round(x["goal_favored_rate"], 3) for f, x in v["by_family"].items()}
                    for w, v in rates.items()
                },
                "eligible_w": sel["eligible_w"],
                "w_star": sel["w_star"],
                "tie_break_path": sel["tie_break_path"],
                "final_prompt_text_sha256": summary.get("final_prompt_text_sha256"),
                "status": summary["status"],
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
