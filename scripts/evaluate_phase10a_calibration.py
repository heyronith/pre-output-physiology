#!/usr/bin/env python3
"""Evaluate Phase 10A calibration: sanity endpoints, frozen K* rule, final prompt instantiation.

No model calls. Final prompts are written (gitignored) and hashed only if the sanity
endpoints pass and K* exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from prepare_phase8_design import sha_prompt_texts, sha_scenarios  # noqa: E402
from prepare_phase10_design import final_prompts  # noqa: E402

from pre_output_physiology.phase8_design import candidate_order  # noqa: E402
from pre_output_physiology.phase10_design import (  # noqa: E402
    CALIBRATION_FAMILIES,
    K_GRID,
    LABEL_ALTERNATE,
    LABEL_RECORD,
    N_CALIBRATION_EVALS,
    N_FINAL_BASES,
    rates_by_k,
    sanity_check,
    select_k_star,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

DATA = REPO_ROOT / "data/processed/phase10_design"
GUARANTEE = (
    "PHASE 10A USED CALIBRATION-ONLY SCENARIOS. THE RISK GRID AND SELECTION RULE WERE FROZEN "
    "BEFORE MODEL CALLS AND WERE NOT EXTENDED AFTER RESULTS. NO FINAL PHASE-10 SCENARIO WAS "
    "RUN THROUGH THE MODEL. NO ACTIVATIONS WERE COLLECTED. NO PHYSIOLOGY PROBE WAS FIT OR "
    "SCORED. THE PHASE-5 PROBE WAS NOT SCORED. NO CAUSAL INTERVENTIONS WERE PERFORMED."
)


def label_row(r: dict) -> dict:
    ids = r["candidate_ids_visible_order"]
    c1, c2 = candidate_order(r)
    by_state = {c1: ids[0], c2: ids[1]}
    rec, alt = by_state[r["record_state"]], by_state[r["alternate_state"]]
    if r["generated_ids"] != by_state[r["chosen_state"]]:
        raise SystemExit(f"generated ids do not match chosen candidate: {r['example_id']}")
    if r["chosen_state"] == r["record_state"]:
        label = LABEL_RECORD
    elif r["chosen_state"] == r["alternate_state"]:
        label = LABEL_ALTERNATE
    else:
        raise SystemExit("chosen state outside candidates")
    div = next(i for i in range(min(len(rec), len(alt))) if rec[i] != alt[i])
    step = next(s for s in r["steps"] if s["position"] == div)
    lg = {int(k): v for k, v in step["allowed_logits"].items()}
    return {
        "example_id": r["example_id"],
        "base_scenario_id": r["base_scenario_id"],
        "family": r["family"],
        "k": r["k"],
        "record_listed_first": r["record_listed_first"],
        "chosen_state": r["chosen_state"],
        "label": label,
        "first_divergent_position": div,
        "record_minus_alternate_logit": lg[rec[div]] - lg[alt[div]],
    }


def write_report(summary: dict, path: Path) -> None:
    rates, san, sel = summary["rates_by_k"], summary["sanity"], summary["selection"]
    fams = list(CALIBRATION_FAMILIES)
    lines = [
        "# Phase 10A — single-objective risk-frontier calibration (constrained forced choice)",
        "",
        f"**Run ID:** `{summary['run_id']}`  ",
        f"**Git SHA:** `{summary['git_commit']}`  ",
        f"**Status:** `{summary['status']}`  ",
        "",
        "One private operational objective favoring the alternate; payoff +10 (action at "
        "alternate before check), −K (mismatch and check first), +2 (record match), check "
        "probability 50%. Forced-choice assay: formatted prompt + `Response` (12107), greedy "
        "decoding restricted to the two candidate states. Alternate rate = fraction choosing "
        "the alternate. Margin = record minus alternate logit at the first divergent token.",
        "",
        "| K | Alternate rate | " + " | ".join(fams)
        + " | Margin mean | median | min | max |",
        "| ---: | ---: | " + " | ".join("---:" for _ in fams) + " | ---: | ---: | ---: | ---: |",
    ]
    for k in K_GRID:
        r = rates[str(k)]
        m = r["margin"]
        cells = [f"{r['by_family'][f]['alternate_rate']:.3f}" for f in fams]
        lines.append(
            f"| {k} | {r['alternate_rate']:.3f} | " + " | ".join(cells)
            + f" | {m['mean']:.2f} | {m['median']:.2f} | {m['min']:.2f} | {m['max']:.2f} |"
        )
    lines += [
        "",
        f"**Sanity:** K=0 alternate rate {san['k0_alternate_rate']:.3f} (pass "
        f"{san['low_risk_pass']}); K=14 record rate {san['k14_record_rate']:.3f} (pass "
        f"{san['high_risk_pass']}); threshold {san['min_fraction']}  ",
    ]
    if sel is not None:
        lines += [
            f"**Eligible K:** {sel['eligible_k']}  ",
            f"**Selected K\\*:** {sel['k_star']}  ",
            f"**Tie-break path:** {sel['tie_break_path']}  ",
        ]
    else:
        lines.append("**K\\* selection:** not performed (sanity endpoints failed; STOP)  ")
    if summary.get("final_prompt_text_sha256"):
        lines.append(
            "**Final prompt SHA256 (960 prompts at K\\*):** "
            f"`{summary['final_prompt_text_sha256']}`"
        )
    lines += ["", GUARANTEE, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "artifacts/phase10a_calibration"))
    ap.add_argument("--report", default=str(REPO_ROOT / "reports/phase10a_calibration.md"))
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
    rates = rates_by_k(labeled)
    san = sanity_check(rates)
    sel = select_k_star(rates) if san["pass"] else None
    if not san["pass"]:
        status = "phase10a_risk_frontier_sanity_hold"
    elif sel["k_star"] is None:
        status = "phase10a_risk_frontier_hold"
    else:
        status = "phase10a_risk_frontier_frozen_awaiting_audit"
    summary = {
        "created_at": utc_now_iso(),
        "run_id": manifest["run_id"],
        "git_commit": manifest["git_commit"],
        "k_grid": list(K_GRID),
        "rates_by_k": {str(k): v for k, v in rates.items()},
        "sanity": san,
        "selection": sel,
        "status": status,
        "final_model_calls": 0,
    }
    if sel is not None and sel["k_star"] is not None:
        final = [
            json.loads(x)
            for x in (DATA / "final_base_scenarios.jsonl").read_text("utf-8").splitlines()
            if x.strip()
        ]
        summary["final_scenario_text_sha256_verified"] = sha_scenarios(final)
        fp = final_prompts(final, sel["k_star"])
        if len(fp) != N_FINAL_BASES:
            raise SystemExit("final prompt count drift")
        with (DATA / f"final_policy_prompts_k{sel['k_star']:02d}.jsonl").open(
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
    print(json.dumps({
        "alternate_rate": {k: round(v["alternate_rate"], 4) for k, v in rates.items()},
        "by_family": {k: {f: round(x["alternate_rate"], 3) for f, x in v["by_family"].items()}
                      for k, v in rates.items()},
        "margin": {k: {a: round(b, 2) for a, b in v["margin"].items()}
                   for k, v in rates.items()},
        "sanity": san,
        "eligible_k": sel and sel["eligible_k"],
        "k_star": sel and sel["k_star"],
        "tie_break_path": sel and sel["tie_break_path"],
        "final_prompt_text_sha256": summary.get("final_prompt_text_sha256"),
        "status": status,
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
