#!/usr/bin/env python3
"""Evaluate Phase 9A framing x decoder assay diagnostic (frozen rules; no model calls)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase8_design import CONTROLLED_PREFIX_TOKEN_ID  # noqa: E402
from pre_output_physiology.phase9_diagnostic import (  # noqa: E402
    CELLS,
    FAMILIES,
    FRAMINGS,
    N_BASES,
    classify_free_output,
    first_divergent_position,
    interpretation,
    margin_summary,
    paired_bootstrap_mean,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

CHOICES = ("record", "alternate", "other")
GUARANTEE = (
    "PHASE 9A WAS AN ASSAY-DIAGNOSTIC EXPERIMENT ONLY. NO ACTIVATIONS WERE COLLECTED. NO "
    "PHYSIOLOGY PROBE WAS FIT OR SCORED. THE PHASE-5 PROBE WAS NOT SCORED. NO POST-RESULT "
    "PROMPT OR DECODER TUNING WAS PERFORMED. NO CAUSAL INTERVENTIONS WERE PERFORMED."
)


def label_row(r: dict) -> dict:
    vis = r["candidate_states_visible_order"]
    ids = r["candidate_ids_visible_order"]
    by_state = dict(zip(vis, ids, strict=True))
    rec, alt = by_state[r["record_state"]], by_state[r["alternate_state"]]
    if r["constrained_generated_ids"] != by_state[r["constrained_chosen_state"]]:
        raise SystemExit(f"constrained ids mismatch: {r['example_id']}")
    div = first_divergent_position(rec, alt)
    step = next(s for s in r["constrained_steps"] if s["position"] == div)
    lg = {int(k): v for k, v in step["allowed_logits"].items()}
    if set(lg) != {rec[div], alt[div]}:
        raise SystemExit("divergent step allowed set mismatch")
    for s in r["constrained_steps"]:
        if s["position"] < div and len(s["allowed_logits"]) != 1:
            raise SystemExit("shared candidate prefix not forced")
    cc = r["constrained_chosen_state"]
    constrained_choice = "record" if cc == r["record_state"] else "alternate"
    free = classify_free_output(r["free_raw_text"], r["record_state"], r["alternate_state"])
    return {
        "example_id": r["example_id"],
        "base_scenario_id": r["base_scenario_id"],
        "family": r["family"],
        "framing": r["framing"],
        "record_listed_first": r["record_listed_first"],
        "first_divergent_position": div,
        "record_minus_alternate_logit": lg[rec[div]] - lg[alt[div]],
        "constrained_choice": constrained_choice,
        "free_choice": free["semantic_choice"],
        "free_format_valid": free["format_valid"],
        "free_format_failure_reason": free["format_failure_reason"],
        "free_first_line": free["first_line"],
        "free_first_token_id": r["free_first_token_id"],
        "free_raw_text": r["free_raw_text"],
        "natural_first_token_id": r["natural_first_token_id"],
        "natural_first_token_is_12107": r["natural_first_token_id"] == CONTROLLED_PREFIX_TOKEN_ID,
    }


def counts(xs: list[str]) -> dict[str, int]:
    c = Counter(xs)
    return {k: c.get(k, 0) for k in CHOICES}


def crosstab(a: list[str], b: list[str]) -> dict[str, dict[str, int]]:
    return {x: {y: sum(1 for i, j in zip(a, b, strict=True) if (i, j) == (x, y))
                for y in CHOICES} for x in CHOICES}


def summarize(labeled: list[dict]) -> dict:
    by = {f: sorted((x for x in labeled if x["framing"] == f),
                    key=lambda x: x["base_scenario_id"]) for f in FRAMINGS}
    if [x["base_scenario_id"] for x in by["P7"]] != [x["base_scenario_id"] for x in by["P8"]]:
        raise SystemExit("P7/P8 base pairing broken")
    cell_choice = {}
    margins = {}
    for f in FRAMINGS:
        m = [x["record_minus_alternate_logit"] for x in by[f]]
        cell_choice[f"{f}_FREE"] = counts([x["free_choice"] for x in by[f]])
        cell_choice[f"{f}_CONSTRAINED"] = counts([x["constrained_choice"] for x in by[f]])
        margins[f"{f}_FREE"] = margin_summary(m)
        margins[f"{f}_CONSTRAINED"] = margin_summary(m)
    diffs = [
        b["record_minus_alternate_logit"] - a["record_minus_alternate_logit"]
        for a, b in zip(by["P7"], by["P8"], strict=True)
    ]
    shift = paired_bootstrap_mean(diffs)
    agreement = {}
    for f in FRAMINGS:
        cl = [x for x in by[f] if x["free_choice"] != "other"]
        agreement[f] = {
            "n_free_classifiable": len(cl),
            "n_agree": sum(1 for x in cl if x["free_choice"] == x["constrained_choice"]),
            "crosstab_free_by_constrained": crosstab(
                [x["free_choice"] for x in by[f]], [x["constrained_choice"] for x in by[f]]
            ),
        }
    order = {}
    for f in FRAMINGS:
        order[f] = {}
        for rlf in (True, False):
            sub = [x for x in by[f] if x["record_listed_first"] is rlf]
            order[f]["record_listed_first" if rlf else "alternate_listed_first"] = {
                "n": len(sub),
                "margin": margin_summary([x["record_minus_alternate_logit"] for x in sub]),
                "constrained": counts([x["constrained_choice"] for x in sub]),
                "free": counts([x["free_choice"] for x in sub]),
            }
    fmt = {}
    for f in FRAMINGS:
        fmt[f"{f}_FREE"] = {
            "n": len(by[f]),
            "n_format_valid": sum(x["free_format_valid"] for x in by[f]),
            "n_format_invalid": sum(not x["free_format_valid"] for x in by[f]),
            "failure_reasons": dict(Counter(
                x["free_format_failure_reason"] for x in by[f] if not x["free_format_valid"]
            )),
            "n_free_first_token_ids": dict(Counter(str(x["free_first_token_id"]) for x in by[f])),
            "n_natural_first_token_12107": sum(x["natural_first_token_is_12107"] for x in by[f]),
        }
    by_family = {
        f: {
            fam: {
                "margin_mean": margin_summary(
                    [x["record_minus_alternate_logit"] for x in by[f] if x["family"] == fam]
                )["mean"],
                "constrained": counts([x["constrained_choice"] for x in by[f]
                                       if x["family"] == fam]),
                "free": counts([x["free_choice"] for x in by[f] if x["family"] == fam]),
            }
            for fam in FAMILIES
        }
        for f in FRAMINGS
    }
    return {
        "cell_choice_counts": cell_choice,
        "margins_by_cell": margins,
        "margin_note": (
            "The first-divergent-token margin depends only on the prompt, so FREE and "
            "CONSTRAINED cells of a framing share identical margins by construction."
        ),
        "paired_shift_p8_minus_p7": shift,
        "constrained_p7_by_p8": crosstab(
            [x["constrained_choice"] for x in by["P7"]],
            [x["constrained_choice"] for x in by["P8"]],
        ),
        "free_p7_by_p8": crosstab(
            [x["free_choice"] for x in by["P7"]], [x["free_choice"] for x in by["P8"]]
        ),
        "free_vs_constrained_agreement": agreement,
        "candidate_order_sensitivity": order,
        "free_format": fmt,
        "by_family": by_family,
        "interpretation": interpretation(cell_choice, margins, shift),
    }


def _ct_lines(title: str, ct: dict, row: str, col: str) -> list[str]:
    lines = [f"**{title}** (rows {row}, columns {col})", "", "| | " + " | ".join(CHOICES) + " |",
             "| --- | ---: | ---: | ---: |"]
    for x in CHOICES:
        lines.append(f"| {x} | " + " | ".join(str(ct[x][y]) for y in CHOICES) + " |")
    return lines + [""]


def write_report(s: dict, path: Path) -> None:
    lines = [
        "# Phase 9A — policy-flip assay diagnostic (framing x decoder)",
        "",
        f"**Run ID:** `{s['run_id']}`  ",
        f"**Git SHA:** `{s['git_commit']}`  ",
        f"**Status:** `{s['status']}`  ",
        "",
        "Assay diagnostic only; not a deception experiment. P7 = operational-objective-only "
        "framing; P8 = dual-objective framing at W=80. FREE = greedy generation after forced "
        "`Response` (12107); CONSTRAINED = Phase-8 two-candidate decoder. Margin = record minus "
        "alternate logit at the first divergent candidate token. " + s["margin_note"],
        "",
        "| Cell | record | alternate | other | margin mean | median | min | max | frac>0 "
        "| frac<0 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for c in CELLS:
        k, m = s["cell_choice_counts"][c], s["margins_by_cell"][c]
        lines.append(
            f"| {c} | {k['record']} | {k['alternate']} | {k['other']} | {m['mean']:.2f} | "
            f"{m['median']:.2f} | {m['min']:.2f} | {m['max']:.2f} | "
            f"{m['fraction_positive_record_favored']:.3f} | "
            f"{m['fraction_negative_alternate_favored']:.3f} |"
        )
    sh = s["paired_shift_p8_minus_p7"]
    lines += [
        "",
        f"**Paired shift margin(P8) − margin(P7):** mean {sh['mean']:.3f} logits, 95% CI "
        f"[{sh['ci_low']:.3f}, {sh['ci_high']:.3f}] (n={sh['n_scenarios']}, "
        f"{sh['n_bootstrap']} bootstrap, seed {sh['seed']})",
        "",
    ]
    lines += _ct_lines("Constrained choices", s["constrained_p7_by_p8"], "P7", "P8")
    lines += _ct_lines("Free semantic choices", s["free_p7_by_p8"], "P7", "P8")
    for f in FRAMINGS:
        a = s["free_vs_constrained_agreement"][f]
        lines.append(
            f"**{f} free vs constrained agreement:** {a['n_agree']}/"
            f"{a['n_free_classifiable']} classifiable free outputs  "
        )
    lines += ["", "**Candidate-order sensitivity**", "",
              "| Framing | Order | n | margin mean | constrained rec/alt | free rec/alt/other |",
              "| --- | --- | ---: | ---: | --- | --- |"]
    for f in FRAMINGS:
        for o, v in s["candidate_order_sensitivity"][f].items():
            c, fr = v["constrained"], v["free"]
            lines.append(
                f"| {f} | {o} | {v['n']} | {v['margin']['mean']:.2f} | "
                f"{c['record']}/{c['alternate']} | {fr['record']}/{fr['alternate']}/"
                f"{fr['other']} |"
            )
    lines += ["", "**FREE format**", ""]
    for c, v in s["free_format"].items():
        lines.append(
            f"- {c}: valid {v['n_format_valid']}/{v['n']}; failures {v['failure_reasons']}; "
            f"natural first token 12107 in {v['n_natural_first_token_12107']}/{v['n']}"
        )
    it = s["interpretation"]
    lines += [
        "",
        "**Interpretation (frozen rules)**",
        "",
        f"- Inputs: {json.dumps(it['inputs'])}",
        f"- Prompt-framing explanation supported: {it['prompt_framing_explanation_supported']}",
        f"- Decoder explanation supported: {it['decoder_explanation_supported']}",
        f"- Mixed: {it['mixed_explanation']}; neither: {it['neither_supported']}",
        "",
        GUARANTEE,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "artifacts/phase9a_diagnostic"))
    ap.add_argument("--report", default=str(REPO_ROOT / "reports/phase9a_diagnostic.md"))
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    rows = [json.loads(x) for x in (run_dir / "diagnostic_outputs.jsonl").read_text(
        "utf-8").splitlines() if x.strip()]
    if len(rows) != 2 * N_BASES:
        raise SystemExit("row count invalid")
    manifest = json.loads((run_dir / "diagnostic_manifest.json").read_text("utf-8"))
    labeled = [label_row(r) for r in rows]
    summary = {
        "created_at": utc_now_iso(),
        "run_id": manifest["run_id"],
        "git_commit": manifest["git_commit"],
        "n_evaluations": 2 * len(labeled),
        "status": "phase9a_policy_flip_diagnostic_complete_awaiting_audit",
        **summarize(labeled),
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "diagnostic_manifest.json", manifest)
    write_json(out / "diagnostic_summary.json", summary)
    write_json(out / "labeled_rows.json", labeled)
    write_report(summary, Path(args.report))
    print(json.dumps({k: summary[k] for k in (
        "cell_choice_counts", "margins_by_cell", "paired_shift_p8_minus_p7",
        "free_format", "interpretation")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
