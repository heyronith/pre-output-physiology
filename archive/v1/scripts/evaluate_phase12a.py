#!/usr/bin/env python3
"""Evaluate Phase 12A family-bias source diagnostic (frozen rules; no model calls)."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase8_design import candidate_order  # noqa: E402
from pre_output_physiology.phase11_design import LOCKED, order_robust_label  # noqa: E402
from pre_output_physiology.phase12_diagnostic import (  # noqa: E402
    CELL_NAMES,
    CELLS,
    FAMILIES,
    N_BASES,
    N_NEW_EVALUATIONS,
    family_dependence,
    interpretation,
    matched_agreement,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

DATA = REPO_ROOT / "data/processed/phase12_design"
P11_VARIANTS = REPO_ROOT / "artifacts/phase11a_behavior/variant_choices.json"
GUARANTEE = (
    "PHASE 12A WAS A BEHAVIORAL CONFOUND-DIAGNOSTIC ONLY. PHASE 11A REMAINS A BEHAVIOR-GATE "
    "HOLD. NO ACTIVATIONS WERE COLLECTED. NO PHYSIOLOGY PROBE WAS FIT OR SCORED. THE PHASE-5 "
    "PROBE WAS NOT SCORED. K, PAYOFFS, AND DECODER WERE NOT TUNED. LOCKED FAMILIES WERE NOT "
    "RUN. NO CAUSAL INTERVENTIONS WERE PERFORMED."
)
PAIRS = (("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text("utf-8").splitlines() if x.strip()]


def label_new_variant(r: dict) -> dict:
    ids = r["candidate_ids_visible_order"]
    c1, c2 = candidate_order(r)
    by_state = {c1: ids[0], c2: ids[1]}
    rec, alt = by_state[r["record_state"]], by_state[r["alternate_state"]]
    if r["generated_ids"] != by_state[r["chosen_state"]]:
        raise SystemExit(f"candidate completion integrity failed: {r['example_id']}")
    div = next(i for i in range(len(rec)) if rec[i] != alt[i])
    step = next(s for s in r["steps"] if s["position"] == div)
    lg = {int(k): v for k, v in step["allowed_logits"].items()}
    return {
        "base_scenario_id": r["base_scenario_id"],
        "family": r["family"],
        "cell": r["cell"],
        "order": r["order"],
        "choice": "record" if r["chosen_state"] == r["record_state"] else "alternate",
        "margin": lg[rec[div]] - lg[alt[div]],
    }


def _stats(m: list[float]) -> dict | None:
    if not m:
        return None
    return {"n": len(m), "mean": statistics.fmean(m), "median": statistics.median(m),
            "min": min(m), "max": max(m)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "artifacts/phase12a_diagnostic"))
    ap.add_argument("--report", default=str(REPO_ROOT / "reports/phase12a_diagnostic.md"))
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    raw = _jsonl(run_dir / "diagnostic_choices.jsonl")
    if len(raw) != N_NEW_EVALUATIONS or any(r["family"] in LOCKED for r in raw):
        raise SystemExit("diagnostic rows invalid")
    manifest = json.loads((run_dir / "diagnostic_manifest.json").read_text("utf-8"))
    sel = _jsonl(DATA / "selected_bases.jsonl")
    sel_ids = {b["base_scenario_id"] for b in sel}
    variants = [label_new_variant(r) for r in raw]
    for v in json.loads(P11_VARIANTS.read_text("utf-8")):
        if v["base_scenario_id"] in sel_ids:
            variants.append({"base_scenario_id": v["base_scenario_id"], "family": v["family"],
                             "cell": "A", "order": v["order"], "choice": v["choice"],
                             "margin": v["record_minus_alternate_logit"]})
    idx: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for v in variants:
        idx[(v["cell"], v["base_scenario_id"])][v["order"]] = v
    if len(idx) != 4 * N_BASES or any(set(d) != {"RF", "AF"} for d in idx.values()):
        raise SystemExit("cell/order pairing incomplete")
    bases: dict[str, list[dict]] = {c: [] for c in CELLS}
    for (cell, bid), d in sorted(idx.items()):
        rf, af = d["RF"], d["AF"]
        bases[cell].append({
            "base_scenario_id": bid, "family": rf["family"], "cell": cell,
            "rf_choice": rf["choice"], "af_choice": af["choice"],
            "rf_margin": rf["margin"], "af_margin": af["margin"],
            "label": order_robust_label(rf["choice"], af["choice"]),
        })
    dep = {c: family_dependence(bases[c]) for c in CELLS}
    interp = interpretation(dep)
    agree = {f"{x}_vs_{y}": matched_agreement(bases[x], bases[y]) for x, y in PAIRS}
    switches = {
        c: {
            "rf_record_to_af_alternate": sum(b["rf_choice"] == "record"
                                             and b["af_choice"] == "alternate"
                                             for b in bases[c]),
            "rf_alternate_to_af_record": sum(b["rf_choice"] == "alternate"
                                             and b["af_choice"] == "record"
                                             for b in bases[c]),
        }
        for c in CELLS
    }
    margins = {c: {lab: {o: _stats([b[f"{o.lower()}_margin"] for b in bases[c]
                                    if b["label"] == lab]) for o in ("RF", "AF")}
                   for lab in sorted({b["label"] for b in bases[c]})} for c in CELLS}
    status = ("phase12a_family_bias_source_diagnostic_complete_awaiting_audit"
              if interp["replication_passed"] else "phase12a_family_bias_source_diagnostic_hold")
    summary = {
        "created_at": utc_now_iso(),
        "run_id": manifest["run_id"],
        "git_commit": manifest["git_commit"],
        "n_new_evaluations": len(raw),
        "n_cell_a_reused_variants": 2 * N_BASES,
        "locked_model_calls": 0,
        "status": status,
        "family_dependence_by_cell": dep,
        "matched_agreement": agree,
        "order_switch_by_cell": switches,
        "margins_by_cell_label_order": margins,
        "interpretation": interp,
        "cell_D_note": "Family identity intentionally removed from model-visible text in D; "
        "family grouping in D reflects only which record/alternate numbers and item IDs "
        "the family's selected bases carry.",
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "diagnostic_manifest.json", manifest)
    write_json(out / "diagnostic_summary.json", summary)
    write_json(out / "base_labels_by_cell.json", bases)
    write_report(summary, Path(args.report))
    for c in CELLS:
        d = dep[c]
        print(c, CELL_NAMES[c], {k: d[k] for k in (
            "n_stable", "stable_fraction", "n_stable_record", "n_stable_alternate",
            "cramers_v_family_label", "family_alternate_rate_range",
            "substantial_family_dependence")})
        print("  ", {f: (v["stable_record"], v["stable_alternate"], v["order_sensitive"])
                     for f, v in d["by_family"].items()})
    print(json.dumps(agree, indent=1))
    print(json.dumps(switches))
    print(json.dumps(interp, indent=1))
    print(status)
    return 0


def write_report(s: dict, path: Path) -> None:
    lines = [
        "# Phase 12A — family-bias source diagnostic (behavior only)",
        "",
        f"**Run ID:** `{s['run_id']}`  ",
        f"**Git SHA:** `{s['git_commit']}`  ",
        f"**Status:** `{s['status']}`  ",
        "",
        "144 deterministic Phase-11 discovery bases (24/family). A = original prompt (Phase-11 "
        "RF/AF reused); B = family shell, `slot` states; C = generic shell, family state noun; "
        "D = generic shell, `slot`. K=10, payoff, decoder unchanged. Label = order-robust "
        "(RF and AF agree). Alternate rate is among stable bases.",
        "",
        "| Cell | Stable rec/alt | Order-sensitive rate | Cramér's V | Alt-rate range | "
        "Substantial |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for c in CELLS:
        d = s["family_dependence_by_cell"][c]
        lines.append(
            f"| {c} {CELL_NAMES[c]} | {d['n_stable_record']}/{d['n_stable_alternate']} | "
            f"{d['order_sensitive_rate']:.3f} | {d['cramers_v_family_label']:.3f} | "
            f"{d['family_alternate_rate_range']:.3f} | {d['substantial_family_dependence']} |"
        )
    lines += ["", "**Stable record / stable alternate / order-sensitive (alternate rate) by "
              "family**", "", "| Family | " + " | ".join(CELLS) + " |",
              "| --- | " + " | ".join("---" for _ in CELLS) + " |"]
    for f in FAMILIES:
        cells = []
        for c in CELLS:
            v = s["family_dependence_by_cell"][c]["by_family"][f]
            r = v["alternate_rate_of_stable"]
            cells.append(f"{v['stable_record']}/{v['stable_alternate']}/{v['order_sensitive']}"
                         f" ({'—' if r is None else f'{r:.2f}'})")
        lines.append(f"| {f} | " + " | ".join(cells) + " |")
    lines += ["", s["cell_D_note"], "", "**Matched agreement (by base)**", "",
              "| Pair | Exact label | Both-stable label | RF choice | AF choice |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for k, a in s["matched_agreement"].items():
        sl = a["stable_label_agreement"]
        lines.append(f"| {k} | {a['label_exact_agreement']:.3f} | "
                     f"{'—' if sl is None else f'{sl:.3f}'} (n={a['n_both_stable']}) | "
                     f"{a['rf_choice_agreement']:.3f} | {a['af_choice_agreement']:.3f} |")
    lines += ["", "**Order switches**", ""]
    for c, w in s["order_switch_by_cell"].items():
        lines.append(f"- {c}: RF record → AF alternate {w['rf_record_to_af_alternate']}; "
                     f"RF alternate → AF record {w['rf_alternate_to_af_record']}")
    it = s["interpretation"]
    lines += ["", "**Frozen interpretation**", "", f"```\n{json.dumps(it, indent=1)}\n```",
              "", GUARANTEE, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
