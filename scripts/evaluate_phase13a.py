#!/usr/bin/env python3
"""Evaluate Phase 13A semantic-component diagnostic (frozen rules; no model calls)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase8_design import candidate_order  # noqa: E402
from pre_output_physiology.phase11_design import LOCKED, order_robust_label  # noqa: E402
from pre_output_physiology.phase13_diagnostic import (  # noqa: E402
    CELLS,
    FAMILIES,
    N_BASES,
    N_NEW_EVALUATIONS,
    REUSE_CELLS,
    ablation_diagnostics,
    family_dependence,
    interpretation,
    substantial,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

DATA = REPO_ROOT / "data/processed/phase13_design"
P12_LABELS = REPO_ROOT / "artifacts/phase12a_diagnostic/base_labels_by_cell.json"
GUARANTEE = (
    "PHASE 13A WAS A BEHAVIORAL CONFOUND-DIAGNOSTIC ONLY. PHASE 11A REMAINS A BEHAVIOR-GATE "
    "HOLD. PHASE 12A IDENTIFIED THE SEMANTIC SHELL AS THE DOMINANT FAMILY-LEVEL CONFOUND. "
    "NO ACTIVATIONS WERE COLLECTED. NO PHYSIOLOGY PROBE WAS FIT OR SCORED. THE PHASE-5 "
    "PROBE WAS NOT SCORED. K, PAYOFFS, AND DECODER WERE NOT TUNED. LOCKED FAMILIES WERE "
    "NOT RUN. NO CAUSAL INTERVENTIONS WERE PERFORMED."
)


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "artifacts/phase13a_diagnostic"))
    ap.add_argument("--report", default=str(REPO_ROOT / "reports/phase13a_diagnostic.md"))
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    raw = _jsonl(run_dir / "diagnostic_choices.jsonl")
    if len(raw) != N_NEW_EVALUATIONS or any(r["family"] in LOCKED for r in raw):
        raise SystemExit("diagnostic rows invalid")
    if any(r["cell"] in REUSE_CELLS for r in raw):
        raise SystemExit("reused cell present in new run")
    manifest = json.loads((run_dir / "diagnostic_manifest.json").read_text("utf-8"))
    sel = _jsonl(DATA / "selected_bases.jsonl")
    sel_ids = {b["base_scenario_id"] for b in sel}

    variants = [label_new_variant(r) for r in raw]
    p12 = json.loads(P12_LABELS.read_text("utf-8"))
    for c13, c12 in REUSE_CELLS.items():
        for b in p12[c12]:
            if b["base_scenario_id"] not in sel_ids:
                continue
            for order, choice, margin_key in (
                ("RF", b["rf_choice"], "rf_margin"),
                ("AF", b["af_choice"], "af_margin"),
            ):
                variants.append(
                    {
                        "base_scenario_id": b["base_scenario_id"],
                        "family": b["family"],
                        "cell": c13,
                        "order": order,
                        "choice": choice,
                        "margin": b[margin_key],
                        "reused_from_phase12_cell": c12,
                    }
                )

    idx: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for v in variants:
        idx[(v["cell"], v["base_scenario_id"])][v["order"]] = v
    if len(idx) != 8 * N_BASES or any(set(d) != {"RF", "AF"} for d in idx.values()):
        raise SystemExit("cell/order pairing incomplete")

    bases: dict[str, list[dict]] = {c: [] for c in CELLS}
    for (cell, bid), d in sorted(idx.items()):
        rf, af = d["RF"], d["AF"]
        bases[cell].append(
            {
                "base_scenario_id": bid,
                "family": rf["family"],
                "cell": cell,
                "rf_choice": rf["choice"],
                "af_choice": af["choice"],
                "rf_margin": rf["margin"],
                "af_margin": af["margin"],
                "label": order_robust_label(rf["choice"], af["choice"]),
            }
        )

    dep = {c: family_dependence(bases[c]) for c in CELLS}
    interp = interpretation(dep)
    ablations = ablation_diagnostics(dep, bases) if interp["replication_111_passed"] else None

    # Replication: 000 must match Phase-12 D exactly (same labels on same bases).
    p12_d = {b["base_scenario_id"]: b for b in p12["D"] if b["base_scenario_id"] in sel_ids}
    zero = {b["base_scenario_id"]: b for b in bases["000"]}
    zero_match = all(
        zero[bid]["label"] == p12_d[bid]["label"]
        and zero[bid]["rf_choice"] == p12_d[bid]["rf_choice"]
        and zero[bid]["af_choice"] == p12_d[bid]["af_choice"]
        for bid in sel_ids
    )
    if not zero_match:
        raise SystemExit("000 does not exactly match reused Phase-12 D labels")

    status = (
        "phase13a_semantic_component_diagnostic_complete_awaiting_audit"
        if interp["replication_111_passed"] and zero_match
        else "phase13a_semantic_component_diagnostic_hold"
    )
    summary = {
        "created_at": utc_now_iso(),
        "run_id": manifest["run_id"],
        "git_commit": manifest["git_commit"],
        "n_new_evaluations": len(raw),
        "n_reused_variants": 2 * 2 * N_BASES,
        "locked_model_calls": 0,
        "status": status,
        "family_dependence_by_cell": dep,
        "substantial_by_cell": {c: substantial(dep[c]) for c in CELLS},
        "interpretation": interp,
        "ablation_diagnostics": ablations,
        "cell_000_exact_match_to_phase12_D": zero_match,
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "diagnostic_manifest.json", manifest)
    write_json(out / "diagnostic_summary.json", summary)
    write_json(out / "base_labels_by_cell.json", bases)
    write_report(summary, Path(args.report))
    for c in CELLS:
        d = dep[c]
        print(
            c,
            {
                k: d[k]
                for k in (
                    "n_stable_record",
                    "n_stable_alternate",
                    "order_sensitive_rate",
                    "cramers_v_family_label",
                    "family_alternate_rate_range",
                    "substantial_family_dependence",
                )
            },
        )
        print(
            " ",
            {
                f: (v["stable_record"], v["stable_alternate"], v["order_sensitive"])
                for f, v in d["by_family"].items()
            },
        )
    print(json.dumps(interp, indent=1))
    if ablations:
        slim = {}
        for k, v in ablations.items():
            slim[k] = {
                a: b for a, b in v.items() if a != "matched_agreement"
            }
            slim[k]["stable_label_agreement"] = v["matched_agreement"][
                "stable_label_agreement"
            ]
        print(json.dumps(slim, indent=1))
    print(status)
    return 0


def write_report(s: dict, path: Path) -> None:
    lines = [
        "# Phase 13A — semantic-component diagnostic (T × Q × E)",
        "",
        f"**Run ID:** `{s['run_id']}`  ",
        f"**Git SHA:** `{s['git_commit']}`  ",
        f"**Status:** `{s['status']}`  ",
        "",
        "144 Phase-12 selected bases. Every cell uses `slot N` states. Factors: T=topic, "
        "Q=question, E=entity. Cells `000`/`111` reuse Phase-12 D/B (byte-identical). "
        "Label = order-robust (RF and AF agree). Substantial = V≥0.50 and range≥0.50.",
        "",
        "| Cell | TQE | Stable rec/alt | OS rate | V | Range | Substantial |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    names = {
        "000": "generic",
        "001": "entity only",
        "010": "question only",
        "011": "Q+E",
        "100": "topic only",
        "101": "T+E",
        "110": "T+Q",
        "111": "full shell",
    }
    for c in CELLS:
        d = s["family_dependence_by_cell"][c]
        lines.append(
            f"| {c} ({names[c]}) | {c} | {d['n_stable_record']}/{d['n_stable_alternate']} | "
            f"{d['order_sensitive_rate']:.3f} | {d['cramers_v_family_label']:.3f} | "
            f"{d['family_alternate_rate_range']:.3f} | "
            f"{d['substantial_family_dependence']} |"
        )
    lines += [
        "",
        "**Stable record / alternate / order-sensitive by family**",
        "",
        "| Family | " + " | ".join(CELLS) + " |",
        "| --- | " + " | ".join("---" for _ in CELLS) + " |",
    ]
    for f in FAMILIES:
        cells = []
        for c in CELLS:
            v = s["family_dependence_by_cell"][c]["by_family"][f]
            cells.append(
                f"{v['stable_record']}/{v['stable_alternate']}/{v['order_sensitive']}"
            )
        lines.append(f"| {f} | " + " | ".join(cells) + " |")
    interp_json = json.dumps(s["interpretation"], indent=1)
    lines += ["", "**Frozen interpretation**", "", f"```\n{interp_json}\n```"]
    if s.get("ablation_diagnostics"):
        lines += ["", "**Single-component ablation (descriptive)**", "",
                  "| Remove | 111 → | ΔV | Stable-label agreement |",
                  "| --- | --- | ---: | ---: |"]
        for k, v in s["ablation_diagnostics"].items():
            sl = v["matched_agreement"]["stable_label_agreement"]
            lines.append(
                f"| {k} | {v['from']}→{v['to']} | {v['delta_V']:+.3f} | "
                f"{'—' if sl is None else f'{sl:.3f}'} |"
            )
    lines += [
        "",
        f"**000 exact match to Phase-12 D:** {s['cell_000_exact_match_to_phase12_D']}  ",
        f"**Locked-family model calls:** {s['locked_model_calls']}",
        "",
        GUARANTEE,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
