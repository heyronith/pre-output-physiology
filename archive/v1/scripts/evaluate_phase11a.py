#!/usr/bin/env python3
"""Evaluate Phase 11A order-robust behavior: frozen labels, gates, descriptive diagnostics.

No model calls.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase8_design import candidate_order  # noqa: E402
from pre_output_physiology.phase11_design import (  # noqa: E402
    LABEL_ALTERNATE,
    LABEL_ORDER_SENSITIVE,
    LABEL_RECORD,
    LOCKED,
    N_DISCOVERY_BASES,
    N_EVALUATIONS,
    order_robust_label,
    split_gate,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

SPLITS = ("discovery_train", "discovery_validation")
LABELS = (LABEL_RECORD, LABEL_ALTERNATE, LABEL_ORDER_SENSITIVE)
GUARANTEE = (
    "PHASE 10A REMAINS A SANITY HOLD. PHASE 11 USES K=10 AS A NEW PROSPECTIVE FIXED DESIGN "
    "SETTING SELECTED ONLY FROM CALIBRATION-ONLY PHASE-10 DATA. NO K SEARCH OR PROMPT TUNING "
    "WAS PERFORMED ON PHASE-11 SCENARIOS. NO ACTIVATIONS WERE COLLECTED. NO PHYSIOLOGY PROBE "
    "WAS FIT OR SCORED. THE PHASE-5 PROBE WAS NOT SCORED. LOCKED FAMILIES WERE NOT RUN "
    "THROUGH THE MODEL. NO CAUSAL INTERVENTIONS WERE PERFORMED."
)


def label_variant(r: dict) -> dict:
    ids = r["candidate_ids_visible_order"]
    c1, c2 = candidate_order(r)
    by_state = {c1: ids[0], c2: ids[1]}
    rec, alt = by_state[r["record_state"]], by_state[r["alternate_state"]]
    if r["generated_ids"] != by_state[r["chosen_state"]]:
        raise SystemExit(f"candidate completion integrity failed: {r['example_id']}")
    if r["chosen_state"] not in (r["record_state"], r["alternate_state"]):
        raise SystemExit("chosen state outside candidates")
    div = next(i for i in range(min(len(rec), len(alt))) if rec[i] != alt[i])
    step = next(s for s in r["steps"] if s["position"] == div)
    lg = {int(k): v for k, v in step["allowed_logits"].items()}
    if set(lg) != {rec[div], alt[div]}:
        raise SystemExit("divergent step allowed set mismatch")
    return {
        "example_id": r["example_id"],
        "base_scenario_id": r["base_scenario_id"],
        "family": r["family"],
        "split": r["split"],
        "order": r["order"],
        "record_state": r["record_state"],
        "alternate_state": r["alternate_state"],
        "record_token_ids": rec,
        "alternate_token_ids": alt,
        "choice": "record" if r["chosen_state"] == r["record_state"] else "alternate",
        "first_divergent_position": div,
        "record_minus_alternate_logit": lg[rec[div]] - lg[alt[div]],
    }


def _stats(m: list[float]) -> dict[str, float] | None:
    if not m:
        return None
    return {"n": len(m), "mean": statistics.fmean(m), "median": statistics.median(m),
            "min": min(m), "max": max(m)}


def _cramers_v(pairs: list[tuple[str, str]]) -> dict[str, float]:
    rows = sorted({a for a, _ in pairs})
    cols = sorted({b for _, b in pairs})
    n = len(pairs)
    obs = Counter(pairs)
    ra = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    chi2 = 0.0
    for a in rows:
        for b in cols:
            e = ra[a] * cb[b] / n
            if e > 0:
                chi2 += (obs[(a, b)] - e) ** 2 / e
    k = min(len(rows), len(cols)) - 1
    return {"chi2": chi2, "dof": (len(rows) - 1) * (len(cols) - 1), "n": n,
            "cramers_v": math.sqrt(chi2 / (n * k)) if k > 0 and n else 0.0}


def _digit(state: str) -> str:
    return state.rsplit(" ", 1)[-1]


def association(items: list[dict], key: str) -> dict:
    """items: {feature, outcome}. Per-feature outcome counts + Cramér's V."""
    table: dict[str, Counter] = defaultdict(Counter)
    for it in items:
        table[str(it["feature"])][it["outcome"]] += 1
    return {
        "feature": key,
        "table": {f: dict(c) for f, c in sorted(table.items())},
        **_cramers_v([(str(it["feature"]), it["outcome"]) for it in items]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "artifacts/phase11a_behavior"))
    ap.add_argument("--report", default=str(REPO_ROOT / "reports/phase11a_behavior.md"))
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    rows = [json.loads(x) for x in (run_dir / "behavior_choices.jsonl").read_text(
        "utf-8").splitlines() if x.strip()]
    if len(rows) != N_EVALUATIONS or any(r["family"] in LOCKED for r in rows):
        raise SystemExit("behavior rows invalid")
    manifest = json.loads((run_dir / "behavior_manifest.json").read_text("utf-8"))
    variants = [label_variant(r) for r in rows]
    by_base: dict[str, dict[str, dict]] = defaultdict(dict)
    for v in variants:
        by_base[v["base_scenario_id"]][v["order"]] = v
    if len(by_base) != N_DISCOVERY_BASES or any(set(d) != {"RF", "AF"} for d in by_base.values()):
        raise SystemExit("RF/AF pairing incomplete")
    bases = []
    for bid, d in sorted(by_base.items()):
        rf, af = d["RF"], d["AF"]
        bases.append({
            "base_scenario_id": bid,
            "family": rf["family"],
            "split": rf["split"],
            "record_state": rf["record_state"],
            "alternate_state": rf["alternate_state"],
            "record_token_ids": rf["record_token_ids"],
            "alternate_token_ids": rf["alternate_token_ids"],
            "rf_choice": rf["choice"],
            "af_choice": af["choice"],
            "rf_margin": rf["record_minus_alternate_logit"],
            "af_margin": af["record_minus_alternate_logit"],
            "label": order_robust_label(rf["choice"], af["choice"]),
            "canonical_prompt_order": "RF",
        })

    gates = {s: split_gate([b for b in bases if b["split"] == s], s) for s in SPLITS}
    passed = all(g["pass"] for g in gates.values())
    choice_counts = {
        s: {o: dict(Counter(b[f"{o.lower()}_choice"] for b in bases if b["split"] == s))
            for o in ("RF", "AF")}
        for s in SPLITS
    }
    switch = {
        s: {
            "rf_record_to_af_alternate": sum(
                b["rf_choice"] == "record" and b["af_choice"] == "alternate"
                for b in bases if b["split"] == s),
            "rf_alternate_to_af_record": sum(
                b["rf_choice"] == "alternate" and b["af_choice"] == "record"
                for b in bases if b["split"] == s),
        }
        for s in SPLITS
    }
    for s in SPLITS:
        switch[s]["order_switch_rate"] = sum(switch[s].values()) / gates[s]["n_bases"]
    margins = {
        s: {
            lab: {o: _stats([b[f"{o.lower()}_margin"] for b in bases
                             if b["split"] == s and b["label"] == lab]) for o in ("RF", "AF")}
            for lab in LABELS
        }
        for s in SPLITS
    }
    stable = [b for b in bases if b["label"] != LABEL_ORDER_SENSITIVE]
    diag = {
        "stable_label_vs_record_digit": association(
            [{"feature": _digit(b["record_state"]), "outcome": b["label"]} for b in stable],
            "record_digit"),
        "stable_label_vs_alternate_digit": association(
            [{"feature": _digit(b["alternate_state"]), "outcome": b["label"]} for b in stable],
            "alternate_digit"),
        "stable_label_vs_alternate_gt_record": association(
            [{"feature": int(_digit(b["alternate_state"]) > _digit(b["record_state"])),
              "outcome": b["label"]} for b in stable], "alternate_digit_gt_record_digit"),
        "stable_label_vs_divergent_token_pair": association(
            [{"feature": f"{b['record_token_ids'][-1]}>{b['alternate_token_ids'][-1]}",
              "outcome": b["label"]} for b in stable], "record_vs_alternate_divergent_token"),
        "stable_label_vs_family": association(
            [{"feature": b["family"], "outcome": b["label"]} for b in stable], "family"),
        "rf_choice_vs_record_digit_all_bases": association(
            [{"feature": _digit(b["record_state"]), "outcome": b["rf_choice"]} for b in bases],
            "record_digit"),
        "rf_choice_vs_alternate_digit_all_bases": association(
            [{"feature": _digit(b["alternate_state"]), "outcome": b["rf_choice"]}
             for b in bases], "alternate_digit"),
    }
    summary = {
        "created_at": utc_now_iso(),
        "run_id": manifest["run_id"],
        "git_commit": manifest["git_commit"],
        "n_evaluations": len(variants),
        "locked_model_calls": 0,
        "status": "phase11a_order_robust_behavior_pass_awaiting_audit" if passed
        else "phase11a_order_robust_behavior_hold",
        "gates": gates,
        "gates_pass": passed,
        "choice_counts_by_order": choice_counts,
        "order_switch": switch,
        "margins_by_split_label_order": margins,
        "descriptive_diagnostics": diag,
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "behavior_manifest.json", manifest)
    write_json(out / "behavior_summary.json", summary)
    write_json(out / "base_labels.json", bases)
    write_json(out / "variant_choices.json", variants)
    write_report(summary, Path(args.report))
    print(json.dumps({k: summary[k] for k in (
        "status", "choice_counts_by_order", "order_switch")}, indent=1))
    for s in SPLITS:
        g = gates[s]
        print(s, {k: g[k] for k in ("n_stable_record", "n_stable_alternate",
                                    "n_order_sensitive", "stable_fraction", "checks", "pass")})
        print(json.dumps(g["by_family"]))
    print(json.dumps(margins, indent=1))
    print({k: round(v["cramers_v"], 3) for k, v in diag.items()})
    return 0


def write_report(s: dict, path: Path) -> None:
    lines = [
        "# Phase 11A — order-robust behavioral labels (fixed K=10)",
        "",
        f"**Run ID:** `{s['run_id']}`  ",
        f"**Git SHA:** `{s['git_commit']}`  ",
        f"**Status:** `{s['status']}`  ",
        "",
        "Every prompt: same goal-vs-record conflict, same single-objective payoff rule, K=10. "
        "Each discovery base decoded in RF (record first) and AF (alternate first) orders with "
        "the unchanged two-candidate constrained decoder after `Response` (12107). Label = "
        "shared choice if RF and AF agree; otherwise `order_sensitive_unlabeled`. Locked "
        "families not run.",
        "",
        "| Split | RF rec/alt | AF rec/alt | Stable record | Stable alternate | "
        "Order-sensitive | Stable fraction | Gate |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for sp in SPLITS:
        g, c = s["gates"][sp], s["choice_counts_by_order"][sp]
        lines.append(
            f"| {sp} | {c['RF'].get('record', 0)}/{c['RF'].get('alternate', 0)} | "
            f"{c['AF'].get('record', 0)}/{c['AF'].get('alternate', 0)} | "
            f"{g['n_stable_record']} | {g['n_stable_alternate']} | {g['n_order_sensitive']} | "
            f"{g['stable_fraction']:.3f} | {'PASS' if g['pass'] else 'FAIL'} {g['checks']} |"
        )
    lines += ["", "| Family | Split | Stable record | Stable alternate | Order-sensitive | "
              "Meets per-class |", "| --- | --- | ---: | ---: | ---: | --- |"]
    for sp in SPLITS:
        for f, v in s["gates"][sp]["by_family"].items():
            lines.append(f"| {f} | {sp} | {v['stable_record']} | {v['stable_alternate']} | "
                         f"{v['order_sensitive']} | {v['meets_per_class']} |")
    lines += ["", "**Order switches**", ""]
    for sp in SPLITS:
        w = s["order_switch"][sp]
        lines.append(f"- {sp}: switch rate {w['order_switch_rate']:.3f}; RF record → AF "
                     f"alternate {w['rf_record_to_af_alternate']}; RF alternate → AF record "
                     f"{w['rf_alternate_to_af_record']}")
    lines += ["", "**Margins (record − alternate at first divergent token)**", "",
              "| Split | Label | Order | n | mean | median | min | max |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for sp in SPLITS:
        for lab, d in s["margins_by_split_label_order"][sp].items():
            for o, m in d.items():
                if m:
                    lines.append(f"| {sp} | {lab} | {o} | {m['n']} | {m['mean']:.2f} | "
                                 f"{m['median']:.2f} | {m['min']:.2f} | {m['max']:.2f} |")
    lines += ["", "**Descriptive associations (Cramér's V; not used for any decision)**", ""]
    for k, v in s["descriptive_diagnostics"].items():
        lines.append(f"- {k}: V={v['cramers_v']:.3f} (chi2={v['chi2']:.1f}, dof={v['dof']}, "
                     f"n={v['n']})")
    lines += ["", f"**Behavior gates pass:** {s['gates_pass']}  ",
              f"**Locked-family model calls:** {s['locked_model_calls']}", "", GUARANTEE, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
