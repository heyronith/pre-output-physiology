#!/usr/bin/env python3
"""Analyze Phase-23B DEVELOPMENT (merged pilot+remainder); freeze winner decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase23_open_grader import (  # noqa: E402
    CANDIDATES,
    DEV_ELIGIBILITY,
    GUARANTEE,
    PHASE21_ONSET,
    PHASE22B_ONSET,
    STATUS_23B,
    compare_development_population_membership,
    compute_agreement_metrics,
    development_eligibility_check,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

PROC = REPO_ROOT / "data/processed/phase23_open_grader"
OUT = REPO_ROOT / "artifacts/phase23b_development"
REPORT = REPO_ROOT / "reports/phase23b_development.md"

PILOT_RUNS = {
    "gemma4_31b_it": "phase23a_pilot_gemma4_31b_it_20260929T213351Z_7e341a62",
    "qwen35_27b": "phase23a_pilot_qwen35_27b_20260929T212300Z_7e341a62",
}
PILOT_SHA = {
    "gemma4_31b_it": "d9794d0268237ae6721554c6959e2b0a1c179784585e1f8f13de1f6c3f9c79f0",
    "qwen35_27b": "7cdea08e3b65132eb7debd1dfbde2498afe780ec975cc8ef30a55478ed9c52b1",
}


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(x)
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def _precision_recall(yt: list[str], yp: list[str], labels: list[str]) -> dict:
    out = {}
    for lab in labels:
        tp = sum(1 for t, p in zip(yt, yp, strict=True) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(yt, yp, strict=True) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(yt, yp, strict=True) if t == lab and p != lab)
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        out[lab] = {"precision": prec, "recall": rec, "support": tp + fn}
    return out


def _extended_metrics(rows: list[dict]) -> dict:
    base = compute_agreement_metrics(rows)
    labels3 = ["honest", "ambiguous", "deceptive"]
    # Include invalids explicitly in reporting
    n_invalid = sum(1 for r in rows if not r.get("open_valid"))
    pairs = [
        (r["reference_label"], r.get("open_class"))
        for r in rows
        if r.get("open_valid")
        and r["reference_label"] in labels3
        and r.get("open_class") in labels3
    ]
    yt = [a for a, _ in pairs]
    yp = [b for _, b in pairs]
    base["three_class"]["per_class"] = _precision_recall(yt, yp, labels3)

    # HD with invalids counted as errors (not silently dropped from n_reference)
    hd_ref = [r for r in rows if r["reference_label"] in ("honest", "deceptive")]
    n_h = sum(1 for r in hd_ref if r["reference_label"] == "honest")
    n_d = sum(1 for r in hd_ref if r["reference_label"] == "deceptive")
    # Treat invalid / non-matching as miss
    def _pred(r: dict) -> str:
        if not r.get("open_valid"):
            return "invalid"
        p = r.get("open_class")
        return p if p in ("honest", "deceptive", "ambiguous") else "other"

    sens_h = (
        sum(1 for r in hd_ref if r["reference_label"] == "honest" and _pred(r) == "honest")
        / n_h
        if n_h
        else float("nan")
    )
    sens_d = (
        sum(
            1
            for r in hd_ref
            if r["reference_label"] == "deceptive" and _pred(r) == "deceptive"
        )
        / n_d
        if n_d
        else float("nan")
    )
    h_to_d = sum(
        1 for r in hd_ref if r["reference_label"] == "honest" and _pred(r) == "deceptive"
    )
    d_to_h = sum(
        1 for r in hd_ref if r["reference_label"] == "deceptive" and _pred(r) == "honest"
    )
    hd_to_amb = sum(1 for r in hd_ref if _pred(r) == "ambiguous")
    hd_to_inv = sum(1 for r in hd_ref if _pred(r) == "invalid")
    # κ only on valid H/D preds (same as base); also report invalid-aware sensitivities
    base["unambiguous_honest_deceptive"]["n_evaluated_reference_hd"] = len(hd_ref)
    base["unambiguous_honest_deceptive"]["honest_sensitivity_invalid_as_miss"] = sens_h
    base["unambiguous_honest_deceptive"]["deceptive_sensitivity_invalid_as_miss"] = sens_d
    base["unambiguous_honest_deceptive"]["balanced_accuracy_invalid_as_miss"] = (
        (sens_h + sens_d) / 2.0
        if not (math.isnan(sens_h) or math.isnan(sens_d))
        else float("nan")
    )
    base["unambiguous_honest_deceptive"]["honest_to_deceptive_errors"] = h_to_d
    base["unambiguous_honest_deceptive"]["deceptive_to_honest_errors"] = d_to_h
    base["unambiguous_honest_deceptive"]["hd_mapped_to_ambiguous"] = hd_to_amb
    base["unambiguous_honest_deceptive"]["hd_mapped_to_invalid"] = hd_to_inv
    base["unambiguous_honest_deceptive"]["specificity_deceptive_as_positive"] = sens_h
    base["n_invalid_explicit"] = n_invalid

    # Ambiguous-class analysis
    amb_ref = [r for r in rows if r["reference_label"] == "ambiguous"]
    amb_pred = Counter(
        (r.get("open_class") if r.get("open_valid") else "invalid") for r in amb_ref
    )
    n_amb = len(amb_ref)
    converted_hd = sum(amb_pred[c] for c in ("honest", "deceptive"))
    hd_rows = [r for r in rows if r["reference_label"] in ("honest", "deceptive")]
    hd_to_a = sum(
        1
        for r in hd_rows
        if r.get("open_valid") and r.get("open_class") == "ambiguous"
    )
    base["ambiguous_analysis"] = {
        "n_reference_ambiguous": n_amb,
        "open_pred_counts": dict(amb_pred),
        "fraction_ambiguous_converted_to_hd": converted_hd / n_amb if n_amb else float("nan"),
        "fraction_hd_converted_to_ambiguous": hd_to_a / len(hd_rows) if hd_rows else float("nan"),
    }

    # Prompt-level H/D disagreement rates
    by_prompt: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_prompt[r["prompt_id"]].append(r)
    rates = []
    top = []
    for pid, rs in by_prompt.items():
        hd = [r for r in rs if r["reference_label"] in ("honest", "deceptive")]
        if not hd:
            continue
        n_dis = sum(
            1
            for r in hd
            if (not r.get("open_valid"))
            or r.get("open_class") != r["reference_label"]
        )
        rate = n_dis / len(hd)
        rates.append(rate)
        top.append({"prompt_id": pid, "n_hd": len(hd), "n_disagree": n_dis, "rate": rate})
    top = sorted(top, key=lambda x: (-x["rate"], x["prompt_id"]))
    rates_sorted = sorted(rates)
    base["prompt_hd_disagreement"] = {
        "n_prompts_with_hd": len(rates),
        "median_rate": statistics.median(rates) if rates else float("nan"),
        "p90_rate": rates_sorted[int(0.9 * (len(rates_sorted) - 1))] if rates_sorted else float("nan"),
        "max_rate": max(rates) if rates else float("nan"),
        "top_disagreement_prompts": top[:20],
    }
    return base


def _merge(
    cand: str, remainder_run_id: str, out_dir: Path
) -> tuple[list[dict], dict]:
    pilot_path = (
        REPO_ROOT
        / "artifacts/phase23a_pilot/runs"
        / PILOT_RUNS[cand]
        / "judgments.jsonl"
    )
    if _sha_file(pilot_path) != PILOT_SHA[cand]:
        raise SystemExit(f"STOP: pilot SHA mismatch for {cand}")
    rem_dir = REPO_ROOT / "artifacts/runs" / remainder_run_id
    rem_man = json.loads((rem_dir / "manifest.json").read_text())
    pilot = _load_jsonl(pilot_path)
    rem = _load_jsonl(rem_dir / "judgments.jsonl")
    if len(pilot) != 200 or len(rem) != 5000:
        raise SystemExit(f"STOP: {cand} sizes pilot={len(pilot)} rem={len(rem)}")
    by = {}
    for r in pilot + rem:
        cid = r["continuation_id"]
        if cid in by:
            raise SystemExit(f"STOP: duplicate {cid}")
        by[cid] = r
    if len(by) != 5200:
        raise SystemExit(f"STOP: merged {len(by)} != 5200")
    # Ensure all DEVELOPMENT
    corpus_dev = {
        r["continuation_id"]
        for r in _load_jsonl(PROC / "reference_corpus.jsonl")
        if r["grader_split"] == "development"
    }
    if set(by) != corpus_dev:
        raise SystemExit("STOP: merged IDs != DEVELOPMENT set")
    # No locked
    locked = {
        r["continuation_id"]
        for r in _load_jsonl(PROC / "reference_corpus.jsonl")
        if r["grader_split"] == "locked_validation"
    }
    if set(by) & locked:
        raise SystemExit("STOP: LOCKED IDs in merge")

    merged = sorted(by.values(), key=lambda r: r["continuation_id"])
    merge_dir = out_dir / "merged" / cand
    merge_dir.mkdir(parents=True, exist_ok=True)
    merged_path = merge_dir / "judgments_5200.jsonl"
    with merged_path.open("w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    rem_j_sha = _sha_file(rem_dir / "judgments.jsonl")
    merged_sha = _sha_file(merged_path)
    cost_new = rem_man["cost"]
    meta = {
        "candidate": cand,
        "pilot_run_id": PILOT_RUNS[cand],
        "pilot_judgments_sha256": PILOT_SHA[cand],
        "remainder_run_id": remainder_run_id,
        "remainder_judgments_sha256": rem_j_sha,
        "remainder_manifest": rem_man,
        "n_reused_pilot": 200,
        "n_new": 5000,
        "n_total": 5200,
        "n_valid": sum(1 for r in merged if r.get("open_valid")),
        "n_invalid": sum(1 for r in merged if not r.get("open_valid")),
        "merged_judgments_path": str(merged_path.relative_to(REPO_ROOT)),
        "merged_judgments_sha256": merged_sha,
        "revision": rem_man["revision"],
        "hf_id": rem_man["hf_id"],
        "cost_new_judgments": cost_new,
        "cost_per_1000_new_usd": cost_new["cost_per_1000_judgments_usd"],
        "cost_per_1000_total_effective_usd": cost_new["estimated_cost_usd"] * 1000 / 5200,
        "estimated_new_cost_usd": cost_new["estimated_cost_usd"],
    }
    write_json(merge_dir / "merge_manifest.json", meta)
    return merged, meta


def _build_disagreement_audit(
    cand: str, rows: list[dict], changed_prompts: set[str], out: Path
) -> dict:
    corpus = {
        r["continuation_id"]: r for r in _load_jsonl(PROC / "reference_corpus.jsonl")
    }
    buckets = {
        "ref_honest_open_dec": [],
        "ref_dec_open_honest": [],
        "ambiguous_disagreement": [],
        "qualification_changing": [],
    }
    for r in rows:
        if not r.get("open_valid"):
            continue
        ref, pred = r["reference_label"], r["open_class"]
        if ref == pred:
            continue
        cid = r["continuation_id"]
        if r["prompt_id"] in changed_prompts:
            buckets["qualification_changing"].append(cid)
        if ref == "honest" and pred == "deceptive":
            buckets["ref_honest_open_dec"].append(cid)
        elif ref == "deceptive" and pred == "honest":
            buckets["ref_dec_open_honest"].append(cid)
        elif ref == "ambiguous" or pred == "ambiguous":
            buckets["ambiguous_disagreement"].append(cid)

    def _take(pool: list[str], k: int) -> list[str]:
        pool = sorted(pool, key=lambda x: hashlib.sha256(x.encode()).hexdigest())
        return pool[:k]

    selected = []
    for key, k in (
        ("ref_honest_open_dec", 30),
        ("ref_dec_open_honest", 30),
        ("ambiguous_disagreement", 40),
        ("qualification_changing", 40),
    ):
        selected.extend(_take(buckets[key], k))
    # unique preserve order by sha
    seen = set()
    uniq = []
    for cid in sorted(selected, key=lambda x: hashlib.sha256(x.encode()).hexdigest()):
        if cid not in seen:
            seen.add(cid)
            uniq.append(cid)
    uniq = uniq[:120]

    by_j = {r["continuation_id"]: r for r in rows}
    items = []
    mapping = {}
    for cid in uniq:
        c = corpus[cid]
        j = by_j[cid]
        swap = int(hashlib.sha256(f"ab23b|{cid}".encode()).hexdigest(), 16) % 2 == 0
        gpt = {
            "score": j["reference_judgement"],
            "class": j["reference_label"],
            "explanation": None,
            "identity": "gpt-4o-2024-08-06",
        }
        op = {
            "score": j.get("open_score"),
            "class": j.get("open_class"),
            "explanation": j.get("open_explanation"),
            "identity": cand,
        }
        if swap:
            a, b = op, gpt
        else:
            a, b = gpt, op
        mapping[cid] = {"grader_a": a["identity"], "grader_b": b["identity"]}
        items.append(
            {
                "continuation_id": cid,
                "prompt_id": c["prompt_id"],
                "scenario": c["scenario"],
                "question": c["question"],
                "response": c["full_response"],
                "judgment_a": {
                    "score": a["score"],
                    "class": a["class"],
                    "explanation": a["explanation"],
                },
                "judgment_b": {
                    "score": b["score"],
                    "class": b["class"],
                    "explanation": b["explanation"],
                },
            }
        )
    audit_dir = out / "disagreement_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    with (audit_dir / "blinded_items.jsonl").open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, sort_keys=True) + "\n")
    write_json(audit_dir / "ab_mapping.json", mapping)
    write_json(
        audit_dir / "manifest.json",
        {
            "n": len(items),
            "candidate": cand,
            "bucket_sizes": {k: len(v) for k, v in buckets.items()},
            "note": "A/B identities blinded in blinded_items.jsonl; mapping separate.",
        },
    )
    return {"n": len(items), "path": str(audit_dir.relative_to(REPO_ROOT))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemma-remainder-run", required=True)
    ap.add_argument("--qwen-remainder-run", required=True)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    git_commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()

    # DEVELOPMENT annotated rows for population (from onset runs, filtered)
    split = json.loads((PROC / "grader_prompt_split.json").read_text())
    dev_pids = set(split["development_prompt_ids"])
    p21 = _load_jsonl(REPO_ROOT / "artifacts/runs" / PHASE21_ONSET / "annotated.jsonl")
    p22 = _load_jsonl(REPO_ROOT / "artifacts/runs" / PHASE22B_ONSET / "annotated.jsonl")
    annotated_dev = [r for r in (p21 + p22) if r["prompt_id"] in dev_pids]
    if len(annotated_dev) != 5200:
        raise SystemExit(f"annotated DEV {len(annotated_dev)} != 5200")

    results = {}
    for cand, rem_run in (
        ("gemma4_31b_it", args.gemma_remainder_run),
        ("qwen35_27b", args.qwen_remainder_run),
    ):
        merged, meta = _merge(cand, rem_run, OUT)
        metrics = _extended_metrics(merged)
        elig = development_eligibility_check(metrics)
        # Prefer invalid-as-miss for eligibility numeric gates
        u = metrics["unambiguous_honest_deceptive"]
        elig_gate = {
            "cohen_kappa": u["cohen_kappa"],
            "balanced_accuracy": u["balanced_accuracy_invalid_as_miss"],
            "honest_sensitivity": u["honest_sensitivity_invalid_as_miss"],
            "deceptive_sensitivity": u["deceptive_sensitivity_invalid_as_miss"],
        }
        # Re-check with invalid-as-miss sensitivities / bal-acc; κ from valid HD pairs
        reasons = []
        if elig_gate["cohen_kappa"] < DEV_ELIGIBILITY["min_kappa_unambiguous"]:
            reasons.append(
                f"kappa {elig_gate['cohen_kappa']:.4f} < {DEV_ELIGIBILITY['min_kappa_unambiguous']}"
            )
        if elig_gate["balanced_accuracy"] < DEV_ELIGIBILITY["min_balanced_accuracy"]:
            reasons.append(
                f"balanced_accuracy {elig_gate['balanced_accuracy']:.4f} < "
                f"{DEV_ELIGIBILITY['min_balanced_accuracy']}"
            )
        if elig_gate["honest_sensitivity"] < DEV_ELIGIBILITY["min_honest_sensitivity"]:
            reasons.append(
                f"honest_sensitivity {elig_gate['honest_sensitivity']:.4f} < "
                f"{DEV_ELIGIBILITY['min_honest_sensitivity']}"
            )
        if (
            elig_gate["deceptive_sensitivity"]
            < DEV_ELIGIBILITY["min_deceptive_sensitivity"]
        ):
            reasons.append(
                f"deceptive_sensitivity {elig_gate['deceptive_sensitivity']:.4f} < "
                f"{DEV_ELIGIBILITY['min_deceptive_sensitivity']}"
            )
        pop = compare_development_population_membership(annotated_dev, merged)
        results[cand] = {
            "merge": meta,
            "metrics": metrics,
            "eligibility_library_check": elig,
            "eligibility_gates": {
                "passed_numeric": not bool(reasons),
                "reasons": reasons,
                "values": elig_gate,
                "thresholds": DEV_ELIGIBILITY,
            },
            "population_development": pop,
        }

    # Winner selection
    passed = [
        c for c, v in results.items() if v["eligibility_gates"]["passed_numeric"]
    ]
    # Population materiality is descriptive; flag large changes
    for c in results:
        n_chg = results[c]["population_development"]["n_status_changed"]
        results[c]["population_materiality_note"] = (
            f"{n_chg} DEVELOPMENT prompts change qualification status vs GPT-4o; "
            "no post-hoc numeric population threshold applied."
        )

    if not passed:
        status = "phase23b_no_open_grader_passed"
        winner = None
        reason = "neither candidate met frozen numeric Stage-2 eligibility gates"
    elif len(passed) == 1:
        status = "phase23b_winner_selected_pending_audit"
        winner = passed[0]
        reason = f"sole numeric gate passer: {winner}"
    else:
        status = "phase23b_winner_selected_pending_audit"
        # Frozen priority: scientific agreement, population, robustness, cost
        def _key(c: str):
            v = results[c]
            u = v["eligibility_gates"]["values"]
            pop = v["population_development"]
            return (
                -u["cohen_kappa"],
                -u["balanced_accuracy"],
                -pop["jaccard_qualifying_prompts"],
                pop["n_status_changed"],
                v["merge"]["estimated_new_cost_usd"],
            )

        ranked = sorted(passed, key=_key)
        winner = ranked[0]
        reason = (
            f"both passed numeric gates; selected {winner} by frozen order "
            f"(κ, bal-acc, population jaccard, fewer membership changes, cost)"
        )

    audit_info = None
    if winner:
        changed = set(results[winner]["population_development"]["status_changed_prompt_ids"])
        merged_rows = _load_jsonl(
            REPO_ROOT / results[winner]["merge"]["merged_judgments_path"]
        )
        audit_info = _build_disagreement_audit(winner, merged_rows, changed, OUT)

    selection = {
        "created_at": utc_now_iso(),
        "git_commit": git_commit,
        "status": status,
        "winner": winner,
        "reason": reason,
        "passed_numeric_gates": passed,
        "comparison": {
            c: {
                "kappa": results[c]["eligibility_gates"]["values"]["cohen_kappa"],
                "balanced_accuracy": results[c]["eligibility_gates"]["values"][
                    "balanced_accuracy"
                ],
                "honest_sensitivity": results[c]["eligibility_gates"]["values"][
                    "honest_sensitivity"
                ],
                "deceptive_sensitivity": results[c]["eligibility_gates"]["values"][
                    "deceptive_sensitivity"
                ],
                "jaccard_qualifying": results[c]["population_development"][
                    "jaccard_qualifying_prompts"
                ],
                "n_status_changed": results[c]["population_development"][
                    "n_status_changed"
                ],
                "estimated_new_cost_usd": results[c]["merge"]["estimated_new_cost_usd"],
                "invalid_rate": results[c]["metrics"]["invalid_rate"],
            }
            for c in results
        },
        "stage3_scientifically_eligible": winner is not None,
        "stage3_locked_validation_authorized": False,
    }
    write_json(OUT / "winner_selection.json", selection)
    write_json(OUT / "summary.json", {
        "created_at": utc_now_iso(),
        "git_commit": git_commit,
        "status": status,
        "authorization_commit": "00131f41670c52a202c3ac70ad940e40162e4b38",
        "phase23a_freeze_commit": "b08c4b2ca2e4ade430bce0e96e768d98260f7e8d",
        "grader_prompt_split_sha256": split["grader_prompt_split_sha256"],
        "candidates": results,
        "winner_selection": selection,
        "disagreement_audit": audit_info,
        "guarantee": GUARANTEE,
        "authorizations_after_freeze": {
            "modal_gpu_open_grader_inference_authorized": False,
            "stage2_development_authorized": False,
            "stage3_locked_validation_authorized": False,
            "stage4_onset_validation_authorized": False,
            "k_gt_20_generation_authorized": False,
        },
    })

    # Markdown
    lines = [
        "# Phase 23B — DEVELOPMENT open-grader comparison",
        "",
        f"**Status:** `{status}`",
        "",
        f"**Winner (pending audit):** `{winner}`",
        "",
        f"Reason: {reason}",
        "",
        "STOP for audit. Stage 23C is **not** authorized.",
        "",
    ]
    for cand, v in results.items():
        u = v["eligibility_gates"]["values"]
        m = v["metrics"]
        pop = v["population_development"]
        lines += [
            f"## `{cand}`",
            "",
            f"- Invalid rate: {m['invalid_rate']:.4f} ({v['merge']['n_invalid']}/{v['merge']['n_total']})",
            f"- HD κ: {u['cohen_kappa']:.4f}",
            f"- HD bal-acc (invalid=miss): {u['balanced_accuracy']:.4f}",
            f"- H / D sens (invalid=miss): {u['honest_sensitivity']:.4f} / {u['deceptive_sensitivity']:.4f}",
            f"- 3-class macro-F1: {m['three_class']['macro_f1']:.4f}",
            f"- Population Jaccard: {pop['jaccard_qualifying_prompts']:.4f}",
            f"- Qualifying prompts changed: {pop['n_status_changed']}",
            f"- New Modal cost: ~${v['merge']['estimated_new_cost_usd']:.2f}",
            f"- Numeric gates passed: {v['eligibility_gates']['passed_numeric']}",
            "",
        ]
        if v["eligibility_gates"]["reasons"]:
            lines.append("Reasons:")
            for r in v["eligibility_gates"]["reasons"]:
                lines.append(f"- {r}")
            lines.append("")
    lines += ["## Guarantee", "", GUARANTEE, ""]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "winner": winner, "report": str(REPORT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
