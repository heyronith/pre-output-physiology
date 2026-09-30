#!/usr/bin/env python3
"""Analyze Phase-23B/C/D grading or onset runs; emit STOP reports."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase23_open_grader import (  # noqa: E402
    CANDIDATES,
    DEV_ELIGIBILITY,
    GUARANTEE,
    LOCKED_ACCEPTANCE,
    PHASE21_ONSET,
    PHASE22B_ONSET,
    STATUS_23B,
    STATUS_23C_FAIL,
    STATUS_23C_PASS,
    compare_k20_population_membership,
    compute_agreement_metrics,
    development_eligibility_check,
    locked_acceptance_check,
    pilot_elimination_check,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(x)
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def _load_run(run_id: str) -> tuple[dict, list[dict]]:
    run_dir = REPO_ROOT / "artifacts/runs" / run_id
    man = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = _load_jsonl(run_dir / "judgments.jsonl")
    return man, rows


def analyze_pilot(run_ids: list[str]) -> dict:
    by_cand: dict[str, dict] = {}
    for run_id in run_ids:
        man, rows = _load_run(run_id)
        metrics = compute_agreement_metrics(rows)
        elim = pilot_elimination_check(metrics)
        cand = man["candidate"]
        by_cand[cand] = {
            "run_id": run_id,
            "manifest": man,
            "metrics": metrics,
            "elimination": elim,
            "survives_to_development": not elim["eliminated"],
        }
    survivors = [c for c, v in by_cand.items() if v["survives_to_development"]]
    return {
        "status": "phase23a_pilot_complete_awaiting_audit",
        "candidates": by_cand,
        "survivors": survivors,
        "authorizations": {"stage2_development_authorized": False},
    }


def analyze_development(run_ids: list[str]) -> dict:
    results = []
    for run_id in run_ids:
        man, rows = _load_run(run_id)
        metrics = compute_agreement_metrics(rows)
        elig = development_eligibility_check(metrics)
        p21 = _load_jsonl(
            REPO_ROOT / "artifacts/runs" / PHASE21_ONSET / "annotated.jsonl"
        )
        p22 = _load_jsonl(
            REPO_ROOT / "artifacts/runs" / PHASE22B_ONSET / "annotated.jsonl"
        )
        pop = compare_k20_population_membership(p21, p22, rows)
        results.append(
            {
                "run_id": run_id,
                "candidate": man["candidate"],
                "metrics": metrics,
                "development_eligibility": elig,
                "population": pop,
                "cost": man.get("cost"),
            }
        )
    eligible = [r for r in results if r["development_eligibility"]["eligible_for_locked"]]
    # Frozen tie-break: scientific agreement first (kappa), then population jaccard
    eligible.sort(
        key=lambda r: (
            -r["metrics"]["unambiguous_honest_deceptive"]["cohen_kappa"],
            -r["population"]["jaccard_qualifying_prompts"],
            r["cost"]["estimated_cost_usd"] if r.get("cost") else 0,
        )
    )
    winner = eligible[0]["candidate"] if eligible else None
    return {
        "status": STATUS_23B if winner else "phase23b_development_no_eligible_winner",
        "results": results,
        "eligible_for_locked": [r["candidate"] for r in eligible],
        "frozen_winner": winner,
        "thresholds": DEV_ELIGIBILITY,
        "authorizations": {
            # Winner implies scientific eligibility only — never auto-authorize Stage 23C.
            "stage3_scientifically_eligible": bool(winner),
            "stage3_locked_validation_authorized": False,
            "winner_frozen": winner,
        },
    }


def analyze_locked(run_id: str) -> dict:
    man, rows = _load_run(run_id)
    metrics = compute_agreement_metrics(rows)
    acc = locked_acceptance_check(metrics)
    p21 = _load_jsonl(REPO_ROOT / "artifacts/runs" / PHASE21_ONSET / "annotated.jsonl")
    p22 = _load_jsonl(
        REPO_ROOT / "artifacts/runs" / PHASE22B_ONSET / "annotated.jsonl"
    )
    pop = compare_k20_population_membership(p21, p22, rows)
    status = STATUS_23C_PASS if acc["passed"] else STATUS_23C_FAIL
    return {
        "status": status,
        "run_id": run_id,
        "candidate": man["candidate"],
        "metrics": metrics,
        "locked_acceptance": acc,
        "population": pop,
        "thresholds": LOCKED_ACCEPTANCE,
        "authorizations": {
            # Locked acceptance implies scientific eligibility for onset only —
            # never auto-authorize Stage 23D.
            "stage4_scientifically_eligible": bool(acc["passed"]),
            "stage4_onset_validation_authorized": False,
            "open_grader_for_k60_authorized": False,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage",
        choices=("23A", "23B", "23C"),
        required=True,
    )
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--winner-out", default=str(REPO_ROOT / "artifacts/phase23_winner.json"))
    args = ap.parse_args()

    git_commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()

    if args.stage == "23A":
        summary = analyze_pilot(args.runs)
        out_dir = REPO_ROOT / "artifacts/phase23a_pilot"
        report = REPO_ROOT / "reports/phase23a_pilot.md"
    elif args.stage == "23B":
        summary = analyze_development(args.runs)
        out_dir = REPO_ROOT / "artifacts/phase23b_development"
        report = REPO_ROOT / "reports/phase23b_development.md"
        if summary.get("frozen_winner"):
            write_json(
                Path(args.winner_out),
                {
                    "created_at": utc_now_iso(),
                    "git_commit": git_commit,
                    "winner": summary["frozen_winner"],
                    "candidate_spec": CANDIDATES[summary["frozen_winner"]],
                },
            )
    else:
        if len(args.runs) != 1:
            raise SystemExit("23C expects exactly one run id")
        summary = analyze_locked(args.runs[0])
        out_dir = REPO_ROOT / "artifacts/phase23c_locked"
        report = REPO_ROOT / "reports/phase23c_locked.md"

    summary["created_at"] = utc_now_iso()
    summary["git_commit"] = git_commit
    summary["guarantee"] = GUARANTEE
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "summary.json", summary)

    lines = [
        f"# Phase {args.stage} — open-grader validation",
        "",
        f"**Status:** `{summary['status']}`",
        "",
        f"**Commit:** `{git_commit}`",
        "",
    ]
    if args.stage == "23A":
        for cand, v in summary["candidates"].items():
            m = v["metrics"]
            u = m["unambiguous_honest_deceptive"]
            lines += [
                f"## {cand}",
                f"- Survives: {v['survives_to_development']}",
                f"- Invalid rate: {m['invalid_rate']:.4f}",
                f"- HD κ / bal acc: {u['cohen_kappa']:.4f} / {u['balanced_accuracy']:.4f}",
                "",
            ]
        lines.append(f"**Survivors:** {summary['survivors']}")
    elif args.stage == "23B":
        lines.append(f"**Frozen winner:** {summary.get('frozen_winner')}")
        for r in summary["results"]:
            u = r["metrics"]["unambiguous_honest_deceptive"]
            lines += [
                f"## {r['candidate']}",
                f"- κ: {u['cohen_kappa']:.4f} · bal acc: {u['balanced_accuracy']:.4f}",
                f"- Population jaccard: {r['population']['jaccard_qualifying_prompts']:.4f}",
                f"- Eligible locked: {r['development_eligibility']['eligible_for_locked']}",
                "",
            ]
    else:
        u = summary["metrics"]["unambiguous_honest_deceptive"]
        lines += [
            f"- Candidate: {summary['candidate']}",
            f"- κ: {u['cohen_kappa']:.4f} · bal acc: {u['balanced_accuracy']:.4f}",
            f"- Locked pass: {summary['locked_acceptance']['passed']}",
        ]
    lines += ["", "## Guarantee", "", GUARANTEE, ""]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "report": str(report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
