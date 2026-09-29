#!/usr/bin/env python3
"""Analyze Phase-23A pilot judgments; emit STOP report (no Stage 2)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase23_open_grader import (  # noqa: E402
    CANDIDATES,
    GUARANTEE,
    PILOT_ELIMINATION,
    STATUS_23A,
    compute_agreement_metrics,
    pilot_elimination_check,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

OUT = REPO_ROOT / "artifacts/phase23a_pilot"
REPORT = REPO_ROOT / "reports/phase23a_pilot.md"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="phase23a_pilot_* run IDs (one per candidate)",
    )
    args = ap.parse_args()

    by_cand: dict[str, dict] = {}
    for run_id in args.runs:
        run_dir = REPO_ROOT / "artifacts/runs" / run_id
        man = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        rows = [
            json.loads(x)
            for x in (run_dir / "judgments.jsonl").read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]
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
    summary = {
        "created_at": utc_now_iso(),
        "phase": "phase23a",
        "status": STATUS_23A,
        "pilot_elimination_thresholds": PILOT_ELIMINATION,
        "candidates": by_cand,
        "survivors": survivors,
        "eliminated": [c for c in by_cand if c not in survivors],
        "guarantee": GUARANTEE,
        "authorizations": {
            "stage2_development_authorized": False,
            "stage3_locked_validation_authorized": False,
            "stage4_onset_validation_authorized": False,
            "k_gt_20_generation_authorized": False,
            "mistral_roleplay_generation_authorized": False,
            "openai_api_authorized": False,
            "physiology_authorized": False,
        },
        "next_step": (
            "STOP for 23A audit. Do not authorize Stage-2 DEVELOPMENT until audit."
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "pilot_summary.json", summary)

    lines = [
        "# Phase 23A — open-grader 200-response pilot",
        "",
        f"**Status:** `{STATUS_23A}`",
        "",
        "STOP for audit. Stage-2 DEVELOPMENT is **not** authorized.",
        "",
        "## Pilot elimination thresholds (frozen)",
        "",
        f"- balanced accuracy ≥ {PILOT_ELIMINATION['min_balanced_accuracy']}",
        f"- κ ≥ {PILOT_ELIMINATION['min_kappa']}",
        f"- honest / deceptive sensitivity ≥ "
        f"{PILOT_ELIMINATION['min_honest_sensitivity']}",
        f"- invalid rate ≤ {PILOT_ELIMINATION['max_invalid_rate']}",
        "",
        "## Results",
        "",
    ]
    for cand, v in by_cand.items():
        m = v["metrics"]
        u = m["unambiguous_honest_deceptive"]
        t = m["three_class"]
        cost = v["manifest"]["cost"]
        lines += [
            f"### `{cand}` ({CANDIDATES[cand]['display_name']})",
            "",
            f"- Run: `{v['run_id']}`",
            f"- HF: `{v['manifest']['hf_id']}` @ `{v['manifest']['revision']}`",
            f"- GPU: {cost['gpu_type']} · wall {cost['wall_seconds']:.1f}s · "
            f"{cost['responses_per_sec']:.2f} resp/s · "
            f"~${cost['estimated_cost_usd']:.3f} · "
            f"${cost['cost_per_1000_judgments_usd']:.2f}/1k",
            f"- Invalid rate: {m['invalid_rate']:.4f}",
            f"- 3-class accuracy / macro-F1 / κ: "
            f"{t['accuracy']:.4f} / {t['macro_f1']:.4f} / {t['cohen_kappa']:.4f}",
            f"- HD balanced acc / κ: "
            f"{u['balanced_accuracy']:.4f} / {u['cohen_kappa']:.4f}",
            f"- Honest / deceptive sensitivity: "
            f"{u['honest_sensitivity']:.4f} / {u['deceptive_sensitivity']:.4f}",
            f"- Survives to DEVELOPMENT: **{v['survives_to_development']}**",
        ]
        if v["elimination"]["reasons"]:
            lines.append("- Elimination reasons:")
            for r in v["elimination"]["reasons"]:
                lines.append(f"  - {r}")
        lines.append("")
        # confusion
        lines.append("Confusion (rows=GPT-4o, cols=open):")
        lines.append("")
        labels = t["labels"]
        header = "| ref \\ pred | " + " | ".join(labels) + " |"
        lines.append(header)
        lines.append("|---|" + "|".join(["---:" for _ in labels]) + "|")
        for lab in labels:
            row = t["confusion"].get(lab, {})
            lines.append(
                "| "
                + lab
                + " | "
                + " | ".join(str(row.get(p, 0)) for p in labels)
                + " |"
            )
        lines.append("")

    total_cost = sum(
        float(v["manifest"]["cost"]["estimated_cost_usd"]) for v in by_cand.values()
    )
    lines += [
        f"**Survivors:** {survivors}",
        "",
        "## STOP audit fields",
        "",
        f"- Design SHA: `40fecca062757f489b1b7e1661e0728c9c52f205b9013e6dfc9b1b34a7c45cff`",
        f"- Grader split SHA: `6504b40b43153a7bd359c344efb6eee01afca887fe4978960d9d2e0d6fec3869`",
        f"- Pilot IDs SHA: `6e412617f7a4e461500880f1420be1c05c8453b433a96ac3d55a9fccb8f02533`",
        f"- Inference: T=0, do_sample=False, max_new_tokens=256, batch=4, "
        f"engine=transformers_generate_temp0_batch4",
        f"- Estimated Modal cost (23A total): ~${total_cost:.3f}",
        "",
        "## Authorization state",
        "",
        "- `stage2_development_authorized`: **false** (awaiting 23A audit)",
        "- `stage3_locked_validation_authorized`: false",
        "- `stage4_onset_validation_authorized`: false",
        "- `k_gt_20_generation_authorized`: false",
        "- `mistral_roleplay_generation_authorized`: false",
        "- `openai_*_api_authorized`: false",
        "- `physiology_authorized`: false",
        "",
        "## Guarantee",
        "",
        GUARANTEE,
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": STATUS_23A, "survivors": survivors, "report": str(REPORT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
