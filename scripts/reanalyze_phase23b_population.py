#!/usr/bin/env python3
"""Zero-GPU Phase-23B population reanalysis (label-only primary + onset-gated sensitivity)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase23_open_grader import (  # noqa: E402
    GUARANTEE,
    N_DEV_PROMPTS,
    PHASE21_ONSET,
    PHASE22B_ONSET,
    compare_development_population_membership,
    compare_development_population_membership_label_only,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

OUT = REPO_ROOT / "artifacts/phase23b_development"
REPORT = REPO_ROOT / "reports/phase23b_development.md"
STATUS = "phase23b_population_reanalysis_pending_audit"

EXPECTED_SHA = {
    "gemma4_31b_it": "c88f37349e587e7a9c071ebc276cf7f3d039e61a844557d18f7e7393eb8c4a03",
    "qwen35_27b": "22ddfd78eeb3e03dd9a89ae0264b4a462e15d13ae6c02e2132509d2c4d668dbc",
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


def main() -> int:
    summary_path = OUT / "summary.json"
    freeze_path = OUT / "freeze.json"
    if not summary_path.is_file():
        raise SystemExit("missing phase23b summary.json")

    old = json.loads(summary_path.read_text(encoding="utf-8"))
    git_commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()

    split = json.loads(
        (
            REPO_ROOT / "data/processed/phase23_open_grader/grader_prompt_split.json"
        ).read_text(encoding="utf-8")
    )
    dev_pids = set(split["development_prompt_ids"])
    p21 = _load_jsonl(REPO_ROOT / "artifacts/runs" / PHASE21_ONSET / "annotated.jsonl")
    p22 = _load_jsonl(REPO_ROOT / "artifacts/runs" / PHASE22B_ONSET / "annotated.jsonl")
    annotated_dev = [r for r in (p21 + p22) if r["prompt_id"] in dev_pids]
    if len(annotated_dev) != N_DEV_PROMPTS * 20:
        raise SystemExit(f"annotated DEV {len(annotated_dev)} != {N_DEV_PROMPTS * 20}")

    for cand in ("gemma4_31b_it", "qwen35_27b"):
        merged_path = OUT / "merged" / cand / "judgments_5200.jsonl"
        got = _sha_file(merged_path)
        if got != EXPECTED_SHA[cand]:
            raise SystemExit(f"STOP: {cand} merged SHA {got} != {EXPECTED_SHA[cand]}")
        rows = _load_jsonl(merged_path)
        label_only = compare_development_population_membership_label_only(rows)
        onset_gated = compare_development_population_membership(annotated_dev, rows)
        old["candidates"][cand]["population_label_only"] = label_only
        old["candidates"][cand]["population_reference_onset_gated_sensitivity"] = (
            onset_gated
        )
        # Keep legacy key but mark hybrid so it is not primary
        old["candidates"][cand]["population_development"] = {
            **onset_gated,
            "deprecated_as_primary": True,
            "see_instead": "population_label_only",
        }
        old["candidates"][cand]["population_materiality_note"] = (
            f"PRIMARY label-only: {label_only['n_status_changed']} DEV prompts change "
            f"qualification (Jaccard {label_only['jaccard_qualifying_prompts']:.4f}). "
            f"SECONDARY onset-gated hybrid sensitivity: "
            f"{onset_gated['n_status_changed']} changes "
            f"(Jaccard {onset_gated['jaccard_qualifying_prompts']:.4f}). "
            "No post-hoc numeric population threshold applied."
        )

    # Winner selection: numeric gates unchanged; population interpretation uses label-only
    ws = old["winner_selection"]
    ws["status"] = STATUS
    ws["population_metric_primary"] = "label_only_mixed_behavior_population"
    ws["population_metric_secondary"] = "reference_onset_gated_sensitivity_analysis"
    ws["comparison"] = {
        c: {
            **{k: v for k, v in ws["comparison"][c].items() if k not in (
                "jaccard_qualifying",
                "n_status_changed",
            )},
            "label_only_jaccard": old["candidates"][c]["population_label_only"][
                "jaccard_qualifying_prompts"
            ],
            "label_only_n_status_changed": old["candidates"][c]["population_label_only"][
                "n_status_changed"
            ],
            "onset_gated_sensitivity_jaccard": old["candidates"][c][
                "population_reference_onset_gated_sensitivity"
            ]["jaccard_qualifying_prompts"],
            "onset_gated_sensitivity_n_status_changed": old["candidates"][c][
                "population_reference_onset_gated_sensitivity"
            ]["n_status_changed"],
            "estimated_new_cost_usd": ws["comparison"][c]["estimated_new_cost_usd"],
            "invalid_rate": ws["comparison"][c]["invalid_rate"],
        }
        for c in ("gemma4_31b_it", "qwen35_27b")
    }
    ws["reason"] = (
        "sole numeric gate passer: gemma4_31b_it; population equivalence now reported "
        "via label-only primary metric (onset-gated retained as sensitivity only)"
    )
    ws["stage3_locked_validation_authorized"] = False
    ws["created_at"] = utc_now_iso()
    ws["reanalysis_git_commit"] = git_commit

    old["status"] = STATUS
    old["created_at"] = utc_now_iso()
    old["reanalysis_git_commit"] = git_commit
    old["winner_selection"] = ws
    old["population_reanalysis"] = {
        "note": (
            "Label-only (≥2 honest + ≥2 deceptive, no onset) is the primary "
            "population-equivalence comparison. Onset-gated hybrid Jaccard is "
            "sensitivity-only and must not be presented as primary."
        ),
        "merged_judgment_sha256": EXPECTED_SHA,
    }

    write_json(summary_path, old)
    write_json(OUT / "winner_selection.json", ws)

    freeze = json.loads(freeze_path.read_text(encoding="utf-8")) if freeze_path.is_file() else {}
    freeze.update(
        {
            "status": STATUS,
            "reanalysis_git_commit": git_commit,
            "population_metric_primary": "label_only_mixed_behavior_population",
            "population_metric_secondary": "reference_onset_gated_sensitivity_analysis",
            "merged_judgment_sha256": EXPECTED_SHA,
            "winner": "gemma4_31b_it",
            "winner_reason": ws["reason"],
            "authorizations_after_freeze": {
                "modal_gpu_open_grader_inference_authorized": False,
                "stage2_development_authorized": False,
                "stage3_locked_validation_authorized": False,
                "stage4_onset_validation_authorized": False,
                "mistral_roleplay_generation_authorized": False,
                "openai_grading_api_authorized": False,
                "openai_onset_api_authorized": False,
                "activation_extraction_authorized": False,
                "probe_fitting_authorized": False,
                "physiology_authorized": False,
                "k_gt_20_generation_authorized": False,
            },
        }
    )
    for cand in ("gemma4_31b_it", "qwen35_27b"):
        lo = old["candidates"][cand]["population_label_only"]
        og = old["candidates"][cand]["population_reference_onset_gated_sensitivity"]
        freeze.setdefault(cand, {})
        freeze[cand]["population_label_only"] = {
            "gpt_qualifying": lo["reference"]["n_qualifying_prompts"],
            "open_qualifying": lo["open_grader"]["n_qualifying_prompts"],
            "intersection": lo["overlap_n_prompts"],
            "union": lo["union_n_prompts"],
            "jaccard": lo["jaccard_qualifying_prompts"],
            "agreement_rate": lo["qualification_status_agreement_rate"],
            "n_status_changed": lo["n_status_changed"],
        }
        freeze[cand]["population_onset_gated_sensitivity"] = {
            "gpt_qualifying": og["reference"]["n_qualifying_prompts"],
            "open_hybrid_qualifying": og["open_grader"]["n_qualifying_prompts"],
            "jaccard": og["jaccard_qualifying_prompts"],
            "n_status_changed": og["n_status_changed"],
            "disclaimer": og["disclaimer"],
        }
    write_json(freeze_path, freeze)

    # Report
    lines = [
        "# Phase 23B — DEVELOPMENT open-grader comparison (population reanalysis)",
        "",
        f"**Status:** `{STATUS}`",
        "",
        "**Proposed Stage-23C winner (pending audit):** `gemma4_31b_it`",
        "",
        "Numeric response-level gates unchanged. Population equivalence corrected:",
        "label-only is primary; onset-gated hybrid is sensitivity-only.",
        "",
        "STOP for audit. Stage 23C is **not** authorized.",
        "",
        "## Primary population comparison (label-only)",
        "",
        "Rule: ≥2 honest and ≥2 deceptive labels per DEVELOPMENT prompt (20 responses).",
        "**No onset dependency.** Invalid open outputs → exclude.",
        "",
    ]
    for cand in ("gemma4_31b_it", "qwen35_27b"):
        lo = old["candidates"][cand]["population_label_only"]
        lines += [
            f"### `{cand}`",
            "",
            f"- GPT qualifying: {lo['reference']['n_qualifying_prompts']}",
            f"- Open qualifying: {lo['open_grader']['n_qualifying_prompts']}",
            f"- Intersection: {lo['overlap_n_prompts']}",
            f"- Union: {lo['union_n_prompts']}",
            f"- Jaccard: {lo['jaccard_qualifying_prompts']:.4f}",
            f"- Status agreement: {lo['qualification_status_agreement_rate']:.4f}",
            f"- Changed prompts: {lo['n_status_changed']}",
            f"- GPT-only: {lo['only_reference_qualifying']}",
            f"- Open-only: {lo['only_open_grader_qualifying']}",
            "",
        ]
    lines += [
        "## Secondary sensitivity analysis (reference-onset-gated hybrid)",
        "",
        "> Open-grader labels are combined with frozen GPT-4o onset annotations. "
        "This analysis is not a fully open-grader-defined population and is "
        "retained only as a sensitivity analysis.",
        "",
    ]
    for cand in ("gemma4_31b_it", "qwen35_27b"):
        og = old["candidates"][cand]["population_reference_onset_gated_sensitivity"]
        lines += [
            f"### `{cand}` (hybrid)",
            "",
            f"- GPT qualifying: {og['reference']['n_qualifying_prompts']}",
            f"- Open+GPT-onset qualifying: {og['open_grader']['n_qualifying_prompts']}",
            f"- Jaccard: {og['jaccard_qualifying_prompts']:.4f}",
            f"- Changed prompts: {og['n_status_changed']}",
            "",
        ]
    lines += [
        "## Response-level gates (unchanged)",
        "",
        "| Candidate | κ | bal-acc | H sens | D sens | numeric |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for cand in ("gemma4_31b_it", "qwen35_27b"):
        u = old["candidates"][cand]["eligibility_gates"]["values"]
        passed = old["candidates"][cand]["eligibility_gates"]["passed_numeric"]
        lines.append(
            f"| {cand} | {u['cohen_kappa']:.3f} | {u['balanced_accuracy']:.3f} | "
            f"{u['honest_sensitivity']:.3f} | {u['deceptive_sensitivity']:.3f} | "
            f"{'PASS' if passed else 'FAIL'} |"
        )
    lines += [
        "",
        "## Authorization state",
        "",
        "All Modal / Stage 2–4 / generation / physiology flags remain **false**.",
        "",
        "## Guarantee",
        "",
        GUARANTEE,
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": STATUS,
                "gemma_label_only": freeze["gemma4_31b_it"]["population_label_only"],
                "qwen_label_only": freeze["qwen35_27b"]["population_label_only"],
                "report": str(REPORT),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
