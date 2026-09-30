#!/usr/bin/env python3
"""Analyze Phase-23C locked-validation run; emit STOP report; consume Stage-23C auth."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase23_open_grader import (  # noqa: E402
    CANDIDATES,
    GUARANTEE,
    LOCKED_ACCEPTANCE,
    N_LOCKED_PROMPTS,
    PHASE21_ONSET,
    PHASE22B_ONSET,
    STATUS_23C_FAIL,
    STATUS_23C_PASS,
    compare_development_population_membership,
    compare_development_population_membership_label_only,
    compute_agreement_metrics,
    locked_acceptance_check,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

OUT = REPO_ROOT / "artifacts/phase23c_locked"
REPORT = REPO_ROOT / "reports/phase23c_locked.md"
CFG = REPO_ROOT / "configs/experiments/phase23_open_grader_validation.yaml"
EXPECTED_WINNER = "gemma4_31b_it"
EXPECTED_REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
EXPECTED_SPLIT_SHA = "6504b40b43153a7bd359c344efb6eee01afca887fe4978960d9d2e0d6fec3869"

AUTH_FALSE = {
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
    "prompt_changes_authorized": False,
    "population_threshold_changes_authorized": False,
    "open_grader_for_k60_authorized": False,
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


def _gate_row(name: str, thr: float, val: float) -> dict:
    return {
        "metric": name,
        "threshold": thr,
        "observed": val,
        "pass": bool(val >= thr),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--authorization-commit", required=True)
    args = ap.parse_args()

    run_dir = REPO_ROOT / "artifacts/runs" / args.run_id
    man = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = _load_jsonl(run_dir / "judgments.jsonl")
    judgments_sha = _sha_file(run_dir / "judgments.jsonl")

    if man.get("candidate") != EXPECTED_WINNER:
        raise SystemExit(f"STOP: candidate {man.get('candidate')} != {EXPECTED_WINNER}")
    if man.get("revision") != EXPECTED_REVISION:
        raise SystemExit("STOP: revision mismatch")
    if man.get("grader_split") != "locked_validation":
        raise SystemExit("STOP: grader_split != locked_validation")
    if man.get("grader_prompt_split_sha256") != EXPECTED_SPLIT_SHA:
        raise SystemExit("STOP: split SHA mismatch in run manifest")
    if len(rows) != N_LOCKED_PROMPTS * 20:
        raise SystemExit(f"STOP: n judgments {len(rows)} != 2220")
    if any(r.get("grader_split") != "locked_validation" for r in rows):
        raise SystemExit("STOP: DEVELOPMENT leak in judgments")
    cids = [r["continuation_id"] for r in rows]
    if len(set(cids)) != 2220:
        raise SystemExit("STOP: duplicate continuation IDs in judgments")

    split = json.loads(
        (
            REPO_ROOT / "data/processed/phase23_open_grader/grader_prompt_split.json"
        ).read_text(encoding="utf-8")
    )
    locked_pids = set(split["locked_validation_prompt_ids"])
    if {r["prompt_id"] for r in rows} != locked_pids:
        raise SystemExit("STOP: judgment prompt set != frozen locked split")

    metrics = compute_agreement_metrics(rows)
    acc = locked_acceptance_check(metrics)
    u = metrics["unambiguous_honest_deceptive"]
    gates = [
        _gate_row("cohen_kappa", LOCKED_ACCEPTANCE["min_kappa"], u["cohen_kappa"]),
        _gate_row(
            "balanced_accuracy",
            LOCKED_ACCEPTANCE["min_balanced_accuracy"],
            u["balanced_accuracy"],
        ),
        _gate_row(
            "honest_sensitivity",
            LOCKED_ACCEPTANCE["min_honest_sensitivity"],
            u["honest_sensitivity"],
        ),
        _gate_row(
            "deceptive_sensitivity",
            LOCKED_ACCEPTANCE["min_deceptive_sensitivity"],
            u["deceptive_sensitivity"],
        ),
    ]
    status = STATUS_23C_PASS if acc["passed"] else STATUS_23C_FAIL

    # Population diagnostics on LOCKED only (descriptive; not rejection gates)
    label_only = compare_development_population_membership_label_only(
        rows, expected_n_prompts=N_LOCKED_PROMPTS
    )
    p21 = _load_jsonl(REPO_ROOT / "artifacts/runs" / PHASE21_ONSET / "annotated.jsonl")
    p22 = _load_jsonl(REPO_ROOT / "artifacts/runs" / PHASE22B_ONSET / "annotated.jsonl")
    annotated_locked = [r for r in (p21 + p22) if r["prompt_id"] in locked_pids]
    if len(annotated_locked) != N_LOCKED_PROMPTS * 20:
        raise SystemExit(
            f"STOP: annotated locked {len(annotated_locked)} != 2220"
        )
    onset_gated = compare_development_population_membership(
        annotated_locked, rows, expected_n_prompts=N_LOCKED_PROMPTS
    )

    git_commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()

    stage4_eligible = bool(acc["passed"])
    authorizations_after = dict(AUTH_FALSE)
    summary = {
        "created_at": utc_now_iso(),
        "status": status,
        "authorization_commit": args.authorization_commit,
        "analysis_git_commit": git_commit,
        "inference_git_commit": man.get("git_commit"),
        "run_id": args.run_id,
        "candidate": EXPECTED_WINNER,
        "candidate_spec": CANDIDATES[EXPECTED_WINNER],
        "hf_id": man.get("hf_id"),
        "revision": man.get("revision"),
        "grader_prompt_split_sha256": EXPECTED_SPLIT_SHA,
        "design_sha256": man.get("design_sha256"),
        "judgments_sha256": judgments_sha,
        "manifest": man,
        "n_requested": 2220,
        "n_completed": len(rows),
        "n_valid": metrics["n_valid"],
        "n_invalid": metrics["n_rows"] - metrics["n_valid"],
        "invalid_rate": metrics["invalid_rate"],
        "metrics": metrics,
        "locked_acceptance": {
            **acc,
            "gates": gates,
            "thresholds": LOCKED_ACCEPTANCE,
        },
        "population_metric_primary": "label_only_mixed_behavior_population",
        "population_metric_secondary": "reference_onset_gated_sensitivity_analysis",
        "population_label_only": label_only,
        "population_reference_onset_gated_sensitivity": onset_gated,
        "population_note": (
            "Population diagnostics on the LOCKED split are descriptive robustness "
            "results only. They are not frozen rejection gates and no post-hoc "
            "Jaccard threshold was applied."
        ),
        "stage4_scientifically_eligible": stage4_eligible,
        "stage4_onset_validation_authorized": False,
        "authorizations_after_freeze": authorizations_after,
        "guarantee": GUARANTEE,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "summary.json", summary)
    write_json(
        OUT / "freeze.json",
        {
            "status": status,
            "authorization_commit": args.authorization_commit,
            "analysis_git_commit": git_commit,
            "inference_git_commit": man.get("git_commit"),
            "run_id": args.run_id,
            "candidate": EXPECTED_WINNER,
            "revision": EXPECTED_REVISION,
            "grader_prompt_split_sha256": EXPECTED_SPLIT_SHA,
            "judgments_sha256": judgments_sha,
            "locked_acceptance": summary["locked_acceptance"],
            "metrics_hd": {
                **u,
                "n": u.get("n_reference_hd"),
            },
            "metrics_three_class": metrics.get("three_class"),
            "invalid_rate": metrics["invalid_rate"],
            "population_label_only": {
                "gpt_qualifying": label_only["reference"]["n_qualifying_prompts"],
                "open_qualifying": label_only["open_grader"]["n_qualifying_prompts"],
                "intersection": label_only["overlap_n_prompts"],
                "union": label_only["union_n_prompts"],
                "jaccard": label_only["jaccard_qualifying_prompts"],
                "n_status_changed": label_only["n_status_changed"],
                "agreement_rate": label_only["qualification_status_agreement_rate"],
                "gpt_only": label_only["only_reference_qualifying"],
                "open_only": label_only["only_open_grader_qualifying"],
            },
            "population_onset_gated_sensitivity": {
                "gpt_qualifying": onset_gated["reference"]["n_qualifying_prompts"],
                "open_hybrid_qualifying": onset_gated["open_grader"][
                    "n_qualifying_prompts"
                ],
                "jaccard": onset_gated["jaccard_qualifying_prompts"],
                "n_status_changed": onset_gated["n_status_changed"],
                "disclaimer": onset_gated["disclaimer"],
            },
            "authorizations_after_freeze": authorizations_after,
            "stage4_scientifically_eligible": stage4_eligible,
            "stage4_onset_validation_authorized": False,
            "open_grader_for_k60_authorized": False,
        },
    )

    # Consume Stage-23C authorization in config (preserve YAML structure)
    text = CFG.read_text(encoding="utf-8")
    # Status line at top
    import re

    text = re.sub(
        r"^status:.*$",
        f"status: {status}",
        text,
        count=1,
        flags=re.M,
    )
    for key in (
        "modal_gpu_open_grader_inference_authorized",
        "stage3_locked_validation_authorized",
        "stage2_development_authorized",
        "stage4_onset_validation_authorized",
    ):
        text = re.sub(
            rf"^(\s*{key}:\s*).*$",
            r"\1false",
            text,
            flags=re.M,
        )
    # Upsert phase23c_locked block before authorizations if missing/replace
    block = (
        "phase23c_locked:\n"
        "  freeze_artifact: artifacts/phase23c_locked/freeze.json\n"
        f"  status: {status}\n"
        f"  run_id: {args.run_id}\n"
        f"  candidate: {EXPECTED_WINNER}\n"
        f"  judgments_sha256: {judgments_sha}\n"
        f"  authorization_commit: {args.authorization_commit}\n"
        f"  stage4_scientifically_eligible: {str(stage4_eligible).lower()}\n"
        "  stage4_onset_validation_authorized: false\n"
        "\n"
    )
    if "phase23c_locked:" in text:
        text = re.sub(
            r"phase23c_locked:.*?(\nauthorizations:)",
            block + r"\1",
            text,
            count=1,
            flags=re.S,
        )
    else:
        text = text.replace("\nauthorizations:", "\n" + block + "authorizations:", 1)
    text = re.sub(
        r"^notes: >$.*?(?=\Z|\n[a-z_])",
        "notes: >\n"
        f"  Stage 23C complete ({status}). Authorization consumed. "
        "Stage 23D / K>20 / physiology remain unauthorized.\n",
        text,
        count=1,
        flags=re.M | re.S,
    )
    CFG.write_text(text, encoding="utf-8")

    # Report
    lo = label_only
    og = onset_gated
    lines = [
        "# Phase 23C — LOCKED validation (Gemma-4-31B-IT)",
        "",
        f"**Status:** `{status}`",
        "",
        f"**Decision:** `{'PASS' if acc['passed'] else 'FAIL'}`",
        "",
        "## Provenance",
        "",
        f"- Authorization commit: `{args.authorization_commit}`",
        f"- Inference commit: `{man.get('git_commit')}`",
        f"- Analysis commit: `{git_commit}`",
        f"- Run ID: `{args.run_id}`",
        f"- Candidate: `{EXPECTED_WINNER}`",
        f"- Model: `{man.get('hf_id')}` @ `{man.get('revision')}`",
        f"- Split hash: `{EXPECTED_SPLIT_SHA}`",
        f"- Judgments SHA256: `{judgments_sha}`",
        f"- Engine: `{man.get('engine')}` · dtype `{man.get('dtype')}`",
        f"- GPU: `{man.get('cost', {}).get('gpu_type')}`",
        f"- n: {len(rows)} completed / 2220 requested "
        f"(valid {metrics['n_valid']}, invalid {metrics['n_rows'] - metrics['n_valid']})",
        f"- Wall seconds: {man.get('cost', {}).get('wall_seconds')}",
        f"- Estimated cost USD: {man.get('cost', {}).get('estimated_cost_usd')}",
        "",
        "## Frozen locked metrics",
        "",
        f"- Invalid rate: {metrics['invalid_rate']:.6f}",
        f"- Three-class accuracy: {metrics['three_class']['accuracy']:.6f}",
        f"- Three-class macro-F1: {metrics['three_class']['macro_f1']:.6f}",
        f"- Three-class κ: {metrics['three_class']['cohen_kappa']:.6f}",
        f"- H/D n: {u['n_reference_hd']}",
        f"- H/D κ: {u['cohen_kappa']:.6f}",
        f"- Balanced accuracy: {u['balanced_accuracy']:.6f}",
        f"- Honest sensitivity: {u['honest_sensitivity']:.6f}",
        f"- Deceptive sensitivity: {u['deceptive_sensitivity']:.6f}",
        "",
        "## Frozen acceptance gates",
        "",
        "| Metric | Threshold | Observed | Pass |",
        "|---|---:|---:|---|",
    ]
    for g in gates:
        lines.append(
            f"| {g['metric']} | {g['threshold']:.2f} | {g['observed']:.6f} | "
            f"{'PASS' if g['pass'] else 'FAIL'} |"
        )
    lines += [
        "",
        "## Population diagnostics (descriptive only — not rejection gates)",
        "",
        "### Primary — label-only mixed-behavior population",
        "",
        f"- GPT qualifying: {lo['reference']['n_qualifying_prompts']}",
        f"- Gemma qualifying: {lo['open_grader']['n_qualifying_prompts']}",
        f"- Intersection: {lo['overlap_n_prompts']}",
        f"- Union: {lo['union_n_prompts']}",
        f"- Jaccard: {lo['jaccard_qualifying_prompts']:.6f}",
        f"- Changed prompts: {lo['n_status_changed']}",
        f"- GPT-only: {lo['only_reference_qualifying']}",
        f"- Gemma-only: {lo['only_open_grader_qualifying']}",
        f"- Qualification-status agreement: "
        f"{lo['qualification_status_agreement_rate']:.6f}",
        "",
        "### Secondary — reference-onset-gated sensitivity",
        "",
        f"- GPT qualifying: {og['reference']['n_qualifying_prompts']}",
        f"- Open hybrid qualifying: {og['open_grader']['n_qualifying_prompts']}",
        f"- Jaccard: {og['jaccard_qualifying_prompts']:.6f}",
        f"- Changed prompts: {og['n_status_changed']}",
        "",
        "> Population Jaccard is **not** a frozen pass/fail threshold.",
        "",
        "## Decision",
        "",
        f"`{status}`",
        "",
        (
            "Gemma passed confirmatory locked label validation. "
            "Stage 23D onset validation remains a separate unauthorized gate."
            if acc["passed"]
            else "Gemma failed confirmatory locked validation under the frozen "
            "contract. STOP. No prompt/threshold/revision retuning; no locked-set "
            "winner search."
        ),
        "",
        "## Authorization state (consumed)",
        "",
    ]
    for k, v in authorizations_after.items():
        lines.append(f"- `{k}`: **{str(v).lower()}**")
    lines += [
        "",
        f"- `stage4_scientifically_eligible`: **{str(stage4_eligible).lower()}**",
        "- `stage4_onset_validation_authorized`: **false**",
        "",
        "## Guarantee",
        "",
        GUARANTEE,
        "",
        "STAGE 23C WAS RUN ONCE AS A CONFIRMATORY LOCKED VALIDATION USING THE "
        "FROZEN GEMMA-4-31B-IT GRADER, FROZEN MODEL REVISION, FROZEN PHASE-23 "
        "PROMPT, FROZEN 111-PROMPT / 2,220-RESPONSE LOCKED SPLIT, AND FROZEN "
        "ACCEPTANCE THRESHOLDS. NO MISTRAL RESPONSES WERE GENERATED. NO OPENAI "
        "API CALLS, ONSET VALIDATION, K>20 GENERATION, ACTIVATION EXTRACTION, "
        "PROBE FITTING, PHYSIOLOGY, PROMPT TUNING, THRESHOLD TUNING, OR "
        "LOCKED-SET-DRIVEN MODEL SELECTION WAS PERFORMED.",
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": status,
                "passed": acc["passed"],
                "judgments_sha256": judgments_sha,
                "report": str(REPORT.relative_to(REPO_ROOT)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
