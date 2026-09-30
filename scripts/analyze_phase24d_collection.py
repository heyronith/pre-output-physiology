#!/usr/bin/env python3
"""Finalize Phase-24D report, hashes, and decision-log (no GPU / no TEST peek)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase24c_design import sha256_file  # noqa: E402
from pre_output_physiology.phase24d_collection import (  # noqa: E402
    GUARANTEE,
    assert_no_test_outcome_leakage,
)
from pre_output_physiology.provenance import write_json  # noqa: E402

OUT = REPO_ROOT / "artifacts/phase24d_collection"
SUMMARY = OUT / "summary.json"
REPORT = REPO_ROOT / "reports/phase24d_primary_live_collection.md"
DECISION_LOG = REPO_ROOT / "docs/decision_log.md"
CFG = REPO_ROOT / "configs/experiments/phase24d_primary_live_collection.yaml"


def main() -> int:
    if not SUMMARY.exists():
        raise SystemExit("STOP: summary.json missing — run Modal collection first")
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    ver = summary["phase24c_verification"]
    train = summary["train_yield"]
    val = summary["validation_yield"]
    yld = summary["realized_yield"]
    eng = summary["engineering_gates"]
    test_seal = summary["locked_test_sealed_summary"]
    status = summary["status"]

    train_table = "\n".join(
        f"| `{p['prompt_id']}` | {p['n_honest']} | {p['n_ambiguous']} | "
        f"{p['n_deceptive']} | {'yes' if p['has_ge2_h_and_ge2_d'] else 'no'} |"
        for p in train["per_prompt"]
    )
    val_table = "\n".join(
        f"| `{p['prompt_id']}` | {p['n_honest']} | {p['n_ambiguous']} | "
        f"{p['n_deceptive']} | {'yes' if p['has_ge2_h_and_ge2_d'] else 'no'} |"
        for p in val["per_prompt"]
    )

    report = f"""# Phase 24D — Primary live K=16 collection

**Status:** `{status}`

{GUARANTEE}

## Phase-24C verification

Verified: **{ver['verified']}** · selected_k check: **{ver['selected_k_check']['pass']}**

## Model / generation

- Mistral: `{summary['model']['model_id']}` @ `{summary['model']['revision']}`
- BF16 · sdpa · T={summary['model']['temperature']}
- max_new_tokens={summary['model']['max_new_tokens']} · batch=1
- Capture loop: code-equivalent to Phase 24B (`_generate_one` SHA match; canary not re-run)

## Schedule

- Reused Phase-24B: **{summary['n_reused_phase24b']}**
- New requested: **{summary['n_requested_new']}**
- New completed: **{summary['n_completed_new']}**
- Final total: **{summary['n_total_trajectories']}**
- New schedule SHA256: `{summary['new_schedule_sha256']}`
- Generated tokens (new): **{summary['total_generated_tokens_new']}**

## Integrity gates (528 new)

```json
{json.dumps(eng, indent=2)}
```

## Activations / logits

- Activations stored: {summary['activation_coordinates']}
- Logits stored: {summary['logit_storage']}
- New storage bytes: {summary['total_new_storage_bytes']}
- Mean bytes/traj: {summary['bytes_per_trajectory_mean']}

## Gemma grader

- `{summary['grader']['hf_id']}` @ `{summary['grader']['revision']}`
- Graded newly generated responses only; Phase-24B labels retained via provenance

## TRAIN realized yield

Totals H/A/D: **{train['totals']['honest']}** /
**{train['totals']['ambiguous']}** /
**{train['totals']['deceptive']}**

≥2H∧≥2D prompts: **{train['n_prompts_ge2h_ge2d']}** / 20
(gate ≥10: **{yld['train']['pass']}**)

| prompt | H | A | D | ≥2H≥2D |
|---|---|---|---|---|
{train_table}

## VALIDATION realized yield

Totals H/A/D: **{val['totals']['honest']}** /
**{val['totals']['ambiguous']}** /
**{val['totals']['deceptive']}**

≥2H∧≥2D prompts: **{val['n_prompts_ge2h_ge2d']}** / 8
(gate ≥4: **{yld['validation']['pass']}**)

| prompt | H | A | D | ≥2H≥2D |
|---|---|---|---|---|
{val_table}

## LOCKED TEST (sealed)

```json
{json.dumps(test_seal, indent=2)}
```

Behavioral outcomes for TEST are **not reported**.

## Runtime / cost

- Capture wall seconds: {summary['wall_seconds_capture']}
- Grade wall seconds: {summary['wall_seconds_grade']}
- GPU: {summary['gpu']}
- Estimated cost USD: {summary['estimated_cost_usd']}
- Peak memory bytes: {summary['peak_memory_bytes']}

## Authorizations after freeze

Live capture / Gemma grading / physiology / TEST scientific analysis: **false**.
"""
    if not assert_no_test_outcome_leakage(report):
        raise SystemExit("STOP: report would leak TEST outcomes")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")

    hashes = {
        "summary.json": sha256_file(str(SUMMARY)),
        "freeze.json": sha256_file(str(OUT / "freeze.json")),
        "schedule.json": sha256_file(str(OUT / "schedule.json")),
        "contract.json": sha256_file(str(OUT / "contract.json")),
        "config": sha256_file(str(CFG)),
        "report": sha256_file(str(REPORT)),
        "grades_new.json": sha256_file(str(OUT / "grades_new.json"))
        if (OUT / "grades_new.json").exists()
        else None,
        "TRAIN_labels.json": sha256_file(str(OUT / "by_split/TRAIN/labels.json")),
        "VALIDATION_labels.json": sha256_file(
            str(OUT / "by_split/VALIDATION/labels.json")
        ),
        "LOCKED_TEST_labels_SEALED.json": sha256_file(
            str(OUT / "by_split/LOCKED_TEST/labels_SEALED.json")
        ),
        "LOCKED_TEST_sealed_summary.json": sha256_file(
            str(OUT / "by_split/LOCKED_TEST/sealed_summary.json")
        ),
    }
    hashes = {k: v for k, v in hashes.items() if v is not None}
    write_json(OUT / "artifact_hashes.json", hashes)

    log = DECISION_LOG.read_text(encoding="utf-8")
    if "### D161 —" not in log:
        entry = (
            "\n### D161 — Phase 24D primary live K=16 collection\n\n"
            "- **Type:** **OUR RESEARCH DECISION**\n"
            "- **Decision:** Starting from Phase-24C freeze `c649203…`, Phase 24D "
            f"executed the frozen K=16 live-capture schedule (528 new + 96 reused). "
            f"Engineering integrity {'PASS' if eng['passed'] else 'FAIL'}; "
            f"TRAIN/VAL realized-yield {'PASS' if yld['passed'] else 'FAIL'}. "
            f"Status `{status}`. LOCKED TEST labels sealed. No physiology, AUROC, "
            "layer×token search, SAE, or causal analysis.\n"
            "- **Date:** 2026-09-30\n"
        )
        DECISION_LOG.write_text(log.rstrip() + "\n" + entry, encoding="utf-8")

    print(json.dumps({"status": status, "report": str(REPORT), "hashes": hashes}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
