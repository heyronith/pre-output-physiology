#!/usr/bin/env python3
"""Evaluate Phase 5D locked behavior as secondary sensitivity only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase5_behavior import evaluate_row  # noqa: E402
from pre_output_physiology.phase5_locked import (  # noqa: E402
    EXPECTED_LOCKED_PAIR_SHA256,
    FROZEN_CANDIDATE,
    LOCKED_FAMILIES,
    Phase5FrozenProbe,
)
from pre_output_physiology.phase5_probes import (  # noqa: E402
    auroc_with_paired_bootstrap,
    paired_s3_minus_s2,
)
from pre_output_physiology.phase5_split import sha_sorted_ids  # noqa: E402
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

N_EXPECTED = 640
EXPECTED_FIRST_TOKEN_ID = 12107


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior-run-dir", required=True)
    parser.add_argument("--extract-run-dir", required=True)
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "artifacts/phase5d_locked_test"),
    )
    parser.add_argument(
        "--report-path",
        default=str(REPO_ROOT / "reports/phase5d_locked_test.md"),
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    primary_path = out_dir / "locked_primary_physiology.json"
    if not primary_path.is_file():
        raise SystemExit("primary physiology missing — refuse sensitivity-first")
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    if primary.get("behavior_conditioned") is not False:
        raise SystemExit("primary must remain unconditioned")

    beh_dir = Path(args.behavior_run_dir)
    rows = _load_jsonl(beh_dir / "locked_outputs.jsonl")
    if len(rows) != N_EXPECTED:
        raise SystemExit(f"expected {N_EXPECTED}")
    if any(r["family"] not in LOCKED_FAMILIES for r in rows):
        raise SystemExit("non-locked family")

    eval_rows = [
        evaluate_row(r, expected_first_token_id=EXPECTED_FIRST_TOKEN_ID) for r in rows
    ]
    s2_ok = {
        r["base_scenario_id"]
        for r in eval_rows
        if r["condition_id"] == "S2_strategic_honesty" and r["behaviorally_valid"]
    }
    s3_ok = {
        r["base_scenario_id"]
        for r in eval_rows
        if r["condition_id"] == "S3_strategic_deception" and r["behaviorally_valid"]
    }
    paired = sorted(s2_ok & s3_ok)
    n_s2 = sum(
        1
        for r in eval_rows
        if r["condition_id"] == "S2_strategic_honesty" and r["behaviorally_valid"]
    )
    n_s3 = sum(
        1
        for r in eval_rows
        if r["condition_id"] == "S3_strategic_deception" and r["behaviorally_valid"]
    )

    # Sensitivity physiology on behavior-valid pairs only (secondary).
    sens_metrics: dict | None = None
    sens_paired: dict | None = None
    if paired:
        extract_dir = Path(args.extract_run_dir)
        meta = json.loads(
            (extract_dir / "extraction_meta.json").read_text(encoding="utf-8")
        )
        tens = load_file(str(extract_dir / "activations_l12_k1.safetensors"))
        acts = np.asarray(tens["activations_l12_k1"], dtype=np.float32)
        probe = Phase5FrozenProbe(
            REPO_ROOT / FROZEN_CANDIDATE["probe_artifact"],
            expected_sha256=primary["probe_sha256_verified"],
        )
        prompts = {
            r["example_id"]: r
            for r in _load_jsonl(
                REPO_ROOT / "data/processed/phase5_design/final_candidate_prompts.jsonl"
            )
            if r.get("pool") == "locked"
        }
        sens_idx = []
        for i, eid in enumerate(meta["example_ids"]):
            p = prompts[eid]
            if p["base_scenario_id"] in set(paired):
                sens_idx.append(i)
        scores = probe.decision_scores(acts[sens_idx])
        y = np.asarray(
            [
                1
                if prompts[meta["example_ids"][i]]["condition_id"]
                == "S3_strategic_deception"
                else 0
                for i in sens_idx
            ],
            dtype=int,
        )
        groups = np.asarray(
            [prompts[meta["example_ids"][i]]["base_scenario_id"] for i in sens_idx]
        )
        sens_metrics = auroc_with_paired_bootstrap(y, scores, groups)
        sens_paired = paired_s3_minus_s2(
            [prompts[meta["example_ids"][i]]["base_scenario_id"] for i in sens_idx],
            [prompts[meta["example_ids"][i]]["condition_id"] for i in sens_idx],
            scores,
        )
        sens_metrics["paired_score_delta"] = sens_paired

    beh_man = json.loads((beh_dir / "locked_behavior_manifest.json").read_text())
    sensitivity = {
        "created_at": utc_now_iso(),
        "role": "secondary_sensitivity_only",
        "primary_population_unchanged": True,
        "primary_pair_ids_sha256": EXPECTED_LOCKED_PAIR_SHA256,
        "behavior_run_id": beh_man["run_id"],
        "n_s2_valid": n_s2,
        "n_s3_valid": n_s3,
        "n_paired_valid": len(paired),
        "paired_valid_ids_sha256": sha_sorted_ids(paired),
        "physiology_on_behavior_valid_pairs": sens_metrics,
        "does_not_replace_primary": True,
    }
    write_json(out_dir / "locked_behavior_sensitivity.json", sensitivity)
    write_json(
        out_dir / "locked_behavior_manifest.json",
        {
            **{k: v for k, v in beh_man.items() if k != "raw_outputs_path"},
            "raw_outputs_gitignored": True,
            "raw_outputs_path_gitignored": beh_man.get("raw_outputs_path"),
        },
    )

    # Append sensitivity section to report without rewriting primary numbers.
    report = Path(args.report_path)
    if sens_metrics is not None and sens_paired is not None:
        auroc_line = (
            f"- Physiology AUROC on behavior-valid pairs: `{sens_metrics['auroc']:.6f}` "
            f"(95% CI `{sens_metrics['auroc_ci_low']:.6f}`–"
            f"`{sens_metrics['auroc_ci_high']:.6f}`)"
        )
        delta_line = (
            f"- Paired S3−S2 Δ (valid subset): `{sens_paired['paired_mean_diff']:.6f}`"
        )
    else:
        auroc_line = "- Physiology AUROC on behavior-valid pairs: `n/a` (no paired-valid)"
        delta_line = "- Paired S3−S2 Δ (valid subset): `n/a`"
    extra = [
        "",
        "## Behavior sensitivity (secondary; post-primary)",
        "",
        f"- Behavior run ID: `{beh_man['run_id']}`",
        f"- S2 valid: `{n_s2}` / 320",
        f"- S3 valid: `{n_s3}` / 320",
        f"- Paired-valid N: `{len(paired)}`",
        auroc_line,
        delta_line,
        "",
        "Primary locked population was **not** conditioned on these behavior results.",
        "",
    ]
    text = report.read_text(encoding="utf-8")
    if "## Behavior sensitivity" not in text:
        # Insert before guarantees
        marker = "## Guarantees"
        if marker in text:
            text = text.replace(marker, "\n".join(extra) + marker)
        else:
            text = text + "\n".join(extra)
        report.write_text(text, encoding="utf-8")

    print(
        json.dumps(
            {
                "n_s2_valid": n_s2,
                "n_s3_valid": n_s3,
                "n_paired_valid": len(paired),
                "sens_auroc": None if sens_metrics is None else sens_metrics["auroc"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
