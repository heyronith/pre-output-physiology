#!/usr/bin/env python3
"""Score Phase 5D locked activations with the frozen Phase-5C candidate probe.

Writes primary locked physiology results before any behavior generation.
Does not retrain, recalibrate, or reselect.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase5_locked import (  # noqa: E402
    BOOTSTRAP_SEED,
    CONFIRMATION_CRITERIA,
    EXPECTED_LOCKED_PAIR_SHA256,
    EXPECTED_LOCKED_PROMPT_TEXT_SHA256,
    FROZEN_CANDIDATE,
    LOCKED_FAMILIES,
    N_BOOTSTRAP,
    Phase5FrozenProbe,
)
from pre_output_physiology.phase5_probes import (  # noqa: E402
    auroc_with_paired_bootstrap,
    paired_s3_minus_s2,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

N_EXPECTED = 640
EXPECTED_PROBE_SHA256 = FROZEN_CANDIDATE["probe_sha256"]


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _evaluate_confirmation(metrics: dict[str, Any]) -> dict[str, Any]:
    ci_low = float(metrics["auroc_ci_low"])
    delta = metrics["paired_score_delta"]
    d_low = float(delta["paired_mean_diff_ci_low"])
    per_fam = metrics["per_family_auroc"]
    checks = {
        "overall_auroc_ci_low_gt_0_50": ci_low
        > CONFIRMATION_CRITERIA["overall_auroc_ci_low_gt"],
        "paired_delta_ci_entirely_gt_0": d_low
        > CONFIRMATION_CRITERIA["paired_delta_ci_entirely_gt"],
        "each_family_auroc_gt_0_50": all(
            float(v) > CONFIRMATION_CRITERIA["each_family_auroc_gt"]
            for v in per_fam.values()
        ),
    }
    return {
        "criteria": CONFIRMATION_CRITERIA,
        "checks": checks,
        "passed": all(checks.values()),
        "observed": {
            "auroc": metrics["auroc"],
            "auroc_ci_low": ci_low,
            "auroc_ci_high": metrics["auroc_ci_high"],
            "paired_mean_diff": delta["paired_mean_diff"],
            "paired_mean_diff_ci_low": d_low,
            "paired_mean_diff_ci_high": delta["paired_mean_diff_ci_high"],
            "per_family_auroc": per_fam,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "artifacts/phase5d_locked_test"),
    )
    parser.add_argument(
        "--report-path",
        default=str(REPO_ROOT / "reports/phase5d_locked_test.md"),
    )
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    man = json.loads((run_dir / "extraction_manifest.json").read_text(encoding="utf-8"))
    meta = json.loads((run_dir / "extraction_meta.json").read_text(encoding="utf-8"))
    freeze = json.loads(
        (
            REPO_ROOT / "artifacts/phase5d_locked_freeze/locked_population.json"
        ).read_text(encoding="utf-8")
    )
    if freeze["pair_ids_sha256"] != EXPECTED_LOCKED_PAIR_SHA256:
        raise SystemExit("locked pair hash changed")
    if man.get("locked_prompt_text_sha256") != EXPECTED_LOCKED_PROMPT_TEXT_SHA256:
        raise SystemExit("locked prompt hash changed")
    if man.get("other_layers_extracted") is not False:
        raise SystemExit("other layers were extracted")
    if man.get("k0_extracted") is not False:
        raise SystemExit("k0 was extracted")

    probe = Phase5FrozenProbe(
        REPO_ROOT / FROZEN_CANDIDATE["probe_artifact"],
        expected_sha256=EXPECTED_PROBE_SHA256,
    )
    tens = load_file(str(run_dir / "activations_l12_k1.safetensors"))
    acts = np.asarray(tens["activations_l12_k1"], dtype=np.float32)
    if acts.shape[0] != N_EXPECTED:
        raise SystemExit(f"expected {N_EXPECTED} rows")
    if int(np.asarray(tens["layer"]).ravel()[0]) != 12:
        raise SystemExit("activation layer != 12")

    example_ids = meta["example_ids"]
    prompts = {
        r["example_id"]: r
        for r in _load_jsonl(
            REPO_ROOT / "data/processed/phase5_design/final_candidate_prompts.jsonl"
        )
        if r.get("pool") == "locked" and r.get("split") == "final"
    }
    rows = []
    for i, eid in enumerate(example_ids):
        p = prompts[eid]
        if p["family"] not in LOCKED_FAMILIES:
            raise SystemExit("non-locked family in join")
        y = 1 if p["condition_id"] == "S3_strategic_deception" else 0
        rows.append(
            {
                "index": i,
                "example_id": eid,
                "base_scenario_id": p["base_scenario_id"],
                "condition_id": p["condition_id"],
                "family": p["family"],
                "y": y,
            }
        )

    scores = probe.decision_scores(acts)
    y = np.asarray([r["y"] for r in rows], dtype=int)
    groups = np.asarray([r["base_scenario_id"] for r in rows])
    base_ids = [r["base_scenario_id"] for r in rows]
    cond_ids = [r["condition_id"] for r in rows]
    families = [r["family"] for r in rows]

    overall = auroc_with_paired_bootstrap(
        y, scores, groups, n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED
    )
    paired = paired_s3_minus_s2(
        base_ids, cond_ids, scores, n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED
    )
    per_family = {}
    per_family_detail = {}
    for fam in LOCKED_FAMILIES:
        idx = [i for i, f in enumerate(families) if f == fam]
        yt = y[idx]
        ys = scores[idx]
        gt = groups[idx]
        fam_m = auroc_with_paired_bootstrap(
            yt, ys, gt, n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED
        )
        per_family[fam] = fam_m["auroc"]
        per_family_detail[fam] = fam_m

    metrics = {
        **overall,
        "paired_score_delta": paired,
        "per_family_auroc": per_family,
        "per_family_detail": per_family_detail,
        "n_pairs": 320,
        "n_rows": N_EXPECTED,
    }
    confirmation = _evaluate_confirmation(metrics)

    primary = {
        "created_at": utc_now_iso(),
        "phase": "phase5d",
        "primary_locked_physiology": True,
        "behavior_conditioned": False,
        "run_id": man["run_id"],
        "frozen_candidate": FROZEN_CANDIDATE,
        "probe_sha256_verified": probe.sha256,
        "locked_pair_ids_sha256": EXPECTED_LOCKED_PAIR_SHA256,
        "locked_prompt_text_sha256": EXPECTED_LOCKED_PROMPT_TEXT_SHA256,
        "prompt_template_revision": 2,
        "metrics": metrics,
        "confirmation": confirmation,
        "activation_integrity": {
            "n_rows": N_EXPECTED,
            "finite": bool(np.isfinite(acts).all()),
            "layer": 12,
            "endpoint": "k1",
            "other_layers_extracted": False,
            "k0_extracted": False,
            "discovery_rows_present": False,
            "condition_labels_on_gpu": False,
            "future_response_tokens_present": False,
        },
        "probe_retrained": False,
        "probe_recalibrated": False,
        "layer_reselected": False,
        "causal_interventions_performed": False,
        "extraction_manifest": {
            "activations_sha256": man["activations_sha256"],
            "preflight_repeatability_min_cosine": man[
                "preflight_repeatability_min_cosine"
            ],
            "wall_seconds": man["wall_seconds"],
            "estimated_cost_usd": man["estimated_cost_usd"],
        },
    }
    # PRIMARY FREEZE — write before any behavior generation.
    write_json(out_dir / "locked_primary_physiology.json", primary)
    write_json(
        out_dir / "extraction_manifest.json",
        {
            **{
                k: v
                for k, v in man.items()
                if "path" not in k.lower() or "gitignored" in k
            },
            "activations_gitignored": True,
            "activations_path_gitignored": man.get("activations_path_gitignored"),
        },
    )

    lines = [
        "# Phase 5D locked generalization test",
        "",
        f"**Extraction run ID:** `{man['run_id']}`  ",
        f"**Frozen probe SHA256:** `{probe.sha256}`  ",
        "**Endpoint:** controlled-prefix k1 / layer 12  ",
        "**Primary population:** all 320 designed locked pairs  ",
        "",
        "## Primary locked physiology",
        "",
        f"- Overall AUROC: `{metrics['auroc']:.6f}` "
        f"(95% CI `{metrics['auroc_ci_low']:.6f}`–`{metrics['auroc_ci_high']:.6f}`)",
        f"- Paired S3−S2 Δ: `{paired['paired_mean_diff']:.6f}` "
        f"(95% CI `{paired['paired_mean_diff_ci_low']:.6f}`–"
        f"`{paired['paired_mean_diff_ci_high']:.6f}`)",
        f"- harbor_dock_slip AUROC: `{per_family['harbor_dock_slip']:.6f}`",
        f"- trail_marker_post AUROC: `{per_family['trail_marker_post']:.6f}`",
        "",
        "## Confirmation criteria",
        "",
        f"- Passed: `{confirmation['passed']}`",
        f"- Checks: `{confirmation['checks']}`",
        "",
        "## Guarantees",
        "",
        "THE LOCKED PROBE WAS NOT RETRAINED, RECALIBRATED, OR RESELECTED.  ",
        "NO ALTERNATIVE LAYERS OR ENDPOINTS WERE USED FOR CANDIDATE SELECTION.  ",
        "THE PRIMARY LOCKED POPULATION WAS NOT CONDITIONED ON DOWNSTREAM BEHAVIOR.  ",
        "NO CAUSAL INTERVENTIONS WERE PERFORMED.",
        "",
    ]
    Path(args.report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "run_id": man["run_id"],
                "auroc": metrics["auroc"],
                "auroc_ci": [metrics["auroc_ci_low"], metrics["auroc_ci_high"]],
                "paired_delta": paired["paired_mean_diff"],
                "per_family": per_family,
                "confirmation_passed": confirmation["passed"],
                "probe_sha256": probe.sha256,
                "primary_frozen_path": str(
                    (out_dir / "locked_primary_physiology.json").relative_to(REPO_ROOT)
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
