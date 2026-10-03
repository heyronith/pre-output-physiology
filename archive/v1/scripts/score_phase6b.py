#!/usr/bin/env python3
"""Phase 6B: score all 2880 activations with the frozen Phase-5 probe; factorial analysis.

Usage:

  uv run python scripts/score_phase6b.py --run-id <phase6b_extract_...>
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from safetensors.numpy import load_file

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology import phase6b_freeze as fz  # noqa: E402
from pre_output_physiology.phase5_locked import Phase5FrozenProbe  # noqa: E402
from pre_output_physiology.phase6_analysis import analyze  # noqa: E402
from pre_output_physiology.phase6_conditions import CONDITION_ORDER  # noqa: E402
from pre_output_physiology.phase6b_report import render_report  # noqa: E402

OUT_DIR = REPO_ROOT / "artifacts/phase6b_primary"
REPORT_PATH = REPO_ROOT / "reports/phase6b_primary.md"
SCIENTIFIC_PATHS = [
    "src/pre_output_physiology/phase6_analysis.py",
    "src/pre_output_physiology/phase6b_freeze.py",
    "src/pre_output_physiology/phase6b_report.py",
    "src/pre_output_physiology/phase5_locked.py",
    "scripts/score_phase6b.py",
    "modal/phase6b_extract.py",
]


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), *args], text=True).strip()


def frozen_logits(probe: Phase5FrozenProbe, x: np.ndarray) -> np.ndarray:
    x_n = (np.asarray(x, dtype=np.float64) - probe.mean) / probe.scale
    return x_n @ probe.coef + probe.intercept


def per_scenario(
    rows: list[dict[str, Any]], scores: np.ndarray
) -> tuple[dict[str, np.ndarray], list[str], list[str]]:
    by_base: dict[str, dict[str, float]] = {}
    fam: dict[str, str] = {}
    for r, sc in zip(rows, scores, strict=True):
        by_base.setdefault(r["base_scenario_id"], {})[r["condition_id"]] = float(sc)
        fam[r["base_scenario_id"]] = r["family"]
    base_ids = sorted(by_base)
    for b in base_ids:
        if set(by_base[b]) != set(CONDITION_ORDER):
            raise SystemExit(f"STOP: incomplete conditions for {b}")
    s = {c: np.asarray([by_base[b][c] for b in base_ids]) for c in CONDITION_ORDER}
    return s, base_ids, [fam[b] for b in base_ids]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()

    if _git("status", "--porcelain"):
        raise SystemExit("STOP: working tree dirty")
    cfg = yaml.safe_load(fz.CONFIG_PATH.read_text(encoding="utf-8"))
    if cfg.get("status") != fz.STATUS_AUTHORIZED:
        raise SystemExit(f"unexpected status {cfg.get('status')}")
    if cfg.get("phase6a_outcome") != fz.PHASE6A_OUTCOME:
        raise SystemExit("phase6a_outcome must remain HOLD")
    if cfg["authorizations"].get("probe_scoring_authorized") is not True:
        raise SystemExit("probe_scoring_authorized must be true")

    run_dir = REPO_ROOT / "artifacts/runs" / args.run_id
    manifest = json.loads((run_dir / "extraction_manifest.json").read_text(encoding="utf-8"))
    pre_run_commit = manifest["git_commit"]
    changed = _git("diff", "--name-only", pre_run_commit, "HEAD", "--", *SCIENTIFIC_PATHS)
    if changed:
        raise SystemExit(f"STOP: scientific code changed after freeze: {changed}")

    prompts, _sc, hashes = fz.load_verified_corpus()
    for k, v in hashes.items():
        if manifest[k] != v:
            raise SystemExit(f"STOP: manifest {k} mismatch")
    probe_sha = fz.verify_probe_sha()
    if manifest["probe_sha256"] != probe_sha:
        raise SystemExit("STOP: manifest probe sha mismatch")
    if manifest["preflight_repeatability_min_cosine"] < 0.9999:
        raise SystemExit("STOP: preflight below threshold")

    act_path = run_dir / "activations_l12_k1.safetensors"
    if fz.sha_bytes(act_path.read_bytes()) != manifest["activations_sha256"]:
        raise SystemExit("STOP: activation sha mismatch")
    index_map = json.loads((run_dir / "row_index_map.json").read_text(encoding="utf-8"))
    if index_map["row_index_to_example_id"] != [r["example_id"] for r in prompts]:
        raise SystemExit("STOP: row index map mismatch")
    tens = load_file(str(act_path))
    acts = tens["activations_l12_k1"]
    integrity = {
        "shape": list(acts.shape),
        "dtype": str(acts.dtype),
        "all_finite": bool(np.isfinite(acts).all()),
        "row_index_contiguous": bool(np.array_equal(tens["row_index"], np.arange(fz.N_PROMPTS))),
        "layer": int(tens["layer"][0]),
        "k1_prefix_token_id": int(tens["k1_prefix_token_id"][0]),
        "n_zero_rows": int((np.abs(acts).sum(axis=1) == 0).sum()),
        "n_duplicate_rows": int(len(acts) - len(np.unique(acts, axis=0))),
    }
    if not (
        integrity["shape"] == [fz.N_PROMPTS, 4096]
        and integrity["dtype"] == "float32"
        and integrity["all_finite"]
        and integrity["row_index_contiguous"]
        and integrity["layer"] == fz.LAYER
        and integrity["k1_prefix_token_id"] == fz.CONTROLLED_PREFIX_TOKEN_ID
        and integrity["n_zero_rows"] == 0
    ):
        raise SystemExit(f"STOP: activation integrity failure {integrity}")
    integrity["passed"] = True

    probe = Phase5FrozenProbe(fz.PROBE_PATH, expected_sha256=fz.EXPECTED_PROBE_SHA256)
    prob = probe.decision_scores(acts)
    logit = frozen_logits(probe, acts)

    s_p, base_ids, families = per_scenario(prompts, prob)
    s_l, base_ids_l, _ = per_scenario(prompts, logit)
    assert base_ids == base_ids_l
    analysis_p = analyze(s_p, base_ids, families)
    analysis_l = analyze(s_l, base_ids, families)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "frozen_probe_scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["example_id", "base_scenario_id", "family", "condition_id", "probability", "logit"]
        )
        for r, p, lg in zip(prompts, prob, logit, strict=True):
            w.writerow(
                [
                    r["example_id"],
                    r["base_scenario_id"],
                    r["family"],
                    r["condition_id"],
                    f"{p:.8f}",
                    f"{lg:.6f}",
                ]
            )
    ex_keys = (
        "run_id",
        "n_rows",
        "activations_sha256",
        "wall_seconds",
        "extraction_seconds",
        "model_load_seconds",
        "estimated_cost_usd",
        "preflight_repeatability_min_cosine",
        "preflight_wall_seconds",
        "preflight_estimated_cost_usd",
        "total_estimated_cost_usd",
        "final_prompt_text_sha256",
        "final_scenario_text_sha256",
        "final_base_scenario_ids_sha256",
        "payload_sha256",
        "extractor_sha256",
        "dtype_compute",
        "dtype_storage",
        "batch_size",
        "quantization",
    )
    summary = {
        "status": fz.STATUS_PRIMARY_COMPLETE,
        "phase6a_outcome": fz.PHASE6A_OUTCOME,
        "pre_run_commit": pre_run_commit,
        "scoring_commit": _git("rev-parse", "HEAD"),
        "probe_sha256": probe_sha,
        "extraction": {k: manifest[k] for k in ex_keys},
        "activation_integrity": integrity,
        "primary_score_scale": "frozen_probe_probability",
        "analysis_probability": analysis_p,
        "analysis_logit_secondary": analysis_l,
        "behavior_conditioned": False,
        "exclusions": 0,
        "generation_performed": False,
        "probe_retrained": False,
        "probe_recalibrated": False,
        "layer_reselected": False,
        "causal_interventions_performed": False,
    }
    (OUT_DIR / "phase6b_primary_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    REPORT_PATH.write_text(render_report(summary), encoding="utf-8")
    c = analysis_p["contrasts"]
    e = analysis_p["effects"]
    print(
        json.dumps(
            {
                **{
                    k: [
                        v["paired_mean_delta"],
                        v["paired_mean_delta_ci_low"],
                        v["paired_mean_delta_ci_high"],
                        v["auroc"],
                        v["auroc_ci_low"],
                        v["auroc_ci_high"],
                    ]
                    for k, v in c.items()
                },
                **{k: [v["mean"], v["ci_low"], v["ci_high"]] for k, v in e.items()},
                "flags": {
                    k: v for k, v in analysis_p["pattern_flags"].items() if k.startswith("pattern")
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
