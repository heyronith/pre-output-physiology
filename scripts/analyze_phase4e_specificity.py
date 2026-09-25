#!/usr/bin/env python3
"""Score Phase 4E activations with frozen probes; compute specificity contrasts.

Labels are loaded only here (local). Eligibility sets are frozen from Phase 4D.
No retraining, recalibration, eligibility changes, or C5 analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase4_scoring import (  # noqa: E402
    EXPECTED_ELIGIBILITY,
    FROZEN_PROBE_L12_K0_SHA256,
    FROZEN_PROBE_L12_K1_SHA256,
    KEY_SECONDARY_CONTRAST,
    N_BOOTSTRAP_MIN,
    PRIMARY_CONTRAST,
    contrast_metrics,
    load_frozen_probe,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

PHASE4D_RUN = "phase4d_final_20260925T174548Z_4d93730c"
ELIGIBILITY_PATH = (
    REPO_ROOT / "artifacts/phase4d_final_behavior/frozen_contrast_eligibility.json"
)
CONTRAST_ORDER = (
    "C3_vs_C2",
    "C3_vs_C4",
    "C3_vs_C1",
    "C3_vs_C6",
    "C2_vs_C1",
    "C4_vs_C1",
)


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_ids(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _verify_eligibility(elig: dict[str, Any]) -> None:
    for name, expected in EXPECTED_ELIGIBILITY.items():
        c = elig["contrasts"][name]
        ids = c["paired_base_scenario_ids"]
        digest = _sha_ids(ids)
        if c["n_paired_valid"] != expected["n"] or digest != expected["sha256"]:
            raise SystemExit(
                f"STOP: eligibility drift for {name}: "
                f"N={c['n_paired_valid']} hash={digest}"
            )
        if c["paired_ids_sha256"] != expected["sha256"]:
            raise SystemExit(f"STOP: stored hash mismatch for {name}")
    if elig.get("held_condition") != "C5_uncertain_honest":
        raise SystemExit("C5 HOLD missing from eligibility")


def _score_all(
    example_ids: list[str],
    acts_k0: np.ndarray,
    acts_k1: np.ndarray,
    probe_k0: Any,
    probe_k1: Any,
) -> dict[str, dict[str, float]]:
    scores_k0 = probe_k0.decision_scores(acts_k0)
    scores_k1 = probe_k1.decision_scores(acts_k1)
    out: dict[str, dict[str, float]] = {}
    for i, eid in enumerate(example_ids):
        out[eid] = {
            "score_l12_k0": float(scores_k0[i]),
            "score_l12_k1": float(scores_k1[i]),
        }
    return out


def _contrast_block(
    name: str,
    left: str,
    right: str,
    paired_ids: list[str],
    by_example: dict[str, dict],
    scores: dict[str, dict[str, float]],
    *,
    endpoint: str,
    n_bootstrap: int,
) -> dict[str, Any]:
    score_key = "score_l12_k1" if endpoint == "k1" else "score_l12_k0"
    pos = []
    neg = []
    used = []
    for sid in paired_ids:
        left_id = f"{sid}__{left}"
        right_id = f"{sid}__{right}"
        if left_id not in scores or right_id not in scores:
            raise SystemExit(f"missing scores for {sid} in {name}")
        # Sanity: condition labels match eligibility join
        if by_example[left_id]["condition_id"] != left:
            raise SystemExit(f"label join mismatch {left_id}")
        if by_example[right_id]["condition_id"] != right:
            raise SystemExit(f"label join mismatch {right_id}")
        pos.append(scores[left_id][score_key])
        neg.append(scores[right_id][score_key])
        used.append(sid)
    metrics = contrast_metrics(
        np.asarray(pos),
        np.asarray(neg),
        used,
        n_bootstrap=n_bootstrap,
        seed=0,
    )
    return {
        "contrast": name,
        "left_condition": left,
        "right_condition": right,
        "endpoint": endpoint,
        "layer": 12,
        "k": 1 if endpoint == "k1" else 0,
        "paired_ids_sha256": _sha_ids(used),
        "n_pairs": len(used),
        **metrics,
    }


def _write_report(
    summary: dict[str, Any],
    extract_man: dict[str, Any],
    path: Path,
) -> None:
    prim = summary["contrasts"]["L12_k1"][PRIMARY_CONTRAST]
    key = summary["contrasts"]["L12_k1"][KEY_SECONDARY_CONTRAST]
    lines = [
        "# Phase 4E specificity report",
        "",
        f"**Extraction run ID:** `{extract_man.get('run_id')}`  ",
        f"**Git SHA (extraction):** `{extract_man.get('git_commit')}`  ",
        "**Status:** `phase4e_specificity_complete_awaiting_audit`  ",
        "",
        "## Scope",
        "",
        "- Frozen Phase-3 probes only (no retrain / recalibrate)",
        "- Primary endpoint: L12 / controlled-prefix k1",
        "- Transfer/specificity test under supplied `Response` (12107) prefix",
        "- C5 HOLD; eligibility sets unchanged from Phase 4D",
        "- No causal interventions",
        "",
        "## Compatibility preflight",
        "",
        f"- Min cosine L12/k0: `{extract_man.get('preflight_min_cosine_l12_k0')}`",
        f"- Min cosine L12/k1: `{extract_man.get('preflight_min_cosine_l12_k1')}`",
        f"- Min cosine all: `{extract_man.get('preflight_min_cosine_all')}` (gate ≥ 0.9999)",
        "",
        "## Primary: C3 vs C2 (L12/k1)",
        "",
        f"- Paired N: `{prim['n_pairs']}`",
        f"- AUROC: `{prim['auroc']:.4f}` "
        f"(95% CI `{prim['auroc_ci_low']:.4f}`–`{prim['auroc_ci_high']:.4f}`)",
        f"- Paired mean score diff C3−C2: `{prim['paired_mean_diff']:.4f}` "
        f"(95% CI `{prim['paired_mean_diff_ci_low']:.4f}`–"
        f"`{prim['paired_mean_diff_ci_high']:.4f}`)",
        "",
        "## Key secondary: C3 vs C4 (L12/k1)",
        "",
        f"- Paired N: `{key['n_pairs']}`",
        f"- AUROC: `{key['auroc']:.4f}` "
        f"(95% CI `{key['auroc_ci_low']:.4f}`–`{key['auroc_ci_high']:.4f}`)",
        f"- Paired mean score diff C3−C4: `{key['paired_mean_diff']:.4f}` "
        f"(95% CI `{key['paired_mean_diff_ci_low']:.4f}`–"
        f"`{key['paired_mean_diff_ci_high']:.4f}`)",
        "",
        "## All L12/k1 contrasts",
        "",
        "| Contrast | N | AUROC | AUROC 95% CI | Mean Δ | Δ 95% CI |",
        "| --- | ---: | ---: | --- | ---: | --- |",
    ]
    for name in CONTRAST_ORDER:
        c = summary["contrasts"]["L12_k1"][name]
        lines.append(
            f"| {name} | {c['n_pairs']} | {c['auroc']:.4f} | "
            f"{c['auroc_ci_low']:.4f}–{c['auroc_ci_high']:.4f} | "
            f"{c['paired_mean_diff']:.4f} | "
            f"{c['paired_mean_diff_ci_low']:.4f}–{c['paired_mean_diff_ci_high']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Secondary L12/k0 (prompt-boundary)",
            "",
            "| Contrast | N | AUROC | AUROC 95% CI | Mean Δ | Δ 95% CI |",
            "| --- | ---: | ---: | --- | ---: | --- |",
        ]
    )
    for name in CONTRAST_ORDER:
        c = summary["contrasts"]["L12_k0"][name]
        lines.append(
            f"| {name} | {c['n_pairs']} | {c['auroc']:.4f} | "
            f"{c['auroc_ci_low']:.4f}–{c['auroc_ci_high']:.4f} | "
            f"{c['paired_mean_diff']:.4f} | "
            f"{c['paired_mean_diff_ci_low']:.4f}–{c['paired_mean_diff_ci_high']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Guarantees",
            "",
            "NO PROBES WERE RETRAINED OR RECALIBRATED.  ",
            "NO ELIGIBILITY SETS WERE CHANGED.  ",
            "C5 REMAINED ON HOLD.  ",
            "NO CAUSAL INTERVENTIONS WERE PERFORMED.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--summary-dir",
        default=str(REPO_ROOT / "artifacts/phase4e_specificity"),
    )
    parser.add_argument(
        "--report-path",
        default=str(REPO_ROOT / "reports/phase4e_specificity.md"),
    )
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP_MIN)
    args = parser.parse_args()
    if args.n_bootstrap < N_BOOTSTRAP_MIN:
        raise SystemExit(f"n_bootstrap must be >= {N_BOOTSTRAP_MIN}")

    run_dir = Path(args.run_dir)
    summary_dir = Path(args.summary_dir)
    report_path = Path(args.report_path)
    extract_man = json.loads(
        (run_dir / "extraction_manifest.json").read_text(encoding="utf-8")
    )
    meta = json.loads((run_dir / "activation_meta.json").read_text(encoding="utf-8"))
    act_path = run_dir / "activations_l12_k0_k1.safetensors"
    if _sha_file(act_path) != extract_man["activations_sha256"]:
        raise SystemExit("activation file hash mismatch")
    if not extract_man.get("preflight_passed"):
        raise SystemExit("preflight not recorded as passed")
    if float(extract_man["preflight_min_cosine_all"]) < 0.9999:
        raise SystemExit("preflight cosine below gate")
    if meta.get("condition_labels_present") is not False:
        raise SystemExit("activation meta must not carry condition labels")
    if meta.get("controlled_prefix_token_id") != 12107:
        raise SystemExit("controlled prefix drift")
    if extract_man.get("prefix_identity_count") != 1200:
        raise SystemExit("prefix identity incomplete")

    tens = load_file(str(act_path))
    acts_k0 = np.asarray(tens["activations_l12_k0"], dtype=np.float32)
    acts_k1 = np.asarray(tens["activations_l12_k1"], dtype=np.float32)
    prefix_ids = np.asarray(tens["k1_prefix_token_ids"])
    if not np.all(prefix_ids == 12107):
        raise SystemExit("k1 prefix identity failure in tensors")
    example_ids = meta["example_ids"]
    if len(example_ids) != 1200 or acts_k0.shape[0] != 1200:
        raise SystemExit("activation row count != 1200")

    # Labels loaded only now
    final_rows = _load_jsonl(
        REPO_ROOT / "artifacts/runs" / PHASE4D_RUN / "final_outputs.jsonl"
    )
    by_example = {r["example_id"]: r for r in final_rows}
    if any(r["condition_id"] == "C5_uncertain_honest" for r in final_rows):
        raise SystemExit("C5 present")
    for eid in example_ids:
        if eid not in by_example:
            raise SystemExit(f"missing label row for {eid}")

    elig = json.loads(ELIGIBILITY_PATH.read_text(encoding="utf-8"))
    _verify_eligibility(elig)

    probe_k1 = load_frozen_probe(
        REPO_ROOT / "artifacts/phase4_models/probe_l12_k1.npz",
        expected_sha256=FROZEN_PROBE_L12_K1_SHA256,
    )
    probe_k0 = load_frozen_probe(
        REPO_ROOT / "artifacts/phase4_models/probe_l12_k0.npz",
        expected_sha256=FROZEN_PROBE_L12_K0_SHA256,
    )
    scores = _score_all(example_ids, acts_k0, acts_k1, probe_k0, probe_k1)

    contrasts: dict[str, dict[str, Any]] = {"L12_k1": {}, "L12_k0": {}}
    for name in CONTRAST_ORDER:
        c = elig["contrasts"][name]
        for endpoint, key in (("k1", "L12_k1"), ("k0", "L12_k0")):
            contrasts[key][name] = _contrast_block(
                name,
                c["left_condition"],
                c["right_condition"],
                c["paired_base_scenario_ids"],
                by_example,
                scores,
                endpoint=endpoint,
                n_bootstrap=args.n_bootstrap,
            )
            # Confirm eligibility hash unchanged for primary/key secondary
            if name in EXPECTED_ELIGIBILITY:
                if contrasts[key][name]["paired_ids_sha256"] != (
                    EXPECTED_ELIGIBILITY[name]["sha256"]
                ):
                    raise SystemExit(f"analysis pair hash drift {name}")

    summary_dir.mkdir(parents=True, exist_ok=True)
    score_rows = []
    for eid in example_ids:
        row = by_example[eid]
        score_rows.append(
            {
                "example_id": eid,
                "base_scenario_id": row["base_scenario_id"],
                "condition_id": row["condition_id"],
                **scores[eid],
            }
        )
    write_json(summary_dir / "probe_scores.json", {"rows": score_rows})

    summary = {
        "created_at": utc_now_iso(),
        "extraction_run_id": extract_man["run_id"],
        "git_commit_extraction": extract_man["git_commit"],
        "phase4d_run_id": PHASE4D_RUN,
        "n_bootstrap": args.n_bootstrap,
        "primary_endpoint": {"layer": 12, "k": 1, "interpretation": "controlled_prefix_k1"},
        "primary_contrast": PRIMARY_CONTRAST,
        "key_secondary_contrast": KEY_SECONDARY_CONTRAST,
        "eligibility_hashes": {
            name: EXPECTED_ELIGIBILITY[name]["sha256"] for name in EXPECTED_ELIGIBILITY
        },
        "frozen_probe_l12_k1_sha256": FROZEN_PROBE_L12_K1_SHA256,
        "frozen_probe_l12_k0_sha256": FROZEN_PROBE_L12_K0_SHA256,
        "probes_retrained": False,
        "probes_recalibrated": False,
        "eligibility_changed": False,
        "c5_status": "HOLD",
        "causal_interventions_performed": False,
        "preflight_min_cosine_all": extract_man["preflight_min_cosine_all"],
        "contrasts": contrasts,
        "interpretation_note": (
            "Transfer/specificity test under controlled-prefix distribution shift. "
            "A null does not prove absence of deception physiology."
        ),
    }
    write_json(summary_dir / "specificity_summary.json", summary)

    commit_manifest = {
        **{k: v for k, v in extract_man.items() if k != "activations_path"},
        "activations_gitignored": True,
        "activations_path_gitignored": extract_man.get("activations_path"),
        "scored_at": utc_now_iso(),
        "n_bootstrap": args.n_bootstrap,
        "probes_retrained": False,
        "probes_recalibrated": False,
        "eligibility_changed": False,
        "c5_status": "HOLD",
        "causal_interventions_performed": False,
    }
    write_json(summary_dir / "extraction_manifest.json", commit_manifest)

    # Copy preflight summary into committed artifacts
    preflight = json.loads(
        (run_dir / "b1_compatibility_preflight.json").read_text(encoding="utf-8")
    )
    write_json(summary_dir / "b1_compatibility_preflight.json", preflight)

    elig_confirm = {
        "verified_at": utc_now_iso(),
        "source": str(ELIGIBILITY_PATH.relative_to(REPO_ROOT)),
        "unchanged": True,
        "contrasts": {
            name: {
                "n_paired_valid": elig["contrasts"][name]["n_paired_valid"],
                "paired_ids_sha256": elig["contrasts"][name]["paired_ids_sha256"],
            }
            for name in CONTRAST_ORDER
        },
    }
    write_json(summary_dir / "eligibility_confirmation.json", elig_confirm)
    _write_report(summary, extract_man, report_path)

    print(
        json.dumps(
            {
                "extraction_run_id": extract_man["run_id"],
                "preflight_min_cosine": extract_man["preflight_min_cosine_all"],
                "primary_C3_vs_C2_k1": contrasts["L12_k1"]["C3_vs_C2"],
                "key_C3_vs_C4_k1": contrasts["L12_k1"]["C3_vs_C4"],
                "report": str(report_path),
            },
            indent=2,
            default=float,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
