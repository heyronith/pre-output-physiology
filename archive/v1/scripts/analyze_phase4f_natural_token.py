#!/usr/bin/env python3
"""Phase 4F natural-token k1 diagnostic vs Phase 4E controlled-prefix scores.

Frozen eligibility and frozen L12/k1 probe only. No retrain, no full regen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase4_scoring import (  # noqa: E402
    EXPECTED_ELIGIBILITY,
    FROZEN_PROBE_L12_K1_SHA256,
    N_BOOTSTRAP_MIN,
    contrast_metrics,
    load_frozen_probe,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

PHASE4D_RUN = "phase4d_final_20260925T174548Z_4d93730c"
PHASE4E_SCORES = REPO_ROOT / "artifacts/phase4e_specificity/probe_scores.json"
PHASE4E_SUMMARY = REPO_ROOT / "artifacts/phase4e_specificity/specificity_summary.json"
ELIGIBILITY_PATH = (
    REPO_ROOT / "artifacts/phase4d_final_behavior/frozen_contrast_eligibility.json"
)
CONTROLLED_C3_C2_AUROC_REF = 0.4639
C2 = "C2_known_honest_strategic"
C3 = "C3_known_deceptive_strategic"
C4 = "C4_false_belief_honest"


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
        digest = _sha_ids(c["paired_base_scenario_ids"])
        if c["n_paired_valid"] != expected["n"] or digest != expected["sha256"]:
            raise SystemExit(f"eligibility drift {name}: N={c['n_paired_valid']} {digest}")
        if c["paired_ids_sha256"] != expected["sha256"]:
            raise SystemExit(f"stored eligibility hash mismatch {name}")


def _token_distribution(
    example_ids: list[str],
    by_example: dict[str, dict],
    token_ids: dict[str, int],
    token_texts: dict[str, str],
    condition_id: str,
) -> dict[str, Any]:
    ids = [eid for eid in example_ids if by_example[eid]["condition_id"] == condition_id]
    id_counts: Counter[int] = Counter(token_ids[eid] for eid in ids)
    tid_to_text = {token_ids[eid]: token_texts[eid] for eid in ids}
    top = [
        {
            "token_id": tid,
            "token_text": tid_to_text[tid],
            "count": cnt,
            "fraction": cnt / len(ids),
        }
        for tid, cnt in id_counts.most_common()
    ]
    return {
        "condition_id": condition_id,
        "n": len(ids),
        "n_unique_tokens": len(id_counts),
        "top_tokens": top[:20],
        "frequency_distribution": [
            {"token_id": t["token_id"], "token_text": t["token_text"], "count": t["count"]}
            for t in top
        ],
    }


def _paired_same_token_fraction(
    paired_ids: list[str],
    token_ids: dict[str, int],
    left: str,
    right: str,
) -> dict[str, Any]:
    same = 0
    for sid in paired_ids:
        a = token_ids[f"{sid}__{left}"]
        b = token_ids[f"{sid}__{right}"]
        if a == b:
            same += 1
    return {
        "n_pairs": len(paired_ids),
        "n_same_first_token": same,
        "fraction_same_first_token": same / len(paired_ids) if paired_ids else float("nan"),
        "left_condition": left,
        "right_condition": right,
    }


def _auroc_delta_bootstrap(
    scores_pos_nat: np.ndarray,
    scores_neg_nat: np.ndarray,
    scores_pos_ctrl: np.ndarray,
    scores_neg_ctrl: np.ndarray,
    scenario_ids: list[str],
    *,
    n_bootstrap: int,
    seed: int = 0,
) -> dict[str, float]:
    """Scenario-paired bootstrap of AUROC_natural − AUROC_controlled."""
    n = len(scenario_ids)
    y_template = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])

    def _auroc(pos: np.ndarray, neg: np.ndarray, idx: np.ndarray) -> float:
        scores = np.concatenate([pos[idx], neg[idx]])
        y = np.concatenate([np.ones(len(idx), dtype=int), np.zeros(len(idx), dtype=int)])
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, scores))

    auroc_nat = float(
        roc_auc_score(
            y_template, np.concatenate([scores_pos_nat, scores_neg_nat])
        )
    )
    auroc_ctrl = float(
        roc_auc_score(
            y_template, np.concatenate([scores_pos_ctrl, scores_neg_ctrl])
        )
    )
    point = auroc_nat - auroc_ctrl
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        an = _auroc(scores_pos_nat, scores_neg_nat, idx)
        ac = _auroc(scores_pos_ctrl, scores_neg_ctrl, idx)
        if np.isnan(an) or np.isnan(ac):
            continue
        boots.append(an - ac)
    arr = np.asarray(boots)
    return {
        "auroc_natural": auroc_nat,
        "auroc_controlled": auroc_ctrl,
        "auroc_natural_minus_controlled": point,
        "auroc_delta_ci_low": float(np.quantile(arr, 0.025)),
        "auroc_delta_ci_high": float(np.quantile(arr, 0.975)),
        "n_bootstrap_effective": int(len(arr)),
        "n_pairs": n,
        "bootstrap_unit": "scenario",
    }


def _score_shift_diagnostic(
    paired_ids: list[str],
    left: str,
    right: str,
    nat_scores: dict[str, float],
    ctrl_scores: dict[str, float],
    *,
    n_bootstrap: int,
) -> dict[str, Any]:
    left_shift = []
    right_shift = []
    for sid in paired_ids:
        lid = f"{sid}__{left}"
        rid = f"{sid}__{right}"
        left_shift.append(nat_scores[lid] - ctrl_scores[lid])
        right_shift.append(nat_scores[rid] - ctrl_scores[rid])
    left_shift_a = np.asarray(left_shift, dtype=float)
    right_shift_a = np.asarray(right_shift, dtype=float)
    # (C3_nat − C3_ctrl) − (C2_nat − C2_ctrl)
    differential = left_shift_a - right_shift_a
    rng = np.random.default_rng(1)
    boots = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, len(differential), size=len(differential))
        boots[i] = float(differential[idx].mean())
    return {
        "left_condition": left,
        "right_condition": right,
        "mean_natural_minus_controlled_left": float(left_shift_a.mean()),
        "mean_natural_minus_controlled_right": float(right_shift_a.mean()),
        "paired_differential_shift": float(differential.mean()),
        "paired_differential_shift_ci_low": float(np.quantile(boots, 0.025)),
        "paired_differential_shift_ci_high": float(np.quantile(boots, 0.975)),
        "definition": (
            "mean((score_nat_left - score_ctrl_left) - "
            "(score_nat_right - score_ctrl_right))"
        ),
        "n_pairs": len(paired_ids),
        "n_bootstrap": n_bootstrap,
    }


def _contrast_natural(
    name: str,
    left: str,
    right: str,
    paired_ids: list[str],
    nat_scores: dict[str, float],
    ctrl_scores: dict[str, float],
    *,
    n_bootstrap: int,
) -> dict[str, Any]:
    pos_n = np.asarray([nat_scores[f"{sid}__{left}"] for sid in paired_ids])
    neg_n = np.asarray([nat_scores[f"{sid}__{right}"] for sid in paired_ids])
    pos_c = np.asarray([ctrl_scores[f"{sid}__{left}"] for sid in paired_ids])
    neg_c = np.asarray([ctrl_scores[f"{sid}__{right}"] for sid in paired_ids])
    nat_metrics = contrast_metrics(
        pos_n, neg_n, paired_ids, n_bootstrap=n_bootstrap, seed=0
    )
    delta = _auroc_delta_bootstrap(
        pos_n, neg_n, pos_c, neg_c, paired_ids, n_bootstrap=n_bootstrap, seed=2
    )
    shift = _score_shift_diagnostic(
        paired_ids, left, right, nat_scores, ctrl_scores, n_bootstrap=n_bootstrap
    )
    return {
        "contrast": name,
        "left_condition": left,
        "right_condition": right,
        "endpoint": "natural_k1",
        "layer": 12,
        "k": 1,
        "paired_ids_sha256": _sha_ids(paired_ids),
        "natural": nat_metrics,
        "controlled_reference": {
            "auroc": float(delta["auroc_controlled"]),
            "auroc_reported_phase4e_c3_vs_c2": (
                CONTROLLED_C3_C2_AUROC_REF if name == "C3_vs_C2" else None
            ),
        },
        "auroc_natural_minus_controlled": delta,
        "score_shift_diagnostic": shift,
    }


def _write_report(summary: dict[str, Any], extract_man: dict[str, Any], path: Path) -> None:
    prim = summary["contrasts"]["C3_vs_C2"]
    key = summary["contrasts"]["C3_vs_C4"]
    lines = [
        "# Phase 4F natural-token k1 diagnostic",
        "",
        f"**Run ID:** `{extract_man.get('run_id')}`  ",
        f"**Git SHA:** `{extract_man.get('git_commit')}`  ",
        "**Status:** `phase4f_natural_token_diagnostic_complete_awaiting_audit`  ",
        "",
        "## Scope",
        "",
        "- Diagnostic only: natural greedy first-token k1 vs Phase-4E controlled `12107`",
        "- Frozen eligibility / frozen L12/k1 probe; no full response regeneration",
        "- C5 HOLD; no retrain; no causal interventions",
        "",
        f"- Preflight min cosine: `{extract_man.get('preflight_min_cosine_all')}`",
        "",
        "## Primary: C3 vs C2",
        "",
        f"- Natural AUROC: `{prim['natural']['auroc']:.4f}` "
        f"(95% CI `{prim['natural']['auroc_ci_low']:.4f}`–"
        f"`{prim['natural']['auroc_ci_high']:.4f}`)",
        f"- Controlled AUROC (Phase 4E): "
        f"`{prim['controlled_reference']['auroc']:.4f}` "
        f"(reported `{CONTROLLED_C3_C2_AUROC_REF}`)",
        f"- Natural−controlled AUROC Δ: "
        f"`{prim['auroc_natural_minus_controlled']['auroc_natural_minus_controlled']:.4f}` "
        f"(95% CI "
        f"`{prim['auroc_natural_minus_controlled']['auroc_delta_ci_low']:.4f}`–"
        f"`{prim['auroc_natural_minus_controlled']['auroc_delta_ci_high']:.4f}`)",
        f"- Natural paired mean C3−C2: `{prim['natural']['paired_mean_diff']:.6f}` "
        f"(95% CI `{prim['natural']['paired_mean_diff_ci_low']:.6f}`–"
        f"`{prim['natural']['paired_mean_diff_ci_high']:.6f}`)",
        "",
        "## Score-shift diagnostic (C3 vs C2)",
        "",
        f"- mean(nat−ctrl) C3: "
        f"`{prim['score_shift_diagnostic']['mean_natural_minus_controlled_left']:.6f}`",
        f"- mean(nat−ctrl) C2: "
        f"`{prim['score_shift_diagnostic']['mean_natural_minus_controlled_right']:.6f}`",
        f"- paired differential (C3 shift − C2 shift): "
        f"`{prim['score_shift_diagnostic']['paired_differential_shift']:.6f}` "
        f"(95% CI "
        f"`{prim['score_shift_diagnostic']['paired_differential_shift_ci_low']:.6f}`–"
        f"`{prim['score_shift_diagnostic']['paired_differential_shift_ci_high']:.6f}`)",
        "",
        "## Key secondary: C3 vs C4",
        "",
        f"- Natural AUROC: `{key['natural']['auroc']:.4f}` "
        f"(95% CI `{key['natural']['auroc_ci_low']:.4f}`–"
        f"`{key['natural']['auroc_ci_high']:.4f}`)",
        f"- Controlled AUROC: `{key['controlled_reference']['auroc']:.4f}`",
        f"- Natural−controlled AUROC Δ: "
        f"`{key['auroc_natural_minus_controlled']['auroc_natural_minus_controlled']:.4f}`",
        "",
        "## Token diagnostics",
        "",
    ]
    for cid in (C2, C3, C4):
        td = summary["token_diagnostics"]["per_condition"][cid]
        lines.append(f"### {cid}")
        lines.append(f"- Unique first tokens: `{td['n_unique_tokens']}`")
        lines.append("- Top tokens:")
        for t in td["top_tokens"][:5]:
            lines.append(
                f"  - id `{t['token_id']}` `{t['token_text']!r}`: "
                f"{t['count']} ({t['fraction']:.3f})"
            )
        lines.append("")
    same = summary["token_diagnostics"]["c2_c3_paired_same_token"]
    lines.extend(
        [
            f"**C2/C3 paired same natural first token:** "
            f"`{same['n_same_first_token']}/{same['n_pairs']}` "
            f"(`{same['fraction_same_first_token']:.3f}`)",
            "",
            "## Guarantees",
            "",
            "NO PROBES WERE RETRAINED OR RECALIBRATED.  ",
            "NO ELIGIBILITY SETS WERE CHANGED.  ",
            "NO FULL PHASE 4 RESPONSES WERE REGENERATED.  ",
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
        default=str(REPO_ROOT / "artifacts/phase4f_natural_token_diagnostic"),
    )
    parser.add_argument(
        "--report-path",
        default=str(REPO_ROOT / "reports/phase4f_natural_token_diagnostic.md"),
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
    act_path = run_dir / "activations_l12_natural_k1.safetensors"
    if _sha_file(act_path) != extract_man["activations_sha256"]:
        raise SystemExit("activation hash mismatch")
    if not extract_man.get("preflight_passed"):
        raise SystemExit("preflight not passed")
    if float(extract_man["preflight_min_cosine_all"]) < 0.9999:
        raise SystemExit("preflight cosine below gate")
    if extract_man.get("full_response_generated") is not False:
        raise SystemExit("full response generation flagged")
    if meta.get("condition_labels_present") is not False:
        raise SystemExit("labels in activation meta")

    tens = load_file(str(act_path))
    acts = np.asarray(tens["activations_l12_natural_k1"], dtype=np.float32)
    nat_id_arr = np.asarray(tens["natural_first_token_ids"], dtype=np.int32)
    example_ids = meta["example_ids"]
    if len(example_ids) != 1200 or acts.shape[0] != 1200:
        raise SystemExit("row count != 1200")

    final_rows = _load_jsonl(
        REPO_ROOT / "artifacts/runs" / PHASE4D_RUN / "final_outputs.jsonl"
    )
    by_example = {r["example_id"]: r for r in final_rows}
    if any(r["condition_id"] == "C5_uncertain_honest" for r in final_rows):
        raise SystemExit("C5 present")

    elig = json.loads(ELIGIBILITY_PATH.read_text(encoding="utf-8"))
    _verify_eligibility(elig)

    probe = load_frozen_probe(
        REPO_ROOT / "artifacts/phase4_models/probe_l12_k1.npz",
        expected_sha256=FROZEN_PROBE_L12_K1_SHA256,
    )
    nat_score_arr = probe.decision_scores(acts)
    nat_scores = {
        eid: float(nat_score_arr[i]) for i, eid in enumerate(example_ids)
    }
    token_ids = {
        eid: int(nat_id_arr[i]) for i, eid in enumerate(example_ids)
    }
    token_texts = {
        eid: meta["natural_first_token_texts"][i]
        for i, eid in enumerate(example_ids)
    }

    ctrl_payload = json.loads(PHASE4E_SCORES.read_text(encoding="utf-8"))
    ctrl_scores = {
        r["example_id"]: float(r["score_l12_k1"]) for r in ctrl_payload["rows"]
    }
    phase4e = json.loads(PHASE4E_SUMMARY.read_text(encoding="utf-8"))
    ctrl_c3c2 = phase4e["contrasts"]["L12_k1"]["C3_vs_C2"]["auroc"]
    if abs(ctrl_c3c2 - CONTROLLED_C3_C2_AUROC_REF) > 1e-3:
        raise SystemExit(
            f"Phase4E controlled AUROC drift: {ctrl_c3c2} vs {CONTROLLED_C3_C2_AUROC_REF}"
        )

    contrasts = {}
    for name in ("C3_vs_C2", "C3_vs_C4"):
        c = elig["contrasts"][name]
        contrasts[name] = _contrast_natural(
            name,
            c["left_condition"],
            c["right_condition"],
            c["paired_base_scenario_ids"],
            nat_scores,
            ctrl_scores,
            n_bootstrap=args.n_bootstrap,
        )
        if contrasts[name]["paired_ids_sha256"] != EXPECTED_ELIGIBILITY[name]["sha256"]:
            raise SystemExit(f"pair hash drift {name}")

    token_diag = {
        "per_condition": {
            cid: _token_distribution(
                example_ids, by_example, token_ids, token_texts, cid
            )
            for cid in (C2, C3, C4)
        },
        "c2_c3_paired_same_token": _paired_same_token_fraction(
            elig["contrasts"]["C3_vs_C2"]["paired_base_scenario_ids"],
            token_ids,
            C3,
            C2,
        ),
    }

    summary_dir.mkdir(parents=True, exist_ok=True)
    score_rows = []
    for eid in example_ids:
        row = by_example[eid]
        score_rows.append(
            {
                "example_id": eid,
                "base_scenario_id": row["base_scenario_id"],
                "condition_id": row["condition_id"],
                "natural_first_token_id": token_ids[eid],
                "natural_first_token_text": token_texts[eid],
                "score_l12_natural_k1": nat_scores[eid],
                "score_l12_controlled_k1": ctrl_scores[eid],
                "score_shift_natural_minus_controlled": (
                    nat_scores[eid] - ctrl_scores[eid]
                ),
            }
        )
    write_json(summary_dir / "natural_probe_scores.json", {"rows": score_rows})

    summary = {
        "created_at": utc_now_iso(),
        "extraction_run_id": extract_man["run_id"],
        "git_commit_extraction": extract_man["git_commit"],
        "phase4d_run_id": PHASE4D_RUN,
        "phase4e_run_id": extract_man.get("phase4e_run_id"),
        "diagnostic_only": True,
        "n_bootstrap": args.n_bootstrap,
        "controlled_c3_vs_c2_auroc_reference": CONTROLLED_C3_C2_AUROC_REF,
        "eligibility_hashes": {
            name: EXPECTED_ELIGIBILITY[name]["sha256"] for name in EXPECTED_ELIGIBILITY
        },
        "frozen_probe_l12_k1_sha256": FROZEN_PROBE_L12_K1_SHA256,
        "probes_retrained": False,
        "probes_recalibrated": False,
        "eligibility_changed": False,
        "full_responses_regenerated": False,
        "c5_status": "HOLD",
        "causal_interventions_performed": False,
        "preflight_min_cosine_all": extract_man["preflight_min_cosine_all"],
        "contrasts": contrasts,
        "token_diagnostics": token_diag,
    }
    write_json(summary_dir / "diagnostic_summary.json", summary)

    commit_manifest = {
        **{k: v for k, v in extract_man.items() if k != "activations_path"},
        "activations_gitignored": True,
        "activations_path_gitignored": extract_man.get("activations_path"),
        "scored_at": utc_now_iso(),
        "n_bootstrap": args.n_bootstrap,
        "diagnostic_only": True,
        "probes_retrained": False,
        "eligibility_changed": False,
        "full_responses_regenerated": False,
        "c5_status": "HOLD",
        "causal_interventions_performed": False,
    }
    write_json(summary_dir / "extraction_manifest.json", commit_manifest)
    preflight = json.loads(
        (run_dir / "b1_compatibility_preflight.json").read_text(encoding="utf-8")
    )
    write_json(summary_dir / "b1_compatibility_preflight.json", preflight)
    write_json(
        summary_dir / "eligibility_confirmation.json",
        {
            "verified_at": utc_now_iso(),
            "unchanged": True,
            "contrasts": {
                name: {
                    "n_paired_valid": elig["contrasts"][name]["n_paired_valid"],
                    "paired_ids_sha256": elig["contrasts"][name]["paired_ids_sha256"],
                }
                for name in EXPECTED_ELIGIBILITY
            },
        },
    )
    _write_report(summary, extract_man, report_path)

    print(
        json.dumps(
            {
                "run_id": extract_man["run_id"],
                "preflight_min_cosine": extract_man["preflight_min_cosine_all"],
                "C3_vs_C2_natural": contrasts["C3_vs_C2"]["natural"],
                "C3_vs_C2_delta_auroc": contrasts["C3_vs_C2"][
                    "auroc_natural_minus_controlled"
                ],
                "C3_vs_C4_natural_auroc": contrasts["C3_vs_C4"]["natural"]["auroc"],
                "same_token_fraction": token_diag["c2_c3_paired_same_token"],
                "report": str(report_path),
            },
            indent=2,
            default=float,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
