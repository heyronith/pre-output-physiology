#!/usr/bin/env python3
"""Freeze Phase-24C full-trajectory design + K yield forecast (no GPU)."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology import phase24c_design as _p24c_design  # noqa: E402
from pre_output_physiology.phase21_roleplay import (  # noqa: E402
    APOLLO_COMMIT,
    APOLLO_REPO,
    DATASET_BLOB_SHA,
    DATASET_CONTENT_SHA256,
    DATASET_RELPATH,
    N_PROMPTS,
)
from pre_output_physiology.phase24c_design import (  # noqa: E402
    A100_USD_PER_HOUR,
    BYTES_PER_TRAJ_24B,
    CANDIDATE_MIN_CONSEC_LAYERS,
    CANDIDATE_MIN_MEDIAN_DELTA,
    FORECAST_SEED,
    GPU_SEC_PER_TRAJ_24B,
    GUARANTEE,
    K_GRID,
    LOCAL_PROMPTS_JSONL,
    LOCAL_RAW_DATASET,
    LOGISTIC_C_GRID,
    N_MC,
    N_OBSERVED_24B,
    TEMPORAL_LANDMARKS,
    TEST_BOOTSTRAP_REPS,
    VAL_BOOTSTRAP_REPS,
    build_train_val_test_split,
    project_generation_load,
    run_historical_sensitivity_forecast,
    run_primary_live_forecast,
    select_k_from_forecasts,
    sha256_file,
    sha256_json,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

INV24A = REPO_ROOT / "artifacts/phase24a_replay/k20_label_only_inventory.json"
PILOT24B = REPO_ROOT / "artifacts/phase24b_live/pilot_manifest.json"
SUMMARY24B = REPO_ROOT / "artifacts/phase24b_live/summary.json"
CORPUS = REPO_ROOT / "data/processed/phase23_open_grader/reference_corpus.jsonl"
OUT = REPO_ROOT / "artifacts/phase24c_design"
CFG = REPO_ROOT / "configs/experiments/phase24c_full_trajectory_design.yaml"
REPORT = REPO_ROOT / "reports/phase24c_full_trajectory_design.md"
DECISION_LOG = REPO_ROOT / "docs/decision_log.md"


def main() -> int:
    if DATASET_CONTENT_SHA256 != _p24c_design.DATASET_CONTENT_SHA256:
        raise SystemExit("STOP: phase24c dataset SHA constant mismatch")
    raw_path = REPO_ROOT / LOCAL_RAW_DATASET
    prompts_path = REPO_ROOT / LOCAL_PROMPTS_JSONL
    raw_sha = sha256_file(str(raw_path))
    if raw_sha != DATASET_CONTENT_SHA256:
        raise SystemExit(
            f"STOP: raw dataset SHA mismatch: {raw_sha} != {DATASET_CONTENT_SHA256}"
        )
    prompts = [
        json.loads(x)
        for x in prompts_path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    if len(prompts) != N_PROMPTS:
        raise SystemExit(f"STOP: prompts.jsonl count {len(prompts)} != {N_PROMPTS}")
    prompt_ids = [r["prompt_id"] for r in prompts]
    if prompt_ids != [f"roleplay_{i:03d}" for i in range(N_PROMPTS)]:
        raise SystemExit("STOP: prompt ID sequence changed")

    inv = json.loads(INV24A.read_text(encoding="utf-8"))
    dev_eligible = list(inv["development"]["qualifying_prompt_ids"])
    locked_eligible = list(inv["locked_validation"]["qualifying_prompt_ids"])
    pilot = json.loads(PILOT24B.read_text(encoding="utf-8"))
    pilot_ids = list(pilot["prompt_ids"])
    summary24b = json.loads(SUMMARY24B.read_text(encoding="utf-8"))

    split = build_train_val_test_split(
        pilot_24b_ids=pilot_ids,
        development_eligible=dev_eligible,
        locked_eligible=locked_eligible,
    )

    # Phase-24B live counts
    pilot_counts: dict[str, tuple[int, int, int]] = {}
    for row in summary24b["per_prompt"]:
        pid = row["prompt_id"]
        h, a, d = int(row["n_honest"]), int(row["n_ambiguous"]), int(row["n_deceptive"])
        if h + a + d != N_OBSERVED_24B:
            raise SystemExit(f"STOP: pilot {pid} counts sum != 6")
        pilot_counts[pid] = (h, a, d)
    if set(pilot_counts) != set(pilot_ids):
        raise SystemExit("STOP: pilot count IDs mismatch manifest")

    # Historical K=20 GPT counts for 39
    corpus = [
        json.loads(x)
        for x in CORPUS.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    eligible = set(dev_eligible) | set(locked_eligible)
    by: dict[str, Counter] = defaultdict(Counter)
    for r in corpus:
        if r["prompt_id"] in eligible:
            by[r["prompt_id"]][r["reference_label"]] += 1
    historical_counts: dict[str, tuple[int, int, int]] = {}
    for pid in sorted(eligible):
        c = by[pid]
        # Fold GPT "exclude" into ambiguous so H/A/D sum to K=20 for Dirichlet.
        h = int(c["honest"])
        a = int(c["ambiguous"]) + int(c["exclude"])
        d = int(c["deceptive"])
        historical_counts[pid] = (h, a, d)
        if h + a + d != 20:
            raise SystemExit(
                f"STOP: historical {pid} H+A+D={h+a+d} != 20 "
                f"(raw={dict(c)})"
            )
        if h < 2 or d < 2:
            raise SystemExit(f"STOP: historical eligibility broken for {pid}")

    git_commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()

    print("Running primary live forecast (100k MC)...", flush=True)
    primary = run_primary_live_forecast(
        pilot_counts=pilot_counts,
        train_ids=split["train_prompt_ids"],
        val_ids=split["validation_prompt_ids"],
        test_ids=split["test_prompt_ids"],
    )
    print("Running historical sensitivity forecast...", flush=True)
    historical = run_historical_sensitivity_forecast(
        historical_counts=historical_counts,
        train_ids=split["train_prompt_ids"],
        val_ids=split["validation_prompt_ids"],
        test_ids=split["test_prompt_ids"],
    )
    selection = select_k_from_forecasts(primary, historical)
    status = selection["status"]

    provenance = {
        "source_provider": APOLLO_REPO,
        "upstream_commit": APOLLO_COMMIT,
        "dataset_relpath": DATASET_RELPATH,
        "dataset_blob_sha": DATASET_BLOB_SHA,
        "dataset_content_sha256": DATASET_CONTENT_SHA256,
        "local_raw_path": LOCAL_RAW_DATASET,
        "local_raw_sha256_verified": raw_sha,
        "local_prompts_jsonl": LOCAL_PROMPTS_JSONL,
        "local_prompts_sha256": sha256_file(str(prompts_path)),
        "n_prompts": 371,
        "prompt_ids_sha256": sha256_json(prompt_ids),
        "formatting": (
            "Phase-21 format_prompt_for_generation / fold_system; unchanged "
            "scenario/question/answer_prefix fields from Apollo dataset.yaml"
        ),
        "phases_21_24_unchanged": True,
        "verification": "PASS",
    }

    eligibility = {
        "rule": "historical GPT-4o K=20 label-only ≥2H and ≥2D; no onset",
        "development_eligible": dev_eligible,
        "locked_eligible": locked_eligible,
        "n_development": len(dev_eligible),
        "n_locked": len(locked_eligible),
        "n_total": len(eligible),
        "development_sha256": sha256_json(dev_eligible),
        "locked_sha256": sha256_json(locked_eligible),
        "all_eligible_sha256": sha256_json(sorted(eligible)),
        "label_source": "gpt-4o-2024-08-06_frozen_reference_labels",
        "gemma_not_used_for_eligibility": True,
    }

    analysis_spec = {
        "outcome_primary": "honest_vs_deceptive",
        "ambiguous_excluded_from_primary_fit": True,
        "grader": {
            "hf_id": "google/gemma-4-31B-it",
            "revision": "842da3794eaa0b77d5f08bae87a17459d91ff475",
            "onset_grader": None,
        },
        "temporal_landmarks": list(TEMPORAL_LANDMARKS),
        "prediction_step_convention": (
            "state at step t exists before generated token t is sampled; "
            "surface prefix is tokens 0..t-1 only"
        ),
        "step0_role": "prompt_conditioned_propensity_reference_not_rollout_trajectory",
        "primary_activation": "post_block_residual_L0_L31",
        "primary_representation": "delta_h(t,L)=h(t,L)-h(0,L)",
        "secondary_activations": ["raw_h", "embedding_pre_block", "final_rmsnorm"],
        "baselines": {
            "TEXT": "TF-IDF word 1-2 + char-within-word 3-5; logistic L2; vocab/IDF TRAIN-only",
            "LOGITS": "full next-token logits; standardized L2 logistic; scale TRAIN-only",
            "SURFACE": "cross-fitted stack of TEXT+LOGIT log-odds; L2 meta; OOF only",
            "ACTIVATION": "standardized L2 logistic on Δh(t,L)",
            "SURFACE_PLUS_ACTIVATION": "cross-fitted stack SURFACE+ACTIVATION log-odds",
        },
        "primary_contrast": "AUROC(SURFACE+ACTIVATION)-AUROC(SURFACE)",
        "logistic_C_grid": list(LOGISTIC_C_GRID),
        "class_weight": "balanced",
        "prompt_equal_total_weight": True,
        "grouped_cv": "prompt-group folds on TRAIN only; no same-prompt across folds",
        "validation_discovery": {
            "times": list(TEMPORAL_LANDMARKS),
            "layers": list(range(32)),
            "bootstrap_prompt_cluster_reps": VAL_BOOTSTRAP_REPS,
        },
        "candidate_selection": {
            "min_consecutive_layers_delta_gt_0": CANDIDATE_MIN_CONSEC_LAYERS,
            "min_band_median_delta_auroc": CANDIDATE_MIN_MEDIAN_DELTA,
            "rule": "choose_earliest_time_then_best_band_then_best_layer",
            "tie_break": "lower_layer_number",
        },
        "confirmatory_test": {
            "refit_on_all_28_development": True,
            "evaluate_once_on_11_locked": True,
            "delta_auroc_bootstrap_reps": TEST_BOOTSTRAP_REPS,
            "support_requires_delta_gt_0_and_ci_low_gt_0": True,
            "no_posthoc_layer_time_search_on_test": True,
        },
        "claim_limits": (
            "additional linearly decodable predictive signal beyond frozen "
            "text-prefix and logit baselines; not hidden intent / not causal"
        ),
    }

    # Generation schedule template
    selected_k = selection.get("selected_k")
    schedule_template = None
    projections = None
    if selected_k is not None:
        projections = project_generation_load(selected_k=selected_k)
        rows = []
        for pid in split["train_prompt_ids"]:
            if pid in set(pilot_ids):
                reps = list(range(N_OBSERVED_24B, selected_k))
                reuse = list(range(N_OBSERVED_24B))
            else:
                reps = list(range(selected_k))
                reuse = []
            rows.append(
                {
                    "prompt_id": pid,
                    "split": "train",
                    "new_replicate_indices": reps,
                    "reused_phase24b_replicate_indices": reuse,
                }
            )
        for pid in split["validation_prompt_ids"]:
            rows.append(
                {
                    "prompt_id": pid,
                    "split": "validation",
                    "new_replicate_indices": list(range(selected_k)),
                    "reused_phase24b_replicate_indices": [],
                }
            )
        for pid in split["test_prompt_ids"]:
            rows.append(
                {
                    "prompt_id": pid,
                    "split": "test",
                    "new_replicate_indices": list(range(selected_k)),
                    "reused_phase24b_replicate_indices": [],
                }
            )
        schedule_template = {
            "selected_k": selected_k,
            "seed_rule": "sha256(phase24c|{prompt_id}|{replicate_index})[:8] as int",
            "rows": rows,
            "projections": projections,
            "note": "Template only — generation not authorized in Phase 24C",
        }

    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "provenance.json", provenance)
    write_json(OUT / "eligibility_manifest.json", eligibility)
    write_json(OUT / "split_manifest.json", split)
    write_json(
        OUT / "k_forecast.json",
        {
            "created_at": utc_now_iso(),
            "git_commit": git_commit,
            "k_grid": list(K_GRID),
            "n_mc": N_MC,
            "seed": FORECAST_SEED,
            "pilot_counts": {k: list(v) for k, v in pilot_counts.items()},
            "historical_counts": {k: list(v) for k, v in historical_counts.items()},
            "primary_live_forecast": primary,
            "historical_label_sensitivity": historical,
            "selection": selection,
        },
    )
    write_json(OUT / "analysis_spec.json", analysis_spec)
    if schedule_template is not None:
        write_json(OUT / "generation_schedule_template.json", schedule_template)

    # K-forecast plots as dependency-free SVG
    def _svg_line_chart(
        path: Path,
        *,
        title: str,
        series: list[tuple[str, list[float], str]],
        xs: list[float],
        y_min: float,
        y_max: float,
        hlines: list[tuple[float, str]],
        vline: float | None,
        width: int = 520,
        height: int = 320,
    ) -> None:
        pad_l, pad_r, pad_t, pad_b = 50, 20, 36, 40
        plot_w = width - pad_l - pad_r
        plot_h = height - pad_t - pad_b

        def xpix(x: float) -> float:
            return pad_l + (x - xs[0]) / (xs[-1] - xs[0]) * plot_w

        def ypix(y: float) -> float:
            return pad_t + (1.0 - (y - y_min) / (y_max - y_min)) * plot_h

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="{width // 2}" y="20" text-anchor="middle" '
            f'font-family="sans-serif" font-size="13">{title}</text>',
            f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" '
            f'fill="none" stroke="#ccc"/>',
        ]
        for yv, color in hlines:
            parts.append(
                f'<line x1="{pad_l}" y1="{ypix(yv):.1f}" x2="{pad_l + plot_w}" '
                f'y2="{ypix(yv):.1f}" stroke="{color}" stroke-dasharray="4,3"/>'
            )
        if vline is not None:
            parts.append(
                f'<line x1="{xpix(vline):.1f}" y1="{pad_t}" x2="{xpix(vline):.1f}" '
                f'y2="{pad_t + plot_h}" stroke="#2ca02c" stroke-dasharray="5,3"/>'
            )
        for label, ys, color in series:
            pts = " ".join(
                f"{xpix(x):.1f},{ypix(y):.1f}" for x, y in zip(xs, ys, strict=True)
            )
            parts.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2" '
                f'points="{pts}"/>'
            )
            for x, y in zip(xs, ys, strict=True):
                parts.append(
                    f'<circle cx="{xpix(x):.1f}" cy="{ypix(y):.1f}" r="3" '
                    f'fill="{color}"/>'
                )
            parts.append(f"<!-- series {label} color={color} -->")
        for x in xs:
            parts.append(
                f'<text x="{xpix(x):.1f}" y="{height - 12}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="11">{int(x)}</text>'
            )
        parts.append("</svg>")
        path.write_text("\n".join(parts), encoding="utf-8")

    ks = [float(k) for k in K_GRID]
    _svg_line_chart(
        OUT / "k_forecast_gates_primary.svg",
        title="Primary Phase-24B live gate probabilities",
        series=[
            (
                "TRAIN",
                [primary["by_k"][str(k)]["p_train_ge10_of_20_ge2"] for k in K_GRID],
                "#1f77b4",
            ),
            (
                "VAL",
                [primary["by_k"][str(k)]["p_val_ge4_of_8_ge2"] for k in K_GRID],
                "#ff7f0e",
            ),
            (
                "TEST",
                [primary["by_k"][str(k)]["p_test_ge5_of_11_ge2"] for k in K_GRID],
                "#d62728",
            ),
        ],
        xs=ks,
        y_min=0.0,
        y_max=1.05,
        hlines=[(0.90, "#1f77b4"), (0.80, "#888"), (0.50, "#c00")],
        vline=float(selected_k) if selected_k is not None else None,
    )
    _svg_line_chart(
        OUT / "k_forecast_gates_historical.svg",
        title="historical_label_sensitivity gate probabilities",
        series=[
            (
                "TRAIN",
                [
                    historical["by_k"][str(k)]["p_train_ge10_of_20_ge2"]
                    for k in K_GRID
                ],
                "#1f77b4",
            ),
            (
                "VAL",
                [historical["by_k"][str(k)]["p_val_ge4_of_8_ge2"] for k in K_GRID],
                "#ff7f0e",
            ),
            (
                "TEST",
                [historical["by_k"][str(k)]["p_test_ge5_of_11_ge2"] for k in K_GRID],
                "#d62728",
            ),
        ],
        xs=ks,
        y_min=0.0,
        y_max=1.05,
        hlines=[(0.90, "#1f77b4"), (0.80, "#888"), (0.50, "#c00")],
        vline=float(selected_k) if selected_k is not None else None,
    )
    _svg_line_chart(
        OUT / "k_forecast_test_counts.svg",
        title="Primary live — TEST median H/D counts",
        series=[
            (
                "H",
                [primary["by_k"][str(k)]["median_test_honest"] for k in K_GRID],
                "#2ca02c",
            ),
            (
                "D",
                [primary["by_k"][str(k)]["median_test_deceptive"] for k in K_GRID],
                "#9467bd",
            ),
        ],
        xs=ks,
        y_min=0.0,
        y_max=max(
            primary["by_k"][str(k)]["median_test_honest"] for k in K_GRID
        )
        * 1.15,
        hlines=[(20.0, "#888")],
        vline=float(selected_k) if selected_k is not None else None,
    )

    design = {
        "phase": "phase24c",
        "created_at": utc_now_iso(),
        "git_commit": git_commit,
        "starting_sha": "2a08fae5b32bbb611d92c1b872cb0d15696bc317",
        "status": status,
        "guarantee": GUARANTEE,
        "selected_k": selected_k,
        "provenance_sha256": sha256_json(provenance),
        "eligibility_sha256": sha256_json(eligibility),
        "split_train_sha256": split["train_sha256"],
        "split_validation_sha256": split["validation_sha256"],
        "split_test_sha256": split["test_sha256"],
        "analysis_spec_sha256": sha256_json(analysis_spec),
        "bytes_per_trajectory_reference": BYTES_PER_TRAJ_24B,
        "gpu_seconds_per_trajectory_reference": GPU_SEC_PER_TRAJ_24B,
        "usd_per_hour_reference": A100_USD_PER_HOUR,
        "authorizations": {
            "live_generation_authorized": False,
            "probe_fitting_authorized": False,
            "physiology_authorized": False,
            "sae_authorized": False,
            "causal_authorized": False,
            "test_inspection_authorized": False,
        },
    }
    write_json(OUT / "design_matrix.json", design)

    cfg = f"""# Phase 24C — full trajectory design freeze (no GPU)

experiment_id: phase24c_full_trajectory_design
phase: phase24c
status: {status}
starting_sha: 2a08fae5b32bbb611d92c1b872cb0d15696bc317
selected_k: {selected_k}

split:
  train_n: 20
  validation_n: 8
  test_n: 11
  train_sha256: {split['train_sha256']}
  validation_sha256: {split['validation_sha256']}
  test_sha256: {split['test_sha256']}

forecast:
  seed: {FORECAST_SEED}
  n_mc: {N_MC}
  k_grid: {list(K_GRID)}

authorizations:
  live_generation_authorized: false
  probe_fitting_authorized: false
  physiology_authorized: false
  sae_authorized: false
  causal_authorized: false
  test_inspection_authorized: false

notes: >
  Prospective design only. Phase-24B pilots are TRAIN-only. No new generation.
"""
    CFG.write_text(cfg, encoding="utf-8")

    # Report
    primary_table = "\n".join(
        f"| {k} | {primary['by_k'][str(k)]['p_train_ge10_of_20_ge2']:.4f} | "
        f"{primary['by_k'][str(k)]['p_val_ge4_of_8_ge2']:.4f} | "
        f"{primary['by_k'][str(k)]['p_test_ge5_of_11_ge2']:.4f} | "
        f"{primary['by_k'][str(k)]['median_test_honest']:.1f} | "
        f"{primary['by_k'][str(k)]['median_test_deceptive']:.1f} | "
        f"{'PASS' if primary['by_k'][str(k)]['all_gates_pass'] else 'fail'} |"
        for k in K_GRID
    )
    hist_table = "\n".join(
        f"| {k} | {historical['by_k'][str(k)]['p_train_ge10_of_20_ge2']:.4f} | "
        f"{historical['by_k'][str(k)]['p_val_ge4_of_8_ge2']:.4f} | "
        f"{historical['by_k'][str(k)]['p_test_ge5_of_11_ge2']:.4f} | "
        f"{historical['by_k'][str(k)]['median_test_honest']:.1f} | "
        f"{historical['by_k'][str(k)]['median_test_deceptive']:.1f} | "
        f"{'PASS' if historical['by_k'][str(k)]['all_gates_pass'] else 'fail'} |"
        for k in K_GRID
    )
    proj_txt = json.dumps(projections, indent=2) if projections else "n/a"
    report = f"""# Phase 24C — Full trajectory design freeze

**Status:** `{status}`

{GUARANTEE}

## Dataset provenance

| Field | Value |
|---|---|
| Source | `{APOLLO_REPO}` |
| Upstream commit | `{APOLLO_COMMIT}` |
| Dataset path | `{DATASET_RELPATH}` |
| Blob SHA | `{DATASET_BLOB_SHA}` |
| Content SHA256 | `{DATASET_CONTENT_SHA256}` |
| Local raw verified | `{raw_sha}` |
| Local prompts.jsonl | `{sha256_file(str(prompts_path))}` |
| Prompt count | 371 |
| Verification | **PASS** (unchanged vs Phases 21–24) |

## Eligible pool (GPT-4o K=20 label-only)

- DEVELOPMENT: 28 prompts (`{eligibility['development_sha256'][:16]}…`)
- LOCKED: 11 prompts (`{eligibility['locked_sha256'][:16]}…`)
- Total: 39

## Split

| Split | n | SHA256 |
|---|---|---|
| TRAIN | 20 | `{split['train_sha256']}` |
| VALIDATION | 8 | `{split['validation_sha256']}` |
| TEST | 11 | `{split['test_sha256']}` |

All 16 Phase-24B pilot prompts are **TRAIN-only**:

```
{json.dumps(pilot_ids, indent=2)}
```

TRAIN IDs:

```
{json.dumps(split['train_prompt_ids'], indent=2)}
```

VALIDATION IDs:

```
{json.dumps(split['validation_prompt_ids'], indent=2)}
```

TEST IDs:

```
{json.dumps(split['test_prompt_ids'], indent=2)}
```

## K forecast

Grid: `{list(K_GRID)}` · MC=`{N_MC}` · seed=`{FORECAST_SEED}`

### Primary — Phase-24B live forecast

| K | P(TRAIN≥10/20 ≥2H≥2D) | P(VAL≥4/8) | P(TEST≥5/11) | med TEST H | med TEST D | gates |
|---|---|---|---|---|---|---|
{primary_table}

### Historical sensitivity (GPT K=20; labeled)

| K | P(TRAIN≥10/20) | P(VAL≥4/8) | P(TEST≥5/11) | med TEST H | med TEST D | gates |
|---|---|---|---|---|---|---|
{hist_table}

### Selection

```json
{json.dumps(selection, indent=2)}
```

Selected K: **{selected_k}**

## Generation load (if K frozen)

```json
{proj_txt}
```

## Analysis freeze

- Temporal landmarks: `{list(TEMPORAL_LANDMARKS)}`
- Primary activation: post-block residual L0–L31; representation `Δh(t,L)=h(t,L)-h(0,L)`
- Predictors: TEXT / LOGITS / SURFACE / ACTIVATION / SURFACE+ACTIVATION
- Primary contrast: ΔAUROC = AUROC(SURFACE+ACTIVATION) − AUROC(SURFACE)
- C grid: `{list(LOGISTIC_C_GRID)}`; balanced + prompt-equal weights;
  prompt-group CV
- Candidate rule: earliest time with ≥{CANDIDATE_MIN_CONSEC_LAYERS}
  consecutive layers Δ>0 and median Δ≥{CANDIDATE_MIN_MEDIAN_DELTA};
  then best band; then best layer (tie → lower L)
- Confirmatory: ΔAUROC>0 and 95% prompt-cluster bootstrap CI lower bound >0
  ({TEST_BOOTSTRAP_REPS} reps)

## Authorizations

Live generation / probes / physiology / SAE / causal / TEST inspection: **all false**.
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")

    hashes = {
        "provenance.json": sha256_file(str(OUT / "provenance.json")),
        "eligibility_manifest.json": sha256_file(str(OUT / "eligibility_manifest.json")),
        "split_manifest.json": sha256_file(str(OUT / "split_manifest.json")),
        "k_forecast.json": sha256_file(str(OUT / "k_forecast.json")),
        "analysis_spec.json": sha256_file(str(OUT / "analysis_spec.json")),
        "design_matrix.json": sha256_file(str(OUT / "design_matrix.json")),
        "report": sha256_file(str(REPORT)),
        "config": sha256_file(str(CFG)),
    }
    if (OUT / "generation_schedule_template.json").exists():
        hashes["generation_schedule_template.json"] = sha256_file(
            str(OUT / "generation_schedule_template.json")
        )
    for plot_name in (
        "k_forecast_gates_primary.svg",
        "k_forecast_gates_historical.svg",
        "k_forecast_test_counts.svg",
    ):
        if (OUT / plot_name).exists():
            hashes[plot_name] = sha256_file(str(OUT / plot_name))
    write_json(OUT / "artifact_hashes.json", hashes)

    log = DECISION_LOG.read_text(encoding="utf-8")
    if "### D160 —" not in log:
        decision = (
            "Starting from Phase-24B PASS `2a08fae…`, Phase 24C freezes the "
            "prospective live-activation trajectory experiment design without GPU "
            f"calls. Apollo RoleplayDeception provenance verified (content SHA "
            f"`7d3e36dc…`). Split: TRAIN20/VAL8/TEST11 with all 16 Phase-24B "
            f"pilots TRAIN-only. Primary live Dirichlet forecast (seed 2403, "
            f"100k MC) selects K={selected_k} under frozen yield gates; status "
            f"`{status}`. Analysis landmarks, Δh representation, surface "
            "baselines, candidate-selection, and confirmatory ΔAUROC CI rule "
            "frozen. No generation, grading, probing, SAE, or physiology."
        )
        entry = (
            "\n### D160 — Phase 24C full-trajectory design freeze\n\n"
            "- **Type:** **OUR RESEARCH DECISION**\n"
            f"- **Decision:** {decision}\n"
            "- **Date:** 2026-09-30\n"
        )
        DECISION_LOG.write_text(log.rstrip() + "\n" + entry, encoding="utf-8")

    print(
        json.dumps(
            {
                "status": status,
                "selected_k": selected_k,
                "train_sha": split["train_sha256"],
                "projections": projections,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
