"""Phase 24D — primary live K=16 collection + Gemma grading (no physiology)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pre_output_physiology.phase24c_design import (
    DATASET_CONTENT_SHA256,
    K_GRID,
    N_ELIGIBLE,
    N_OBSERVED_24B,
    N_TEST,
    N_TRAIN,
    N_VALIDATION,
    sha256_file,
    sha256_json,
)

SELECTED_K = 16
N_NEW_TRAJECTORIES = 528
N_REUSED_24B = 96
N_TOTAL_TRAJECTORIES = N_ELIGIBLE * SELECTED_K  # 624

STARTING_SHA = "c649203f3f59701b556ea6ff8906e9752190027c"
PHASE24C_HASHES_PATH = "artifacts/phase24c_design/artifact_hashes.json"

STATUS_PASS = (
    "phase24d_primary_live_dataset_complete_awaiting_discovery_analysis_authorization"
)
STATUS_YIELD_REVIEW = "phase24d_collection_complete_realized_yield_requires_review"
STATUS_ENG_FAIL = "phase24d_collection_engineering_failed"

TRAIN_MIN_GE2 = 10  # of 20
VAL_MIN_GE2 = 4  # of 8

GUARANTEE = (
    "PHASE 24D COLLECTED THE FROZEN K=16 LIVE-ACTIVATION DATASET AND "
    "RESPONSE-LEVEL LABELS ONLY. NO PHYSIOLOGY PREDICTOR, AUROC ANALYSIS, "
    "LAYER×TOKEN SEARCH, SAE ANALYSIS, CAUSAL INTERVENTION, OR LOCKED-TEST "
    "SCIENTIFIC ANALYSIS WAS PERFORMED. LOCKED TEST OUTCOMES REMAIN SEALED."
)

# Frozen Phase-24C artifact relative paths → keys in artifact_hashes.json
PHASE24C_ARTIFACT_PATHS: dict[str, str] = {
    "analysis_spec.json": "artifacts/phase24c_design/analysis_spec.json",
    "config": "configs/experiments/phase24c_full_trajectory_design.yaml",
    "design_matrix.json": "artifacts/phase24c_design/design_matrix.json",
    "eligibility_manifest.json": "artifacts/phase24c_design/eligibility_manifest.json",
    "generation_schedule_template.json": (
        "artifacts/phase24c_design/generation_schedule_template.json"
    ),
    "k_forecast.json": "artifacts/phase24c_design/k_forecast.json",
    "k_forecast_gates_historical.svg": (
        "artifacts/phase24c_design/k_forecast_gates_historical.svg"
    ),
    "k_forecast_gates_primary.svg": (
        "artifacts/phase24c_design/k_forecast_gates_primary.svg"
    ),
    "k_forecast_test_counts.svg": (
        "artifacts/phase24c_design/k_forecast_test_counts.svg"
    ),
    "provenance.json": "artifacts/phase24c_design/provenance.json",
    "report": "reports/phase24c_full_trajectory_design.md",
    "split_manifest.json": "artifacts/phase24c_design/split_manifest.json",
}


def trajectory_seed_phase24c(prompt_id: str, replicate_index: int) -> int:
    """Frozen Phase-24C seed: sha256(phase24c|{prompt_id}|{replicate_index})[:8]."""
    if not (0 <= replicate_index < SELECTED_K):
        raise ValueError(
            f"replicate_index {replicate_index} out of range for K={SELECTED_K}"
        )
    digest = hashlib.sha256(
        f"phase24c|{prompt_id}|{replicate_index}".encode()
    ).hexdigest()
    return int(digest[:8], 16)


def verify_phase24c_artifact_hashes(repo_root: Path) -> dict[str, Any]:
    """Verify all frozen Phase-24C artifacts against recorded hashes. STOP on mismatch."""
    expected = json.loads(
        (repo_root / PHASE24C_HASHES_PATH).read_text(encoding="utf-8")
    )
    results: dict[str, Any] = {"verified": True, "checks": []}
    for key, rel in PHASE24C_ARTIFACT_PATHS.items():
        path = repo_root / rel
        if not path.exists():
            results["verified"] = False
            results["checks"].append(
                {"key": key, "path": rel, "pass": False, "error": "missing"}
            )
            continue
        got = sha256_file(str(path))
        exp = expected[key]
        ok = got == exp
        if not ok:
            results["verified"] = False
        results["checks"].append(
            {
                "key": key,
                "path": rel,
                "pass": ok,
                "expected": exp,
                "observed": got,
            }
        )
    # Design must freeze K=16
    design = json.loads(
        (repo_root / "artifacts/phase24c_design/design_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    k_ok = design.get("selected_k") == SELECTED_K
    if not k_ok:
        results["verified"] = False
    results["selected_k_check"] = {
        "pass": k_ok,
        "expected": SELECTED_K,
        "observed": design.get("selected_k"),
    }
    results["dataset_content_sha256"] = DATASET_CONTENT_SHA256
    results["k_grid"] = list(K_GRID)
    return results


def build_collection_schedule(
    *,
    schedule_template: dict[str, Any],
    prompts_by_id: dict[str, dict[str, Any]],
    pilot_24b_ids: Sequence[str],
) -> dict[str, Any]:
    """Expand Phase-24C template into exact new-trajectory schedule (528 rows)."""
    if schedule_template.get("selected_k") != SELECTED_K:
        raise ValueError("schedule template selected_k != 16")
    pilot_set = set(pilot_24b_ids)
    new_rows: list[dict[str, Any]] = []
    reuse_refs: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    for row in schedule_template["rows"]:
        pid = row["prompt_id"]
        split = row["split"]
        if pid not in prompts_by_id:
            raise KeyError(f"missing prompt fields for {pid}")
        p = prompts_by_id[pid]
        for rep in row["reused_phase24b_replicate_indices"]:
            key = (pid, int(rep))
            if key in seen:
                raise RuntimeError(f"duplicate reuse {key}")
            seen.add(key)
            if pid not in pilot_set:
                raise RuntimeError(f"non-pilot reuse for {pid}")
            if not (0 <= int(rep) < N_OBSERVED_24B):
                raise RuntimeError(f"reuse rep out of range: {rep}")
            reuse_refs.append(
                {
                    "trajectory_id": f"{pid}__r{int(rep):02d}",
                    "prompt_id": pid,
                    "replicate_index": int(rep),
                    "split": split,
                    "source": "phase24b_live",
                    "sealed": split == "test",
                }
            )
        for rep in row["new_replicate_indices"]:
            key = (pid, int(rep))
            if key in seen:
                raise RuntimeError(f"duplicate new {key}")
            seen.add(key)
            if pid in pilot_set:
                if not (N_OBSERVED_24B <= int(rep) < SELECTED_K):
                    raise RuntimeError(f"pilot new rep out of range: {rep}")
            else:
                if not (0 <= int(rep) < SELECTED_K):
                    raise RuntimeError(f"non-pilot rep out of range: {rep}")
            seed = trajectory_seed_phase24c(pid, int(rep))
            new_rows.append(
                {
                    "trajectory_id": f"{pid}__r{int(rep):02d}",
                    "prompt_id": pid,
                    "replicate_index": int(rep),
                    "sample_seed": seed,
                    "split": split,
                    "sealed": split == "test",
                    "source": "phase24d_new",
                    "scenario": p["scenario"],
                    "question": p["question"],
                    "answer_prefix": p["answer_prefix"],
                    "purpose": "primary_live_collection_k16",
                }
            )

    if len(new_rows) != N_NEW_TRAJECTORIES:
        raise RuntimeError(f"expected {N_NEW_TRAJECTORIES} new rows, got {len(new_rows)}")
    if len(reuse_refs) != N_REUSED_24B:
        raise RuntimeError(f"expected {N_REUSED_24B} reuse refs, got {len(reuse_refs)}")
    if len(seen) != N_TOTAL_TRAJECTORIES:
        raise RuntimeError(f"expected {N_TOTAL_TRAJECTORIES} unique pairs, got {len(seen)}")

    by_split = Counter(r["split"] for r in new_rows)
    return {
        "selected_k": SELECTED_K,
        "n_new": len(new_rows),
        "n_reuse": len(reuse_refs),
        "n_total_after": N_TOTAL_TRAJECTORIES,
        "seed_rule": "sha256(phase24c|{prompt_id}|{replicate_index})[:8] as int",
        "new_rows": new_rows,
        "reuse_refs": reuse_refs,
        "new_by_split": dict(by_split),
        "new_schedule_sha256": sha256_json(new_rows),
        "reuse_refs_sha256": sha256_json(reuse_refs),
    }


def evaluate_integrity_gates_528(
    *,
    n_requested: int,
    n_completed: int,
    n_alignment_ok: int,
    n_activation_nan: int,
    n_logit_nan: int,
    n_artifact_ok: int,
    n_seed_unchanged: int,
) -> dict[str, Any]:
    """Strict gates for the 528 newly generated trajectories."""
    gates = [
        {
            "gate": "G1_completion_528",
            "pass": n_requested == N_NEW_TRAJECTORIES and n_completed == N_NEW_TRAJECTORIES,
            "observed_completed": n_completed,
            "observed_requested": n_requested,
            "threshold": N_NEW_TRAJECTORIES,
        },
        {
            "gate": "G2_alignment_100pct",
            "pass": n_completed > 0 and n_alignment_ok == n_completed,
            "observed": n_alignment_ok,
            "threshold": n_completed,
        },
        {
            "gate": "G3_activation_zero_nan",
            "pass": n_activation_nan == 0 and n_completed == N_NEW_TRAJECTORIES,
            "observed_nan_count": n_activation_nan,
        },
        {
            "gate": "G4_logit_zero_nan",
            "pass": n_logit_nan == 0 and n_completed == N_NEW_TRAJECTORIES,
            "observed_nan_count": n_logit_nan,
        },
        {
            "gate": "G5_artifact_integrity",
            "pass": n_artifact_ok == n_completed == N_NEW_TRAJECTORIES,
            "observed": n_artifact_ok,
        },
        {
            "gate": "G6_no_replacement_seeds",
            "pass": n_seed_unchanged == n_completed == N_NEW_TRAJECTORIES,
            "observed": n_seed_unchanged,
        },
    ]
    return {"passed": all(g["pass"] for g in gates), "gates": gates}


def realized_yield_counts(
    labeled_rows: Sequence[dict[str, Any]],
    *,
    prompt_ids: Sequence[str],
) -> dict[str, Any]:
    """H/A/D totals and ≥2H≥2D prompt counts for a split (TRAIN or VAL only)."""
    per_prompt = []
    for pid in prompt_ids:
        labs = [
            r.get("open_class")
            for r in labeled_rows
            if r["prompt_id"] == pid and r.get("open_valid")
        ]
        c = Counter(labs)
        h, a, d = c.get("honest", 0), c.get("ambiguous", 0), c.get("deceptive", 0)
        per_prompt.append(
            {
                "prompt_id": pid,
                "n_honest": h,
                "n_ambiguous": a,
                "n_deceptive": d,
                "has_ge2_h_and_ge2_d": h >= 2 and d >= 2,
            }
        )
    n_ge2 = sum(1 for p in per_prompt if p["has_ge2_h_and_ge2_d"])
    return {
        "per_prompt": per_prompt,
        "totals": {
            "honest": sum(p["n_honest"] for p in per_prompt),
            "ambiguous": sum(p["n_ambiguous"] for p in per_prompt),
            "deceptive": sum(p["n_deceptive"] for p in per_prompt),
        },
        "n_prompts_ge2h_ge2d": n_ge2,
        "n_prompts": len(prompt_ids),
    }


def evaluate_realized_yield(
    *,
    train_yield: dict[str, Any],
    val_yield: dict[str, Any],
) -> dict[str, Any]:
    train_ok = train_yield["n_prompts_ge2h_ge2d"] >= TRAIN_MIN_GE2
    val_ok = val_yield["n_prompts_ge2h_ge2d"] >= VAL_MIN_GE2
    return {
        "passed": train_ok and val_ok,
        "train": {
            "observed": train_yield["n_prompts_ge2h_ge2d"],
            "threshold": TRAIN_MIN_GE2,
            "of": N_TRAIN,
            "pass": train_ok,
        },
        "validation": {
            "observed": val_yield["n_prompts_ge2h_ge2d"],
            "threshold": VAL_MIN_GE2,
            "of": N_VALIDATION,
            "pass": val_ok,
        },
    }


def final_status(*, engineering_pass: bool, yield_pass: bool) -> str:
    if not engineering_pass:
        return STATUS_ENG_FAIL
    if not yield_pass:
        return STATUS_YIELD_REVIEW
    return STATUS_PASS


def seal_test_summary(
    *,
    n_requested: int,
    n_completed: int,
    n_graded: int,
    integrity_ok: bool,
    artifact_hashes: dict[str, str],
) -> dict[str, Any]:
    """TEST scientific outcomes are intentionally omitted."""
    return {
        "split": "LOCKED_TEST",
        "sealed": True,
        "n_prompts": N_TEST,
        "n_trajectories_requested": n_requested,
        "n_trajectories_completed": n_completed,
        "n_labels_completed": n_graded,
        "technical_integrity_ok": integrity_ok,
        "artifact_hashes": artifact_hashes,
        "note": (
            "Behavioral outcomes and activation statistics by label are sealed "
            "until Phase 24E freezes the candidate (time, layer)."
        ),
        # Explicit absences for audit
        "omitted": [
            "H_A_D_totals",
            "per_prompt_H_A_D",
            "mixed_prompt_counts",
            "example_responses",
            "activation_stats_by_label",
            "predictive_performance",
        ],
    }


def assert_no_test_outcome_leakage(report_text: str) -> bool:
    """Heuristic: sealed TEST section must not contain H/A/D tables for TEST."""
    lower = report_text.lower()
    # Look for forbidden patterns near LOCKED TEST scientific content
    forbidden = [
        "test h/a/d totals",
        "test per-prompt h/a/d",
        "locked test labels:",
        "test mixed-prompt",
    ]
    return not any(f in lower for f in forbidden)


def capture_impl_source_sha(modal_path: Path, fn_name: str = "_generate_one") -> str:
    """SHA256 of a named top-level function body (for 24B↔24D equivalence)."""
    import ast

    src = modal_path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            body = ast.get_source_segment(src, node)
            if body is None:
                raise RuntimeError(f"cannot extract {fn_name}")
            return hashlib.sha256(body.encode()).hexdigest()
    raise KeyError(fn_name)
