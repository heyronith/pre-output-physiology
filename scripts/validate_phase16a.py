#!/usr/bin/env python3
"""Phase 16A sampling-scaling forecast integrity checks."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase15_onset import (  # noqa: E402
    PHASE14_RAW_CONTINUATIONS_SHA256,
    PHASE14_RUN_ID,
    RULE_HASH,
    RULE_VERSION,
)
from pre_output_physiology.phase16_forecast import (  # noqa: E402
    DESIGN_HASH,
    GUARANTEE,
    MONTE_CARLO_SEED,
    N_GRID,
    N_REPLICATES,
    STATUS_AUTHORIZED,
    STATUS_EXPENSIVE,
    STATUS_PLAUSIBLE,
    STATUS_UNSUPPORTED,
    TEMPERATURE_GRID,
    design_manifest,
    empirical_prompt_counts,
    interpret_forecast,
    run_primary_forecast,
)

PHASE15_RULE_HASH = RULE_HASH
PHASE15_RULE_VERSION = RULE_VERSION

STATUSES = {
    STATUS_AUTHORIZED,
    STATUS_PLAUSIBLE,
    STATUS_EXPENSIVE,
    STATUS_UNSUPPORTED,
}
CFG = REPO_ROOT / "configs/experiments/phase16_same_prompt_scaling_forecast.yaml"
ART = REPO_ROOT / "artifacts/phase16a_scaling_forecast"
P15_SUMMARY = REPO_ROOT / "artifacts/phase15a_onset_reanalysis/reanalysis_summary.json"
P14_CFG = REPO_ROOT / "configs/experiments/phase14_same_prompt_trajectory.yaml"
P15_CFG = REPO_ROOT / "configs/experiments/phase15_same_prompt_onset_reanalysis.yaml"
RAW = (
    REPO_ROOT
    / "artifacts/runs"
    / PHASE14_RUN_ID
    / "calibration_continuations.jsonl"
)


def main() -> int:
    fails: list[str] = []
    n_pass = 0

    def check(cond: bool, msg: str) -> None:
        nonlocal n_pass
        if cond:
            n_pass += 1
            print(f"PASS  {msg}")
        else:
            fails.append(msg)
            print(f"FAIL  {msg}")

    cfg = yaml.safe_load(CFG.read_text("utf-8"))
    status = cfg["status"]
    check(status in STATUSES, f"status {status}")
    auth = cfg["authorizations"]
    for k in (
        "model_calls_authorized",
        "final_model_calls_authorized",
        "locked_family_model_calls_authorized",
        "activation_extraction_authorized",
        "probe_fitting_authorized",
        "phase5_probe_scoring_authorized",
        "resampling_from_model_authorized",
        "prompt_changes_authorized",
        "temperature_extension_authorized",
        "causal_intervention_authorized",
    ):
        check(auth.get(k) is False, f"{k}=false")
    check(auth.get("monte_carlo_simulation_authorized") is True, "MC authorized")
    pre = status == STATUS_AUTHORIZED
    if pre:
        check(auth.get("forecast_authorized") is True, "pre-run forecast authorized")
    else:
        check(auth.get("forecast_authorized") is False, "post-run forecast closed")

    check(
        cfg["phase15_merge_sha"] == "3263ae8a7f4bd6984692cd303ffbe03ed9f7c4b3",
        "phase15 merge",
    )
    p14 = yaml.safe_load(P14_CFG.read_text("utf-8"))
    p15c = yaml.safe_load(P15_CFG.read_text("utf-8"))
    check(
        p14["status"] == "phase14a_same_prompt_trajectory_calibration_hold",
        "Phase 14A remains HOLD",
    )
    check(
        p15c["status"] == "phase15a_same_prompt_onset_reanalysis_hold",
        "Phase 15A remains HOLD",
    )
    dec = (REPO_ROOT / "docs/decision_log.md").read_text("utf-8")
    if pre:
        check("### D123" in dec, "D123")
    else:
        check(all(f"### {d}" in dec for d in ("D123", "D124")), "D123-D124")

    check(list(cfg["design"]["n_grid"]) == list(N_GRID), "N grid")
    check(list(cfg["design"]["temperature_grid"]) == list(TEMPERATURE_GRID), "T grid")
    check(cfg["design"]["monte_carlo_seed"] == MONTE_CARLO_SEED, "MC seed")
    check(cfg["design"]["n_replicates"] == N_REPLICATES, "MC replicates")
    check(cfg["design"]["phase14_run_id"] == PHASE14_RUN_ID, "source run")
    check(
        cfg["design"]["raw_continuations_sha256"] == PHASE14_RAW_CONTINUATIONS_SHA256,
        "raw sha config",
    )
    check(cfg["design"]["phase15_rule_hash"] == PHASE15_RULE_HASH, "P15 rule hash")
    check(cfg["design"]["phase15_rule_version"] == PHASE15_RULE_VERSION, "P15 rule ver")

    check(RAW.is_file(), "raw present")
    raw_sha = hashlib.sha256(RAW.read_bytes()).hexdigest()
    check(raw_sha == PHASE14_RAW_CONTINUATIONS_SHA256, "raw sha verifies")
    p15 = json.loads(P15_SUMMARY.read_text("utf-8"))
    check(p15["rule_hash"] == PHASE15_RULE_HASH, "P15 summary rule hash")
    check(len(p15["annotated_rows"]) == 1536, "1536 annotated")

    fresh = design_manifest()
    check(fresh["design_hash"] == DESIGN_HASH, "design hash stable")

    if not pre:
        man = json.loads((ART / "design_manifest.json").read_text("utf-8"))
        check(man["design_hash"] == DESIGN_HASH, "frozen design artifact")
        summary = json.loads((ART / "forecast_summary.json").read_text("utf-8"))
        check(summary["status"] == status, "summary status")
        check(summary["model_calls"] == 0, "zero model calls")
        check(
            summary["phase14_status_unchanged"]
            == "phase14a_same_prompt_trajectory_calibration_hold",
            "keeps P14 HOLD",
        )
        check(
            summary["phase15_status_unchanged"]
            == "phase15a_same_prompt_onset_reanalysis_hold",
            "keeps P15 HOLD",
        )
        check(summary["monte_carlo_seed"] == MONTE_CARLO_SEED, "summary seed")
        # Recompute interpretation deterministically from primary (reload counts)
        counts = empirical_prompt_counts(p15["annotated_rows"])
        # Spot-check: re-run smaller is not required; verify stored interpretation
        # matches recompute from stored primary probabilities for the gate.
        stored_interp = summary["interpretation"]
        # Rebuild primary keys as ints for interpret_forecast
        primary_int: dict[str, dict[int, dict]] = {}
        for t, by_n in summary["primary_forecast"].items():
            primary_int[t] = {int(n): cell for n, cell in by_n.items()}
        recomputed = interpret_forecast(primary_int)
        check(recomputed["outcome"] == stored_interp["outcome"], "outcome recomputes")
        check(recomputed["status"] == status, "status recomputes from primary")
        check(
            cfg["forecast_result"]["status"] == status,
            "result status in config",
        )
        check(
            cfg["forecast_result"]["outcome"] == stored_interp["outcome"],
            "outcome in config",
        )
        report = REPO_ROOT / "reports/phase16a_scaling_forecast.md"
        check(report.is_file(), "report present")
        check(GUARANTEE in report.read_text("utf-8"), "guarantee")
        check(
            not [
                p
                for p in (REPO_ROOT / "artifacts").rglob("*")
                if p.is_file()
                and "phase16" in str(p)
                and p.suffix in {".safetensors", ".npz", ".pt", ".pth"}
            ],
            "no activations",
        )
        # Determinism smoke: re-run primary with same seed matches stored key probs
        primary2 = run_primary_forecast(counts)
        for t in TEMPERATURE_GRID:
            for n in (16, 128):
                a = summary["primary_forecast"][str(t)][str(n)]["prob_groups_ge4_ge20"]
                b = primary2[str(t)][n]["prob_groups_ge4_ge20"]
                check(abs(a - b) < 1e-12, f"deterministic P ge4@20 T={t} N={n}")

    print()
    print(f"{n_pass} passed, {len(fails)} failed")
    for f in fails:
        print(f"  - {f}")
    if fails:
        return 1
    print("Phase 16A validation OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
