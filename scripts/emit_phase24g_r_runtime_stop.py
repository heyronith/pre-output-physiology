#!/usr/bin/env python3
"""Emit Phase 24G-R runtime STOP package (no protocol change)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pre_output_physiology.phase24c_design import sha256_file  # noqa: E402
from pre_output_physiology.phase24g_r_correction import (  # noqa: E402
    GUARANTEE,
    NESTED_REPS,
    STARTING_SHA,
    STATUS_RUNTIME_STOP,
    assert_phase24f_immutable,
    snapshot_prior_24g_hashes,
    verify_prior_24g_unchanged,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

OUT = REPO / "artifacts/phase24g_r_correction"
REPORT = REPO / "reports/phase24g_r_diagnostic_correction.md"
AUDIT = REPO / "reports/phase24g_r_code_audit_manifest.md"
CFG = REPO / "configs/experiments/phase24g_r_diagnostic_correction.yaml"
DECISION_LOG = REPO / "docs/decision_log.md"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    imm = assert_phase24f_immutable(REPO)
    prior = snapshot_prior_24g_hashes(REPO)
    prior_check = verify_prior_24g_unchanged(REPO, prior)

    ckpt = OUT / "lopo_full_checkpoint.jsonl"
    rows = []
    if ckpt.exists():
        rows = [
            json.loads(line)
            for line in ckpt.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    n_partial = len(rows)
    exact = sum(
        1
        for r in rows
        if r.get("candidate")
        and r["candidate"].get("candidate_time") == 1
        and r["candidate"].get("candidate_layer") == 20
    )

    measured_s_per_grid = 98.5  # clean single-process microbench
    observed_s_per_grid = 140.0  # under host swap pressure during serial LOPO
    nested_hours_clean = NESTED_REPS * measured_s_per_grid / 3600
    nested_hours_observed = NESTED_REPS * observed_s_per_grid / 3600

    runtime = {
        "verdict": "PROHIBITIVE",
        "status": STATUS_RUNTIME_STOP,
        "reason": (
            "Full frozen SURFACE+ACTIVATION stacked discovery grids are "
            "~98.5s clean / ~140s under memory pressure on this 24GB host. "
            "Multiprocess fork pools caused ~17GB swap thrash and were "
            "abandoned. Serial in-process 1000 nested would require "
            f"~{nested_hours_observed:.1f}h wall "
            f"(~{nested_hours_clean:.1f}h clean). Protocol forbids silently "
            "reducing repetitions or substituting the logits-only proxy for "
            "primary diagnostics."
        ),
        "seconds_per_full_grid_clean": measured_s_per_grid,
        "seconds_per_full_grid_observed_under_pressure": observed_s_per_grid,
        "nested_1000_serial_hours_clean": nested_hours_clean,
        "nested_1000_serial_hours_observed": nested_hours_observed,
        "parallel_attempt": {
            "workers_tried": [4, 6, 10, 16],
            "outcome": "swap_thrash",
            "peak_swap_gb_observed": 17.5,
        },
        "partial_lopo_completed": n_partial,
        "partial_lopo_exact_t1_l20": exact,
        "protocol_unchanged": {
            "nested_reps_required": NESTED_REPS,
            "proxy_forbidden_for_primary": True,
            "full_stacked_required": True,
        },
        "action_required": (
            "Authorize either (a) a higher-RAM host / cloud runner for the "
            "unchanged 1000-rep full-pipeline protocol, or (b) an explicit "
            "protocol amendment. Do not silently reduce reps or use proxy."
        ),
        "created_at": utc_now_iso(),
    }
    write_json(OUT / "runtime_stop.json", runtime)
    # Keep benchmark aligned with STOP gate
    write_json(
        OUT / "runtime_benchmark.json",
        {
            **runtime,
            "minutes_per_full_grid": measured_s_per_grid / 60,
            "seconds_per_full_grid": measured_s_per_grid,
            "n_finite_delta_cells": 224,
            "action": runtime["action_required"],
        },
    )
    write_json(OUT / "phase24f_immutability.json", imm)
    write_json(OUT / "prior_phase24g_artifact_hashes.json", prior)
    write_json(OUT / "prior_phase24g_unchanged_verification.json", prior_check)
    write_json(
        OUT / "partial_lopo_full_pipeline.json",
        {
            "note": "PARTIAL only — STOP before completing 28/1000",
            "n": n_partial,
            "rows": rows,
            "frac_exact_t1_l20_among_partial": (
                exact / n_partial if n_partial else float("nan")
            ),
        },
    )

    git_commit = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO)
        .decode()
        .strip()
    )
    write_json(
        OUT / "correction_manifest.json",
        {
            "created_at": utc_now_iso(),
            "starting_sha": STARTING_SHA,
            "git_commit": git_commit,
            "status": STATUS_RUNTIME_STOP,
            "nested_reps_required": NESTED_REPS,
            "nested_reps_completed": 0,
            "lopo_completed": n_partial,
            "proxy_used_in_primary": False,
            "phase24f_immutable": True,
            "prior_24g_unchanged": prior_check["unchanged"],
            "guarantee_intent": GUARANTEE,
            "stopped": True,
        },
    )

    CFG.write_text(
        f"""# Phase 24G-R — STOPPED (runtime prohibitive on this host)

experiment_id: phase24g_r_diagnostic_correction
phase: phase24g_r
status: {STATUS_RUNTIME_STOP}
starting_sha: {STARTING_SHA}
nested_reps_required: {NESTED_REPS}
nested_reps_completed: 0
lopo_partial_completed: {n_partial}
proxy_used_in_primary: false
phase24f_primary_pass: false

authorizations:
  exploratory_corrected_diagnostics: true
  primary_endpoint_redefinition: false
  sae_analysis_authorized: false
  causal_intervention_authorized: false
  protocol_amendment_to_reduce_reps_or_use_proxy: false
""",
        encoding="utf-8",
    )

    report = f"""# Phase 24G-R — RUNTIME STOP

**Status:** `{STATUS_RUNTIME_STOP}`

## Why we stopped

Full frozen stacked SURFACE+ACTIVATION discovery grids measure
**{measured_s_per_grid:.1f}s** clean (and **~{observed_s_per_grid:.0f}s** under
host memory pressure). Multiprocess pools induced **~17GB swap thrash** and
were abandoned. Serial 1,000 nested would require
**~{nested_hours_observed:.1f} hours** wall on this 24GB host.

Per authorization: **do not silently reduce repetitions or use a proxy**.

## Phase-24F immutability

Verified: **{imm['immutable']}** (ΔAUROC={imm['delta_auroc']:.4f}, CI={imm['ci95']})

## Prior Phase-24G artifacts

Unchanged: **{prior_check['unchanged']}**

## Partial progress (not a primary result)

Full-pipeline LOPO completed **{n_partial}/28** before STOP.
Among those, exact t=1/L20 selections: **{exact}/{n_partial}**.

Nested 1000: **0/{NESTED_REPS}** completed.

## Required next step

Authorize a higher-RAM / cloud runner for the **unchanged** protocol, or an
explicit protocol amendment. No scientific reclassification is issued under
STOP.
"""
    REPORT.write_text(report, encoding="utf-8")

    diff_stat = subprocess.check_output(
        ["git", "diff", "--stat", STARTING_SHA], cwd=REPO
    ).decode()
    audit = f"""# Phase 24G-R code audit manifest (RUNTIME STOP)

Starting SHA: `{STARTING_SHA}`
Working tree: `{git_commit}`
Status: `{STATUS_RUNTIME_STOP}`

## Changed files (purpose)

| File | Purpose |
|---|---|
| `src/pre_output_physiology/phase24g_r_correction.py` | Constants, immutability, summaries |
| `scripts/run_phase24g_r_correction.py` | Full-pipeline LOPO + nested runner |
| `scripts/benchmark_phase24g_r_full_grid.py` | Runtime gate |
| `scripts/emit_phase24g_r_runtime_stop.py` | STOP package emitter |
| `scripts/run_phase24g_diagnostics.py` | mmap NPZ load + slice Δh (numerically identical speedup) |
| `tests/test_phase24g_r_correction.py` | Invariant tests |
| `configs/experiments/phase24g_r_diagnostic_correction.yaml` | Config |
| `reports/phase24g_r_diagnostic_correction.md` | STOP report |
| `reports/phase24g_r_code_audit_manifest.md` | This audit |
| `artifacts/phase24g_r_correction/*` | Runtime/partial artifacts only |
| `docs/decision_log.md` | D165 STOP entry |

## Critical functions (implemented; not fully executed)

### LOPO
- `scripts/run_phase24g_r_correction.py::_lopo_worker`
- `scripts/run_phase24g_r_correction.py::_run_one_discovery` (`proxy=False`)

### Nested
- `scripts/run_phase24g_r_correction.py::_make_nested_splits`
- `scripts/run_phase24g_r_correction.py::_nested_rep_full`
- `NESTED_REPS = 1000`

### Candidate selection
- `phase24c_design.select_candidate_region`

### Stacking / OOF
- `run_phase24g_diagnostics.discovery_maps_fast(..., proxy_delta=False)`
- `run_phase24g_diagnostics.eval_coordinate`

### Tests
- `tests/test_phase24g_r_correction.py`

## Confirmations

- Prior Phase-24G artifacts not modified: **{prior_check['unchanged']}**
- Phase-24F primary not overwritten: **{imm['immutable']}**
- Primary path forbids proxy (`assert_no_proxy_in_primary`)
- Nested completed: **0 / 1000** (STOP)

## git diff --stat from starting SHA

```
{diff_stat}
```

## Final commit SHA

To be filled after freeze commit.
"""
    AUDIT.write_text(audit, encoding="utf-8")

    hashes = {
        p.name: sha256_file(str(p))
        for p in sorted(OUT.iterdir())
        if p.is_file() and p.name != "artifact_hashes.json"
    }
    for p in (CFG, REPORT, AUDIT):
        hashes[p.name] = sha256_file(str(p))
    write_json(OUT / "artifact_hashes.json", hashes)

    dlog = DECISION_LOG.read_text(encoding="utf-8")
    if "### D165 — Phase 24G-R" not in dlog:
        entry = (
            "\n### D165 — Phase 24G-R runtime STOP (full-pipeline 1000)\n\n"
            "- **Type:** **OUR RESEARCH DECISION**\n"
            f"- **Decision:** From `{STARTING_SHA[:7]}…`, Phase 24G-R "
            "attempted corrected LOPO/nested using full frozen stacked "
            "pipelines and exactly 1000 nested reps. Multiprocess pools "
            "swap-thrashed (~17GB); serial full grids remain "
            f"~{nested_hours_observed:.0f}h for 1000 nested on this host. "
            "Per protocol we STOP without reducing reps or using proxy. "
            "Phase-24F FAIL unchanged. Prior 24G artifacts retained by hash. "
            f"Status `{STATUS_RUNTIME_STOP}`.\n"
            "- **Date:** 2026-10-01\n"
        )
        DECISION_LOG.write_text(dlog.rstrip() + "\n" + entry, encoding="utf-8")

    print(json.dumps({"status": STATUS_RUNTIME_STOP, **runtime}, indent=2)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
