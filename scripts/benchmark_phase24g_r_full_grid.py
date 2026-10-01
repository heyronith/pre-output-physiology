#!/usr/bin/env python3
"""Phase 24G-R runtime benchmark: one full-pipeline 7×32 discovery grid.

STOP gate before committing to 28 LOPO + 1000 nested full-pipeline runs.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_phase24g_diagnostics import (  # noqa: E402
    _build_all_records,
    _load_npz,
    _load_tokenizer,
    discovery_delta_grid_fast,
)

OUT = REPO_ROOT / "artifacts/phase24g_r_correction"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading tokenizer + records + NPZs…", flush=True)
    tok = _load_tokenizer()
    by_split = _build_all_records(tok)
    train = by_split["train"]
    val = by_split["validation"]
    cache = {}
    for r in train + val:
        cache[r["trajectory_id"]] = _load_npz(Path(r["npz_path"]))

    print("Benchmarking ONE full-pipeline discovery grid (proxy_delta=False)…", flush=True)
    t0 = time.perf_counter()
    grid = discovery_delta_grid_fast(
        train, val, cache, tok, proxy_delta=False
    )
    elapsed = time.perf_counter() - t0
    n_finite = sum(
        1
        for t in grid
        for L, v in grid[t].items()
        if v != float("-inf")
    )
    estimate = {
        "seconds_per_full_grid": elapsed,
        "minutes_per_full_grid": elapsed / 60.0,
        "n_finite_delta_cells": n_finite,
        "lopo_28_serial_hours": 28 * elapsed / 3600.0,
        "nested_1000_serial_hours": 1000 * elapsed / 3600.0,
        "nested_1000_plus_lopo_serial_hours": 1028 * elapsed / 3600.0,
        "parallel_8_workers_hours": 1028 * elapsed / 3600.0 / 8.0,
        "parallel_16_workers_hours": 1028 * elapsed / 3600.0 / 16.0,
        "protocol": {
            "nested_reps_required": 1000,
            "lopo_runs_required": 28,
            "proxy_forbidden_for_primary": True,
            "full_stacked_SURFACE_plus_ACTIVATION_required": True,
        },
        "verdict": None,
    }
    # Prohibitive if serial > 24h OR even 16-way parallel > 12h without cluster
    serial_h = estimate["nested_1000_plus_lopo_serial_hours"]
    p16_h = estimate["parallel_16_workers_hours"]
    if serial_h > 24 and p16_h > 12:
        estimate["verdict"] = "PROHIBITIVE"
        estimate["action"] = (
            "STOP before changing protocol: do not silently reduce reps "
            "or reintroduce proxy. Report runtime and await authorization."
        )
    elif serial_h > 24 and p16_h <= 12:
        estimate["verdict"] = "FEASIBLE_WITH_PARALLELISM"
        estimate["action"] = (
            "Proceed with parallel full-pipeline nested (16 workers) "
            "without reducing repetitions or using proxy."
        )
    else:
        estimate["verdict"] = "FEASIBLE_SERIAL"
        estimate["action"] = "Proceed with full-pipeline LOPO+nested."

    (OUT / "runtime_benchmark.json").write_text(
        json.dumps(estimate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(estimate, indent=2), flush=True)
    return 0 if estimate["verdict"] != "PROHIBITIVE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
