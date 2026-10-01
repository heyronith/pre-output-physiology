# Phase 24G-R code audit manifest (RUNTIME STOP)

Starting SHA: `8f2cbb67d0ad6a8ae4d0c8afb304729166f11a17`
Working tree: `8f2cbb67d0ad6a8ae4d0c8afb304729166f11a17`
Status: `phase24g_r_stopped_full_pipeline_1000_nested_runtime_prohibitive`

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

- Prior Phase-24G artifacts not modified: **True**
- Phase-24F primary not overwritten: **True**
- Primary path forbids proxy (`assert_no_proxy_in_primary`)
- Nested completed: **0 / 1000** (STOP)

## git diff --stat from starting SHA

```
 .../phase24g_r_correction/artifact_hashes.json     |  15 +
 .../phase24g_r_correction/correction_manifest.json |  15 +
 .../lopo_full_checkpoint.jsonl                     |   4 +
 .../lopo_full_checkpoint_partial.jsonl             |   4 +
 .../partial_lopo_full_pipeline.json                | 203 ++++++
 .../phase24f_immutability.json                     |  15 +
 .../prior_phase24g_artifact_hashes.json            | 100 +++
 .../prior_phase24g_unchanged_verification.json     |   4 +
 .../phase24g_r_correction/runtime_benchmark.json   |  32 +
 artifacts/phase24g_r_correction/runtime_stop.json  |  28 +
 .../phase24g_r_diagnostic_correction.yaml          |  18 +
 docs/decision_log.md                               |   6 +
 reports/phase24g_r_code_audit_manifest.md          |  61 ++
 reports/phase24g_r_diagnostic_correction.md        |  34 +
 scripts/benchmark_phase24g_r_full_grid.py          |  95 +++
 scripts/emit_phase24g_r_runtime_stop.py            | 303 ++++++++
 scripts/run_phase24g_diagnostics.py                |  28 +-
 scripts/run_phase24g_r_correction.py               | 785 +++++++++++++++++++++
 src/pre_output_physiology/phase24g_r_correction.py | 330 +++++++++
 tests/test_phase24g_r_correction.py                | 239 +++++++
 20 files changed, 2309 insertions(+), 10 deletions(-)
```

## Final commit SHA

`922613c36aed11768eb52fabcdb988e19b635216`
