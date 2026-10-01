# Phase 24G-R2 — Modal 500 corrected diagnostics (freeze)

**Status:** `phase24g_r2_modal500_corrected_diagnostics_complete_awaiting_code_audit`

**Branch:** `phase24/diagnostic-correction-modal500`

**Starting SHA:** 

**Ending / freeze SHA:** `be8bbdd8012ed0869d2b4b32c0000a417e45e0b7`

## Prospective amendment

The exploratory Phase-24G-R nested resampling target is amended from 1,000 to
**500** full-pipeline repetitions solely for computational feasibility. No result
from the corrected nested analysis has yet been observed. All other analysis
rules remain unchanged.

PHASE 24G-R2 USED A PROSPECTIVELY AMENDED 500-REPETITION FULL-PIPELINE NESTED
DIAGNOSTIC ON MODAL CPU. THE AMENDMENT WAS MADE FOR COMPUTATIONAL FEASIBILITY
BEFORE CORRECTED RESULTS WERE OBSERVED. PHASE-24F REMAINS THE FINAL CONFIRMATORY
RESULT. NO NEW MODEL DATA, LABEL CHANGES, PROXY PRIMARY ANALYSIS, SAE ANALYSIS,
CAUSAL INTERVENTION, OR PRIMARY-ENDPOINT REDEFINITION WAS PERFORMED.

## Modal execution

| Field | Value |
| --- | --- |
| App ID | `ap-dDyWrXnyiMKUNMQ8jf8JIL` |
| Function call ID | `fc-01M3TRD9GTJ2NDCTZ6E9P9CG8H` |
| Volume | `pre-output-physiology-phase24g-r2` |
| Resources | 16 CPU / 128 GiB RAM / 8 workers |
| Threads/worker | `OMP/MKL/OPENBLAS_NUM_THREADS=1` |
| Detached | yes (`.spawn()`) |
| Resumable | yes (per-rep atomic artifacts + resume scan) |
| Compute commit | `650b81f3614a0afec124d8614af217cae5430383` |
| Config hash | `ae83d67bb4742fecc1cb9c8f91b9210590aa51c65ebf17c922221583c9247949` |
| Wall time | ~5.53 h (relaunch `2026-10-01T03:34:14Z` → complete `2026-10-01T09:06:19Z`) |
| Actual Modal cost (app) | **$9.91** (CPU $4.21 + Memory $5.70) |

## Completeness

- LOPO full-pipeline: **28 / 28**
- Nested full-pipeline: **500 / 500**
- Failed reps: **none**
- Resume/retry events after successful detach: **none** (one deliberate
  cancel/relaunch before detach for tokenizer + `.spawn()` fix; no completed
  valid reps were overwritten)

## Results summary

| Metric | Value |
| --- | --- |
| Candidate-production frequency | **1.000** (500/500) |
| Discovery ΔAUROC mean / median | **0.1706** / **0.1408** |
| Held-out ΔAUROC mean / median | **0.0487** / **0.0383** |
| Optimism gap mean / median | **0.1219** / **0.1041** |
| Optimism gap 95% interval | **[-0.1328, 0.4429]** |
| Nested t=1/L20 frequency | **0.064** (32/500) |
| LOPO t=1/L20 frequency | **0.357** (10/28) |
| LOPO neighborhood frequency | **0.750** |
| Corrected diagnostic category | **B** (vs prior 24G Category C: **changes**) |
| Phase-24F | **FAIL unchanged** (`immutable: true`) |

### Selected time/layer (nested, among candidate-producing reps)

- Time hist: t1=247, t2=121, t4=64, t8=32, t16=25, t32=11
- Layer hist: peak at L14 (53), L15 (45), L13 (40), L20 (36); see
  `artifacts/phase24g_r2_modal500/nested_full_pipeline_500.json`

### Proxy vs full (audit-only secondary)

- LOPO (28): ΔAUROC Pearson **0.566**; time agree **0.929**; layer agree **0.179**
- Nested first 50: ΔAUROC Pearson **0.165**; time agree **0.600**; layer agree **0.140**

## Artifact paths (local freeze mirror)

- `artifacts/phase24g_r2_modal500/` — summaries, manifests, hashes
- `artifacts/phase24g_r2_modal500/results/lopo/full/` — 28 full LOPO JSON
- `artifacts/phase24g_r2_modal500/results/nested/` — 500 nested JSON
- `artifacts/phase24g_r2_modal500/results/lopo/proxy/` — 28 proxy LOPO
- `artifacts/phase24g_r2_modal500/results/nested_proxy/` — 50 proxy nested
- `reports/phase24g_r2_modal500_diagnostic_correction.md` — this report
- `reports/phase24g_r2_modal500_code_audit_manifest.md` — auditor checklist

## Tests

- `uv run pytest`: **359 passed**
- `uv run ruff check .`: **All checks passed**
