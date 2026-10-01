# Phase 24G-R2 — Modal 500 corrected diagnostics (freeze + audit repair)

**Status:** `phase24g_r2_modal500_corrected_diagnostics_complete_awaiting_code_audit`

**Branch:** `phase24/diagnostic-correction-modal500`

## Provenance SHAs (do not conflate)

| Role | SHA |
| --- | --- |
| Authoritative starting SHA | `be8bbdd8012ed0869d2b4b32c0000a417e45e0b7` |
| Analysis-logic / Volume-upload commit | `650b81f3614a0afec124d8614af217cae5430383` |
| Successful execution-wrapper commit (tokenizer path + `.spawn()` detach) | `94add8a178ff84775660cf49a3106bb6c3394d48` |
| Freeze commit (results + report mirror) | `76fc2c4cedb0a31b33b7021b6e9a3f6db0a0348d` |
| HEAD at audit-repair start | `a9c5f303512c5758dc2ba09111149344054fa231` |

**Starting SHA:** `be8bbdd8012ed0869d2b4b32c0000a417e45e0b7`

**Freeze commit SHA:** `76fc2c4cedb0a31b33b7021b6e9a3f6db0a0348d`

**Note on `code_commit.txt`:** The Volume file `code_commit.txt` records
`650b81f…` because it was written at upload time (analysis-logic commit). That is
**not** the final Modal execution-wrapper provenance. The job that completed
LOPO 28/28 + nested 500/500 ran under the later wrapper commit `94add8a…`
(spawn detach + Volume tokenizer). Config hash remains
`ae83d67bb4742fecc1cb9c8f91b9210590aa51c65ebf17c922221583c9247949`.

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
| Resumable | yes (per-rep atomic Volume artifacts) |
| Analysis-logic commit | `650b81f3614a0afec124d8614af217cae5430383` |
| Execution-wrapper commit | `94add8a178ff84775660cf49a3106bb6c3394d48` |
| Config hash | `ae83d67bb4742fecc1cb9c8f91b9210590aa51c65ebf17c922221583c9247949` |
| Wall time | ~5.53 h (relaunch `2026-10-01T03:34:14Z` → complete `2026-10-01T09:06:19Z`) |
| Actual Modal cost (app) | **$9.91** (CPU $4.21 + Memory $5.70) |

## Completeness

- LOPO full-pipeline: **28 / 28**
- Nested full-pipeline: **500 / 500**
- Failed reps: **none**
- Resume/retry events after successful detach: **none**

## Results summary

| Metric | Value |
| --- | --- |
| Candidate-production frequency | **1.000** (500/500) |
| Discovery ΔAUROC mean / median | **0.1706** / **0.1408** |
| Held-out ΔAUROC mean / median | **0.0487** / **0.0383** |
| Optimism gap mean / median | **0.1219** / **0.1041** |
| Optimism empirical 2.5–97.5% interval | **[-0.1328, 0.4429]** |
| Nested t=1/L20 frequency | **0.064** (32/500) |
| LOPO t=1/L20 frequency | **0.357** (10/28) |
| LOPO neighborhood frequency | **0.750** |
| Corrected diagnostic category | **B** (vs prior 24G Category C: **changes**) |
| Phase-24F | **FAIL unchanged** (`immutable: true`) |

**Interval semantics:** the optimism bounds are the **empirical 2.5th and 97.5th
percentiles of the 500 per-repetition optimism gaps** (discovery − held-out).
They are **not** a confidence interval for the mean optimism gap.

### Selected time/layer (nested)

- Time hist: t1=247, t2=121, t4=64, t8=32, t16=25, t32=11
- Layer hist: peak at L14 (53), L15 (45), L13 (40), L20 (36)

### Proxy vs full (audit-only secondary)

- LOPO (28): ΔAUROC Pearson **0.566**; time agree **0.929**; layer agree **0.179**
- Nested first 50: ΔAUROC Pearson **0.165**; time agree **0.600**; layer agree **0.140**

## Artifact paths

- `artifacts/phase24g_r2_modal500/`
- `reports/phase24g_r2_modal500_diagnostic_correction.md`
- `reports/phase24g_r2_modal500_code_audit_manifest.md`
- `artifacts/phase24g_r2_modal500/provenance_sha_map.json` (audit repair)

## Tests

- `uv run pytest` / `uv run ruff check .` (re-run after audit repair)
