# Phase 24G-R — RUNTIME STOP

**Status:** `phase24g_r_stopped_full_pipeline_1000_nested_runtime_prohibitive`

## Why we stopped

Full frozen stacked SURFACE+ACTIVATION discovery grids measure
**98.5s** clean (and **~140s** under
host memory pressure). Multiprocess pools induced **~17GB swap thrash** and
were abandoned. Serial 1,000 nested would require
**~38.9 hours** wall on this 24GB host.

Per authorization: **do not silently reduce repetitions or use a proxy**.

## Phase-24F immutability

Verified: **True** (ΔAUROC=0.0792, CI=[-0.1133442475603493, 0.3102389342794961])

## Prior Phase-24G artifacts

Unchanged: **True**

## Partial progress (not a primary result)

Full-pipeline LOPO completed **4/28** before STOP.
Among those, exact t=1/L20 selections: **1/4**.

Nested 1000: **0/1000** completed.

## Required next step

Authorize a higher-RAM / cloud runner for the **unchanged** protocol, or an
explicit protocol amendment. No scientific reclassification is issued under
STOP.
