# Phase 3B1 development scan — PREFLIGHT HOLD

**Status:** `CAUSAL/BATCH PREFLIGHT HOLD` — full train/validation extraction was **not** run.

**Date:** 2026-09-25  
**Branch:** `phase3/preoutput-trajectory`  
**Starting HEAD:** `d225e5f4883d2fceedbf8c7d12d69ff471e725f6`

---

## 1. Objective

Execute Phase 3B1 Regime A/B coarse scan on train+validation only, after mandatory causal and batching equivalence preflights.

## 2. Exact revisions

| Item | Pin |
| --- | --- |
| Model | `mistralai/Mistral-7B-Instruct-v0.2` |
| Model revision | `63a8b081895390a26e140280378bc85ec8bce07a` |
| Dataset revision | `bf93584916fbd23121eca6f2017017df0ef3184f` |
| GPU | 1× Modal L40S |
| Cosine gate | `>= 0.9999` |

## 3. Frozen grid / endpoints (unchanged)

Layers `[0,4,8,12,16,20,24,28,31]`; k `[0,1,2,4,8,16]`.  
Primary: Regime A = 12/k0; Regime B = 12/k1.

## 4. Surface baseline freeze (completed locally before GPU)

Artifact: `artifacts/phase3b_dev/surface_baseline_freeze.json`

Selected C by k (all **10.0**):

| k | selected C | eligible train | eligible val |
| --- | --- | --- | --- |
| 0 | 10.0 | 2361 | 639 |
| 1 | 10.0 | 2360 | 639 |
| 2 | 10.0 | 2317 | 629 |
| 4 | 10.0 | 2312 | 620 |
| 8 | 10.0 | 2227 | 599 |
| 16 | 10.0 | 1578 | 492 |

`locked_test_used: false`

## 5. Causal full-vs-truncated preflight

Run: `phase3b1_preflight_20260925T144544Z_1b427baa`  
Sample: 8 train examples with response length >16; layers `[0,12,31]`; k `[0,1,4,16]`.

| Metric | Value |
| --- | --- |
| Comparisons | 96 |
| Failures (`cosine < 0.9999`) | **19** |
| Minimum cosine | **0.999839765386112** |
| Gate | 0.9999 |
| Result | **FAIL** |

Failures concentrate at **layer 31** (18/19); one failure at layer 12. All tested k values appear among failures. Worst max-abs diffs ~0.05–0.22.

## 6. Batching-equivalence preflight

Same sample; right-padded batch vs unpadded single.

| Metric | Value |
| --- | --- |
| Comparisons | 96 |
| Failures | **24** |
| Minimum cosine | **0.9997808368214288** |
| Result | **FAIL** |

All 24 batch failures are at **layer 31**.

## 7. Modal usage / cost (preflight only)

| Item | Value |
| --- | --- |
| Wall | ~14.5 s (plus image/model load in prior attempt) |
| Estimated cost | ~$0.008 |
| Full extraction | **not started** |

## 8–17. Activation analysis

**Not executed** (preflight gate blocked full extraction).

## 18. Limitations

Default Hugging Face / torch attention numerics for this pin yield cosines in ~0.99978–0.99987 at late layers — below the pre-registered 0.9999 gate. Per Phase 3B1 protocol: do **not** silently switch attention implementations or invent a workaround.

Next decision for the research lead (not executed here):

1. require truncated-prefix forward passes for Phase 3B extraction; and/or  
2. revisit the numerical gate / attention backend with an explicit decision log entry.

## 19. Regime C

`regime_c_status: hold_insufficient_onset_resolution` — not run.

## 20. Engineering verdict

**PREFLIGHT HOLD — PHASE 3B1 FULL EXTRACTION NOT AUTHORIZED BY GATE**

- LOCKED TEST ACTIVATIONS WERE NOT COLLECTED OR EVALUATED.
- REGIME C WAS NOT RUN.
- NO CAUSAL INTERVENTIONS WERE PERFORMED.
- NO PHASE 3 TRAIN/VAL ACTIVATION GRID WAS EXTRACTED AFTER THE FAILED PREFLIGHT.

Commit-safe summary: `artifacts/phase3b_dev/preflight_hold_summary.json`  
Raw preflight: `artifacts/runs/phase3b1_preflight_20260925T144544Z_1b427baa/preflight.json` (gitignored runs/).
