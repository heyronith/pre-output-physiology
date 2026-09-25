# Phase 3B1 development scan

**Status:** `phase3b_dev_complete_awaiting_audit`  
**Extraction mode:** `truncated_prefix_single_example` (`batch_size=1`)  
**Branch:** `phase3/preoutput-trajectory`  
**Extract run:** `phase3b1_extract_20260925T151054Z_f19e058f`

This is **development evidence** (train/validation only). Not held-out proof. Not a pre-deceptive biomarker claim.

---

## 1. Original preflight HOLD (preserved)

Historical artifact: `artifacts/phase3b_dev/preflight_hold_summary.json` (D034).

| Gate | Min cosine | Required | Result |
| --- | ---: | ---: | --- |
| Full-vs-truncated causal | ≈0.9998398 | 0.9999 | **FAIL** |
| Batch-vs-single | ≈0.9997808 | 0.9999 | **FAIL** |

Full-sequence teacher-forced extraction was **not** used as the scientific path. Thresholds were **not** relaxed.

---

## 2. Research-lead recovery decision

Canonical Path (D035–D037):

- Inputs truncated exactly at the measurement point (no future response tokens supplied).
- Scientific forwards use **batch size 1**.
- Original 0.9999 full-sequence/batch gates remain historical; method changed instead.

---

## 3. Truncated-prefix repeatability preflight

Run: `phase3b1_preflight_truncated_20260925T150853Z_5677efe3`  
Summary: `artifacts/phase3b_dev/preflight_truncated_summary.json`

| Check | Result |
| --- | --- |
| Same-input / same-shape / bs=1 repeatability min cosine | **0.999999999974304** |
| Gate | ≥ 0.9999 |
| Verdict | **PASS** |
| Future tokens present | false |
| Cost | ≈$0.007 |

---

## 4. Cost benchmark (D038)

Summary: `artifacts/phase3b_dev/benchmark_summary.json`

| Metric | Value |
| --- | ---: |
| Benchmark forwards | 100 |
| Forwards/sec | ≈21.69 |
| Projected scientific forwards | 14079 |
| Projected cost | ≈**$0.37** |
| Soft budget | $20 |
| Continue? | **yes** |

---

## 5. Full train/validation extraction

| Item | Value |
| --- | --- |
| Mode | truncated_prefix_single_example |
| Batch size | 1 |
| Scientific forwards | **14079** |
| Examples | 3000 (train 2361 + val 639) |
| Prompt groups (k=0) | 306 |
| Wall / estimated cost | ≈$0.252 |
| Artifact bytes | ≈1.35 GB |
| Locked test present | **false** |
| Regime C | **not run** |
| Causal interventions | **none** |

Surface baseline freeze hash unchanged:  
`b25c5095c1cba440103343f90408c4caac0b59785bdc28b57a46abea405bcd00`  
Selected C = **10.0** for all k.

Eligibility counts match freeze (unchanged).

---

## 6. Artifacts

- Raw (gitignored): `artifacts/runs/phase3b1_extract_20260925T151054Z_f19e058f/`
- Commit-safe pointer: `artifacts/phase3b_dev/latest_extract_manifest.json`
- Metrics: `artifacts/phase3b_dev/phase3b_dev_metrics.json`
- Cell table: `artifacts/phase3b_dev/phase3b_dev_cells.json`

k=0 within-group activation max abs diff (layer 12): **0.0**

---

## 7. Validation activation AUROC (all 54 cells)

Primary probe: standardized LR, C=0.01, train-only norm, group-bootstrap ≥2000.

Full numeric table in `phase3b_dev_metrics.json` / `phase3b_dev_cells.json`.

Descriptive extremes (validation AUROC):

| | layer | k | AUROC |
| --- | ---: | ---: | ---: |
| Strongest | 16 | 4 | 0.967 |
| Weakest | 4 | 0 | 0.526 |

Descriptive ΔAUROC extremes (activation − prompt+prefix surface):

| | layer | k | ΔAUROC |
| --- | ---: | ---: | ---: |
| Largest positive | 20 | 2 | +0.242 |
| Most negative | 0 | 16 | −0.215 |

---

## 8. Surface baselines (frozen; not retuned)

Prompt+prefix TF-IDF+LR, C=10.0 for all k (validation AUROC):

| k | surface AUROC | eligible val N |
| ---: | ---: | ---: |
| 0 | 0.720 | 639 |
| 1 | 0.737 | 639 |
| 2 | 0.715 | 629 |
| 4 | 0.765 | 620 |
| 8 | 0.743 | 599 |
| 16 | 0.753 | 492 |

---

## 9. Primary development endpoints

### Regime A — block 12 / k=0 (prompt-boundary propensity)

| Metric | Value |
| --- | ---: |
| Activation AUROC | **0.713** |
| Group-bootstrap 95% CI | [0.596, 0.813] |
| Prompt-only / prompt+prefix surface AUROC | **0.720** |
| ΔAUROC (act − surface) | **−0.007** |
| Paired Δ 95% CI | [−0.119, 0.106] |

Interpretation: context-conditioned deception **propensity** only. No claim of sample-specific intent. Activation does not beat the matched prompt text baseline on validation (CI includes 0).

### Regime B — block 12 / k=1 (early-trajectory prediction)

| Metric | Value |
| --- | ---: |
| Activation AUROC | **0.933** |
| Group-bootstrap 95% CI | [0.869, 0.977] |
| Prompt+1-token surface AUROC | **0.737** |
| ΔAUROC (act − surface) | **+0.196** |
| Paired Δ 95% CI | [0.097, 0.310] |

Interpretation: **early-trajectory prediction** of eventual deception given prompt + first visible response token. Incremental predictive utility vs matched surface baseline on development validation. Not “pre-deceptive-output physiology.”

---

## 10. Negative controls

Across all 54 cells (mean):

| Control | Mean val AUROC |
| --- | ---: |
| Shuffled-label probe (seed 42) | ≈0.539 |
| Random-direction control | ≈0.532 |

Primary cells (12/k0, 12/k1): shuffled ≈0.516 / 0.508; random-direction ≈0.508 / 0.643.

---

## 11. Output-logit summary baseline (C=1.0, frozen)

| k | logit-summary AUROC |
| ---: | ---: |
| 0 | 0.402 |
| 1 | 0.665 |
| 2 | 0.556 |
| 4 | 0.597 |
| 8 | 0.459 |
| 16 | 0.530 |

---

## 12. H2 layer/time pattern (descriptive)

- Late-mid blocks with small k>0 show the strongest development AUROCs.
- Prompt-boundary (k=0) is weaker and closer to surface text.
- Layer 31 remains in the frozen exploratory scan despite prior numerical sensitivity under *different* (full-sequence/batched) forwards; here it is measured with truncated bs=1 inputs.

---

## 13. Methodological limitations (must remain visible)

1. Initial full-sequence/batch preflight failed the pre-registered 0.9999 gate (D034).
2. **No threshold relaxation** occurred (D037).
3. Primary method was changed **prospectively** to truncated-prefix bs=1 before Phase 3 activation collection (D035–D036).
4. Final scientific extraction used truncated inputs; future tokens absent by construction.
5. Results are development/validation only; locked test not touched.
6. ΔAUROC is incremental predictive utility vs the specified baseline, **not** proof of information-theoretically unavailable information (D030).

---

## 14. Regime C

`regime_c_status: hold_insufficient_onset_resolution` — **not run**.

---

## 15. Engineering / development verdict

**PHASE 3B1 DEVELOPMENT COMPLETE — AWAITING AUDIT**

- Truncated-prefix repeatability preflight passed.
- Cost within soft budget.
- 54/54 cells analyzed on train/validation.
- Locked test untouched.
- Regime C not run.
- No causal interventions.
- Original 0.9999 full-sequence/batch gates were not relaxed.

`LOCKED TEST ACTIVATIONS WERE NOT COLLECTED OR EVALUATED.`  
`REGIME C WAS NOT RUN.`  
`NO CAUSAL INTERVENTIONS WERE PERFORMED.`  
`ORIGINAL 0.9999 FULL-SEQUENCE/BATCH GATES WERE NOT RELAXED.`  
`PRIMARY SCIENTIFIC EXTRACTION USED TRUNCATED-PREFIX BATCH-SIZE-1 FORWARDS.`
