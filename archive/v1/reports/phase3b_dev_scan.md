# Phase 3B1 development scan

**Status:** `phase3b2_locked_complete_awaiting_audit`  
**Extraction mode:** `truncated_prefix_single_example` (`batch_size=1`)  
**Branch:** `phase3/preoutput-trajectory`  
**Canonical B1 extract:** `phase3b1_extract_20260925T154005Z_39505f42`  
**Original development extract (preserved):** `phase3b1_extract_20260925T151054Z_f19e058f`  
**Locked-test extract (B2):** `phase3b2_extract_20260925T161009Z_ae18a95e` — see `reports/phase3b_locked_test.md`

This document covers **development validation** and **canonical reproducibility**. The confirmatory locked-test evaluation is reported separately.

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

## 3. Original float16-storage development run (preserved)

### 3a. Truncated-prefix repeatability preflight (original)

Run: `phase3b1_preflight_truncated_20260925T150853Z_5677efe3`  
Preserved summary: `artifacts/phase3b_dev/original_dev_run_f19e058f_preflight_truncated_summary.json`

| Check | Result |
| --- | --- |
| Same-input / same-shape / bs=1 repeatability min cosine | **0.999999999974304** |
| Gate | ≥ 0.9999 |
| Verdict | **PASS** |

### 3b. Cost benchmark (D038)

| Metric | Value |
| --- | ---: |
| Benchmark forwards | 100 |
| Forwards/sec | ≈21.69 |
| Projected scientific forwards | 14079 |
| Projected cost | ≈**$0.37** |
| Soft budget | $20 |
| Continue? | **yes** |

### 3c. Original full train/validation extraction

| Item | Value |
| --- | --- |
| Run ID | `phase3b1_extract_20260925T151054Z_f19e058f` |
| Mode | truncated_prefix_single_example |
| Batch size | 1 |
| Compute | BF16 |
| **Activation storage** | **float16** (historical; not canonical) |
| Scientific forwards | **14079** |
| Wall / estimated cost | ≈$0.252 |
| Artifact bytes | ≈1.35 GB |
| Locked test present | **false** |

Preserved metrics: `artifacts/phase3b_dev/original_dev_run_f19e058f_metrics.json`

### 3d. Original primary development endpoints

| Endpoint | Act AUROC | Surface | ΔAUROC |
| --- | ---: | ---: | ---: |
| Regime A (12/k0) | 0.713 [0.596, 0.813] | 0.720 | −0.007 [−0.119, 0.106] |
| Regime B (12/k1) | 0.933 [0.869, 0.977] | 0.737 | +0.196 [0.097, 0.310] |

These original results remain historical evidence. They are **not erased**.

---

## 4. Canonical provenance rerun

Research-lead correction (D039–D040): one reproducibility rerun with float32 storage and exact Git provenance **before** Phase 3B2.

| Item | Value |
| --- | --- |
| Provenance freeze commit | `7eb8a61e498d9c1fa3a72555cda60eed4a7bbfab` |
| Preflight commit (clean tree at launch) | `8b374fb2cf099ac1b44427917c2af6443e567e4a` |
| Canonical run code SHA (extract) | `8b374fb2cf099ac1b44427917c2af6443e567e4a` |
| Working tree clean at launch | **true** |
| Extractor SHA256 | `18461c8886409d1778c775a55b63f82a5e8ddd93d6e300face95c018d3b52d47` |
| Analysis script SHA256 (at extract) | `66eaa02679691bf790e64f194d2a6a6158b77d9a6826df701664fe2799e0c47c` |
| Surface freeze SHA256 | `b25c5095c1cba440103343f90408c4caac0b59785bdc28b57a46abea405bcd00` |

Canonical truncated-prefix preflight: `phase3b1_preflight_truncated_20260925T153935Z_2ff68b57`  
Repeatability min cosine: **0.999999999974304** (gate ≥ 0.9999) — **PASS**.

---

## 5. Exact committed run SHA

Canonical scientific extract launched only after a clean tree at:

`git_commit = 8b374fb2cf099ac1b44427917c2af6443e567e4a`

Remote manifest records `working_tree_clean: true` plus extractor/analysis/freeze hashes.

---

## 6. Float32 storage correction

| Field | Value |
| --- | --- |
| `compute_dtype` | `bfloat16` (unchanged model precision) |
| `activation_storage_dtype` | `float32` |
| Reason | BF16 values are exactly representable in float32; avoids the prior float16 storage conversion |

Logit summaries remain float32. Stored tensors are **not** labeled BF16.

Canonical extract:

| Item | Value |
| --- | --- |
| Run ID | `phase3b1_extract_20260925T154005Z_39505f42` |
| Scientific forwards | **14079** |
| Wall | ≈469.7 s |
| Estimated cost | ≈**$0.254** |
| Artifact bytes | ≈**2.70 GB** |
| Activation integrity | all finite; min −27.25; max 16.5; max-abs 27.25 |
| k0 within-group act max abs (L12) | **0.0** |
| k0 within-group surface-score max abs | **0.0** |

---

## 7. Metric comparison: original vs canonical

Comparison artifact: `artifacts/phase3b_dev/canonical_vs_original_comparison.json`

| Cell | ΔAUROC (canonical − original) |
| --- | ---: |
| block12 / k0 (Regime A) | **0.0** |
| block12 / k1 (Regime B) | **0.0** |
| Max \|ΔAUROC\| across all 54 cells | **≈0.000235** |

No unexpectedly large metric drift. Primary endpoints are numerically identical at reported precision.

### Canonical primary endpoints (authoritative for audit)

### Regime A — block 12 / k=0 (prompt-boundary propensity)

| Metric | Value |
| --- | ---: |
| Activation AUROC | **0.713** |
| Group-bootstrap 95% CI | [0.596, 0.813] |
| Prompt-only surface AUROC | **0.720** |
| ΔAUROC (act − surface) | **−0.007** |
| Paired Δ 95% CI | [−0.119, 0.106] |

Interpretation: **prompt-conditioned deception propensity** only.

### Regime B — block 12 / k=1 (early-trajectory prediction)

| Metric | Value |
| --- | ---: |
| Activation AUROC | **0.933** |
| Group-bootstrap 95% CI | [0.869, 0.977] |
| Prompt+1-token surface AUROC | **0.737** |
| ΔAUROC (act − surface) | **+0.196** |
| Paired Δ 95% CI | [0.097, 0.310] |

Interpretation: **early-trajectory prediction**. Not “pre-deceptive-output physiology.”

### Negative controls (terminology)

| Control | Role |
| --- | --- |
| Shuffled-label probe (seed 42) | Chance / null control |
| Fixed random-projection diagnostic | **Not** a chance/null control; may retain distributed class signal |

Primary cells (12/k0, 12/k1): shuffled ≈0.516 / 0.508; fixed random-projection diagnostic ≈0.508 / 0.643.

---

## 8. Final development interpretation

- Engineering/provenance gate met for canonical float32 storage + clean-tree extract.
- Scientific conclusions from the original development scan are **unchanged**: Regime A shows no incremental utility vs prompt surface; Regime B shows clear early-trajectory incremental utility on development validation.
- Original float16-storage run remains documented history.
- Original 0.9999 full-sequence/batch gates were **not** relaxed.
- Locked test untouched. Regime C not run. No causal interventions.
- **STOP before Phase 3B2** pending research-lead audit.

---

## Appendix — methodological limitations (must remain visible)

1. Initial full-sequence/batch preflight failed the pre-registered 0.9999 gate (D034).
2. **No threshold relaxation** occurred (D037).
3. Primary method was changed **prospectively** to truncated-prefix bs=1 before Phase 3 activation collection (D035–D036).
4. Canonical scientific extraction uses truncated inputs + batch size 1; future tokens absent by construction; activations stored as float32 after BF16 compute (D039).
5. Results are development/validation only; locked test not touched.
6. ΔAUROC is incremental predictive utility vs the specified baseline, **not** proof of information-theoretically unavailable information (D030).

`LOCKED TEST ACTIVATIONS WERE NOT COLLECTED OR EVALUATED.`  
`REGIME C WAS NOT RUN.`  
`NO CAUSAL INTERVENTIONS WERE PERFORMED.`  
`ORIGINAL 0.9999 FULL-SEQUENCE/BATCH GATES WERE NOT RELAXED.`  
`PRIMARY SCIENTIFIC EXTRACTION USED TRUNCATED-PREFIX BATCH-SIZE-1 FORWARDS.`  
`CANONICAL ACTIVATION STORAGE IS FLOAT32; MODEL COMPUTE REMAINS BFLOAT16.`
