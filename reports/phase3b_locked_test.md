# Phase 3B2 locked-test evaluation

**Status:** `phase3b2_locked_complete_awaiting_audit`  
**One-shot locked-test evaluation:** yes  
**Locked-test hyperparameter tuning:** none  
**Run ID:** `phase3b2_extract_20260925T161009Z_ae18a95e`  
**Branch:** `phase3/preoutput-trajectory`

This is the **confirmatory** RoleplayDeception locked-test evaluation. Development validation and the canonical B1 reproducibility rerun remain separate.

---

## 1. One-shot locked-test protocol

1. Freeze analysis code, surface C=10, probe C=0.01, endpoints (D029/D041–D044).
2. Freeze locked-test eligibility (lengths/IDs only; no predictive metrics).
3. Commit with clean tree; record Git/code hashes.
4. Truncated-prefix bs=1 repeatability preflight (gate ≥ 0.9999).
5. Extract locked-test activations **without labels in the GPU payload**.
6. Integrity gate (dtype, finiteness, hashes, eligibility, splits).
7. **Only then** load labels and evaluate once.
8. Do not retune or re-run with alternate parameters.

---

## 2. Pre-analysis Git / code hashes

| Item | Value |
| --- | --- |
| B2 analysis freeze commit | `c1608d499ab1ff8aef2d4f07b0042f573ec37d52` |
| Extract launch Git SHA (clean tree) | `05a13a5ba0c327ce6a7f5774241c09916976a677` |
| Working tree clean at launch | **true** |
| Extractor SHA256 | `e56746511623645903047e45c21521eff5ea2cf49674b8f8d599a445e767ccdf` |
| Locked-analysis-script SHA256 | `6b3dc9faa879fe10325a2906f94263e7db58585e8e87dac097a413c9d1f1eb2a` |
| Surface freeze SHA256 | `b25c5095c1cba440103343f90408c4caac0b59785bdc28b57a46abea405bcd00` |
| Eligibility freeze SHA256 | `30371dcaba266ebb2acec06563df9916f27d9e5afd4555e9fa1b8c206ae9ea2a` |
| Canonical B1 run (fit source) | `phase3b1_extract_20260925T154005Z_39505f42` |

---

## 3. Eligibility freeze

Artifact: `artifacts/phase3b_locked/locked_eligibility_freeze.json`  
500 locked-test rows; **53** prompt groups. Labels not used for decisions.

| k | N eligible | N excluded short | Prompt groups |
| ---: | ---: | ---: | ---: |
| 0 | 500 | 0 | 53 |
| 1 | 500 | 0 | 53 |
| 2 | 497 | 3 | 53 |
| 4 | 497 | 3 | 53 |
| 8 | 486 | 14 | 53 |
| 16 | 374 | 126 | 49 |

---

## 4. Extraction provenance

| Item | Value |
| --- | --- |
| Mode | `truncated_prefix_single_example` |
| Batch size | 1 |
| Compute dtype | BF16 |
| Activation storage | float32 |
| Model revision | `63a8b081…` |
| Scientific forwards | **2407** |
| Wall | ≈89.8 s |
| Estimated cost | ≈**$0.049** |
| Artifact bytes | ≈450 MB |
| Labels in GPU payload | **false** |
| Splits | locked_test only |
| Preflight min cosine | **0.9999999999742442** (≥ 0.9999) |

---

## 5. Integrity checks

- Expected 500 locked rows; no train/validation rows.
- Eligibility hashes match freeze.
- Shard SHA256s match; k0 SHA256 matches.
- Activation dtype float32; all finite.
- min / max / max-abs: **−25.125 / 14.5 / 25.125**
- k0 within-group activation max abs (L12): **0.0**
- k0 within-group surface-score max abs: **0.0**

Integrity gate **PASSED** before labels entered predictive evaluation.

---

## 6. Regime A — locked test (block 12 / k=0)

Interpretation: **prompt-boundary deception propensity**

| Metric | Value |
| --- | ---: |
| N / prompt groups | 500 / 53 |
| Activation AUROC | **0.823** |
| Group-bootstrap 95% CI | [0.712, 0.908] |
| Prompt-only surface AUROC | **0.760** |
| ΔAUROC (act − surface) | **+0.063** |
| Paired Δ 95% CI | [−0.049, 0.183] |
| AUPRC | 0.854 |

Paired CI **includes zero**. Failure to reject zero difference is **not** an equivalence claim.

---

## 7. Regime B — locked test (block 12 / k=1) — confirmatory

Interpretation: **early-trajectory prediction**

| Metric | Value |
| --- | ---: |
| N / prompt groups | 500 / 53 |
| Activation AUROC | **0.959** |
| Group-bootstrap 95% CI | [0.921, 0.984] |
| Prompt+1-token surface AUROC | **0.778** |
| ΔAUROC (act − surface) | **+0.181** |
| Paired Δ 95% CI | [0.087, 0.286] |
| AUPRC | **0.966** |

Paired 95% CI **excludes zero**.

Allowed claim:

> On the locked RoleplayDeception test set, the block-12 representation after one visible response token provided greater predictive utility for eventual deception than the frozen prompt+1-token surface classifier.

Not claimed: universal biomarker; spontaneous intent; causal mechanism; pre-deceptive-output physiology; cross-domain generalization.

---

## 8. Paired ΔAUROCs (primary)

| Endpoint | ΔAUROC | 95% CI |
| --- | ---: | --- |
| Regime A L12/k0 | +0.063 | [−0.049, 0.183] |
| Regime B L12/k1 | +0.181 | [0.087, 0.286] |

---

## 9. Surface / logit / structural baselines (locked test)

Surface C=10.0 for all k (refit on combined development only).

| k | Surface AUROC | Logit-summary AUROC | Structural AUROC |
| ---: | ---: | ---: | ---: |
| 0 | 0.760 | 0.502 | (see metrics) |
| 1 | 0.778 | 0.445 | (see metrics) |

Full tables: `artifacts/phase3b_locked/phase3b_locked_metrics.json`.

---

## 10. Negative controls (primary cells)

| Cell | Shuffled-label (null) | Fixed random-projection diagnostic |
| --- | ---: | ---: |
| L12/k0 | 0.527 | 0.608 |
| L12/k1 | 0.447 | 0.562 |

Shuffled-label is the chance/null control. Fixed random projection is **not** a null control.

---

## 11. All 54 secondary cells

Full table: `artifacts/phase3b_locked/phase3b_locked_cells.json`  
Descriptive only (D044): strongest locked AUROC ≈ L16/k8 **0.975**; weakest ≈ L0/k16 **0.625**.  
No multiple-comparison significance claims. No promotion of best cell to primary.

---

## 12. Development vs locked primary endpoints

| Metric | Dev (canonical B1) | Locked |
| --- | ---: | ---: |
| Regime A AUROC | 0.713 | 0.823 |
| Regime A ΔAUROC | −0.007 | +0.063 |
| Regime B AUROC | 0.933 | 0.959 |
| Regime B ΔAUROC | +0.196 | +0.181 |

Regime B incremental utility replicates directionally on the locked test (CI excludes 0).

---

## 13. Methodological limitations

1. RoleplayDeception is a positive-control domain with explicit incentives (D002).
2. Original full-sequence/batch 0.9999 gates failed and were **not** relaxed (D034/D037).
3. Canonical path is truncated-prefix bs=1 with float32 storage (D035–D039).
4. ΔAUROC is incremental predictive utility, not information-theoretic unavailability (D030).
5. One-shot locked evaluation only (D041); no post-hoc retuning.
6. Regime C HOLD; no onset-aligned claims.

---

## 14. Regime C

`hold_insufficient_onset_resolution` — **not run**.

---

## 15. Final Phase 3B observational verdict

**PHASE 3B2 LOCKED-TEST EVALUATION COMPLETE — AWAITING AUDIT**

- Confirmatory Regime B (L12/k1) shows positive incremental predictive utility vs frozen prompt+1 surface on the untouched locked test (paired CI excludes 0).
- Regime A does not reject zero ΔAUROC on locked test.
- Secondary 54-cell grid is exploratory only.
- Locked test never used for tuning.
- Regime C not run. No causal interventions.

`THIS WAS THE ONE-SHOT LOCKED-TEST EVALUATION.`  
`NO LOCKED-TEST HYPERPARAMETER TUNING WAS PERFORMED.`  
`REGIME C WAS NOT RUN.`  
`NO CAUSAL INTERVENTIONS WERE PERFORMED.`  
`ORIGINAL 0.9999 FULL-SEQUENCE/BATCH GATES WERE NOT RELAXED.`
