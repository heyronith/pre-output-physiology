# Phase 24G — Confirmatory failure diagnostics

**Status:** `phase24g_confirmatory_failure_diagnostics_complete`

**Diagnostic category:** **C** — Candidate-selection instability

PHASE 24G WAS AN EXPLORATORY POST-CONFIRMATION DIAGNOSTIC ANALYSIS. THE PHASE-24F PRIMARY CONFIRMATORY FAILURE REMAINS FINAL AND UNCHANGED. NO NEW TRAJECTORIES, SAE ANALYSIS, CAUSAL INTERVENTION, LABEL CHANGES, OR PRIMARY-ENDPOINT REDEFINITION WAS PERFORMED.

## Phase-24F immutability

Verified hash match and `primary_pass=false`. Frozen ΔAUROC=0.0792,
CI95=[-0.1133442475603493, 0.3102389342794961].

## 1. Candidate stability (t=1, L20)

| Split | SURFACE | ACTIVATION | COMBINED | ΔAUROC |
|---|---|---|---|---|
| TRAIN (OOF) | 0.8124 | 0.8288 | 0.7967 | -0.0157 |
| VALIDATION | 0.6436 | 0.7913 | 0.7913 | 0.1477 |
| TEST (exploratory refit) | 0.5096 | 0.4869 | 0.5888 | 0.0792 |
| TEST Phase-24F frozen | 0.5096 | 0.4869 | 0.5888 | 0.0792 |

## 2. LOPO candidate stability

Exact t=1/L20: **0.000**
Neighborhood: **0.036**
Time hist: `{"1": 27, "2": 1}`
Layer hist: `{"0": 26, "14": 1, "1": 1}`

Note: LOPO/nested used a **selection-stability proxy**
(Δ ≈ ACTIVATION−LOGITS-only SURFACE; no TEXT TF-IDF / no stacked meta)
for computational budget. Exploratory TEST 7×32 and candidate-stability
AUROCs use the full stacked pipeline.

## 3. Nested resampling optimism (50 reps)

Expected discovery ΔAUROC: **0.1416**
Expected held-out ΔAUROC: **0.0424**
Optimism gap: **0.0992**
Frac exact t=1/L20: **0.000**

## 4. Exploratory TEST best ΔAUROC region

`{"time": 32, "layer": 14, "surface_auroc": 0.6846683581007611, "activation_auroc": 0.8550199347589706, "combined_auroc": 0.8539325842696628, "delta_auroc": 0.1692642261689017, "text_auroc": 0.4353026458861906, "logits_auroc": 0.6866618339978253, "analysis_label": "EXPLORATORY_POST_CONFIRMATION"}`

Label: `EXPLORATORY_POST_CONFIRMATION` — not confirmed.

## 5. VAL↔TEST topology

Overall Pearson/Spearman: `{"pearson": -0.07438151743841141, "spearman": -0.13248989478765344, "n": 224}`
Jaccard high-q75 overlap: **0.179**

## 6–7. Heterogeneity / same-prompt

TEST ≥2H+≥2D combined fraction>0: **0.8571428571428571**
Summaries in `same_prompt_robustness.json`.

## 8. Probe-direction stability

`{
  "frozen_candidate": {
    "time": 1,
    "layer": 20,
    "stability": {
      "n": 30,
      "mean_pairwise_cosine": 0.6704994731417859,
      "median_pairwise_cosine": 0.674919986176818,
      "mean_norm": 0.23934035775053722,
      "sign_consistency": 0.8265002080732418
    }
  },
  "best_val": {
    "time": 32,
    "layer": 17,
    "stability": {
      "n": 30,
      "mean_pairwise_cosine": 0.6690865068119511,
      "median_pairwise_cosine": 0.6717741705514446,
      "mean_norm": 0.476518361425507,
      "sign_consistency": 0.8172836418209104
    }
  },
  "best_exploratory_test": {
    "time": 32,
    "layer": 14,
    "stability": {
      "n": 30,
      "mean_pairwise_cosine": 0.6932115549766512,
      "median_pairwise_cosine": 0.6973830645846333,
      "mean_norm": 0.8251832167958902,
      "sign_consistency": 0.829406850459482
    }
  },
  "shared_high_q75": {
    "time": 1,
    "layer": 14,
    "stability": {
      "n": 30,
      "mean_pairwise_cosine": 0.6380033150658319,
      "median_pairwise_cosine": 0.6429687846891932,
      "mean_norm": 0.5018846635425495,
      "sign_consistency": 0.8174599198396794
    }
  }
}`

## 9. TEXT vs LOGITS

`{
  "train_oof": {
    "text_auroc": 0.5793817204301075,
    "logits_auroc": 0.8434677419354839,
    "surface_auroc": 0.8123924731182796
  },
  "validation": {
    "text_auroc": 0.505420054200542,
    "logits_auroc": 0.6436314363143631,
    "surface_auroc": 0.6436314363143631
  },
  "test_exploratory": {
    "text_auroc": 0.5189516129032258,
    "logits_auroc": 0.5020161290322581,
    "surface_auroc": 0.5095967741935484,
    "analysis_label": "EXPLORATORY_POST_CONFIRMATION"
  },
  "note": "TEST SURFACE ~chance may inflate \u0394AUROC interpretation; inspect TEXT vs LOGITS separately."
}`

## 10. Power projections

See `power_projections.json` (planning only; no new data).

## 11. Category

**C**: Candidate-selection instability

Phase-24F FAIL is unchanged.
