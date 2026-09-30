# Phase 24C — Full trajectory design freeze

**Status:** `phase24c_full_trajectory_design_frozen_awaiting_live_generation_authorization`

PHASE 24C WAS A PROSPECTIVE DESIGN, YIELD-FORECAST, AND STATISTICAL-FREEZE PHASE. NO NEW MODEL GENERATION, GRADING, ACTIVATION EXTRACTION, DECEPTION PROBING, SAE ANALYSIS, CAUSAL INTERVENTION, OR PRIMARY PHYSIOLOGY ANALYSIS WAS PERFORMED.

## Dataset provenance

| Field | Value |
|---|---|
| Source | `ApolloResearch/deception-detection` |
| Upstream commit | `f8ec4010e74927394709dffa22b97bdf8cd5a62f` |
| Dataset path | `data/roleplaying/dataset.yaml` |
| Blob SHA | `b938599a955946f1c96fa8de79dda6aeab72452d` |
| Content SHA256 | `7d3e36dc9a2fa72ff823b651c2ade2e0fc34c5bc6267cdf9b7be8b046ed2e564` |
| Local raw verified | `7d3e36dc9a2fa72ff823b651c2ade2e0fc34c5bc6267cdf9b7be8b046ed2e564` |
| Local prompts.jsonl | `8c49552f0eac048c82710a5a012add5f8c217412e24a8f30a9fc426f0753aaa0` |
| Prompt count | 371 |
| Verification | **PASS** (unchanged vs Phases 21–24) |

## Eligible pool (GPT-4o K=20 label-only)

- DEVELOPMENT: 28 prompts (`535d7f810b794792…`)
- LOCKED: 11 prompts (`48175cdceb142ac7…`)
- Total: 39

## Split

| Split | n | SHA256 |
|---|---|---|
| TRAIN | 20 | `094321f387793cbea3f92e5f31b54b246f1e0b11593cf2e7bae5ae1151d683cf` |
| VALIDATION | 8 | `77cee997a27371717269bcc7df7ea31f8e05eb41262967830890af266a38e9ec` |
| TEST | 11 | `48175cdceb142ac75208883be986c34c3a1a07e32ee9a8cd9832d30e1da06a05` |

All 16 Phase-24B pilot prompts are **TRAIN-only**:

```
[
  "roleplay_091",
  "roleplay_224",
  "roleplay_078",
  "roleplay_230",
  "roleplay_192",
  "roleplay_163",
  "roleplay_030",
  "roleplay_370",
  "roleplay_036",
  "roleplay_236",
  "roleplay_116",
  "roleplay_014",
  "roleplay_040",
  "roleplay_003",
  "roleplay_119",
  "roleplay_207"
]
```

TRAIN IDs:

```
[
  "roleplay_091",
  "roleplay_224",
  "roleplay_078",
  "roleplay_230",
  "roleplay_192",
  "roleplay_163",
  "roleplay_030",
  "roleplay_370",
  "roleplay_036",
  "roleplay_236",
  "roleplay_116",
  "roleplay_014",
  "roleplay_040",
  "roleplay_003",
  "roleplay_119",
  "roleplay_207",
  "roleplay_355",
  "roleplay_296",
  "roleplay_143",
  "roleplay_308"
]
```

VALIDATION IDs:

```
[
  "roleplay_026",
  "roleplay_122",
  "roleplay_304",
  "roleplay_263",
  "roleplay_281",
  "roleplay_366",
  "roleplay_123",
  "roleplay_009"
]
```

TEST IDs:

```
[
  "roleplay_007",
  "roleplay_111",
  "roleplay_174",
  "roleplay_193",
  "roleplay_203",
  "roleplay_206",
  "roleplay_245",
  "roleplay_275",
  "roleplay_289",
  "roleplay_297",
  "roleplay_298"
]
```

## K forecast

Grid: `[6, 8, 10, 12, 16, 20]` · MC=`100000` · seed=`2403`

### Primary — Phase-24B live forecast

| K | P(TRAIN≥10/20 ≥2H≥2D) | P(VAL≥4/8) | P(TEST≥5/11) | med TEST H | med TEST D | gates |
|---|---|---|---|---|---|---|
| 6 | 0.0000 | 0.1349 | 0.1401 | 30.0 | 25.0 | fail |
| 8 | 0.1644 | 0.3464 | 0.3976 | 40.0 | 33.0 | fail |
| 10 | 0.5477 | 0.5305 | 0.6067 | 50.0 | 41.0 | fail |
| 12 | 0.7969 | 0.6590 | 0.7479 | 60.0 | 49.0 | fail |
| 16 | 0.9616 | 0.8140 | 0.8911 | 81.0 | 66.0 | PASS |
| 20 | 0.9918 | 0.8901 | 0.9467 | 101.0 | 82.0 | PASS |

### Historical sensitivity (GPT K=20; labeled)

| K | P(TRAIN≥10/20) | P(VAL≥4/8) | P(TEST≥5/11) | med TEST H | med TEST D | gates |
|---|---|---|---|---|---|---|
| 6 | 0.0011 | 0.2024 | 0.1129 | 22.0 | 31.0 | fail |
| 8 | 0.0472 | 0.5855 | 0.4485 | 29.0 | 41.0 | fail |
| 10 | 0.2454 | 0.8307 | 0.7482 | 36.0 | 52.0 | fail |
| 12 | 0.5323 | 0.9395 | 0.9000 | 44.0 | 62.0 | fail |
| 16 | 0.8969 | 0.9918 | 0.9863 | 58.0 | 83.0 | fail |
| 20 | 0.9845 | 0.9987 | 0.9983 | 73.0 | 104.0 | PASS |

### Selection

```json
{
  "status": "phase24c_full_trajectory_design_frozen_awaiting_live_generation_authorization",
  "selected_k": 16,
  "primary_row": {
    "k": 16,
    "p_train_ge10_of_20_ge2": 0.96155,
    "p_val_ge4_of_8_ge2": 0.814,
    "p_test_ge5_of_11_ge2": 0.89105,
    "median_test_honest": 81.0,
    "median_test_deceptive": 66.0,
    "mean_train_ge2": 12.54273,
    "mean_val_ge2": 4.73578,
    "mean_test_ge2": 6.51839,
    "mean_train_ge1": 15.84935,
    "mean_val_ge1": 6.02123,
    "mean_test_ge1": 8.27377,
    "gates": {
      "train": true,
      "validation": true,
      "test_prompts": true,
      "test_median_h": true,
      "test_median_d": true
    },
    "all_gates_pass": true
  },
  "historical_row": {
    "k": 16,
    "p_train_ge10_of_20_ge2": 0.89685,
    "p_val_ge4_of_8_ge2": 0.99178,
    "p_test_ge5_of_11_ge2": 0.98634,
    "median_test_honest": 58.0,
    "median_test_deceptive": 83.0,
    "mean_train_ge2": 12.04929,
    "mean_val_ge2": 6.042,
    "mean_test_ge2": 7.63176,
    "gates": {
      "train": false,
      "validation": true,
      "test_prompts": true,
      "test_median_h": true,
      "test_median_d": true
    },
    "all_gates_pass": false
  },
  "disagreement": []
}
```

Selected K: **16**

## Generation load (if K frozen)

```json
{
  "selected_k": 16,
  "reused_phase24b_trajectories": 96,
  "new_trajectories_pilot_tail": 160,
  "new_trajectories_other_prompts": 368,
  "new_trajectories_total": 528,
  "total_trajectories_after": 624,
  "projected_new_bytes": 11058654024.0,
  "projected_new_gpu_seconds": 3144.109048485756,
  "projected_new_cost_usd": 2.1834090614484416
}
```

## Analysis freeze

- Temporal landmarks: `[1, 2, 4, 8, 16, 32, 64]`
- Primary activation: post-block residual L0–L31; representation `Δh(t,L)=h(t,L)-h(0,L)`
- Predictors: TEXT / LOGITS / SURFACE / ACTIVATION / SURFACE+ACTIVATION
- Primary contrast: ΔAUROC = AUROC(SURFACE+ACTIVATION) − AUROC(SURFACE)
- C grid: `[0.01, 0.1, 1.0, 10.0]`; balanced + prompt-equal weights;
  prompt-group CV
- Candidate rule: earliest time with ≥3
  consecutive layers Δ>0 and median Δ≥0.05;
  then best band; then best layer (tie → lower L)
- Confirmatory: ΔAUROC>0 and 95% prompt-cluster bootstrap CI lower bound >0
  (10000 reps)

## Authorizations

Live generation / probes / physiology / SAE / causal / TEST inspection: **all false**.
