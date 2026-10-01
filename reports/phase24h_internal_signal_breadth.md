# Phase 24H — Internal-signal-breadth representation discovery

**Status:** `phase24h_internal_signal_breadth_complete_awaiting_independent_code_audit`  
**Label:** `EXPLORATORY_DEVELOPMENT_ONLY_PHASE24H`

> PHASE 24H WAS EXPLORATORY REPRESENTATION DISCOVERY ON DEVELOPMENT (TRAIN+VALIDATION) DATA ONLY. LOCKED TEST LABELS/ACTIVATIONS AND PHASE-24F TEST ACTIVATIONS WERE NOT ACCESSED. PHASE-24F REMAINS THE FINAL CONFIRMATORY FAIL. NO GPU, MODEL GENERATION, SAE, OR CAUSAL WORK WAS PERFORMED.

Phase-24F remains the **final confirmatory FAIL** (t=1, L20). Nothing here changes it. All ΔAUROC values are `AUROC(SURFACE+ACTIVATION) − AUROC(SURFACE)` at the **same time** (multitime uses SURFACE@t8). Activation-alone AUROC is descriptive only and is never incremental evidence.

## Data and protocol

- DEV prompts: 28 (20 TRAIN + 8 VALIDATION); trajectories 388 (196 honest / 192 deceptive; 60 ambiguous excluded)
- Early times frozen [1, 2, 4, 8]; outer salts [0, 1, 2] × 7 folds (4 held-out prompts each); inner prompt-group CV 5-fold
- Bootstrap: 2000 prompt-cluster resamples; shuffle control: 100 prompt-preserving activation shuffles (salt 0)
- Protocol freeze commit: `d527849bfb81821d2be63a080b39bcf35ae67ccc`; run HEAD `d527849bfb81821d2be63a080b39bcf35ae67ccc`
- Multiplicity: 36 exploratory rows were screened; promotion is a replication-screening label, **not** confirmatory evidence.

## Outcome

- Rows meeting all five promotion criteria: 0; revoked by shuffle control: 0

- **Final `promising_for_independent_replication` rows: 0**

## All rows (pooled over outer salts)

| row | ΔAUROC | 95% CI | Δ salt0/1/2 | SURF AUROC | COMB AUROC | ΔAUPRC | n mixed | med Δ within | frac>0 | label |
|---|---|---|---|---|---|---|---|---|---|---|
| `A_RAW_SINGLE\|t1` | 0.068 | [-0.002, 0.147] | 0.053/0.093/0.057 | 0.746 | 0.814 | 0.116 | 18 | 0.000 | 0.17 | not_promoted |
| `A_RAW_SINGLE\|t2` | 0.087 | [-0.009, 0.196] | 0.050/0.068/0.142 | 0.713 | 0.800 | 0.058 | 18 | 0.000 | 0.44 | not_promoted |
| `A_RAW_SINGLE\|t4` | 0.014 | [-0.053, 0.086] | -0.017/0.003/0.055 | 0.705 | 0.719 | -0.015 | 18 | 0.038 | 0.61 | not_promoted |
| `A_RAW_SINGLE\|t8` | 0.061 | [-0.026, 0.149] | 0.061/0.030/0.092 | 0.754 | 0.815 | 0.054 | 18 | 0.000 | 0.44 | not_promoted |
| `A_RAW_SINGLE\|search` | 0.013 | [-0.042, 0.068] | -0.004/0.023/0.020 | 0.767 | 0.780 | -0.027 | 18 | 0.015 | 0.50 | not_promoted |
| `B_DELTA_SINGLE\|t1` | 0.070 | [-0.001, 0.142] | 0.075/0.072/0.065 | 0.746 | 0.816 | 0.104 | 18 | 0.000 | 0.17 | not_promoted |
| `B_DELTA_SINGLE\|t2` | 0.087 | [-0.034, 0.222] | 0.029/0.055/0.176 | 0.713 | 0.800 | 0.031 | 18 | 0.000 | 0.39 | not_promoted |
| `B_DELTA_SINGLE\|t4` | -0.009 | [-0.099, 0.078] | -0.035/0.030/-0.022 | 0.705 | 0.696 | -0.040 | 18 | 0.000 | 0.44 | not_promoted |
| `B_DELTA_SINGLE\|t8` | 0.044 | [-0.044, 0.127] | 0.054/-0.002/0.080 | 0.754 | 0.798 | 0.053 | 18 | 0.000 | 0.44 | not_promoted |
| `B_DELTA_SINGLE\|search` | 0.041 | [-0.050, 0.133] | 0.046/0.046/0.030 | 0.784 | 0.824 | 0.008 | 18 | 0.000 | 0.39 | not_promoted |
| `C_MULTILAYER\|RAW\|t1` | 0.066 | [-0.022, 0.148] | 0.041/0.069/0.087 | 0.746 | 0.812 | 0.112 | 18 | 0.000 | 0.17 | not_promoted |
| `C_MULTILAYER\|RAW\|t2` | 0.091 | [-0.004, 0.206] | 0.087/0.060/0.127 | 0.713 | 0.804 | 0.069 | 18 | 0.000 | 0.44 | not_promoted |
| `C_MULTILAYER\|RAW\|t4` | -0.015 | [-0.072, 0.052] | 0.007/-0.010/-0.043 | 0.705 | 0.689 | -0.063 | 18 | 0.014 | 0.50 | not_promoted |
| `C_MULTILAYER\|RAW\|t8` | 0.054 | [-0.007, 0.127] | 0.070/0.035/0.057 | 0.754 | 0.808 | 0.038 | 18 | 0.000 | 0.44 | not_promoted |
| `C_MULTILAYER\|DELTA\|t1` | 0.104 | [0.031, 0.191] | 0.083/0.125/0.106 | 0.746 | 0.850 | 0.138 | 18 | 0.000 | 0.22 | not_promoted |
| `C_MULTILAYER\|DELTA\|t2` | 0.066 | [-0.103, 0.221] | 0.006/0.060/0.133 | 0.713 | 0.779 | 0.049 | 18 | 0.000 | 0.39 | not_promoted |
| `C_MULTILAYER\|DELTA\|t4` | -0.006 | [-0.103, 0.089] | -0.030/-0.017/0.029 | 0.705 | 0.699 | -0.022 | 18 | 0.000 | 0.44 | not_promoted |
| `C_MULTILAYER\|DELTA\|t8` | 0.015 | [-0.064, 0.092] | 0.006/0.039/0.000 | 0.754 | 0.769 | 0.029 | 18 | 0.000 | 0.44 | not_promoted |
| `D_MULTITIME\|RAW` | 0.055 | [-0.033, 0.150] | 0.062/0.031/0.074 | 0.754 | 0.809 | 0.054 | 18 | 0.000 | 0.44 | not_promoted |
| `D_MULTITIME\|DELTA` | 0.056 | [-0.067, 0.177] | 0.068/0.038/0.063 | 0.754 | 0.810 | 0.057 | 18 | 0.000 | 0.44 | not_promoted |
| `E_LOWDIM\|RAW\|t1` | 0.073 | [-0.018, 0.157] | 0.045/0.084/0.089 | 0.746 | 0.819 | 0.114 | 18 | 0.000 | 0.22 | not_promoted |
| `E_LOWDIM\|RAW\|t2` | 0.095 | [-0.024, 0.204] | 0.055/0.089/0.140 | 0.713 | 0.808 | 0.080 | 18 | 0.000 | 0.39 | not_promoted |
| `E_LOWDIM\|RAW\|t4` | 0.017 | [-0.050, 0.097] | 0.030/0.001/0.022 | 0.705 | 0.722 | -0.045 | 18 | 0.019 | 0.56 | not_promoted |
| `E_LOWDIM\|RAW\|t8` | 0.046 | [-0.015, 0.117] | 0.073/0.011/0.053 | 0.754 | 0.800 | 0.024 | 18 | 0.000 | 0.44 | not_promoted |
| `E_LOWDIM\|DELTA\|t1` | 0.065 | [-0.003, 0.138] | 0.038/0.089/0.067 | 0.746 | 0.811 | 0.106 | 18 | 0.000 | 0.22 | not_promoted |
| `E_LOWDIM\|DELTA\|t2` | 0.070 | [-0.082, 0.215] | 0.019/0.044/0.146 | 0.713 | 0.783 | 0.033 | 18 | 0.000 | 0.39 | not_promoted |
| `E_LOWDIM\|DELTA\|t4` | 0.041 | [-0.048, 0.135] | 0.028/0.054/0.042 | 0.705 | 0.746 | 0.023 | 18 | 0.000 | 0.44 | not_promoted |
| `E_LOWDIM\|DELTA\|t8` | 0.025 | [-0.043, 0.086] | 0.007/0.051/0.017 | 0.754 | 0.778 | 0.031 | 18 | 0.008 | 0.56 | not_promoted |
| `F_NONLINEAR\|RAW\|t1` | -0.026 | [-0.113, 0.048] | -0.064/0.004/-0.017 | 0.746 | 0.720 | 0.002 | 18 | 0.000 | 0.17 | not_promoted |
| `F_NONLINEAR\|RAW\|t2` | -0.002 | [-0.110, 0.099] | -0.050/-0.028/0.071 | 0.713 | 0.711 | -0.017 | 18 | 0.000 | 0.33 | not_promoted |
| `F_NONLINEAR\|RAW\|t4` | -0.033 | [-0.095, 0.048] | -0.048/-0.039/-0.012 | 0.705 | 0.672 | -0.070 | 18 | 0.005 | 0.50 | not_promoted |
| `F_NONLINEAR\|RAW\|t8` | 0.047 | [-0.018, 0.121] | 0.051/0.025/0.064 | 0.754 | 0.801 | -0.004 | 18 | 0.005 | 0.50 | not_promoted |
| `F_NONLINEAR\|DELTA\|t1` | -0.034 | [-0.102, 0.017] | -0.059/-0.036/-0.007 | 0.746 | 0.712 | 0.000 | 18 | 0.000 | 0.22 | not_promoted |
| `F_NONLINEAR\|DELTA\|t2` | 0.011 | [-0.153, 0.158] | -0.034/0.005/0.061 | 0.713 | 0.724 | -0.025 | 18 | 0.000 | 0.39 | not_promoted |
| `F_NONLINEAR\|DELTA\|t4` | -0.032 | [-0.115, 0.050] | 0.001/-0.069/-0.027 | 0.705 | 0.673 | -0.045 | 18 | 0.022 | 0.56 | not_promoted |
| `F_NONLINEAR\|DELTA\|t8` | 0.008 | [-0.054, 0.065] | -0.009/0.023/0.009 | 0.754 | 0.761 | -0.020 | 18 | 0.000 | 0.44 | not_promoted |

## Same-prompt analysis (cross-fitted predictions only; no per-prompt models)

Prompts with ≥2 honest and ≥2 deceptive; within-prompt Δ averaged over salts.

| row | n qual. | median Δ | mean Δ | frac>0 | median 95% CI | mean 95% CI |
|---|---|---|---|---|---|---|
| `A_RAW_SINGLE\|t1` | 18 | 0.000 | 0.076 | 0.17 | [0.000, 0.000] | [0.000, 0.200] |
| `A_RAW_SINGLE\|t2` | 18 | 0.000 | 0.093 | 0.44 | [0.000, 0.088] | [0.018, 0.189] |
| `A_RAW_SINGLE\|t4` | 18 | 0.038 | 0.079 | 0.61 | [0.000, 0.156] | [0.024, 0.134] |
| `A_RAW_SINGLE\|t8` | 18 | 0.000 | 0.035 | 0.44 | [-0.082, 0.113] | [-0.075, 0.146] |
| `A_RAW_SINGLE\|search` | 18 | 0.015 | 0.052 | 0.50 | [0.000, 0.122] | [-0.014, 0.121] |
| `B_DELTA_SINGLE\|t1` | 18 | 0.000 | 0.075 | 0.17 | [0.000, 0.000] | [0.000, 0.206] |
| `B_DELTA_SINGLE\|t2` | 18 | 0.000 | 0.090 | 0.39 | [0.000, 0.037] | [0.014, 0.198] |
| `B_DELTA_SINGLE\|t4` | 18 | 0.000 | 0.043 | 0.44 | [-0.006, 0.116] | [-0.020, 0.110] |
| `B_DELTA_SINGLE\|t8` | 18 | 0.000 | 0.020 | 0.44 | [-0.048, 0.083] | [-0.097, 0.145] |
| `B_DELTA_SINGLE\|search` | 18 | 0.000 | 0.032 | 0.39 | [-0.014, 0.071] | [-0.022, 0.090] |
| `C_MULTILAYER\|RAW\|t1` | 18 | 0.000 | 0.061 | 0.17 | [0.000, 0.000] | [-0.002, 0.176] |
| `C_MULTILAYER\|RAW\|t2` | 18 | 0.000 | 0.075 | 0.44 | [0.000, 0.048] | [-0.019, 0.179] |
| `C_MULTILAYER\|RAW\|t4` | 18 | 0.014 | 0.030 | 0.50 | [0.000, 0.089] | [-0.028, 0.089] |
| `C_MULTILAYER\|RAW\|t8` | 18 | 0.000 | 0.019 | 0.44 | [-0.067, 0.071] | [-0.083, 0.132] |
| `C_MULTILAYER\|DELTA\|t1` | 18 | 0.000 | 0.076 | 0.22 | [0.000, 0.000] | [0.003, 0.199] |
| `C_MULTILAYER\|DELTA\|t2` | 18 | 0.000 | 0.085 | 0.39 | [0.000, 0.037] | [0.010, 0.178] |
| `C_MULTILAYER\|DELTA\|t4` | 18 | 0.000 | 0.033 | 0.44 | [-0.016, 0.122] | [-0.037, 0.099] |
| `C_MULTILAYER\|DELTA\|t8` | 18 | 0.000 | 0.024 | 0.44 | [-0.069, 0.038] | [-0.047, 0.120] |
| `D_MULTITIME\|RAW` | 18 | 0.000 | 0.045 | 0.44 | [-0.023, 0.119] | [-0.052, 0.156] |
| `D_MULTITIME\|DELTA` | 18 | 0.000 | 0.055 | 0.44 | [-0.046, 0.113] | [-0.031, 0.156] |
| `E_LOWDIM\|RAW\|t1` | 18 | 0.000 | 0.078 | 0.22 | [0.000, 0.000] | [0.003, 0.209] |
| `E_LOWDIM\|RAW\|t2` | 18 | 0.000 | 0.091 | 0.39 | [0.000, 0.041] | [0.017, 0.202] |
| `E_LOWDIM\|RAW\|t4` | 18 | 0.019 | 0.070 | 0.56 | [0.000, 0.122] | [0.024, 0.120] |
| `E_LOWDIM\|RAW\|t8` | 18 | 0.000 | 0.015 | 0.44 | [-0.071, 0.071] | [-0.095, 0.131] |
| `E_LOWDIM\|DELTA\|t1` | 18 | 0.000 | 0.074 | 0.22 | [0.000, 0.000] | [0.003, 0.200] |
| `E_LOWDIM\|DELTA\|t2` | 18 | 0.000 | 0.083 | 0.39 | [0.000, 0.037] | [0.005, 0.186] |
| `E_LOWDIM\|DELTA\|t4` | 18 | 0.000 | 0.015 | 0.44 | [-0.013, 0.057] | [-0.045, 0.077] |
| `E_LOWDIM\|DELTA\|t8` | 18 | 0.008 | 0.042 | 0.56 | [-0.023, 0.054] | [-0.031, 0.131] |
| `F_NONLINEAR\|RAW\|t1` | 18 | 0.000 | 0.062 | 0.17 | [0.000, 0.000] | [0.000, 0.176] |
| `F_NONLINEAR\|RAW\|t2` | 18 | 0.000 | 0.048 | 0.33 | [-0.011, 0.013] | [-0.037, 0.153] |
| `F_NONLINEAR\|RAW\|t4` | 18 | 0.005 | 0.008 | 0.50 | [-0.004, 0.089] | [-0.079, 0.075] |
| `F_NONLINEAR\|RAW\|t8` | 18 | 0.005 | 0.020 | 0.50 | [-0.083, 0.067] | [-0.088, 0.125] |
| `F_NONLINEAR\|DELTA\|t1` | 18 | 0.000 | 0.072 | 0.22 | [0.000, 0.000] | [0.002, 0.195] |
| `F_NONLINEAR\|DELTA\|t2` | 18 | 0.000 | 0.069 | 0.39 | [0.000, 0.021] | [0.000, 0.173] |
| `F_NONLINEAR\|DELTA\|t4` | 18 | 0.022 | 0.005 | 0.56 | [0.000, 0.067] | [-0.086, 0.077] |
| `F_NONLINEAR\|DELTA\|t8` | 18 | 0.000 | 0.020 | 0.44 | [-0.060, 0.060] | [-0.036, 0.094] |

## Promotion criteria detail

| row | pooled>0 | CI low>0 | all salts>0 | within median>0 | frac≥0.60 | promoted |
|---|---|---|---|---|---|---|
| `A_RAW_SINGLE\|t1` | yes | no | yes | no | no | no |
| `A_RAW_SINGLE\|t2` | yes | no | yes | no | no | no |
| `A_RAW_SINGLE\|t4` | yes | no | no | yes | yes | no |
| `A_RAW_SINGLE\|t8` | yes | no | yes | no | no | no |
| `A_RAW_SINGLE\|search` | yes | no | no | yes | no | no |
| `B_DELTA_SINGLE\|t1` | yes | no | yes | no | no | no |
| `B_DELTA_SINGLE\|t2` | yes | no | yes | no | no | no |
| `B_DELTA_SINGLE\|t4` | no | no | no | no | no | no |
| `B_DELTA_SINGLE\|t8` | yes | no | no | no | no | no |
| `B_DELTA_SINGLE\|search` | yes | no | yes | no | no | no |
| `C_MULTILAYER\|RAW\|t1` | yes | no | yes | no | no | no |
| `C_MULTILAYER\|RAW\|t2` | yes | no | yes | no | no | no |
| `C_MULTILAYER\|RAW\|t4` | no | no | no | yes | no | no |
| `C_MULTILAYER\|RAW\|t8` | yes | no | yes | no | no | no |
| `C_MULTILAYER\|DELTA\|t1` | yes | yes | yes | no | no | no |
| `C_MULTILAYER\|DELTA\|t2` | yes | no | yes | no | no | no |
| `C_MULTILAYER\|DELTA\|t4` | no | no | no | no | no | no |
| `C_MULTILAYER\|DELTA\|t8` | yes | no | yes | no | no | no |
| `D_MULTITIME\|RAW` | yes | no | yes | no | no | no |
| `D_MULTITIME\|DELTA` | yes | no | yes | no | no | no |
| `E_LOWDIM\|RAW\|t1` | yes | no | yes | no | no | no |
| `E_LOWDIM\|RAW\|t2` | yes | no | yes | no | no | no |
| `E_LOWDIM\|RAW\|t4` | yes | no | yes | yes | no | no |
| `E_LOWDIM\|RAW\|t8` | yes | no | yes | no | no | no |
| `E_LOWDIM\|DELTA\|t1` | yes | no | yes | no | no | no |
| `E_LOWDIM\|DELTA\|t2` | yes | no | yes | no | no | no |
| `E_LOWDIM\|DELTA\|t4` | yes | no | yes | no | no | no |
| `E_LOWDIM\|DELTA\|t8` | yes | no | yes | yes | no | no |
| `F_NONLINEAR\|RAW\|t1` | no | no | no | no | no | no |
| `F_NONLINEAR\|RAW\|t2` | no | no | no | no | no | no |
| `F_NONLINEAR\|RAW\|t4` | no | no | no | yes | no | no |
| `F_NONLINEAR\|RAW\|t8` | yes | no | yes | yes | no | no |
| `F_NONLINEAR\|DELTA\|t1` | no | no | no | no | no | no |
| `F_NONLINEAR\|DELTA\|t2` | yes | no | no | no | no | no |
| `F_NONLINEAR\|DELTA\|t4` | no | no | no | yes | no | no |
| `F_NONLINEAR\|DELTA\|t8` | yes | no | no | no | no | no |

## Negative control (prompt-preserving activation shuffle)

No row met the promotion rule, so no shuffle control was required.

## Compute

- Wall: 0.41 h; workers 6, BLAS threads 1; CPU only, no GPU/model generation.
- Peak RSS: {'children_max_rss_mb': 1810.546875, 'self_max_rss_mb': 1059.3125}


Input NPZ aggregate SHA256: `d94982adb6db42b8fd290b36e228cb522edfc08b6a797a73236f2e712fb55465`

