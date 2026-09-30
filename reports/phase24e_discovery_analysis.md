# Phase 24E — Discovery analysis (TRAIN + VALIDATION)

**Status:** `phase24e_candidate_discovered_awaiting_locked_test_authorization`

PHASE 24E USED TRAIN AND VALIDATION DATA ONLY. LOCKED TEST LABELS, ACTIVATIONS, AND SCIENTIFIC OUTCOMES REMAINED SEALED. NO SAE ANALYSIS, CAUSAL INTERVENTION, OR LOCKED-TEST CONFIRMATORY ANALYSIS WAS PERFORMED.

## Input verification

Verified: **True**

## TEST seal

LOCKED TEST was not read for labels, activations, or scientific metrics.
Existence of sealed artifacts was verified only.

## Usable primary H/D counts

| Split | Honest | Deceptive | Trajectories | Prompts |
|---|---|---|---|---|
| TRAIN | 155 | 120 | 275 | 20 |
| VALIDATION | 41 | 72 | 113 | 8 |

## Risk sets (survive landmark t)

```json
{
  "1": {
    "train": 275,
    "validation": 113
  },
  "2": {
    "train": 275,
    "validation": 113
  },
  "4": {
    "train": 275,
    "validation": 113
  },
  "8": {
    "train": 275,
    "validation": 113
  },
  "16": {
    "train": 269,
    "validation": 103
  },
  "32": {
    "train": 234,
    "validation": 99
  },
  "64": {
    "train": 156,
    "validation": 60
  }
}
```

## Hyperparameters

C grid `[0.01, 0.1, 1.0, 10.0]` selected by prompt-group CV on TRAIN only.
Weighting: balanced × prompt-equal (`prompt_equal_balanced_weights`).

## Grid

224 coordinates completed: **224** /
224 evaluated rows (including skips: 224).

| Metric | Best value | time | layer |
|---|---|---|---|
| SURFACE AUROC | 0.7515 | 8 | 0 |
| ACTIVATION AUROC | 0.8614 | 1 | 0 |
| SURFACE+ACTIVATION AUROC | 0.8238 | 32 | 17 |
| ΔAUROC | 0.2115 | 32 | 17 |

## Qualifying candidate bands

```json
[
  {
    "time": 1,
    "band": [
      8,
      23
    ],
    "median_delta_auroc": 0.11365176151761514
  },
  {
    "time": 4,
    "band": [
      25,
      31
    ],
    "median_delta_auroc": 0.07757452574525758
  },
  {
    "time": 32,
    "band": [
      9,
      24
    ],
    "median_delta_auroc": 0.13078216989066427
  },
  {
    "time": 64,
    "band": [
      0,
      31
    ],
    "median_delta_auroc": 0.12333333333333335
  }
]
```

## Selected candidate

```json
{
  "candidate_time": 1,
  "candidate_layer": 20,
  "band": [
    8,
    23
  ],
  "surface_auroc": 0.6436314363143631,
  "activation_auroc": 0.7913279132791329,
  "surface_plus_activation_auroc": 0.7913279132791329,
  "delta_auroc": 0.14769647696476973,
  "delta_auroc_bootstrap": {
    "n_reps": 2000,
    "mean": 0.13024750229138402,
    "ci95": [
      0.0,
      0.3629062754133911
    ],
    "n_finite": 2000
  }
}
```

## Heatmaps

- `heatmap_surface_auroc.svg`
- `heatmap_activation_auroc.svg`
- `heatmap_surface_plus_activation_auroc.svg`
- `heatmap_delta_auroc.svg`

## Authorizations

Locked TEST analysis / SAE / causal: **false**.
