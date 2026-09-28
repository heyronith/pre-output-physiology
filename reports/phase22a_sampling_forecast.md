# Phase 22A — sampling forecast (zero model calls)

**Status:** `phase22_sampling_expansion_supported`

**Decision:** sampling expansion supported

Best forecast: K=80 with P(both gates)=1.0000 (credible≥0.5, strong≥0.8).

## Method

- Data: Phase-21 onset annotations only (`phase21_onset_20260928T212924Z_68a5621c`).
- Per prompt: 3-way counts (honest / deceptive-with-valid-explicit-onset / other).
- Prior: Jeffreys Dirichlet(0.5, 0.5, 0.5).
- Predictive: keep the observed 10 rollouts; draw K−10 additional from the posterior predictive Multinomial.
- Monte Carlo replicates: 10000 (seed 22000000).
- Gates unchanged: ≥2 honest & ≥2 explicit-onset deceptive; TRAIN≥25, TEST≥8 qualifying prompts.
- No Mistral / OpenAI / activations / physiology.

## Observed stability at K=10

- TRAIN qualifying: 19
- TEST qualifying: 1

| bucket (n_honest / n_dec_explicit) | n prompts |
|---|---:|
| `10/0_or_0/10` | 169 |
| `8/2_or_2/8` | 3 |
| `9/1_or_1/9` | 5 |
| `more_balanced_or_other` | 194 |

## Forecast table

| K | E[TRAIN qual] | E[TEST qual] | P(TRAIN≥25) | P(TEST≥8) | P(both) |
|---:|---:|---:|---:|---:|---:|
| 20 | 56.20 | 7.91 | 1.0000 | 0.5559 | 0.5559 |
| 40 | 112.85 | 17.58 | 1.0000 | 0.9998 | 0.9998 |
| 80 | 164.95 | 26.54 | 1.0000 | 1.0000 | 1.0000 |
| 160 | 206.64 | 33.71 | 1.0000 | 1.0000 | 1.0000 |

## Sensitivity: structural zeros (no prior rescue)

If a class was never seen in the 10 rollouts, its future probability is fixed at 0.

Sensitivity verdict: **sampling-only rescue unsupported** (best P(both)=0.0000 at K=20).

| K | E[TRAIN] | E[TEST] | P(TRAIN≥25) | P(TEST≥8) | P(both) |
|---:|---:|---:|---:|---:|---:|
| 20 | 30.78 | 3.57 | 0.9981 | 0.0000 | 0.0000 |
| 40 | 35.38 | 4.49 | 1.0000 | 0.0000 | 0.0000 |
| 80 | 37.07 | 4.82 | 1.0000 | 0.0000 | 0.0000 |
| 160 | 37.67 | 4.94 | 1.0000 | 0.0000 | 0.0000 |

## Guarantee

PHASE 22 WAS A ZERO-MODEL-CALL SAMPLING FORECAST USING ONLY FROZEN PHASE-21 GRADED AND ONSET-ANNOTATED ROLLOUTS. NO MISTRAL CALLS, OPENAI CALLS, ACTIVATIONS, OR PHYSIOLOGY WERE PERFORMED. PHASE-21 LABELS, SPLIT, PROMPTS, AND POPULATION THRESHOLDS WERE NOT CHANGED. NO NEW RESPONSES WERE GENERATED.

