# Phase 22C — updated sampling forecast from observed K=20

**Status:** `phase22c_sampling_expansion_supported`

**Decision:** sampling expansion supported at K≥60 (≥2 non-structural-zero models P(BOTH)≥0.8 and conservative rare-event P(BOTH)≥0.5)

**Smallest supported K:** 60

**Interpretation:** The observed K=20 result provided evidence that prompt-level behavior was more stable than the optimistic K=10 Jeffreys forecast assumed.

## Sources

- Phase-21 onset: `phase21_onset_20260928T212924Z_68a5621c`
- Phase-22B onset: `phase22b_onset_20260929T130149Z_96f69da6`
- Combined: 20 rollouts × 371 prompts

## K=20 reproduction (required)

- TRAIN qualifying: **34** (expected 34)
- TEST qualifying: **4** (expected 4)
- Reproduced: **True** → HOLD

## Method

- 3-way multinomial: honest / deceptive-explicit / other.
- Condition on observed 20; draw K−20 additional.
- Monte Carlo: 20000 (seed 22300000).
- Gates unchanged: ≥2/2; TRAIN≥25 / TEST≥8.
- No Mistral / OpenAI / activations / physiology / new generation.

### Model definitions

**`jeffreys`:** Dirichlet(0.5,0.5,0.5) + observed counts

**`empirical_weak`:** Dirichlet(κ·π̂_global) with κ=1.0; π̂ from pooled K=20 counts; prompt data dominate

**`structural_zeros`:** Never-observed class fixed at p=0 (no prior rescue); remaining classes use Jeffreys+data renorm via Dirichlet

**`conservative_rare_event`:** For class i with count>0: α_i=count_i. For count=0: α_i=Clopper–Pearson one-sided (1−0.05) UCB for 0/20 = 1−0.05^(1/20)≈0.139108. Draw p~Dirichlet(α); additional Multinomial(K−20, p). Does not fully Jeffreys-rescue never-seen classes.

## Observed at K=20

- Qualifying: TRAIN 34 / TEST 4
- Prompts with zero honest: 177
- Prompts with zero deceptive-explicit: 137
- Global proportions (h, d, other): [0.3213, 0.429, 0.2497]

### Phase-22B switching (descriptive)

| bucket | acquired / n |
|---|---:|
| honest>0, dec=0 → acquire dec | 11/134 |
| dec>0, honest=0 → acquire honest | 8/171 |
| exact 10/0 → acquire dec | 3/63 |
| exact 0/10 → acquire honest | 2/106 |

## Forecast tables

### `jeffreys`

| K | E[TRAIN] [95% PI] | E[TEST] [95% PI] | P(TRAIN≥25) | P(TEST≥8) | P(both) |
|---:|---|---|---:|---:|---:|
| 30 | 53.94 [47, 62] | 7.55 [5, 11] | 1.0000 | 0.4867 | 0.4867 |
| 40 | 75.12 [65, 86] | 11.18 [7, 16] | 1.0000 | 0.9604 | 0.9604 |
| 60 | 108.07 [95, 122] | 16.74 [11, 23] | 1.0000 | 0.9999 | 0.9999 |
| 80 | 130.96 [116, 146] | 20.62 [15, 27] | 1.0000 | 1.0000 | 1.0000 |

### `empirical_weak`

| K | E[TRAIN] [95% PI] | E[TEST] [95% PI] | P(TRAIN≥25) | P(TEST≥8) | P(both) |
|---:|---|---|---:|---:|---:|
| 30 | 50.54 [44, 58] | 6.99 [4, 10] | 1.0000 | 0.3428 | 0.3428 |
| 40 | 67.17 [58, 77] | 9.85 [6, 14] | 1.0000 | 0.8875 | 0.8875 |
| 60 | 92.94 [81, 106] | 14.27 [10, 20] | 1.0000 | 0.9979 | 0.9979 |
| 80 | 111.40 [98, 125] | 17.44 [12, 23] | 1.0000 | 1.0000 | 1.0000 |

### `structural_zeros`

| K | E[TRAIN] [95% PI] | E[TEST] [95% PI] | P(TRAIN≥25) | P(TEST≥8) | P(both) |
|---:|---|---|---:|---:|---:|
| 30 | 43.46 [39, 48] | 5.81 [4, 8] | 1.0000 | 0.0417 | 0.0417 |
| 40 | 47.68 [43, 52] | 6.60 [5, 8] | 1.0000 | 0.1771 | 0.1771 |
| 60 | 51.42 [48, 55] | 7.23 [5, 8] | 1.0000 | 0.4245 | 0.4245 |
| 80 | 52.98 [50, 56] | 7.49 [6, 8] | 1.0000 | 0.5771 | 0.5771 |

### `conservative_rare_event`

| K | E[TRAIN] [95% PI] | E[TEST] [95% PI] | P(TRAIN≥25) | P(TEST≥8) | P(both) |
|---:|---|---|---:|---:|---:|
| 30 | 43.74 [39, 49] | 5.82 [4, 8] | 1.0000 | 0.0790 | 0.0790 |
| 40 | 51.96 [45, 59] | 7.26 [5, 10] | 1.0000 | 0.4144 | 0.4144 |
| 60 | 64.37 [56, 74] | 9.40 [6, 13] | 1.0000 | 0.8542 | 0.8542 |
| 80 | 73.39 [64, 83] | 10.88 [7, 15] | 1.0000 | 0.9608 | 0.9608 |

## Decision rule application

| K | Jeffreys P(both) | Empirical P(both) | Conservative P(both) | Structural P(both) | Meets support? |
|---:|---:|---:|---:|---:|:---:|
| 30 | 0.4867 | 0.3428 | 0.0790 | 0.0417 | no |
| 40 | 0.9604 | 0.8875 | 0.4144 | 0.1771 | no |
| 60 | 0.9999 | 0.9979 | 0.8542 | 0.4245 | yes |
| 80 | 1.0000 | 1.0000 | 0.9608 | 0.5771 | yes |

Support requires ≥2 non-structural-zero models with P(BOTH)≥0.8 at the same K **and** conservative rare-event P(BOTH)≥0.5.

## Authorizations (all false)

- model / Mistral / OpenAI calls
- activation extraction / probe fitting / physiology
- K>20 generation / adaptive resampling
- prompt / threshold changes

## Guarantee

PHASE 22C WAS A ZERO-MODEL-CALL UPDATED SAMPLING FORECAST USING ONLY FROZEN PHASE-21 AND PHASE-22B ONSET-ANNOTATED ROLLOUTS (20 PER PROMPT). NO MISTRAL CALLS, OPENAI CALLS, ACTIVATIONS, OR PHYSIOLOGY WERE PERFORMED. NO NEW RESPONSES WERE GENERATED. PHASE-21/22B LABELS, SPLIT, PROMPTS, AND POPULATION THRESHOLDS WERE NOT CHANGED. THE OBSERVED K=20 RESULT PROVIDED EVIDENCE THAT PROMPT-LEVEL BEHAVIOR WAS MORE STABLE THAN THE OPTIMISTIC K=10 JEFFREYS FORECAST ASSUMED.

