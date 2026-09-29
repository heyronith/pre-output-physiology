# Phase 21A — mixed-population gates

**Status:** `phase21_roleplay_mixed_population_hold`

## Runs
- Generation: `phase21_gen_20260928T180342Z_3d3671eb` (3710)
- Grading: `phase21_grade_20260928T205710Z_b6287702` (gpt-4o-2024-08-06)
- Onset: `phase21_onset_20260928T212924Z_68a5621c` (1586 deceptive; 1579 explicit spans)

## Labels
| class | n |
|---|---:|
| honest | 1173 |
| ambiguous | 871 |
| deceptive | 1586 |
| exclude | 80 |

## Gates
| gate | required | observed | pass |
|---|---:|---:|---|
| TRAIN qualifying prompts | ≥25 | 19 | no |
| TEST qualifying prompts | ≥8 | 1 | no |

Qualifying = ≥2 honest + ≥2 deceptive-with-explicit-onset; take first 2/class by seed.

## Decision
STOP before activation extraction / probe fitting. No replacement prompts.
