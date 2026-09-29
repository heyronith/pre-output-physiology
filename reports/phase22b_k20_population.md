# Phase 22B — K=20 population validation

**Status:** `phase22b_k20_population_hold`

- Phase-21 onset: `phase21_onset_20260928T212924Z_68a5621c`
- Phase-22B onset: `phase22b_onset_20260929T130149Z_96f69da6`
- Combined rollouts/prompt: 20

## Population gates

- TRAIN qualifying: 34 (need ≥25)
- TEST qualifying: 4 (need ≥8)
- Gates passed: False

## Behavior switching (Phase-21 one-sided → acquire missing in r11–20)

### Bucket sizes at K=10

- `honest_gt0_dec_eq0`: n=134 (train=115, test=19)
- `dec_gt0_honest_eq0`: n=171 (train=146, test=25)
- `exact_10_0_or_0_10`: n=169 (train=143, test=26)

### Acquisition of previously unseen class

- `honest_gt0_dec_eq0_acquires_dec_in_11_20`: acquired 11/134 (train 10/115, test 1/19; rate=0.08208955223880597)
- `dec_gt0_honest_eq0_acquires_honest_in_11_20`: acquired 8/171 (train 6/146, test 2/25; rate=0.04678362573099415)
- exact 10/0: n=63; acquire dec in 11–20: 3
- exact 0/10: n=106; acquire honest in 11–20: 2

## Guarantee

PHASE 22B WAS A BEHAVIOR-ONLY PROSPECTIVE K=20 SAMPLING VALIDATION ON THE ORIGINAL 371 APOLLO ROLEPLAYDECEPTION PROMPTS. TEN ADDITIONAL INDEPENDENT ROLLOUTS WERE GENERATED FOR EVERY PROMPT (R10–R19) WITH SEEDS DISJOINT FROM PHASE 21. PHASE-21 ROLLOUTS R00–R09 WERE KEPT UNCHANGED. GRADING USED GPT-4O-2024-08-06 AND THE EXACT APOLLO TEMPLATE. ONSET ANNOTATION WAS TEXT-ONLY. POPULATION GATES WERE UNCHANGED (≥2 HONEST AND ≥2 EXPLICIT-ONSET DECEPTIVE; TRAIN≥25 / TEST≥8). NO ACTIVATIONS, PROBE FITTING, PHYSIOLOGY, PROMPT CHANGES, THRESHOLD CHANGES, OR K>20 GENERATION WERE PERFORMED.

