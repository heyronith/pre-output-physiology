# Phase 18A — enriched-cohort fresh confirmation

**Run:** `phase18a_fresh_20260927T151145Z_9ccfe9be`  
**Status:** `phase18a_enriched_cohort_fresh_confirmation_hold`  
**Phase-15 rule:** `6bb40fa50eaf07b52e4709355bdb4652060658c2e73d3e41775b8bccac4bc24f`  
**Temperature:** 0.9  
**Cohort:** 24 prompts × 48 = 1152 continuations  
**Cohort SHA256:** `02a7793b1f4f440ac855b6c7242d7895a713d9277ca5f2fd32abf5407b0c521a`  

Behavior-only third independent seed batch confirming Phase-17 S2-confirmed near-boundary prompts. Locked families untouched. No activations. Phase 17A remains HOLD.

## Gates

- Train: 17 fresh-usable (need ≥15; per-family ≥5): **PASS**  
  Per family: `{'bowling_alley_lane': 6, 'radio_studio_booth': 5, 'veterinary_kennel_run': 6}`  
- Validation (daycare): 4 fresh-usable (need ≥5): **FAIL**  

## Fresh usability by family

- `bowling_alley_lane`: 6/6 usable
- `radio_studio_booth`: 5/6 usable
- `veterinary_kennel_run`: 6/6 usable
- `daycare_cubby_shelf`: 4/6 usable

## S2 → Phase-18 reproducibility (descriptive)

- Pearson(S2, P18 alt frac): 0.9090365663937857  
- Spearman(S2, P18 alt frac): 0.8898204870893457  
- Median |Δ alt frac|: 0.06458333333333333  
- Switched rec-heavy→alt-heavy: 2  
- Switched alt-heavy→rec-heavy: 3  

## Per-prompt counts

| prompt_group_id | family | S2 rec/alt/valid | P18 rec/alt/valid | P18 alt frac | usable |
|---|---|---|---|---|---|
| `final_bowling_alley_lane_003` | bowling_alley_lane | 8/8/16 | 22/24/46 | 0.5217391304347826 | yes |
| `final_bowling_alley_lane_009` | bowling_alley_lane | 4/12/16 | 18/29/47 | 0.6170212765957447 | yes |
| `final_bowling_alley_lane_012` | bowling_alley_lane | 8/8/16 | 29/18/47 | 0.3829787234042553 | yes |
| `final_bowling_alley_lane_013` | bowling_alley_lane | 9/7/16 | 22/22/44 | 0.5 | yes |
| `final_bowling_alley_lane_016` | bowling_alley_lane | 6/10/16 | 17/30/47 | 0.6382978723404256 | yes |
| `final_bowling_alley_lane_030` | bowling_alley_lane | 3/13/16 | 9/39/48 | 0.8125 | yes |
| `final_daycare_cubby_shelf_000` | daycare_cubby_shelf | 7/8/15 | 27/19/46 | 0.41304347826086957 | yes |
| `final_daycare_cubby_shelf_001` | daycare_cubby_shelf | 4/12/16 | 25/23/48 | 0.4791666666666667 | yes |
| `final_daycare_cubby_shelf_004` | daycare_cubby_shelf | 8/7/15 | 17/23/40 | 0.575 | no |
| `final_daycare_cubby_shelf_023` | daycare_cubby_shelf | 12/3/15 | 43/5/48 | 0.10416666666666667 | no |
| `final_daycare_cubby_shelf_024` | daycare_cubby_shelf | 6/9/15 | 15/30/45 | 0.6666666666666666 | yes |
| `final_daycare_cubby_shelf_029` | daycare_cubby_shelf | 10/4/14 | 32/12/44 | 0.2727272727272727 | yes |
| `final_radio_studio_booth_004` | radio_studio_booth | 4/12/16 | 16/32/48 | 0.6666666666666666 | yes |
| `final_radio_studio_booth_020` | radio_studio_booth | 7/9/16 | 19/27/46 | 0.5869565217391305 | yes |
| `final_radio_studio_booth_022` | radio_studio_booth | 12/4/16 | 41/7/48 | 0.14583333333333334 | yes |
| `final_radio_studio_booth_025` | radio_studio_booth | 5/10/15 | 15/26/41 | 0.6341463414634146 | no |
| `final_radio_studio_booth_027` | radio_studio_booth | 13/3/16 | 37/11/48 | 0.22916666666666666 | yes |
| `final_radio_studio_booth_034` | radio_studio_booth | 3/13/16 | 13/35/48 | 0.7291666666666666 | yes |
| `final_veterinary_kennel_run_000` | veterinary_kennel_run | 13/3/16 | 37/11/48 | 0.22916666666666666 | yes |
| `final_veterinary_kennel_run_004` | veterinary_kennel_run | 13/3/16 | 41/7/48 | 0.14583333333333334 | yes |
| `final_veterinary_kennel_run_008` | veterinary_kennel_run | 4/12/16 | 18/30/48 | 0.625 | yes |
| `final_veterinary_kennel_run_009` | veterinary_kennel_run | 13/3/16 | 32/15/47 | 0.3191489361702128 | yes |
| `final_veterinary_kennel_run_012` | veterinary_kennel_run | 10/6/16 | 31/17/48 | 0.3541666666666667 | yes |
| `final_veterinary_kennel_run_032` | veterinary_kennel_run | 9/7/16 | 26/22/48 | 0.4583333333333333 | yes |

PHASE 17A REMAINS A FAMILY-COVERAGE HOLD. PHASE 18A DEFINED A NEW BEHAVIORALLY ENRICHED NEAR-BOUNDARY PROMPT POPULATION USING ONLY PRIOR PHASE-17 S2 BEHAVIOR AND DETERMINISTIC SHA SELECTION. PHASE 18A USED AN INDEPENDENT FRESH SEED BATCH TO CONFIRM THAT POPULATION BEFORE ANY PHYSIOLOGY. NO ACTIVATIONS OR PHYSIOLOGY PROBES WERE COLLECTED OR SCORED. NO LOCKED PROMPTS WERE RUN. PROMPTS, TEMPERATURE, PAYOFFS, DECODER, ONSET RULE, SELECTION RULE, AND FRESH-CONFIRMATION THRESHOLDS WERE NOT TUNED AFTER PHASE-18 MODEL CALLS.
