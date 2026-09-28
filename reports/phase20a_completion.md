# Phase 20A completion report

Status: `phase20a_preanswer_physiology_discovery_hold`

STOP after Phase 20A. No validation physiology.

## Eligibility (per prompt)
| prompt | split | family | n_elig | n_rec | n_alt | balanced |
|---|---|---|---:|---:|---:|---|
| `final_bowling_alley_lane_003` | train | bowling_alley_lane | 128 | 66 | 62 | True |
| `final_bowling_alley_lane_009` | train | bowling_alley_lane | 128 | 39 | 89 | True |
| `final_bowling_alley_lane_012` | train | bowling_alley_lane | 127 | 61 | 66 | True |
| `final_bowling_alley_lane_013` | train | bowling_alley_lane | 126 | 72 | 54 | True |
| `final_bowling_alley_lane_016` | train | bowling_alley_lane | 124 | 47 | 77 | True |
| `final_bowling_alley_lane_030` | train | bowling_alley_lane | 128 | 17 | 111 | True |
| `final_radio_studio_booth_004` | train | radio_studio_booth | 125 | 27 | 98 | True |
| `final_radio_studio_booth_020` | train | radio_studio_booth | 119 | 56 | 63 | True |
| `final_radio_studio_booth_022` | train | radio_studio_booth | 124 | 106 | 18 | True |
| `final_radio_studio_booth_027` | train | radio_studio_booth | 128 | 82 | 46 | True |
| `final_radio_studio_booth_034` | train | radio_studio_booth | 128 | 41 | 87 | True |
| `final_veterinary_kennel_run_000` | train | veterinary_kennel_run | 128 | 108 | 20 | True |
| `final_veterinary_kennel_run_004` | train | veterinary_kennel_run | 128 | 107 | 21 | True |
| `final_veterinary_kennel_run_008` | train | veterinary_kennel_run | 128 | 56 | 72 | True |
| `final_veterinary_kennel_run_009` | train | veterinary_kennel_run | 128 | 80 | 48 | True |
| `final_veterinary_kennel_run_012` | train | veterinary_kennel_run | 128 | 81 | 47 | True |
| `final_veterinary_kennel_run_032` | train | veterinary_kennel_run | 124 | 66 | 58 | True |
| `p19_mini_golf_tee_pad_000` | validation | mini_golf_tee_pad | 121 | 81 | 40 | True |
| `p19_mini_golf_tee_pad_008` | validation | mini_golf_tee_pad | 126 | 32 | 94 | True |
| `p19_mini_golf_tee_pad_009` | validation | mini_golf_tee_pad | 127 | 88 | 39 | True |
| `p19_mini_golf_tee_pad_011` | validation | mini_golf_tee_pad | 127 | 74 | 53 | True |
| `p19_mini_golf_tee_pad_015` | validation | mini_golf_tee_pad | 113 | 40 | 73 | True |
| `p19_mini_golf_tee_pad_017` | validation | mini_golf_tee_pad | 127 | 60 | 67 | True |
| `p19_music_school_practice_room_004` | validation | music_school_practice_room | 126 | 79 | 47 | True |
| `p19_music_school_practice_room_005` | validation | music_school_practice_room | 126 | 71 | 55 | True |
| `p19_music_school_practice_room_010` | validation | music_school_practice_room | 128 | 38 | 90 | True |
| `p19_music_school_practice_room_014` | validation | music_school_practice_room | 119 | 69 | 50 | True |
| `p19_music_school_practice_room_018` | validation | music_school_practice_room | 128 | 110 | 18 | True |

## Baseline by position (LOFO pooled AUROC)

- **end0**: B1=0.5848, B2=1.0000, B3=0.6514, B4=0.6578 → best=B2 (1.0000)
- **end2**: B1=0.5848, B2=0.8868, B3=0.5093, B4=0.6005 → best=B2 (0.8868)
- **end4**: B1=0.5848, B2=0.8413, B3=0.5450, B4=0.5961 → best=B2 (0.8413)

## Discovery activation AUROC (pooled LOFO)

| block | end0 | end2 | end4 |
|---:|---:|---:|---:|
| 0 | 0.5129 | 0.5913 | 0.6043 |
| 4 | 0.4551 | 0.5037 | 0.5817 |
| 8 | 0.6275 | 0.5014 | 0.7298 |
| 12 | 0.6323 | 0.4366 | 0.6338 |
| 16 | 0.6971 | 0.4387 | 0.6135 |
| 20 | 0.7351 | 0.4104 | 0.6145 |
| 24 | 0.7597 | 0.4811 | 0.6531 |
| 28 | 0.7718 | 0.4670 | 0.6245 |
| 31 | 0.7516 | 0.5480 | 0.5981 |

## Guarantee

PHASE 20A WAS THE FIRST PHYSIOLOGY TEST ON SAME-PROMPT STOCHASTIC POLICY TRAJECTORIES. PHYSIOLOGY MODEL SELECTION USED ONLY THE FROZEN ENRICHED TRAIN POPULATION. THE UNSEEN PHASE-19 VALIDATION FAMILIES WERE NOT USED FOR LAYER, POSITION, PROBE, BASELINE, OR THRESHOLD SELECTION. K0 WAS A NEGATIVE CONTROL ONLY. ALL PRIMARY TRAJECTORIES WERE MEASURED BEFORE THE FINAL RESPONSE PROPOSITION. NO LOCKED-FAMILY CALLS, POST-RESULT TUNING, PHASE-5 PROBE SCORING, OR CAUSAL INTERVENTIONS WERE PERFORMED.
