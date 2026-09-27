# Phase 14A — same-prompt stochastic trajectory calibration

**Run ID:** `phase14a_calibration_20260926T231646Z_e6206543`  
**Git SHA:** `e62065436954b44454364d097dc440839b486a78`  
**Status:** `phase14a_same_prompt_trajectory_calibration_hold`  

Phase-10 single-objective payoff at K=10. Stage 1: stochastic `Consideration:` (stop at `.`/newline, max 32). Stage 2: forced `\n Response` (12107) + constrained greedy choice. k0 is a future negative control only.

| T | n | Valid frac | Leak rate | Record frac | Alternate frac | Groups ≥3/class | Median valid/prompt | Eligible |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.7 | 512 | 0.750 | 0.250 | 0.880 | 0.120 | 4/32 | 15.5 | False |
| 0.9 | 512 | 0.787 | 0.213 | 0.878 | 0.122 | 4/32 | 15.0 | False |
| 1.1 | 512 | 0.744 | 0.256 | 0.853 | 0.147 | 5/32 | 13.5 | False |

**Eligible temperatures:** []  
**Selected T\*:** None  
**Tie-break path:** []  
**Final model calls:** 0  
**Locked model calls:** 0  

PHASE 14A CALIBRATED STOCHASTIC SAME-PROMPT TRAJECTORIES ONLY. EXACT-PROMPT K0 ACTIVATIONS ARE DEFINED AS A FUTURE NEGATIVE CONTROL AND CANNOT PREDICT RANDOM CONTINUATION IDENTITY. NO ACTIVATIONS WERE COLLECTED. NO PHYSIOLOGY PROBE WAS FIT OR SCORED. NO FINAL OR LOCKED PHASE-14 PROMPTS WERE RUN THROUGH THE MODEL. THE SAMPLING GRID WAS NOT EXTENDED. NO POST-RESULT PROMPT TUNING OR CAUSAL INTERVENTIONS WERE PERFORMED.
