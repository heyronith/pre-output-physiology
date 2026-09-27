# Phase 17A — policy-unstable exact-prompt enrichment screen

**S1 run:** `phase17a_s1_20260927T004928Z_e76c3592`  
**S2 run:** `phase17a_s2_20260927T014319Z_ec5bc192`  
**Status:** `phase17a_policy_unstable_screen_hold`  
**Phase-15 rule:** `6bb40fa50eaf07b52e4709355bdb4652060658c2e73d3e41775b8bccac4bc24f`  
**Temperature:** 0.9  

Behavior-only enrichment of frozen Phase-14 discovery finals. Locked families untouched. No activations.

## Gates

- Train: 34 confirmed (need ≥20; per-family ≥4): **FAIL**  
  Per family: `{'climbing_gym_route': 2, 'radio_studio_booth': 9, 'bowling_alley_lane': 15, 'veterinary_kennel_run': 8}`  
- Validation: 6 confirmed (need ≥10; per-family ≥4): **FAIL**  
  Per family: `{'subway_turnstile_bank': 0, 'daycare_cubby_shelf': 6}`  

**S1→S2 confirmation rate:** 0.597 (40/67)  

## Reproducibility (descriptive)

- Pearson(S1,S2 alt frac): 0.8745888783457285  
- Median |Δ alt frac|: 0.09999999999999998  
- Switched rec-heavy→alt-heavy: 7  
- Switched alt-heavy→rec-heavy: 3  

PHASE 17A WAS A BEHAVIOR-ONLY PROSPECTIVE ENRICHMENT SCREEN FOR EXACT PROMPTS WITH REPRODUCIBLE STOCHASTIC POLICY DIVERSITY. PHASES 14A AND 15A REMAIN HOLDS, AND PHASE 16A FOUND SAMPLING-ONLY RESCUE OF RANDOM PROMPTS UNSUPPORTED. NO ACTIVATIONS OR PHYSIOLOGY PROBES WERE USED FOR SCREENING. NO LOCKED PROMPTS WERE RUN. TEMPERATURE, PROMPTS, PAYOFFS, DECODER, ONSET RULE, AND SCREENING THRESHOLDS WERE NOT TUNED AFTER MODEL CALLS.
