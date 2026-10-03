# Phase 12A — family-bias source diagnostic (behavior only)

**Run ID:** `phase12a_diagnostic_20260926T224325Z_860e81be`  
**Git SHA:** `860e81be19a0ffd0eb3df93bff898bf420b0618b`  
**Status:** `phase12a_family_bias_source_diagnostic_complete_awaiting_audit`  

144 deterministic Phase-11 discovery bases (24/family). A = original prompt (Phase-11 RF/AF reused); B = family shell, `slot` states; C = generic shell, family state noun; D = generic shell, `slot`. K=10, payoff, decoder unchanged. Label = order-robust (RF and AF agree). Alternate rate is among stable bases.

| Cell | Stable rec/alt | Order-sensitive rate | Cramér's V | Alt-rate range | Substantial |
| --- | --- | ---: | ---: | ---: | --- |
| A FULL_FAMILY | 47/67 | 0.208 | 0.911 | 1.000 | True |
| B SEMANTIC_ONLY | 25/97 | 0.153 | 0.843 | 0.900 | True |
| C STATE_ONLY | 4/122 | 0.125 | 0.436 | 0.273 | False |
| D FULL_NEUTRAL | 0/144 | 0.000 | 0.000 | 0.000 | False |

**Stable record / stable alternate / order-sensitive (alternate rate) by family**

| Family | A | B | C | D |
| --- | --- | --- | --- | --- |
| bus_depot_bay | 0/22/2 (1.00) | 0/24/0 (1.00) | 0/24/0 (1.00) | 0/24/0 (1.00) |
| cafeteria_serving_window | 1/18/5 (0.95) | 7/9/8 (0.56) | 0/24/0 (1.00) | 0/24/0 (1.00) |
| gym_locker_row | 16/0/8 (0.00) | 18/2/4 (0.10) | 0/24/0 (1.00) | 0/24/0 (1.00) |
| library_return_cart | 0/19/5 (1.00) | 0/24/0 (1.00) | 0/21/3 (1.00) | 0/24/0 (1.00) |
| concert_hall_door | 23/0/1 (0.00) | 0/15/9 (1.00) | 1/21/2 (0.95) | 0/24/0 (1.00) |
| summer_camp_cabin | 7/8/9 (0.53) | 0/23/1 (1.00) | 3/8/13 (0.73) | 0/24/0 (1.00) |

Family identity intentionally removed from model-visible text in D; family grouping in D reflects only which record/alternate numbers and item IDs the family's selected bases carry.

**Matched agreement (by base)**

| Pair | Exact label | Both-stable label | RF choice | AF choice |
| --- | ---: | ---: | ---: | ---: |
| A_vs_B | 0.535 | 0.760 (n=96) | 0.701 | 0.674 |
| A_vs_C | 0.438 | 0.606 (n=99) | 0.660 | 0.493 |
| B_vs_D | 0.674 | 0.795 (n=122) | 0.819 | 0.681 |
| C_vs_D | 0.847 | 0.968 (n=126) | 0.958 | 0.861 |

**Order switches**

- A: RF record → AF alternate 4; RF alternate → AF record 26
- B: RF record → AF alternate 1; RF alternate → AF record 21
- C: RF record → AF alternate 2; RF alternate → AF record 16
- D: RF record → AF alternate 0; RF alternate → AF record 0

**Frozen interpretation**

```
{
 "thresholds": {
  "substantial": "V >= 0.5 and range >= 0.5",
  "replication": "A V >= 0.7"
 },
 "replication_passed": true,
 "A_cramers_v": 0.9113448198984814,
 "state_vocabulary_contributor": false,
 "semantic_shell_contributor": true,
 "mixed_contributors": false,
 "interaction_or_other_family_structure": false
}
```

PHASE 12A WAS A BEHAVIORAL CONFOUND-DIAGNOSTIC ONLY. PHASE 11A REMAINS A BEHAVIOR-GATE HOLD. NO ACTIVATIONS WERE COLLECTED. NO PHYSIOLOGY PROBE WAS FIT OR SCORED. THE PHASE-5 PROBE WAS NOT SCORED. K, PAYOFFS, AND DECODER WERE NOT TUNED. LOCKED FAMILIES WERE NOT RUN. NO CAUSAL INTERVENTIONS WERE PERFORMED.
