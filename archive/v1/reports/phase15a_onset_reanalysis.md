# Phase 15A — same-prompt proposition-onset reanalysis

**Source run:** `phase14a_calibration_20260926T231646Z_e6206543`  
**Raw continuations SHA256:** `b1c003b3282387ef43e0e3d8b31a14978fbfe32105eac8cdffd740ff19aa44b8`  
**Annotation rule:** `phase15_onset_v1` / `6bb40fa50eaf07b52e4709355bdb4652060658c2e73d3e41775b8bccac4bc24f`  
**Status:** `phase15a_same_prompt_onset_reanalysis_hold`  
**Phase 14A status (unchanged):** `phase14a_same_prompt_trajectory_calibration_hold`  

Calibration-data reanalysis only. Phase-15 validity admits exact candidate mentions without an assignment proposition (`candidate_mention_only`). No T* selection; no final prompts; no model calls.

## Onset category counts

| Category | n |
| --- | ---: |
| `clean_no_candidate_mention` | 1168 |
| `candidate_assignment_proposition` | 32 |
| `candidate_mention_only` | 336 |

## Old vs new validity

| Cross-tab | n |
| --- | ---: |
| `phase14_valid|phase15_valid` | 1168 |
| `phase14_invalid|phase15_invalid` | 32 |
| `phase14_invalid|phase15_valid` | 336 |

## Per-temperature Phase-15-valid feasibility

| T | Valid frac | Assign leak | Record frac | Alternate frac | ≥1/class | ≥3/class | Median valid/prompt | Replication-worthy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.7 | 0.992 | 0.008 | 0.866 | 0.134 | 6/32 | 5/32 | 16.0 | False |
| 0.9 | 0.975 | 0.025 | 0.870 | 0.130 | 13/32 | 6/32 | 16.0 | False |
| 1.1 | 0.971 | 0.029 | 0.841 | 0.159 | 12/32 | 6/32 | 16.0 | False |

**Replication-worthy temperatures:** []  
**Selects T\*:** false  

## Newly recovered vs Phase-14 validity

| T | Newly valid | New record | New alternate | Groups ↑ alternate | Groups newly with alternate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.7 | 124 | 102 | 22 | 3 | 2 |
| 0.9 | 96 | 80 | 16 | 5 | 4 |
| 1.1 | 116 | 93 | 23 | 8 | 3 |

## Final policy rates by Stage-1 onset category (descriptive)

### T=0.7

| Category | n | Record frac | Alternate frac |
| --- | ---: | ---: | ---: |
| `clean_no_candidate_mention` | 384 | 0.880 | 0.120 |
| `candidate_mention_only` | 124 | 0.823 | 0.177 |
| `candidate_assignment_proposition` | 4 | 0.500 | 0.500 |

### T=0.9

| Category | n | Record frac | Alternate frac |
| --- | ---: | ---: | ---: |
| `clean_no_candidate_mention` | 403 | 0.878 | 0.122 |
| `candidate_mention_only` | 96 | 0.833 | 0.167 |
| `candidate_assignment_proposition` | 13 | 0.385 | 0.615 |

### T=1.1

| Category | n | Record frac | Alternate frac |
| --- | ---: | ---: | ---: |
| `clean_no_candidate_mention` | 381 | 0.853 | 0.147 |
| `candidate_mention_only` | 116 | 0.802 | 0.198 |
| `candidate_assignment_proposition` | 15 | 0.333 | 0.667 |

PHASE 14A REMAINS A HOLD UNDER ITS ORIGINAL FROZEN RULES. PHASE 15A ONLY REANALYZED THE EXISTING CALIBRATION TRAJECTORIES USING A PROPOSITION-ONSET VALIDITY DEFINITION FROZEN BEFORE RECOMPUTING BEHAVIOR STATISTICS. NO MODEL CALLS, ACTIVATIONS, PROBES, FINAL/LOCKED PROMPTS, RESAMPLING, PROMPT TUNING, TEMPERATURE EXTENSION, OR CAUSAL INTERVENTIONS WERE PERFORMED.
