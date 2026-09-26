# Phase 13A — semantic-component diagnostic (T × Q × E)

**Run ID:** `phase13a_diagnostic_20260926T225741Z_095895f5`  
**Git SHA:** `095895f508830223fe165b1aaccc9a6b4f281ec6`  
**Status:** `phase13a_semantic_component_diagnostic_complete_awaiting_audit`  

144 Phase-12 selected bases. Every cell uses `slot N` states. Factors: T=topic, Q=question, E=entity. Cells `000`/`111` reuse Phase-12 D/B (byte-identical). Label = order-robust (RF and AF agree). Substantial = V≥0.50 and range≥0.50.

| Cell | TQE | Stable rec/alt | OS rate | V | Range | Substantial |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 000 (generic) | 000 | 0/144 | 0.000 | 0.000 | 0.000 | False |
| 001 (entity only) | 001 | 0/144 | 0.000 | 0.000 | 0.000 | False |
| 010 (question only) | 010 | 0/144 | 0.000 | 0.000 | 0.000 | False |
| 011 (Q+E) | 011 | 0/144 | 0.000 | 0.000 | 0.000 | False |
| 100 (topic only) | 100 | 5/118 | 0.146 | 0.425 | 0.250 | False |
| 101 (T+E) | 101 | 21/97 | 0.181 | 0.819 | 0.875 | True |
| 110 (T+Q) | 110 | 3/125 | 0.111 | 0.383 | 0.167 | False |
| 111 (full shell) | 111 | 25/97 | 0.153 | 0.843 | 0.900 | True |

**Stable record / alternate / order-sensitive by family**

| Family | 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bus_depot_bay | 0/24/0 | 0/24/0 | 0/24/0 | 0/24/0 | 0/24/0 | 0/24/0 | 0/24/0 | 0/24/0 |
| cafeteria_serving_window | 0/24/0 | 0/24/0 | 0/24/0 | 0/24/0 | 0/24/0 | 7/9/8 | 0/24/0 | 7/9/8 |
| gym_locker_row | 0/24/0 | 0/24/0 | 0/24/0 | 0/24/0 | 4/12/8 | 14/2/8 | 3/15/6 | 18/2/4 |
| library_return_cart | 0/24/0 | 0/24/0 | 0/24/0 | 0/24/0 | 0/24/0 | 0/24/0 | 0/24/0 | 0/24/0 |
| concert_hall_door | 0/24/0 | 0/24/0 | 0/24/0 | 0/24/0 | 1/13/10 | 0/15/9 | 0/18/6 | 0/15/9 |
| summer_camp_cabin | 0/24/0 | 0/24/0 | 0/24/0 | 0/24/0 | 0/21/3 | 0/23/1 | 0/20/4 | 0/23/1 |

**Frozen interpretation**

```
{
 "thresholds": {
  "substantial": "V >= 0.5 and range >= 0.5",
  "replication_111": "111 substantial and V >= 0.7"
 },
 "replication_111_passed": true,
 "V_111": 0.8434159800856974,
 "topic_alone_contributor": false,
 "question_alone_contributor": false,
 "entity_alone_contributor": false,
 "TQ_synergistic": false,
 "TE_synergistic": true,
 "QE_synergistic": false,
 "higher_order_interaction": false,
 "substantial_cells": [
  "101",
  "111"
 ]
}
```

**Single-component ablation (descriptive)**

| Remove | 111 → | ΔV | Stable-label agreement |
| --- | --- | ---: | ---: |
| remove_topic | 111→011 | -0.843 | 0.795 |
| remove_question | 111→101 | -0.025 | 1.000 |
| remove_entity | 111→110 | -0.460 | 0.850 |

**000 exact match to Phase-12 D:** True  
**Locked-family model calls:** 0

PHASE 13A WAS A BEHAVIORAL CONFOUND-DIAGNOSTIC ONLY. PHASE 11A REMAINS A BEHAVIOR-GATE HOLD. PHASE 12A IDENTIFIED THE SEMANTIC SHELL AS THE DOMINANT FAMILY-LEVEL CONFOUND. NO ACTIVATIONS WERE COLLECTED. NO PHYSIOLOGY PROBE WAS FIT OR SCORED. THE PHASE-5 PROBE WAS NOT SCORED. K, PAYOFFS, AND DECODER WERE NOT TUNED. LOCKED FAMILIES WERE NOT RUN. NO CAUSAL INTERVENTIONS WERE PERFORMED.
