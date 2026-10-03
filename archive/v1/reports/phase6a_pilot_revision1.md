# Phase 6A factorial behavior pilot (revision 1)

**Run ID:** `phase6a_pilot_r1_20260926T195943Z_5864a14b`  
**Git SHA:** `5864a14b7972f12b53464c5b9fb9d06738f9449b`  
**Pilot prompt SHA256:** `486c38d96a270218c3f44754f459366521086bdfd2fc81fe7bd9eade10267035`  
**Response prefix token ID:** `12107`  
**GPU wall / cost:** `29.3s` / `$0.0159`  

## Scope

- 96 pilot prompts (24 base × F00/F10/F01/F11)
- Greedy BF16 Mistral-7B-Instruct-v0.2 (pinned revision)
- Deterministic one-line `Response <STATE>` parser; valid = communication target
- No activations, no frozen-probe scoring, no final-dataset generation

**All operational gates passed:** `False`  
**All-four paired-valid:** `18` / 24 (gate ≥ 20)  
**Valid runs all first-token 12107:** `True`  

## Per-condition

| Condition | Expected | N | Schema | First-token | Valid | Gate |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| F00 | record | 24 | 24 | 24 | 23 | True |
| F10 | record | 24 | 22 | 24 | 21 | False |
| F01 | alternate | 24 | 21 | 24 | 19 | False |
| F11 | alternate | 24 | 24 | 24 | 23 | True |

## Failure reasons and first-token distributions

### F00
- `state_not_record_or_alternate`: 1

First-token ID counts:
- `12107`: 24

### F10
- `not_exact_one_nonempty_line`: 2
- `state_not_record_or_alternate`: 1

First-token ID counts:
- `12107`: 24

### F01
- `not_exact_one_nonempty_line`: 3
- `state_not_record_or_alternate`: 2

First-token ID counts:
- `12107`: 24

### F11
- `state_not_record_or_alternate`: 1

First-token ID counts:
- `12107`: 24

## Guarantees

NO PHASE 6 ACTIVATIONS WERE COLLECTED.  
THE FROZEN PHASE-5 PROBE WAS NOT SCORED, RETRAINED, RECALIBRATED, OR MODIFIED.  
NO FINAL PHASE-6 SCENARIOS WERE RUN THROUGH THE MODEL.  
NO CAUSAL INTERVENTIONS WERE PERFORMED.

