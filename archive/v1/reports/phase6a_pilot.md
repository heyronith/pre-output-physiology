# Phase 6A factorial behavior pilot (revision 2)

**Run ID:** `phase6a_pilot_r2_20260926T200124Z_028b033c`  
**Git SHA:** `028b033c6f311cff546c08e9c93bb2e66c55df16`  
**Pilot prompt SHA256:** `4777311e8452b5e85e1518c2dee21d13012bada19074a7f6a1ea1ee321623325`  
**Response prefix token ID:** `12107`  
**GPU wall / cost:** `32.9s` / `$0.0178`  

## Scope

- 96 pilot prompts (24 base × F00/F10/F01/F11)
- Greedy BF16 Mistral-7B-Instruct-v0.2 (pinned revision)
- Deterministic one-line `Response <STATE>` parser; valid = communication target
- No activations, no frozen-probe scoring, no final-dataset generation

**All operational gates passed:** `False`  
**All-four paired-valid:** `17` / 24 (gate ≥ 20)  
**Valid runs all first-token 12107:** `True`  

## Per-condition

| Condition | Expected | N | Schema | First-token | Valid | Gate |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| F00 | record | 24 | 24 | 24 | 24 | True |
| F10 | record | 24 | 24 | 24 | 24 | True |
| F01 | alternate | 24 | 18 | 24 | 17 | False |
| F11 | alternate | 24 | 24 | 24 | 24 | True |

## Failure reasons and first-token distributions

### F00
- (no failures)

First-token ID counts:
- `12107`: 24

### F10
- (no failures)

First-token ID counts:
- `12107`: 24

### F01
- `not_exact_one_nonempty_line`: 6
- `state_not_record_or_alternate`: 1

First-token ID counts:
- `12107`: 24

### F11
- (no failures)

First-token ID counts:
- `12107`: 24

## Outcome

**HOLD** (D082). This is the second and final permitted pilot. F01 failed the ≥22/24 gate
(17/24) and all-four paired-valid failed the ≥20/24 gate (17/24). Every F01 failure produced the
correct first line `Response <alternate>` followed by an appended disclaimer note; F11 (same
communication target plus conflicting operational target) produced no such notes. No further
revision is permitted.

## Guarantees

NO PHASE 6 ACTIVATIONS WERE COLLECTED.  
THE FROZEN PHASE-5 PROBE WAS NOT SCORED, RETRAINED, RECALIBRATED, OR MODIFIED.  
NO FINAL PHASE-6 SCENARIOS WERE RUN THROUGH THE MODEL.  
NO CAUSAL INTERVENTIONS WERE PERFORMED.

