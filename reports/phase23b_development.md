# Phase 23B — DEVELOPMENT open-grader comparison (FROZEN)

**Status:** `phase23b_winner_selected_pending_audit`

**Proposed Stage-23C winner (pending audit):** `gemma4_31b_it`

Reason: sole numeric gate passer.

STOP for audit. Stage 23C is **not** authorized.

## Provenance

| Field | Value |
|---|---|
| Authorization commit | `00131f41670c52a202c3ac70ad940e40162e4b38` |
| Phase-23A freeze | `b08c4b2ca2e4ade430bce0e96e768d98260f7e8d` |
| Split SHA | `6504b40b43153a7bd359c344efb6eee01afca887fe4978960d9d2e0d6fec3869` |
| Design SHA | `40fecca062757f489b1b7e1661e0728c9c52f205b9013e6dfc9b1b34a7c45cff` |
| Gemma revision | `842da3794eaa0b77d5f08bae87a17459d91ff475` |
| Qwen revision | `fc05daec18b0a78c049392ed2e771dde82bdf654` |
| Gemma remainder run | `phase23b_dev_remainder_gemma4_31b_it_20260930T031505Z_ff2d709d` |
| Qwen remainder run | `phase23b_dev_remainder_qwen35_27b_20260930T050455Z_ff2d709d` |
| Inference | Transformers T=0, do_sample=false, max_new_tokens=256, batch=4, A100-80GB |

## Results (merged 5200 = 200 pilot reused + 5000 new)

| Candidate | invalid | HD κ | bal-acc | H sens | D sens | macro-F1 | pop Jaccard | Δ qualify | new $ | gates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **gemma4_31b_it** | 0.00 | 0.983 | 0.956 | 0.934 | 0.978 | 0.833 | 0.656 | 11 | 6.46 | **PASS** |
| qwen35_27b | 0.00 | 0.978 | 0.914 | 0.873 | 0.955 | 0.826 | 0.606 | 13 | 11.03 | FAIL |

Qwen fails: balanced_accuracy 0.914 < 0.95; honest_sensitivity 0.873 < 0.90.

Population membership changes are descriptive (no post-hoc numeric threshold). Gemma changes 11 DEVELOPMENT prompts vs GPT-4o; Qwen 13.

## Decision

- Winner: **gemma4_31b_it**
- Stage 23C scientifically eligible (numeric): **yes** (pending audit)
- Stage 23C authorized: **false**

## Authorization state (after freeze)

All false: Modal GPU, Stage 2/3/4, Mistral, OpenAI grade/onset, activations, probes, physiology, K>20.

## Evidence

See `artifacts/phase23b_development/freeze.json`, merged judgments under `artifacts/phase23b_development/merged/`, disagreement audit under `artifacts/phase23b_development/disagreement_audit/`.

## Guarantee

PHASE 23 VALIDATES OPEN-WEIGHT DECEPTION GRADERS AGAINST FROZEN GPT-4O-2024-08-06 LABELS ON THE EXISTING K=20 APOLLO ROLEPLAYDECEPTION CORPUS (7420 RESPONSES). NO NEW MISTRAL GENERATION, OPENAI API CALLS, ACTIVATIONS, OR PHYSIOLOGY WERE PERFORMED. CANDIDATES, SPLITS, INFERENCE SETTINGS, AND THRESHOLDS WERE FROZEN BEFORE EVALUATION. K>20 GENERATION REMAINS UNAUTHORIZED UNTIL PHASE 23D PASSES.
