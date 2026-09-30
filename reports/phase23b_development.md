# Phase 23B — DEVELOPMENT open-grader comparison (post-audit fixes)

**Status:** `phase23b_postaudit_fixes_complete_awaiting_authorization`

**Proposed Stage-23C winner (pending authorization):** `gemma4_31b_it`

**Stage 23C:** `scientifically_eligible: true` · `authorized: false`

STOP. Stage 23C is **not** authorized.

## Frozen numeric DEVELOPMENT results

| Candidate | invalid | HD κ | bal-acc | H sens | D sens | numeric gate |
|---|---:|---:|---:|---:|---:|---|
| gemma4_31b_it | 0 | 0.983 | 0.956 | 0.934 | 0.978 | **PASS** |
| qwen35_27b | 0 | 0.978 | 0.914 | 0.873 | 0.955 | **FAIL** |

Gemma remains the sole frozen numeric-gate passer and the proposed Stage-23C winner.

## Primary population reanalysis — label only

Rule: ≥2 honest and ≥2 deceptive labels per DEVELOPMENT prompt (20 responses).
**No onset requirement.** Invalid open outputs → exclude.

This is the **canonical primary** population-agreement analysis.

### `gemma4_31b_it`

- GPT qualifying: 28
- Open qualifying: 40
- Intersection: 23
- Union: 45
- Jaccard: 23/45 ≈ 0.5111
- Changed prompts: 22
- GPT-only: 5
- Open-only: 17
- Qualification-status agreement over all 260 DEVELOPMENT prompts: 238/260 ≈ 0.9154

### `qwen35_27b`

- GPT qualifying: 28
- Open qualifying: 41
- Intersection: 20
- Union: 49
- Jaccard: 20/49 ≈ 0.4082
- Changed prompts: 29
- GPT-only: 8
- Open-only: 21
- Qualification-status agreement over all 260 DEVELOPMENT prompts: 231/260 ≈ 0.8885

Do not confuse all-prompt status agreement with Jaccard over the much smaller qualifying set.

## Secondary sensitivity — GPT onset gated

Open-grader labels combined with frozen GPT-4o onset annotations.
**Not** a fully open-grader-defined population. Retained only as sensitivity analysis.

### `gemma4_31b_it` (hybrid)

- GPT qualifying: 27
- Open+GPT-onset qualifying: 26
- Jaccard: 0.65625
- Changed prompts: 11

### `qwen35_27b` (hybrid)

- GPT qualifying: 27
- Open+GPT-onset qualifying: 26
- Jaccard ≈ 0.6060606
- Changed prompts: 13

## Audit corrections

1. **Population semantics corrected.** Label-only is primary; onset-gated hybrid is secondary sensitivity only. No post-hoc population rejection threshold was introduced.
2. **Disagreement audit properly blinded.** Both A/B judgments now expose only `score` / `class` (explanation omitted on both sides). Grader identities live solely in `ab_mapping.json`. Sample continuation IDs and deterministic A/B assignment preserved.
3. **Stage-23C authorization logic corrected.** A DEVELOPMENT winner implies `stage3_scientifically_eligible = true` only. `stage3_locked_validation_authorized` remains `false` until an explicit future authorization commit.

## Interpretation

Response-level agreement with GPT-4o is excellent for Gemma (κ ≈ 0.983, bal-acc ≈ 0.956), but prompt-level qualification membership is more sensitive to relatively small label changes near the ≥2 H / ≥2 D threshold (label-only Jaccard ≈ 0.511; 22 changed prompts).

This observation is **descriptive** and does **not** retroactively change the frozen winner-selection criterion. No population-agreement rejection threshold was frozen before evaluation, and none may be introduced post hoc.

## Authorization state

All of the following remain **false**:

- `modal_gpu_open_grader_inference_authorized`
- `stage2_development_authorized`
- `stage3_locked_validation_authorized`
- `stage4_onset_validation_authorized`
- `mistral_roleplay_generation_authorized`
- `openai_grading_api_authorized`
- `openai_onset_api_authorized`
- `activation_extraction_authorized`
- `probe_fitting_authorized`
- `physiology_authorized`
- `k_gt_20_generation_authorized`
- `prompt_changes_authorized`
- `population_threshold_changes_authorized`

Stage 23C:

- `scientifically_eligible: true`
- `authorized: false`

## Future note (Stage 23D — do not implement now)

The current Stage-23D onset analyzer contains a methodological issue if it estimates token error using `abs(char_offset_difference) / 4`. That is not an actual token-index error. Before Stage 23D is eventually authorized, token error should be computed using the exact frozen tokenizer and actual token positions.

## Guarantee

PHASE 23 VALIDATES OPEN-WEIGHT DECEPTION GRADERS AGAINST FROZEN GPT-4O-2024-08-06 LABELS ON THE EXISTING K=20 APOLLO ROLEPLAYDECEPTION CORPUS (7420 RESPONSES). NO NEW MISTRAL GENERATION, OPENAI API CALLS, ACTIVATIONS, OR PHYSIOLOGY WERE PERFORMED. CANDIDATES, SPLITS, INFERENCE SETTINGS, AND THRESHOLDS WERE FROZEN BEFORE EVALUATION. K>20 GENERATION REMAINS UNAUTHORIZED UNTIL PHASE 23D PASSES.

NO STAGE 23C INFERENCE WAS RUN. NO LOCKED-VALIDATION RESPONSES WERE GRADED. NO MODEL OR API CALLS WERE MADE. NO K>20 GENERATION, ACTIVATION EXTRACTION, PROBE FITTING, OR PHYSIOLOGY WAS PERFORMED. GEMMA REMAINS THE PROPOSED STAGE-23C WINNER, BUT STAGE 23C REMAINS UNAUTHORIZED.
