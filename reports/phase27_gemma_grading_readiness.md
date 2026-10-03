# Phase 27 — Gemma grading pipeline readiness (pre-execution)

**Grading status:** `phase27_gemma_grading_preflight_ready`  
**`grader_execution_authorized`:** `false`

## Frozen sources

| Artifact | SHA256 |
|---|---|
| Raw generations | `61fe3ce1a5568e58ce1393bb4d0260ba7eec1b58dca9b21b56edba3723dcd226` |
| Inference manifest | `331444a42699f613411f545d94600de38dc5de53e13c6d322fee726869aa40d5` |
| Grader prompt | `a82284ae9113bfac4f9726ff85e0229d7a9e85c56b06b23c5aa0a35bb4c153df` |
| Grading spec | `0df0f93544e65c3fed3b7a45ff6af5fd24488160ca9da818f6f4fbe4fe44dc34` |

## Grader

- Model: `google/gemma-4-31B-it` @ `842da3794eaa0b77d5f08bae87a17459d91ff475`
- Phase 23 **response-level** implementation (not Phase 23D onset)
- T=0.0, do_sample=false, top_p=1.0, max_new_tokens=256, max_model_len=8192, bf16, A100-80GB
- Inputs: Scenario / Question / Response only

## Execution environment (Phase 23C-matched)

- Image packages: torch 2.6.0 / transformers 5.17.0 / accelerate 1.15.0 / huggingface_hub 1.5.0 / sentencepiece 0.2.0 / protobuf 5.29.4 / numpy 1.26.4 / pyyaml 6.0.2 / safetensors 0.8.0
- Cache Volume: `preoutput-open-grader-cache`
- Engine: `transformers_generate_temp0_batch4` (batch size 4)
- Tokenizer: `trust_remote_code=True`, `padding_side=left`

## Pipeline freeze

- 288 grading jobs; exact raw job-set equality
- Frozen Gemma chat-template model-input text + input-id SHAs per job
- Modal Volume: `phase27-extreme-roleplay-grading-results`
- Checkpoint interval: 24
- Analysis helpers for MIXED_ELIGIBLE / PASS–HOLD prepared but **not run**

## Confirmation

**PHASE 27 GEMMA GRADING REMAINS UNAUTHORIZED PENDING INDEPENDENT AUDIT.**  
No Gemma weights loaded for grading execution. No behavioral labels produced.
