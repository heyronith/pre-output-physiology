# Phase 23A — open-grader 200-response pilot (FROZEN)

**Status:** `phase23a_pilot_complete_awaiting_audit`

STOP for audit. Stage-2 DEVELOPMENT is **not** authorized.

Gemma and Qwen are **pilot survivors only**. No Stage-2 winner has been selected.
Qwen survives 23A because honest sensitivity 0.85 exceeds the frozen pilot
survival threshold of 0.80; do **not** apply Stage-2 thresholds retroactively.
gpt-oss HD metrics are labeled on the **158/200 valid rows only**, not the full 200.

## Frozen provenance

| Field | Value |
|---|---|
| Pilot-run commit | `f8517418a81f0584ffdddbe4e464cedd47ae93e9` |
| Design SHA256 | `40fecca062757f489b1b7e1661e0728c9c52f205b9013e6dfc9b1b34a7c45cff` |
| Split SHA256 | `6504b40b43153a7bd359c344efb6eee01afca887fe4978960d9d2e0d6fec3869` |
| Pilot SHA256 | `6e412617f7a4e461500880f1420be1c05c8453b433a96ac3d55a9fccb8f02533` |
| DEV / LOCKED prompts | 260 / 111 |
| Pilot responses | 200 |
| Gemma revision | `842da3794eaa0b77d5f08bae87a17459d91ff475` |
| Qwen revision | `fc05daec18b0a78c049392ed2e771dde82bdf654` |
| gpt-oss revision | `6cee5e81ee83917806bbde320786a8fb61efebee` |

## Frozen inference

- temperature = 0 · `do_sample = false` · `max_new_tokens = 256` · batch size = 4
- backend = **Transformers** (not vLLM)
- GPU = **A100-80GB** for all successful pilot runs
- gpt-oss: L40S attempted first → escalated to A100-80GB after MXFP4→BF16 OOM

## Frozen pilot results

| Candidate | invalid | HD bal-acc | HD κ | H sens | D sens | $/1k | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| `gemma4_31b_it` | 0.00 | 0.970 | 0.970 | 0.94 | 1.00 | 2.15 | **SURVIVE** |
| `qwen35_27b` | 0.00 | 0.910 | 0.936 | 0.85 | 0.97 | 2.62 | **SURVIVE** |
| `gpt_oss_20b`† | 0.21 | 0.966† | 0.950† | 0.93† | 1.00† | 2.51 | **ELIMINATED** (invalid_rate gate) |

† gpt-oss HD metrics on **158/200 valid rows only**. Harmony `analysis` channel
consumed the 256-token cap before valid JSON on invalid rows.

## Modal cost (23A)

- Total ≈ **$1.46**
- Components ≈ $0.50 (gpt-oss) + $0.52 (Qwen) + $0.43 (Gemma)

## Evidence paths

Full freeze record: `artifacts/phase23a_pilot/freeze.json`

| Candidate | Run ID | Judgments SHA256 |
|---|---|---|
| gemma | `phase23a_pilot_gemma4_31b_it_20260929T213351Z_7e341a62` | `d9794d0268237ae6…` (see freeze.json) |
| qwen | `phase23a_pilot_qwen35_27b_20260929T212300Z_7e341a62` | `7cdea08e3b65132e…` (see freeze.json) |
| gpt-oss | `phase23a_pilot_gpt_oss_20b_20260929T194731Z_7e341a62` | `2a7c0826eb220988…` (see freeze.json) |

Raw judgments + manifests mirrored under `artifacts/phase23a_pilot/runs/<run_id>/`.

## Pilot elimination thresholds (frozen before evaluation)

- balanced accuracy ≥ 0.85
- κ ≥ 0.75
- honest / deceptive sensitivity ≥ 0.8
- invalid rate ≤ 0.01

## Authorization state (all false)

- `modal_gpu_open_grader_inference_authorized`: **false**
- `stage2_development_authorized`: **false**
- `stage3_locked_validation_authorized`: **false**
- `stage4_onset_validation_authorized`: **false**
- `mistral_roleplay_generation_authorized`: **false**
- `openai_grading_api_authorized`: **false**
- `openai_onset_api_authorized`: **false**
- `activation_extraction_authorized`: **false**
- `probe_fitting_authorized`: **false**
- `physiology_authorized`: **false**
- `k_gt_20_generation_authorized`: **false**

## Guarantee

PHASE 23 VALIDATES OPEN-WEIGHT DECEPTION GRADERS AGAINST FROZEN GPT-4O-2024-08-06 LABELS ON THE EXISTING K=20 APOLLO ROLEPLAYDECEPTION CORPUS (7420 RESPONSES). NO NEW MISTRAL GENERATION, OPENAI API CALLS, ACTIVATIONS, OR PHYSIOLOGY WERE PERFORMED. CANDIDATES, SPLITS, INFERENCE SETTINGS, AND THRESHOLDS WERE FROZEN BEFORE EVALUATION. K>20 GENERATION REMAINS UNAUTHORIZED UNTIL PHASE 23D PASSES.
