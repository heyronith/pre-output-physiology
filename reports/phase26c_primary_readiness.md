# Phase 26C primary — readiness report (preflight only)

**Branch:** `phase26/behavioral-feasibility-primary-v2`  
**Base commit:** `ca0aaf6a8bba28a6a78d690189afa9ca3d824b2e` (Phase 26B approved)  
**Protocol:** `phase26c_primary_behavior_v1`  
**Status:** `phase26c_primary_preflight_ready` — **no inference / GPU / model load executed**

## Scope (authoritative)

| Item | Value |
|---|---|
| Scenario families | 24 |
| Active state | ADVERSE only |
| SAFE in execution | **0** |
| Unique prompts | **312** |
| K / N / X / E prompts | 24 / 96 / 96 / 96 |
| Generations K / N / X / E | 216 / 384 / 1536 / 768 |
| **Total planned generations** | **2904** |

SAFE prompts remain untouched in the Phase 26B bank for later robustness testing.

## Model contract

- Model: `mistralai/Mistral-7B-Instruct-v0.2`
- Revision / tokenizer: `63a8b081895390a26e140280378bc85ec8bce07a`
- dtype `bfloat16`, attention `sdpa`, batch 1, no quantization
- Stochastic: T=1.0 full-vocab multinomial; no top_p/top_k
- Seeds: `SHA256("phase26c_primary_behavior_v1\|{prompt_sha256}\|{condition}\|{rollout_index}")` → uint32

## Template distribution (24 families)

- X templates per class: **8 / 8 / 8**
- N templates per class: **12 / 12**
- E inherits paired X template

## Immutability

- Phase 26A sources byte-identical to `bfe441ac4901b4fb057c2c97df0a0a5d18d76ddb`
- Phase 26B sources byte-identical to `ca0aaf6a8bba28a6a78d690189afa9ca3d824b2e`
- Selected prompt texts/hashes copied unchanged from `production_prompts_v1.jsonl`

## Physiology / GPU

- Activations / hidden states / logits / probes / SAE / graders: **disabled**
- `modal_gpu_mistral_screening_authorized: false`
- Runner refuses launch until status flips to `phase26c_primary_generation_authorized`

## Tests

```bash
PYTHONPATH=src python3 scripts/preflight_phase26c_primary.py --repo-root .
PYTHONPATH=src python3 -m pytest -q tests/test_phase26c_primary.py
```

Preflight: **PASS**  
Pytest: **15 passed**

## Artifacts

- `configs/phase26c_primary_behavioral_feasibility.yaml`
- `data/phase26/phase26c_primary_prompt_selection.jsonl` (312)
- `artifacts/phase26c_primary/inference_manifest.jsonl` (2904)
- `artifacts/phase26c_primary/seed_manifest.jsonl` (2880)
- `artifacts/phase26c_primary/preflight_validation.json`
- `artifacts/phase26c_primary/config_hash_manifest.json`
- `src/pre_output_physiology/phase26c_primary.py`
- `scripts/preflight_phase26c_primary.py`
- `scripts/summarize_phase26c_primary.py`
- `modal/phase26c_primary_behavioral_feasibility.py` (blocked until authorization)
- `docs/phase26c_primary_behavioral_feasibility.md`
- `tests/test_phase26c_primary.py`

## STOP

Do **not** begin the 2,904-generation experiment from this commit. Await independent audit and explicit GPU authorization.
