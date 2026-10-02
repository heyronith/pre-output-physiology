# Phase 26B — Production Prompt Construction

**Status:** PASS (static construction + validation only)  
**Protocol:** `phase26b_prompt_construction_v1`  
**Parent Phase 26A (frozen):** `bfe441ac4901b4fb057c2c97df0a0a5d18d76ddb`  
**Target model (construction only; not executed):** `mistralai/Mistral-7B-Instruct-v0.2`

## Scope

Phase 26B authors the production prompt bank and runs **static validation only**.

Not performed:

- model inference / Mistral execution
- weight loading
- GPU use
- output generation
- behavioral screening
- activation/logit collection
- wording tuning from model responses
- any change to 24 families, 48 states, C1–C4, K rule, labels, or controls

Phase 26A source artifacts remain byte-identical to the approved commit.

## Counts

| Quantity | Value |
|---|---|
| Families | 24 |
| State instances | 48 (24 ADVERSE + 24 SAFE) |
| Conditions per instance | 13 |
| **Production prompts** | **624** |

| Condition | Count |
|---|---|
| K | 48 |
| N_C1 … N_C4 | 48 each |
| X_C1 … X_C4 | 48 each |
| E_C1 … E_C4 | 48 each |

## Template assignment

Prospective, state-independent:

- `x_index = ((family_index - 1 + class_index - 1) mod 3) + 1`
- `n_index = ((family_index - 1 + class_index - 1) mod 2) + 1`

ADVERSE and SAFE within a family share the same N/X template IDs. E uses the paired X template.

Per consequence class over 24 families:

- X templates: **8 / 8 / 8**
- N templates: **12 / 12**

Frozen table: `data/phase26/template_assignment_v1.json`

## Rendering rules (summary)

1. Single user instruction (no invented system message).
2. Order: operational role → world context → verified evidence (3 bullets) → additional operational signal → condition block (if any) → frozen question → fixed response requirement (`PRIVATE ASSESSMENT` then `REPORT: YES or NO`).
3. K: stem only (no C-class / severe / escape language).
4. N_Cx: class-specific `n_template` only.
5. X_Cx: one class `x_template` with `{trigger}`/`{avoid}` from gold (ADVERSE YES/NO; SAFE NO/YES); C3/C4 frozen beneficiary/dependency fields filled.
6. E_Cx: **byte-identical** X consequence paragraph + class-specific escape route + same question/response.

Designer metadata (`gold`, `ADVERSE`, `SAFE` as state labels, behavioral labels, etc.) is metadata-only and never model-visible, except where the English word `SAFE` already appears in frozen Phase-26A operational prose (source-authorized; not state-label leakage).

## Artifacts

| Path | Role |
|---|---|
| `configs/phase26b_render.yaml` | Render config |
| `src/pre_output_physiology/phase26b_render.py` | Deterministic renderer |
| `scripts/render_phase26b_prompts.py` | Render entrypoint |
| `scripts/validate_phase26b.py` | Static validator |
| `data/phase26/production_prompts_v1.jsonl` | 624 prompts + metadata |
| `data/phase26/template_assignment_v1.json` | Frozen assignment table |
| `tests/test_phase26b_prompts.py` | Unit tests |
| `artifacts/phase26b/validation_report.json` | Validation report |
| `artifacts/phase26b/manifest_sha256.json` | SHA256 manifest |

## Exit gate

- Renderer deterministic; re-render SHA256s identical
- All 624 prompts pass static validation
- Zero inference / model loading / GPU
- Phase 26A sources byte-identical to `bfe441ac4901b4fb057c2c97df0a0a5d18d76ddb`

**Phase 26C behavioral screening was not started.**
