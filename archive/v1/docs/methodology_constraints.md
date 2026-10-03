# Methodology constraints

Phase 1 freezes integrity rules before primary results exist.

## Conceptual constraints

1. **No “intent” inflation.** Pre-first-token activations are at most **context-conditioned propensity** under fixed weights + fixed prompt. Do not call them deceptive intent solely because a probe is predictive.
2. **Distinguish constructs.** Prompt encoding, propensity, true-state knowledge, uncertainty, strategic reasoning, emerging trajectory, deceptive execution, and causal mechanism must remain separable in labels, analyses, and claims.
3. **Positive control ≠ phenomenon of interest.** RoleplayDeception validates instrumentation; explicit role/incentive structure is a confound risk for spontaneous-intent claims.
4. **Onset over autopsy.** Prefer analyses that locate predictive information **before** the first deceptive span, with prefix-text baselines at the same time index.
5. **Correlation ≠ causation.** H5 requires interventions with matched sham/random controls and preservation checks (knowledge, competence, fluency, honest answering, non-deceptive strategy). Refusal-rate bumps are not selective mechanism evidence.

## Instrumentation constraints

**OUR RESEARCH DECISION:** Prefer standard Hugging Face `output_hidden_states` / forward hooks for primary instrumentation.

**OUR RESEARCH DECISION:** Do **not** make the entire project dependent on TransformerLens. TransformerLens may be added later only with a concrete, documented need (e.g., a specific intervention API unavailable via HF hooks). Phase 1 does not include TransformerLens as a hard dependency.

## Compute policy

| Venue | Allowed uses |
| --- | --- |
| **Local machine** | Documentation; configs; later dataset preprocessing; probe/statistical analysis; plotting; lightweight tests; orchestration |
| **Modal GPU (later)** | Canonical full-precision Mistral inference; primary activation extraction; experimental trajectories; activation patching/steering; GPU-dependent validation |

### Research-integrity rule

Do **not** use quantized local-model activations as evidence for the primary scientific result.

Quantized models may be used **only** for software debugging and must be labelled as such in configs, run manifests, and any plots/tables.

**No Modal jobs in Phase 1.**

## Configuration constraints

Primary model config must include at least:

- `model_id`
- `revision`
- `dtype`
- `quantization`
- `trust_remote_code`
- `device_policy`
- `seed`

Frozen values:

- `model_id: mistralai/Mistral-7B-Instruct-v0.2`
- `quantization: null`
- `revision: TO_BE_PINNED_BEFORE_PHASE2` until an exact HF commit is recorded (never silently treat `main` as the scientific revision)

## Data and statistics constraints

- Scenario/group-level splits; no leakage across related variants.
- Report AUROC primarily; always log prevalence, sample counts, and uncertainty.
- Paired comparisons vs surface baselines with uncertainty on the **difference**.
- Log **all** tested layers; no cherry-picked layer reporting.
- Version label-generation and any LLM-judge prompts exactly.

## Artifact constraints

- Primary raw experimental artifacts must never be overwritten.
- Derived artifacts retain provenance to the raw run.
- Every run records the reproducibility field set in `docs/reproducibility.md`.

## Phase 1 stop conditions

Do not:

- download/run the full Mistral experiment for primary results;
- generate deception samples;
- collect primary activations;
- train primary probes;
- launch Modal;
- perform causal interventions;
- claim scientific results from this repository.
