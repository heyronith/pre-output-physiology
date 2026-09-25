# Research brief

**Working title:** Pre-Output Physiological Signatures of Deception in Language Models  
**Phase:** 1 — specification and infrastructure freeze only  
**Primary model (frozen):** `mistralai/Mistral-7B-Instruct-v0.2`

## Primary research question

Can internal model states reveal that a model is moving toward deceptive behavior **before** the deceptive content appears in its externally visible output, and can those states eventually be distinguished from ordinary knowledge, uncertainty, prompt context, and generic strategic reasoning?

## Longer-term causal question

Are any predictive internal states merely biomarkers of deception, or are they causally involved in producing deceptive behavior?

Causal tests are **pre-registered as a future standard** (hypothesis H5) and are **not** executed in Phase 1.

## Critical conceptual constraint

Do **not** describe any pre-first-token activation as “deceptive intent” merely because it predicts deception.

For a standard autoregressive transformer:

> same weights + same tokenized prompt → same pre-first-token activations.

Therefore pre-first-token activations may encode **context-conditioned deception propensity**, but cannot by themselves demonstrate a sample-specific hidden decision that occurred before generation.

The research must explicitly distinguish:

1. prompt/context encoding,
2. deception propensity,
3. knowledge of the true state,
4. uncertainty,
5. strategic reasoning,
6. an emerging deceptive trajectory,
7. actual deceptive execution,
8. causal mechanisms of deception.

These distinctions are expanded in `docs/research_protocol.md` and `docs/methodology_constraints.md`.

## Evidence posture

| Label | Meaning |
| --- | --- |
| **FACT FROM SOURCE** | Claim attributable to a cited paper, dataset card, or pinned upstream commit |
| **OUR RESEARCH DECISION** | Design choice frozen for this project |
| **OUR HYPOTHESIS** | Claim to be tested later; not established |

Phase 1 asserts no empirical results about deception physiology in this repository.

## Positive control vs novel target

- **Positive control:** RoleplayDeception — known behavior with published activation-detection precedent on Mistral-7B; used later to validate instrumentation. Not sufficient to establish spontaneous deceptive intent.
- **Novel target:** onset of deception — especially **pre-deceptive-output physiology** during generation, compared against information already present in the emitted text prefix.

## Immediate next gate

Independent review of branch `phase1/research-spec` before any Phase 2 authorization.
