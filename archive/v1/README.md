# Pre-Output Physiological Signatures of Deception in Language Models

**Status: PHASE 1 ONLY — research specification, reproducibility contract, and infrastructure freeze.**

No primary experimental data has been collected. No model weights have been downloaded for scientific results. No probes have been trained on primary data. No Modal GPU jobs have been launched.

## Scientific question

Can internal model states reveal that a language model is moving toward deceptive behavior **before** deceptive content appears in its externally visible output? If so, can those states be distinguished from ordinary knowledge, uncertainty, prompt context, and generic strategic reasoning?

A longer-term causal question (not tested in Phase 1): are any predictive internal states merely biomarkers of deception, or are they causally involved in producing deceptive behavior?

## Why this matters

Output monitoring alone is late: by the time a deceptive proposition is emitted, the harm channel is already open. Activation monitoring could, in principle, provide earlier warning. Prior work shows that linear probes can detect strategic deception in completed responses, but generalisation is fragile—especially for intent-like, text-ambiguous behaviours. This project targets the **onset** of deception along the generation trajectory, not only post-hoc classification of finished text.

## Prediction vs causation

Finding that activations **predict** later deception is not the same as showing they **cause** it. Phase 1 freezes correlational hypotheses (H1–H4) separately from a future causal standard (H5). Probe accuracy alone will never be treated as causal evidence.

**Critical constraint:** for a standard autoregressive transformer, the same weights and same tokenized prompt imply the same pre-first-token activations. Those activations may encode **context-conditioned deception propensity**, but they cannot by themselves demonstrate a sample-specific hidden decision that occurred before generation. This project will not label pre-first-token activations as “deceptive intent.”

## Starting model

Frozen primary model: **`mistralai/Mistral-7B-Instruct-v0.2`**.

Rationale: open-weight 7B-class model; manageable on a single cloud GPU; published activation-probing evidence for deception exists; ACL 2026 probe-generalisation work used this checkpoint for RoleplayDeception (published probe layer: 12). The same paper’s Mistral-family InsiderTrading experiment used **Mixtral-8x7B**, not Mistral-7B—do not conflate those results.

## Positive control

**RoleplayDeception** is frozen as the initial positive-control dataset because Mistral-7B has published activation-probe precedent on it. It validates instrumentation later; it is **not** evidence of spontaneous deceptive intent. The dataset includes explicit role/incentive structure and therefore has potential contextual confounds.

## Phase structure

| Phase | Scope |
| --- | --- |
| **1 (current)** | Research protocol, literature basis, configs, reproducibility contract, validation tests |
| **2 (not authorized)** | Positive-control pipeline validation on RoleplayDeception |
| **3+ (not authorized)** | Pre-output physiology scans, specificity controls, causal interventions |

## Local development

Requires Python 3.11 and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
uv run pytest
uv run python scripts/validate_phase1.py
uv run ruff check .
```

These commands must not download the 7B model or launch paid GPU jobs.

## Repository layout

- `docs/` — protocol, literature basis, methodology constraints, decision log, reproducibility
- `configs/` — frozen model and experiment YAML contracts
- `src/pre_output_physiology/` — minimal typed config/schema scaffolding
- `scripts/validate_phase1.py` — inexpensive Phase 1 integrity checks
- `tests/` — config/protocol contract tests (no GPU, no downloads)
- `data/`, `artifacts/` — placeholders; primary raw runs must never be overwritten

## License / contribution

This is an auditable research foundation. Do not merge Phase 2 work until an independent review of this branch authorizes data collection.
