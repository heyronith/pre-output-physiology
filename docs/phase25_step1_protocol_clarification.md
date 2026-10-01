# Phase 25 Step 1 — Protocol Clarification

## Reason
During implementation, one wording inconsistency was identified in the prospectively frozen Step-1 protocol before any model inference or outcome analysis.

The protocol simultaneously requires the response-format instruction to contain the allowed categorical labels (`DECISION: YES or NO`, with `INSUFFICIENT_INFORMATION` additionally allowed for D) and states under balancing requirements that “answer labels … must not appear in prompt text.” These two requirements cannot both be satisfied literally.

## Prospective clarification
The balancing requirement is interpreted as follows:

- categorical answer labels may appear **only in the standardized response-format instruction**;
- answer labels must not appear in scenario facts, rule text, objective text, question text, scenario titles, organization names, or other semantic prompt content;
- scenario/base IDs must never appear in rendered prompt text;
- conflicting-objective text may describe a business preference for more or fewer affirmative outcomes but may not explicitly instruct `YES` or `NO`.

This clarification changes no scientific hypothesis, scenario content, model behavior, label, analysis method, or outcome. It was recorded before Step-1 dataset freeze and before any Mistral/Gemma inference, activation capture, or response-level outcome observation.
