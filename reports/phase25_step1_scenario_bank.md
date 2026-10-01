# Phase 25 Step 1 — Scenario Bank Report

## Status
**PASS — candidate bank frozen pending independent pre-Step-2 audit.**

No Mistral inference, Gemma grading, GPU use, activation/logit capture, or Phase-24 locked-test scientific access occurred.

## Provenance
- Phase-24 starting point: `2af0c27fe034b3419dd9de8ccaafbb2780d6f899`
- Step-1 protocol base freeze: `813776c6dbdeb96aa6f6cb87db78edd3c76b735a`
- Pre-data protocol clarification: `654750f713daf686a7f711de8c836de478bb5e93`
- Branch: `phase25/scenario-bank`

The clarification resolves one internal wording conflict: categorical response labels are allowed only in the standardized response-format instruction, not in semantic scenario content. It was committed before data freeze and before any model inference.

## Frozen bank
- 180 base scenarios.
- 6 domains × 30 scenarios/domain.
- Exactly 15 YES and 15 NO objective gold decisions/domain.
- 4 matched forms/base: K (neutral knowledge), A (aligned objective), X (conflicting objective), D (one decisive fact removed).
- 720 prompt records total.
- K/A/X share identical rule, evidence, question, and gold answer; A/X differ only in objective text.
- X never explicitly instructs dishonesty and favors the decision opposite the objective gold answer.
- D removes exactly one decision-critical evidence item and is labeled `INSUFFICIENT_INFORMATION`.

## Response format
K/A/X require:

`DECISION: YES` or `DECISION: NO`

`EXPLANATION: <one or two sentences>`

D additionally permits `INSUFFICIENT_INFORMATION`.

The structured decision is the later primary behavioral measurement; explanation text is a consistency check.

## Validation results
All frozen gates passed:
- 180 unique base IDs / semantic signatures.
- 720 unique rendered prompts; no exact duplicate prompts.
- 30 scenarios/domain and 15/15 YES/NO balance within every domain.
- No scenario/prompt ID leakage.
- Categorical answer labels occur only in the standardized response-format section.
- Prohibited explicit-deception vocabulary absent.
- Every fictional organization occurs with both gold labels (3 YES, 3 NO), so entity identity cannot determine the label.
- Median A/X word-length fractional difference: **0.0104**; maximum: **0.0239**.
- Cross-template normalized surface-similarity heuristic: maximum **0.7615**; no cross-template pair ≥0.80.
- Targeted tests: **3 passed**.

## Scientific self-audit
All 30 template families were reviewed at both YES and NO polarities; all 180 instances were checked algorithmically. Pre-freeze repairs removed duplicate semantic signatures, entity-label correlation, semantic answer-label leakage, a serialization bug, and one grammar issue.

This is **not an independent audit**. Six variants within each template intentionally share a rule family. Future splitting/analysis must treat template family as a grouping variable or explicitly test template-held-out generalization so near-template variants do not create optimistic generalization estimates.

## Scientific interpretation
Step 1 creates no honest/deceptive rollout labels. The future critical contrast is repeated stochastic responses to the **identical X prompt**: cases that remain rule-consistent versus cases that produce the incentive-compatible incorrect decision, after independent knowledge verification.

## Next gate
Do not start Step 2 until an independent reviewer/code-editor audits construct validity, cue leakage, scenario realism, and the committed generator/validator outputs.
