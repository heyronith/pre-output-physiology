# Decision log

Append-only record of Phase 1 research decisions. Do not silently rewrite history; add superseding entries instead.

Evidence labels: **FACT FROM SOURCE**, **OUR RESEARCH DECISION**, **OUR HYPOTHESIS**.

---

## D001 — Choose Mistral-7B-Instruct-v0.2 rather than 3B/70B

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Freeze `mistralai/Mistral-7B-Instruct-v0.2` as the initial primary model.
- **Reason:** Published deception-probe precedent on RoleplayDeception for this checkpoint family, plus manageable single-GPU compute. A 3B model would reduce continuity with that precedent; a 70B model would exceed the intended Phase 2 instrumentation budget.
- **Limitation recorded:** The ACL 2026 paper’s Mistral-family **InsiderTrading** experiment used **Mixtral-8x7B**, not Mistral-7B. Do not conflate those results.
- **Date:** 2026-09-24

## D002 — RoleplayDeception is a positive control, not proof of spontaneous hidden intent

- **Type:** **OUR RESEARCH DECISION** (motivated by **FACT FROM SOURCE** datasets)
- **Decision:** Use RoleplayDeception only to validate instrumentation and measurement pipelines later.
- **Reason:** Known behavior + published activation-detection precedent on Mistral-7B.
- **Limitation:** Explicit role/incentive structure ⇒ potential contextual confounds; insufficient for spontaneous deceptive-intent claims.
- **Date:** 2026-09-24

## D003 — Separate prompt-boundary “propensity” from emerging trajectory / intent claims

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Label prompt-boundary physiology as context-conditioned **propensity**. Reserve stronger trajectory/onset claims for pre-deceptive-output analyses that beat prefix-text baselines. Never auto-label pre-first-token states as intent.
- **Reason:** Autoregressive determinism: same weights + same tokenized prompt ⇒ same pre-first-token activations.
- **Date:** 2026-09-24

## D004 — Local compute for analysis; Modal only for canonical model forward passes

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Documentation, configs, preprocessing, probes/statistics, plotting, tests, and orchestration run locally. Canonical full-precision inference, primary activation extraction, trajectory generation, and GPU interventions run on Modal (or equivalent) later.
- **Reason:** Preserve scientific integrity of activations while keeping iteration cheap.
- **Phase 1:** No Modal jobs.
- **Date:** 2026-09-24

## D005 — Require strong surface-information baselines before claiming white-box advantage

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Pre-register majority, prompt-text, emitted-prefix-text, logit-stat, random-direction, and shuffled-label baselines (self-report only if appropriate). Paired evaluation with uncertainty on differences.
- **Reason:** High AUROC alone does not show that internals add information beyond surface cues.
- **Date:** 2026-09-24

## D006 — Causal claims require intervention; probe accuracy alone is insufficient

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Keep H5 separate. Require patching/steering/replacement with sham and random-direction controls, plus preservation of knowledge, competence, fluency, honest answering, and non-deceptive strategy. Refusal inflation ≠ selective deception mechanism.
- **Date:** 2026-09-24

---

## Additional Phase 1 decisions

### D007 — Do not vendor LASR-probe-gen code in Phase 1

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Cite and pin upstream SHA only (`f4c6ad69b10a5436a2e819c69009431802a0f5f7`); do not copy upstream code into this repository during Phase 1.
- **Date:** 2026-09-24

### D008 — Prefer HF hidden states over TransformerLens as default instrumentation

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Standard Hugging Face `output_hidden_states` / forward hooks are the default. TransformerLens is optional and only if a concrete need appears; it is not a Phase 1 dependency.
- **Date:** 2026-09-24

### D009 — Model revision left explicitly unpinned rather than silently using `main`

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Set `revision: TO_BE_PINNED_BEFORE_PHASE2` in the primary model config. Do not invent a revision hash or treat `main` as the final scientific revision.
- **Date:** 2026-09-24

### D010 — Minor structural additions beyond the requested skeleton

- **Type:** **OUR RESEARCH DECISION**
- **Decision / deviation notes:**
  - Added `docs/research_brief.md` as a one-page conceptual entry point alongside the full protocol.
  - Placed the installable package under `src/pre_output_physiology/` (src layout) for clean imports with hatchling.
  - Added optional `[project.optional-dependencies] modal` without executing Modal.
  - Added `dependency-groups.dev` for uv + pytest/ruff.
  - Experiment YAMLs use `status: scaffolded` to make non-authorization explicit.
- **Date:** 2026-09-24
