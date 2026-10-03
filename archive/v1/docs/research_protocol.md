# Research protocol (Phase 1 freeze)

**Status:** Frozen for audit before any primary data collection.  
**Phase 1 scope:** specification only. No activations collected. No probes trained on primary data. No Modal jobs.

Evidence labels used below: **FACT FROM SOURCE**, **OUR RESEARCH DECISION**, **OUR HYPOTHESIS**.

---

## 1. Conceptual distinctions (mandatory)

The following must never be collapsed into a single “deception” label without justification:

| # | Construct | Working meaning in this project |
| --- | --- | --- |
| 1 | Prompt/context encoding | Information present because of the tokenized input, role text, incentives, or dialogue state |
| 2 | Deception propensity | Context-conditioned tendency toward later deceptive outputs (may appear at the prompt boundary) |
| 3 | Knowledge of the true state | Internal/private representation of facts the model can use |
| 4 | Uncertainty | Epistemic indeterminacy / lack of evidence |
| 5 | Strategic reasoning | Planning/optimization that need not involve misleading anyone |
| 6 | Emerging deceptive trajectory | Internal-state dynamics that become predictive of later deception **before** deceptive text is emitted |
| 7 | Actual deceptive execution | Extern or span that asserts or implies something the model’s private state treats as false, under a deception-benefit objective |
| 8 | Causal mechanisms of deception | Representations whose intervention selectively changes deceptive behavior |

### Pre-first-token rule

**OUR RESEARCH DECISION:** Never automatically call pre-first-token activations “deceptive intent.”

**FACT FROM SOURCE (autoregressive determinism):** Given fixed weights and a fixed tokenized prompt, pre-first-token hidden states are determined. They may support claims about **context-conditioned propensity**, not sample-specific hidden decisions that “occurred before generation” independent of the prompt.

---

## 2. Measurement regimes (pre-registered)

### A. Prompt-boundary physiology

Measure hidden states immediately before generation begins.

**Allowed interpretation:** context-conditioned propensity toward later deception.  
**Forbidden automatic interpretation:** sample-specific deceptive intent.

### B. Pre-deceptive-output physiology

During generation, identify states occurring **before** the first externally deceptive proposition/span.

**Research question:** Does the internal trajectory become predictive of eventual deception before the visible text itself becomes deceptive?

**Analysis requirement:** Compare internal-state prediction against information already present in the emitted text prefix at the same temporal point. This is the stronger research target.

---

## 3. Core hypotheses

### H1 — Predictive signal (**OUR HYPOTHESIS**)

Internal activations contain information predictive of later deceptive behavior before deceptive content appears.

### H2 — Temporal onset (**OUR HYPOTHESIS**)

Predictive information changes systematically across layers and/or decoding time, allowing an earliest detectable region to be estimated.

### H3 — Incremental information (**OUR HYPOTHESIS**)

Internal activations predict later deception better than appropriate surface-information baselines available at the same point in the trajectory.

### H4 — Specificity (**OUR HYPOTHESIS**)

The signal is not fully explained by uncertainty, truth knowledge, explicit deception instructions, generic strategic reasoning, or lexical prompt cues.

### H5 — Causal involvement (**OUR HYPOTHESIS**; not tested in Phase 1)

A subset of predictive representations may causally influence deceptive behavior.

**OUR RESEARCH DECISION:** H5 remains explicitly separate from correlational detection (H1–H4). Probe AUROC alone never licenses causal language.

---

## 4. Null / failure conditions (pre-registered)

Evidence does **not** support the strong hypothesis if:

- activation probes perform no better than appropriate prompt/text/logit baselines;
- performance disappears on held-out scenarios;
- probes primarily recognize explicit deception wording or incentives;
- the signal primarily tracks uncertainty or truth/falsity rather than deception;
- predictive directions cannot survive basic distribution shift;
- later causal interventions fail to affect deception beyond matched sham/random interventions;
- interventions merely destroy model capability or produce refusal rather than selectively affecting deception.

**OUR RESEARCH DECISION:** Negative results must be retained and reported. Absence of a white-box advantage is a valid scientific outcome.

---

## 5. Required control conditions (conceptual pre-registration)

Exact datasets for conditions 1–5 will be finalized before Phase 2 data collection. **Do not generate those datasets in Phase 1.**

1. **KNOWN-TRUTH / HONEST** — Model has the relevant true information and is incentivized to report it honestly.
2. **KNOWN-TRUTH / DECEPTIVE** — Same underlying truth, but deception benefits the objective.
3. **UNCERTAINTY** — Model lacks sufficient evidence to know the answer.
4. **FALSE-BELIEF** — Model is given incorrect private information but is instructed to answer honestly (separates intentional deception from sincerely expressing a false belief).
5. **STRATEGIC-NONDECEPTIVE** — Model performs planning/optimization without needing to mislead anyone.
6. **EXPLICIT-DECEPTION / ROLEPLAY** — Positive control where deception is directly induced or role-conditioned.

**OUR RESEARCH DECISION:** Freeze **RoleplayDeception** as the initial positive-control instance of condition 6 because Mistral-7B has published activation-probe precedent on it (**FACT FROM SOURCE:** ACL 2026 Findings / Kirch, Dower, Skapars, Lubana, Krasheninnikov; see `docs/literature_basis.md`).

RoleplayDeception is **not** sufficient to establish spontaneous deceptive intent. It contains explicit role/incentive structure and therefore has potential contextual confounds.

---

## 6. Baselines to pre-register

Later experiments must compare activation probes against:

- class prevalence / majority baseline;
- prompt-text-only classifier;
- emitted-prefix-text classifier at the same temporal point;
- output-logit statistics where applicable;
- random projection / random activation direction controls;
- shuffled-label probe;
- simple model self-report only if methodologically appropriate.

**OUR RESEARCH DECISION:** The scientific claim of interest is whether internal states provide information unavailable or less accessible from appropriate surface baselines—not merely high AUROC in isolation.

---

## 7. Data splitting requirements

Prevent leakage:

- scenario-level / group-level splitting, not random response-level splitting where related variants can cross train/test;
- train / validation / held-out test;
- separate held-out scenario/domain evaluation;
- no test-set-driven hyperparameter tuning;
- seeds recorded;
- label-generation procedure versioned;
- any LLM judge prompts/version/settings recorded exactly.

If multiple outputs come from one underlying scenario, they must remain within the same split unless a specific experiment explicitly studies within-scenario variation.

---

## 8. Statistical requirements

**Primary metric:** AUROC.

Also report:

- AUPRC;
- accuracy only with class balance stated;
- sensitivity/specificity or precision/recall;
- calibration where probabilistic interpretation is used;
- bootstrap confidence intervals;
- sample counts;
- class prevalence.

For comparing white-box versus surface baselines: use paired evaluation on the same held-out examples and report uncertainty on the performance **difference**.

**No cherry-picked layer reporting.** All tested layers must be logged.

---

## 9. Future causal standard (document now, execute later)

Correlation is insufficient for H5.

Candidate causal tests may include:

- activation patching;
- representation replacement;
- controlled steering;
- matched sham interventions;
- random-direction intervention.

A successful causal result must alter deception while preserving as much as possible:

- underlying factual knowledge;
- task competence;
- fluency;
- ability to answer honestly;
- non-deceptive strategic reasoning.

“Model refused more often” is **not** sufficient evidence that a deception mechanism was selectively affected.

---

## 10. Model and compute freeze

- **Primary model:** `mistralai/Mistral-7B-Instruct-v0.2` (**OUR RESEARCH DECISION**; see D001).
- **Revision:** `TO_BE_PINNED_BEFORE_PHASE2` until an exact Hugging Face commit is recorded (**OUR RESEARCH DECISION**).
- **Quantization:** `null` for primary scientific results. Quantized local activations are debugging-only and must be labelled as such.
- **No Modal jobs in Phase 1.**

See `docs/methodology_constraints.md` and `docs/reproducibility.md`.

---

## 11. Phase gates

| Gate | Requirement |
| --- | --- |
| End of Phase 1 | Protocol, literature basis, configs, validation suite merged only after review of `phase1/research-spec` |
| Start of Phase 2 | Explicit authorization; RoleplayDeception positive-control pipeline only |
| Start of Phase 3+ | Pre-output scans and specificity controls after positive-control instrumentation review |
| Causal (H5) | Only after correlational claims and intervention ethics/compute review |

**PHASE 1 STOP:** Do not download/run the full Mistral experiment, generate deception samples, collect primary activations, train primary probes, launch Modal, perform causal interventions, or claim scientific results from this repository.
