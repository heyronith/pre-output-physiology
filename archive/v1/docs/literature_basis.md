# Literature basis

Evidence labels: **FACT FROM SOURCE**, **OUR RESEARCH DECISION**, **OUR HYPOTHESIS**.

This document records prior work that motivates the Phase 1 freeze. It does **not** claim that experiments in this repository have replicated or extended those results.

Upstream code is **referenced**, not copied, during Phase 1.

---

## Evidence table

### 1. Kirch / Dower / Skapars / Lubana / Krasheninnikov (ACL 2026 Findings)

| Field | Content |
| --- | --- |
| **Citation** | Adrians Skapars, Nathalie Maria Kirch, Samuel Dower, Ekdeep Singh Lubana, Dmitrii Krasheninnikov. *The Impact of Off-Policy Training Data on Probe Generalisation.* Findings of ACL 2026. |
| **ACL Anthology ID** | `2026.findings-acl.1139` |
| **URL** | https://aclanthology.org/2026.findings-acl.1139/ |
| **Evidence class** | **FACT FROM SOURCE** (paper text / appendices) |

#### Definition of deception (**FACT FROM SOURCE**)

The paper defines deception as situations in which an LLM attempts to convince the user of something it knows is false (citing Barkur et al., 2025).

#### Datasets used for deception (**FACT FROM SOURCE**)

| Dataset | Origin | Notes from source |
| --- | --- | --- |
| **RoleplayDeception** | Goldowsky-Dill et al., 2025 (via Apollo `deception-detection` roleplaying data) | ~371 unique prompts; paper resamples responses rather than reusing the original generated responses; features instructions to roleplay as a character in a situation where lying is beneficial |
| **InsiderTrading** | Scheurer et al., 2024 | Multi-turn dialogue in which the model has chosen insider trading and is asked what information it used; largely a single prompt template from which responses are generated |

#### Models used for deception experiments (**FACT FROM SOURCE**)

Primary activation model in the main text is Llama-3.2-3B-Instruct; additional verification includes Ministral-8B-Instruct-2410 and Gemma-3-27B-it.

For Mistral-family deception data generation / activations (Appendix D / F):

| Dataset | Mistral-family model used | Important limitation |
| --- | --- | --- |
| **RoleplayDeception** | **Mistral-7B-Instruct-v0.2** | Published precedent for this checkpoint |
| **InsiderTrading** | **Mixtral-8x7B-Instruct-v0.1** (with DeepSeek-V3 also listed in off-policy pairing) | **Do NOT state that the ACL work validated Mistral-7B on InsiderTrading** |

#### Probe architectures (**FACT FROM SOURCE**)

1. **Linear probes** — scikit-learn logistic regression on sequence-averaged activations; L2 regularisation with inverse strength \(C\); activations normalised to mean 0, std 1 before probing.
2. **Attention probes** — learned query/value attention over token positions; AdamW optimisation; binary cross-entropy.

#### Activation layers for deception (**FACT FROM SOURCE**)

| Activations model | Layer (linear & attention tables) | Linear \(C\) | Attention LR / weight decay |
| --- | --- | --- | --- |
| Llama 3B | 12 | 0.01 | 0.001 / 1e-5 |
| **Mistral 7B** | **12** | 0.01 | 0.0001 / 1e-5 |
| Mixtral 8x7B | 16 | 0.01 | 0.0001 / 1e-5 |

#### Relevant generalisation limitations (**FACT FROM SOURCE**)

- Largest generalisation failures arise for behaviours defined by response “intent” (e.g., strategic deception) rather than text-level content (“text-ambiguous” vs “text-evident”).
- Domain shift often hurts more than response-strategy shift.
- The authors propose on-policy incentivised data as a proxy test when on-policy natural data is unavailable; using that proxy, they **predict** that current deception probes may fail to generalise to real monitoring scenarios.
- Off-policy training data can sometimes yield more reliable probes than on-policy data from a sufficiently different setting—but deception remains a hard case.
- Even when private/public setups cause models to state deceptive intent in chain-of-thought, probes trained on such data can still fail to generalise, suggesting stated intent ≠ same activation signature as genuine strategic reasoning.

#### How this project uses the paper (**OUR RESEARCH DECISION**)

- Adopt RoleplayDeception + Mistral-7B-Instruct-v0.2 as the **positive-control** instrumentation target (not as proof of spontaneous intent).
- Treat layer 12 as a **literature-informed starting layer** to log, not as a cherry-picked sole reporting layer; all tested layers must be logged.
- Import the paper’s caution about off-policy / distribution-shift fragility into our null conditions and splitting rules.

---

### 2. Goldowsky-Dill et al. (strategic deception / activation probing)

| Field | Content |
| --- | --- |
| **Citation** | Nicholas Goldowsky-Dill, Bilal Chughtai, Stefan Heimersheim, Marius Hobbhahn. *Detecting Strategic Deception Using Linear Probes.* arXiv:2502.03407 (also PMLR / ICML 2025 proceedings entry as *Detecting Strategic Deception with Linear Probes*). |
| **Code / data** | https://github.com/ApolloResearch/deception-detection ; probe outputs https://data.apolloresearch.ai/dd/ |
| **Evidence class** | **FACT FROM SOURCE** |

#### Key facts recorded (**FACT FROM SOURCE**)

- Motivation: output monitoring alone is insufficient if internal reasoning is misaligned while outputs look benign.
- Method: linear probes on residual-stream activations.
- Training sets studied include **Instructed-Pairs** (contrasting honest vs deceptive instructions, following Zou et al., 2023) and **roleplaying** scenarios (the RoleplayDeception lineage used above).
- Primary evaluation model in the paper: **Llama-3.3-70B-Instruct**; reported probe depth in that setting is approximately **layer 22 of 80**.
- Reported in-distribution / evaluation AUROCs often very high (paper abstract: between ~0.96 and 0.999 on their evaluation datasets), with the authors still concluding that current performance is **insufficient as a robust defence** against deception.
- Cross-setting generalisation is a central concern and is a major bridge into the ACL 2026 off-policy generalisation study.

#### How this project uses the paper (**OUR RESEARCH DECISION**)

- Treat RoleplayDeception as a **known positive-control behaviour family**, not as spontaneous scheming.
- Inherit the stance that white-box probes are promising but not yet a robust monitor.
- Do not import their 70B layer index as our Mistral layer; use ACL Mistral-7B layer 12 as the literature-informed starting point for this checkpoint.

---

### 3. SamDower/LASR-probe-gen (public implementation)

| Field | Content |
| --- | --- |
| **Repository** | https://github.com/SamDower/LASR-probe-gen |
| **Association** | Public implementation associated with the ACL 2026 probe-generalisation work (paper footnotes link this repo) |
| **Pinned upstream commit (Phase 1 record)** | `f4c6ad69b10a5436a2e819c69009431802a0f5f7` |
| **Commit metadata** | Committer date `2026-01-02T19:25:12Z`; message `Delete output.png`; tip of default branch `main` as queried during Phase 1 setup |
| **Evidence class** | **FACT FROM SOURCE** (git metadata) |

#### Phase 1 policy (**OUR RESEARCH DECISION**)

- Record the SHA above; do **not** write “latest.”
- **Do not copy upstream code into this repository during Phase 1.**
- Any later reuse must pin this or a newer explicitly approved SHA, document diffs, and avoid silent drift.

---

## Synthesis for this project

| Claim type | Statement |
| --- | --- |
| **FACT FROM SOURCE** | Activation probes can detect strategic deception in completed responses in published settings; Mistral-7B-Instruct-v0.2 has published RoleplayDeception probe precedent at layer 12; generalisation for intent-like behaviours is fragile. |
| **OUR RESEARCH DECISION** | Start with that checkpoint and positive control; target **onset** / pre-deceptive-output physiology; require surface baselines and specificity controls; keep causation separate. |
| **OUR HYPOTHESIS** | Internal trajectories may become predictive of deception before deceptive text appears, and may contain incremental information beyond surface baselines (H1–H4); some predictive directions may be causal (H5, later). |

**No primary experimental data was collected while assembling this literature basis.**
