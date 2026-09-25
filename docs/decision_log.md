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
- **Superseded by:** D011 (Phase 2 exact pin).

### D010 — Minor structural additions beyond the requested skeleton

- **Type:** **OUR RESEARCH DECISION**
- **Decision / deviation notes:**
  - Added `docs/research_brief.md` as a one-page conceptual entry point alongside the full protocol.
  - Placed the installable package under `src/pre_output_physiology/` (src layout) for clean imports with hatchling.
  - Added optional `[project.optional-dependencies] modal` without executing Modal.
  - Added `dependency-groups.dev` for uv + pytest/ruff.
  - Experiment YAMLs use `status: scaffolded` to make non-authorization explicit.
- **Date:** 2026-09-24

---

## Phase 2 decisions

### D011 — Pin exact Mistral revision

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Replace `TO_BE_PINNED_BEFORE_PHASE2` with Hugging Face commit `63a8b081895390a26e140280378bc85ec8bce07a` for `mistralai/Mistral-7B-Instruct-v0.2`.
- **Date:** 2026-09-24

### D012 — Pin Apollo RoleplayDeception source revision

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Pin `ApolloResearch/deception-detection` at `f8ec4010e74927394709dffa22b97bdf8cd5a62f`; canonical file `data/roleplaying/dataset.yaml` with SHA256 `7d3e36dc9a2fa72ff823b651c2ade2e0fc34c5bc6267cdf9b7be8b046ed2e564`.
- **Date:** 2026-09-24

### D013 — Pin exact LASR HF dataset revision

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Pin `lasrprobegen/deception-activations` at `bf93584916fbd23121eca6f2017017df0ef3184f`. Download only RoleplayDeception/Mistral JSONL text+label files; never snapshot-download the TB-scale activation repository.
- **Date:** 2026-09-24

### D014 — Teacher-forced forward passes over existing labelled outputs

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** For the Phase 2 positive control, use fixed RoleplayDeception outputs from LASR JSONL (`input_formatted + model_outputs`) and run teacher-forced canonical forward passes rather than generating a new dataset.
- **Reason:** Validates activation extraction while minimizing stochasticity and GPU cost.
- **Date:** 2026-09-24

### D015 — Collect only block-12 mean activations in Phase 2

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Register a forward hook only on `model.model.layers[12]`; immediately mean-pool non-padding tokens; store `[N, hidden]` float32 safetensors. No all-layer / trajectory / causal data.
- **Reason:** No scientific need to pay for all-layer physiology before instrumentation is validated.
- **Date:** 2026-09-24

### D016 — Train/evaluate probe locally after GPU extraction

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Modal performs extraction only. Logistic regression + metrics + bootstrap CIs run on the local Mac.
- **Date:** 2026-09-24

### D017 — Tokenize with `add_special_tokens=False`

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Because upstream `input_formatted` already includes the Mistral BOS token `<s>`, Phase 2 tokenization uses `add_special_tokens=False` to avoid double-BOS.
- **Date:** 2026-09-24

### D018 — Class-1 semantics for scale_labels

- **Type:** **FACT FROM SOURCE** + documentation decision
- **Decision:** Record that `scale_labels` is an honesty grade; `<=5 → class 1` means deceptive/dishonest (`labels=="positive"`); `>5 → class 0` means honest (`labels=="negative"`), matching LASR `activation_handlers.py`.
- **Date:** 2026-09-24

---

## Phase 3A decisions (methodology freeze; no activations)

### D019 — Reuse fixed RoleplayDeception trajectories

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Reuse the Phase 2 pinned LASR RoleplayDeception JSONL trajectories (no new generation).
- **Reason:** Avoids unnecessary generation/GPU cost and provides multiple behavior outcomes across repeated prompts.
- **Date:** 2026-09-25

### D020 — Prompt-boundary states represent propensity

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** At the prompt boundary, identical tokenized prompts share one activation; repeated rollouts are observations under a shared prompt state. Interpret as context-conditioned deception propensity, not sample-specific intent.
- **Reason:** Deterministic transformers make independent hidden “decisions” per identical prompt methodologically invalid.
- **Date:** 2026-09-25

### D021 — Group by exact prompt hash

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Primary scientific grouping key is `prompt_sha256 = SHA256(exact input_formatted)`. ID-prefix grouping is diagnostic only; on disagreement, prompt hash wins.
- **Reason:** Avoid assumptions about upstream ID format.
- **Date:** 2026-09-25

### D022 — Freeze coarse Phase 3 layer/time grid before activations

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Pre-register transformer blocks `[0, 4, 8, 12, 16, 20, 24, 28, 31]` and prefix lengths `[0, 1, 2, 4, 8, 16]` before any Phase 3 activation extraction.
- **Date:** 2026-09-25

### D023 — Block 12 is the primary Phase 3 anchor

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Block 12 is the pre-registered primary anchor layer for Phase 3, chosen before Phase 3 based on the independent Phase 2 literature-aligned positive control. Do not re-select after inspecting Phase 3 results.
- **Date:** 2026-09-25

### D024 — Strong H3 claims require matched surface baselines

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Report `ΔAUROC = AUROC_activation − AUROC_surface` with paired group-aware bootstrap. Activation AUROC > 0.5 alone is insufficient for incremental white-box claims.
- **Date:** 2026-09-25

### D025 — Separate early-trajectory prediction from true pre-deceptive analysis

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Fixed small `k` results are early-trajectory prediction, not automatically “pre-deceptive.” Reserve “pre-deceptive-output physiology” for onset-aligned analyses that pass the annotation audit gate.
- **Date:** 2026-09-25

### D026 — Conservative onset annotations must be text/judge-grounded and activation-independent

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Candidate onsets use only response text + upstream deception label + existing judge explanation fields with exact substring evidence. No probe-driven or auto-LLM span labels. Unresolved when no defensible exact span exists.
- **Date:** 2026-09-25

### D027 — Canonical Phase 3 prompt/response token boundary

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Define Phase 3 response-token indices from the suffix of `tokenizer(input_formatted + model_outputs, add_special_tokens=False)` only after verifying (1) exact prompt-id prefix equality and (2) no token straddles the prompt/response character boundary. Matched surface baselines and onset token indices must use these canonical offsets. Standalone `tokenizer(model_outputs)` mismatches are diagnostic only and do not redefine the canonical suffix.
- **Reason:** Prevents scientifically incorrect `k` indexing if the tokenizer re-segments across the character join under teacher-forced full-sequence extraction.
- **Date:** 2026-09-25

### D028 — Require unseen future response token

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** For `k>0`, eligibility requires `canonical_response_token_length > k` (strict). Rows with response length exactly `k` are excluded because the full response may already be visible.
- **Date:** 2026-09-25

### D029 — Primary Phase 3B endpoints

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Freeze primary endpoints before activation extraction: Regime A = block 12 / `k=0`; Regime B = block 12 / `k=1`. Other k at block 12 are secondary temporal endpoints; other layers are the pre-registered H2 coarse scan.
- **Date:** 2026-09-25

### D030 — H3 interpretation (predictive utility, not new information)

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** `ΔAUROC` vs matched surface baseline measures incremental predictive utility / accessibility of structure in the representation, not proof of information-theoretically unavailable information.
- **Date:** 2026-09-25

### D031 — k0 extracted once per prompt group

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Prompt-boundary physiology is deterministic; compute once per unique `prompt_sha256` from prompt-only input and broadcast to rows sharing that group.
- **Date:** 2026-09-25

### D032 — Right padding for trajectory extraction

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Use right padding for Phase 3 teacher-forced trajectory batches so scientific token positions match unpadded coordinates; verify batch vs single-example cosine ≥ 0.9999 before full extraction.
- **Date:** 2026-09-25

### D033 — Locked test withheld during Phase 3B1

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Phase 3B1 extracts and analyzes train+validation only. Locked-test GPU extraction and predictive evaluation require a separate Phase 3B2 authorization after independent audit.
- **Date:** 2026-09-25

### D034 — Phase 3B1 full extraction blocked by preflight cosine gate

- **Type:** **OUR RESEARCH DECISION** / operational hold
- **Decision:** After Modal preflight on the pinned stack, full-vs-truncated causal min cosine was ≈0.99984 and batch-vs-single min cosine ≈0.99978 (gate 0.9999), with failures concentrated at late layer 31. Full Phase 3B1 extraction was not started. No attention-implementation switch and no threshold relaxation without a separate explicit decision.
- **Date:** 2026-09-25

### D035 — Canonical truncated-prefix extraction

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Phase 3B Regime A/B scientific activations are extracted from inputs truncated exactly at the measurement point (`prompt_ids` for k=0; `prompt_ids + response_suffix_ids[:k]` for k>0). Future response tokens are never supplied.
- **Reason:** Eliminates future-token exposure by construction instead of relying on numerical equivalence between differently shaped forward passes.
- **Date:** 2026-09-25

### D036 — Single-example primary forwards

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Primary scientific Phase 3B activations use batch size 1 (no scientific multi-example padding/batching).
- **Reason:** The prior BF16 preflight showed batch-vs-single numerical sensitivity, concentrated at late layers. We avoid relaxing the original batch-equivalence gate.
- **Date:** 2026-09-25

### D037 — Original numerical gate not relaxed

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** The failed `0.9999` full-sequence and batch-equivalence gates remain historical results (D034). The extraction method changed; the threshold was not loosened.
- **Date:** 2026-09-25

### D038 — Cost benchmark before full extraction

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Benchmark single-example truncated-prefix throughput; stop if projected Phase 3B1 spend exceeds the soft budget ($20) before full extraction.
- **Date:** 2026-09-25

### D039 — Canonical activation storage is float32

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Scientific activations are computed in BF16 and stored as float32 (BF16 values are exactly representable in float32). Do not store activations as float16. Model compute remains `torch.bfloat16`. Manifests record `compute_dtype: bfloat16` and `activation_storage_dtype: float32` separately.
- **Reason:** Avoid an unnecessary second precision conversion that was present in the original development extract.
- **Date:** 2026-09-25

### D040 — Clean-tree provenance freeze before Modal

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Before any canonical Phase 3B1 GPU launch, commit extraction/provenance code with a clean working tree. Local entrypoint refuses Modal if `git status --porcelain` is non-empty. Remote manifests must include `git_commit`, clean-tree marker, extractor/analysis SHA256s, freeze hash, model/dataset revisions, and dtype fields.
- **Date:** 2026-09-25
