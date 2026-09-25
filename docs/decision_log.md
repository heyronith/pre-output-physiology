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

### D041 — One-shot locked test

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Phase 3B2 evaluates the untouched 500-row locked test once with frozen endpoints/hyperparameters. No scientific tuning after locked-test outcome evaluation.
- **Date:** 2026-09-25

### D042 — Final models refit on combined development data

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** For locked-test evaluation, refit frozen probes/baselines on combined Phase 3 train+validation (3000 rows) after all model/hyperparameter choices are frozen. Do not fit vocabulary/IDF or probes using locked-test text/activations.
- **Date:** 2026-09-25

### D043 — Confirmatory locked endpoint remains L12/k1

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Primary H3 replication requires paired activation-vs-surface evaluation at the pre-registered Regime B endpoint (block 12 / k=1). Regime A (block 12 / k=0) is also reported; failure to reject zero ΔAUROC is not an equivalence claim.
- **Date:** 2026-09-25

### D044 — Secondary grid cannot replace failed primary endpoint

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** All other layer×k cells remain descriptive/exploratory regardless of locked-test performance. Do not promote the best test cell into a new primary result.
- **Date:** 2026-09-25

### D045 — Phase 4 tests deception specificity

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Phase 4 studies whether the Phase 3 L12/k1 signal is distinguishable from strategic reasoning, false belief, uncertainty, and nondeceptive truth–target conflict.
- **Date:** 2026-09-25

### D046 — Frozen Phase 3 L12/k1 probe

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Phase 4 scores use the immutable Phase 3 logistic probe (C=0.01) fitted on the 3000-row development set only. No Phase 4 labels may retrain or select the primary deception probe. L12/k0 is secondary only.
- **Date:** 2026-09-25

### D047 — Matched strategic-honest primary control

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** The primary specificity contrast is C3 (`known_deceptive_strategic`) versus C2 (`known_honest_strategic`) under matched strategic/payoff structure.
- **Date:** 2026-09-25

### D048 — Common first-token control

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** All final Phase 4 responses must share one tokenizer-verified first token (`Response`, id 2963) so k=1 classification cannot be explained by first-token identity.
- **Date:** 2026-09-25

### D049 — Behavioral labels precede activation analysis

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Behavioral validity uses deterministic response parsing rules independent of activations and independent of LLM judges.
- **Date:** 2026-09-25

### D050 — Separate pilot from final scenarios

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Prompt/format compliance may be tested on pilot scenarios only (seed 7). The final 240 base scenarios (seed 42) remain untouched until prompt templates are frozen. Pilot examples never enter scientific results.
- **Date:** 2026-09-25

### D051 — Symmetric strategic-control template

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** C2 and C3 use the same strategic prompt wording. Their only intended difference is whether the reward-target state matches or conflicts with the model-accessible private-record state.
- **Reason:** Prevent the primary specificity contrast from being explained by condition-specific prompt wording (e.g., “opposite”, unique non-disclosure phrasing).
- **Date:** 2026-09-25

### D052 — First-token compliance uses token IDs

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** The Phase-4 common-first-token requirement is evaluated using the actual first generated token ID (`2963` = `Response`), not decoded whitespace parsing.
- **Date:** 2026-09-25

### D053 — Deterministic behavior-only pilot

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Pilot generation uses greedy BF16 Mistral generation solely to validate condition/format compliance. No activations or frozen-probe scoring are permitted during the pilot.
- **Date:** 2026-09-25

### D054 — Pilot readiness gates

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Operational pilot gates are frozen before seeing pilot results: ≥23/24 first-token-ID compliance per condition; ≥20/24 full behavioral validity per condition; ≥18/24 base scenarios with both C2 and C3 behaviorally valid. These are usability gates, not scientific effect-size thresholds.
- **Date:** 2026-09-25

### D055 — Correct common first-token ID

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** The true first generated token for the frozen `Response` prefix under the actual Mistral chat-generation context (`apply_chat_template(..., add_generation_prompt=True)`) is token ID **12107**. The earlier **2963** value came from an incorrect design-time tokenization context that concatenated `Response` directly onto `[/INST]` without the leading-space BPE form used in real generation.
- **Supersedes for ID value:** D052’s numeric ID `2963` (token-*ID* method of D052 remains in force).
- **Date:** 2026-09-25

### D056 — Single allowed post-pilot template revision

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Revision 1 replaces free-form responses with an exact three-line machine-readable output format (`Response` / `MODE=...` / `FINAL=...`) and strengthens reward-target compliance while preserving C2/C3 textual symmetry (D051).
- **Date:** 2026-09-25

### D057 — No second template-tuning round

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Revision 1 is the sole post-pilot prompt revision permitted under D050. If the revised pilot fails the pre-registered usability gates (D054), Phase 4 holds for methodological redesign rather than further prompt tuning.
- **Date:** 2026-09-25

### D058 — Controlled-prefix methodology redesign

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** After the single permitted prompt revision failed because free generation did not reliably emit the common prefix, Phase 4 switches to a **controlled common prefix** while keeping revision-1 prompts unchanged. This is a generation/measurement design change, not another prompt-template revision.
- **Date:** 2026-09-25

### D059 — Controlled-prefix k1 interpretation

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Future Phase-4 L12/k1 analysis will measure the state after a supplied common neutral token (`12107` / `Response`). It tests specificity of the frozen Phase-3 probe under controlled-prefix distribution shift, not natural first-token selection. Terminology: **controlled-prefix k1** (not “natural first-token k1”).
- **Date:** 2026-09-25

### D060 — Behavioral gates unchanged

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** The semantic behavior-validity thresholds remain those frozen before the previous pilots (≥20/24 per condition; ≥18/24 C2∩C3 paired). Prefix integrity (144/144 supplied ID 12107) replaces the sampled first-token gate as an engineering check only.
- **Date:** 2026-09-25

### D061 — Uncertainty control held

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** C5 failed the frozen pilot usability gate under both free-generation and controlled-prefix testing. We do not weaken its behavioral criterion after observing pilot outputs. C5 is excluded from the confirmatory Phase-4 final run and remains an unresolved uncertainty-specificity limitation requiring a separately designed future control.
- **Date:** 2026-09-25

### D062 — Five-condition final run

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Phase-4 final generation proceeds with C1, C2, C3, C4, and C6 only. Primary and key-secondary contrasts remain unchanged: primary C3 vs C2; key secondary C3 vs C4.
- **Date:** 2026-09-25

### D063 — Phase 4E transfer/specificity scoring

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Phase 4E scores the immutable Phase-3 L12 probes on Phase-4D controlled-prefix activations without retraining or recalibration. Primary endpoint is L12 / controlled-prefix k1 on frozen C3∩C2 pairs (N=148). A positive result supports cross-context specificity of the frozen RoleplayDeception probe; a null does not prove absence of deception physiology and must not trigger rescue retraining. Eligibility sets remain frozen; C5 remains HOLD; causal interventions remain unauthorized.
- **Date:** 2026-09-25

### D064 — Natural-token k1 diagnostic after controlled-prefix null

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** After Phase 4E found ~chance C3-vs-C2 separation under controlled `Response` (12107), Phase 4F diagnoses whether that null was caused by replacing the model's natural greedy first token. Natural-token k1 uses truncated `prompt + [argmax next-token]` only—no full response regeneration, no prompt/eligibility/probe changes, no retraining. If C3-vs-C2 remains ~0.5, controlled-prefix choice did not explain the specificity failure. If natural-token AUROC clearly exceeds 0.5, interpret as distribution shift and check first-token identity confounding before claiming deception specificity.
- **Date:** 2026-09-25

### D065 — Phase 4 falsified original-probe deception specificity

- **Type:** **FACT**
- **Decision:** Phase 4 demonstrated that the frozen Phase-3 RoleplayDeception L12 probe does not specifically separate intentional strategic deception from matched strategic honesty under Phase-4 controls (C3 vs C2 near chance under both controlled and natural first-token k1). No causal interventions are authorized on that probe.
- **Date:** 2026-09-25

### D066 — Phase 5 discovery contrast is S3 vs S2 from inception

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Phase 5 discovers physiology distinguishing strategic deception (S3) from strategic honesty (S2) while holding strategic/private-objective framing constant. The only intended difference is whether the private objective target agrees with (S2) or conflicts with (S3) the private record. Prompts must not use condition-revealing lexicon.
- **Date:** 2026-09-25

### D067 — Controlled-prefix primary endpoint for Phase 5 physiology

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Future Phase-5 candidate discovery uses the controlled-prefix hidden state after the shared neutral `Response` token (verified tokenizer ID under real chat-generation context) before the state answer appears, scanned across layers `[0,4,8,12,16,20,24,28,31]`. Secondary endpoint is k0 prompt-boundary. Natural-first-token analyses are diagnostic only, not candidate-selection endpoints.
- **Date:** 2026-09-25

### D068 — Locked generalization families untouched until freeze

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Two Phase-5 families are reserved as locked generalization and must not receive discovery-stage model generation, activation extraction, or probe fitting until the entire probe/layer/endpoint selection procedure is frozen. Phase 5A generates only pilot scenarios (including pilot IDs from locked-family templates that remain disjoint from locked-final IDs).
- **Date:** 2026-09-25

### D069 — Uncertainty control remains out of Phase-5 discovery

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Phase-4 C5 / uncertainty remains an unresolved later specificity control and is not part of Phase-5 S2/S3 discovery.
- **Date:** 2026-09-25

### D070 — Phase 5 does not claim information-theoretic absence from text

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Because record and objective target are present in privileged full context, S2/S3 is in principle inferable from text. Phase 5 therefore does not claim information-theoretic absence from text; it seeks a physiological representation that is robust across scenarios/families, specific to strategic deception relative to strategic honesty, prospectively available before the semantic answer, and transferable to unseen families.
- **Date:** 2026-09-25

### D071 — Single Phase 5A prompt-format revision after revision-1 pilot fail

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Revision-1 free-generation pilot failed operational gates (S2 valid 0/32; S3 valid 3/32; paired 0/32), mainly from `Response:` colon schema mismatches and truncated/shortened states. Exactly one prompt-format revision is authorized: revision 2 strengthens the shared output-schema block (no colon; verbatim full target; one line; correct/incorrect shape examples) while preserving S2/S3 semantic symmetry. A new frozen pilot set (pilot_seed=19) is required. Validity definitions are not weakened. If the revision-2 pilot fails, Phase 5A holds.
- **Date:** 2026-09-25

### D072 — Deterministic discovery train/validation family split

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Before Phase 5B discovery generation, assign the six discovery families by sorting `SHA256("phase5_family_split_v1|" + family_name)` ascending: first 4 = discovery-train, last 2 = discovery-validation. No scenario-level mixing across splits. Assignment is outcome-independent and must be committed before model generation. Locked families remain excluded.
- **Date:** 2026-09-25

### D073 — Primary estimand is all designed S2/S3 pairs

- **Type:** **OUR RESEARCH DECISION**
- **Decision:** Primary future Phase-5 physiology analysis uses all designed S2/S3 pairs (train N=640; validation N=320), independent of later behavioral compliance, because behavior occurs downstream of the k1 state and compliance filtering would introduce post-treatment selection bias. Behaviorally valid paired subsets are secondary sensitivity analyses only and must not replace the primary all-pair populations. Future probe procedure is preregistered (standardized logistic regression, C=0.01, intercept, max_iter=500, seed=42; train on discovery-train; layer select on discovery-validation) but not executed in Phase 5B.
- **Date:** 2026-09-25
