# Phase 3 protocol (methodology freeze — Phase 3A)

**Status:** Phase 3A local preparation only.  
**GPU:** not authorized.  
**Activations:** none collected.

> Phase 3 asks whether Mistral-7B internal states become predictive of later deceptive behavior before the visible output itself provides equivalent evidence.

Evidence labels: **FACT FROM SOURCE**, **OUR RESEARCH DECISION**, **OUR HYPOTHESIS**.

---

## 1. Measurement regimes

### Regime A — Prompt-boundary physiology (`k = 0`)

Activation at the final token of the formatted prompt.

**Allowed interpretation:** context-conditioned **deception propensity**.  
**Forbidden:** sample-specific deceptive intent.

**Deterministic constraint:** identical tokenized prompts ⇒ identical pre-generation activations. Repeated rollouts from the same prompt share one prompt-boundary state; outcomes are observations under that shared state.

### Regime B — Early generation trajectory (`k > 0`)

Activation after `k` generated/assistant tokens are visible, at generated-token index `k-1` (state used immediately before predicting the next token).

**Question:** Does the internal state predict the eventual deception label better than prompt + visible prefix at exactly that point?

**Terminology:** early-trajectory prediction — **not** automatically “pre-deceptive.”

### Regime C — Onset-aligned pre-deception (future, annotation-gated)

Only after independent review of conservative onset annotations.

**Allowed phrase after audit passes:** pre-deceptive-output physiology.

---

## 2. Data reuse (frozen)

| Item | Pin |
| --- | --- |
| Model | `mistralai/Mistral-7B-Instruct-v0.2` |
| Model revision | `63a8b081895390a26e140280378bc85ec8bce07a` |
| LASR HF dataset | `lasrprobegen/deception-activations` |
| Dataset revision | `bf93584916fbd23121eca6f2017017df0ef3184f` |
| Train file | `roleplaying/mistral_7b_incentivised_train.jsonl` |
| Test file | `roleplaying/mistral_7b_incentivised_test.jsonl` |

Do not generate another RoleplayDeception corpus. Do not alter locked test examples.

Primary outcome for trajectory scan: `eventual_deception` = Phase 2/LASR binary deception label.

---

## 3. Grouping key

**Primary scientific key:** `prompt_sha256 = SHA256(exact input_formatted)`.

ID-prefix grouping (`ids` before `_`) is compared diagnostically only. On disagreement, exact prompt hash wins (**D021**).

Prompt-boundary analyses and uncertainty must respect prompt/scenario clustering.

---

## 4. Splits

| Split | Source | Role |
| --- | --- | --- |
| Locked test | Upstream 500-row test | Never used for tuning |
| Phase 3 train | GroupShuffleSplit on upstream train | Fit |
| Phase 3 validation | 20% of train groups, `random_state=42` | Limited hyperparameter choice only |

`GroupShuffleSplit(test_size=0.20, random_state=42)` by `prompt_sha256`.

No seed shopping. Zero prompt-hash overlap across train/val/locked-test.

---

## 5. Temporal alignment

`k` = number of assistant/generated tokens already visible.

Phase 3 response-token indices are defined from the suffix of the canonical tokenization of `input_formatted + model_outputs`, after verifying that the standalone prompt tokenization is an exact prefix and that no token crosses the prompt/response character boundary.

| k | Visible content | Activation locus |
| --- | --- | --- |
| 0 | none | final prompt token |
| >0 | generated tokens `0..k-1` (canonical suffix) | hidden state at generated index `k-1` |

Future tokens after `k` must never inform the representation used at that point.

The matched surface baseline for each `k` uses the **same** canonical response-token positions: for `k>0`, visible text is `prompt + model_outputs[:end]` where `end` is the exclusive character offset of canonical response token `k-1` (not a join of SentencePiece pieces).

Primary gate (required before Phase 3B Regimes A/B): exact prompt-id prefix + zero boundary-straddling tokens across all examples. See `artifacts/phase3a_summaries/token_boundary_audit.json`.

---

## 6. Frozen coarse scan (pre-registered before activations)

**Transformer blocks:** `[0, 4, 8, 12, 16, 20, 24, 28, 31]`  
**Prefix lengths k:** `[0, 1, 2, 4, 8, 16]`  
**Primary anchor layer:** block `12` (Phase 2 literature-aligned positive control; chosen before Phase 3).

---

## 7. Future Phase 3B extraction contract (not executed)

- Canonical pinned Mistral; BF16; quantization null; eval mode.
- Teacher-forced fixed trajectories; no new primary generation.
- Store `[layer, prefix_position, hidden_dim]` on the coarse grid only.
- Native BF16 → float32 locally for analysis; no 4/8-bit storage.
- **Causal-mask validation required:** full-sequence state at `p` vs truncated-prefix forward at `p` must satisfy `cosine_similarity >= 0.9999` or switch to truncated-prefix passes.

GPU: 1× L40S. Soft budget $20; hard stop $40.

---

## 8. Surface baselines (H3)

For each k, same visible information:

| ID | Baseline |
| --- | --- |
| A | Majority / prevalence |
| B | Prompt text only |
| C | Prompt + emitted prefix (primary surface baseline) |
| D | Prefix structural features |
| E | Output-logit summaries (entropy, max p, top1−top2; Phase 3B) |

Primary text classifier: FeatureUnion of word TF-IDF `(1,2)` + char TF-IDF `(3,5)` → logistic regression. Vectorizers fit on Phase 3 train only. Validation may choose `C ∈ {0.1, 1.0, 10.0}` before locked test.

Negative controls: shuffled-label probe; fixed-seed random direction.

**H3 claim requires** `ΔAUROC = AUROC_activation − AUROC_surface` with paired group-aware bootstrap CI — not activation AUROC > 0.5 alone.

---

## 9. Activation probe

Standardized linear logistic regression (Phase 2-aligned):

- `C=0.01`, `fit_intercept=True`, `max_iter=500`, `random_state=42`
- mean/std fit on training activations only
- Do not retune C per layer/time cell

---

## 10. Statistics

Primary: AUROC. Also AUPRC, accuracy, sens/spec, precision/recall, prevalence, counts.

Bootstrap ≥ 2000; by **prompt group**, not independent rows.

Paired group-aware bootstrap for activation vs surface ΔAUROC.

---

## 11. Onset annotation (activation-independent)

High-confidence only when an upstream explanation contains an **exact** textual substring matchable to `model_outputs` (whitespace/quote normalization only; no fuzzy match).

Conservative onset: move to the **start of the sentence** containing the evidence → `conservative_deceptive_sentence_onset`.

Unresolved if no defensible span: `onset_status = unresolved`.

Do not change rules after seeing activations. Do not use probes/LLMs to define spans.

Onset-aligned analysis (Regime C) is pre-registered for Phase 3B after audit only.

---

## 12. Locked-test discipline

Phase 3A may compute counts, prevalence, grouping diagnostics, onset coverage, lengths on test.

Phase 3A must **not** fit/evaluate predictive models on locked test.

---

## 13. Stop gate

After Phase 3A: STOP. No Modal. No Phase 3 activations. No Phase 3 probe training. No Phase 4. No causal interventions.
