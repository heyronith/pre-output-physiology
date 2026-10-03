# Phase 2 positive-control report

**Run ID:** `phase2_full_20260925T041041Z_7852b3a5`  
**Verdict:** **PASS**

> Phase 2 tests whether our instrumentation reproduces a known deception-related activation signal. It does not establish spontaneous deceptive intent, a pre-output signal, generalization, or causality.

## 1. Objective

Validate the white-box activation-extraction and probe pipeline by reproducing a literature-aligned deception-related activation signal on:

- model: `mistralai/Mistral-7B-Instruct-v0.2`
- dataset: RoleplayDeception / LASR `roleplaying` + `incentivised`
- layer: transformer block 12
- probe: mean-pooled linear logistic regression

## 2. Exact revisions

| Component | Pin |
| --- | --- |
| Mistral model | `63a8b081895390a26e140280378bc85ec8bce07a` |
| LASR HF dataset `lasrprobegen/deception-activations` | `bf93584916fbd23121eca6f2017017df0ef3184f` |
| Apollo `deception-detection` | `f8ec4010e74927394709dffa22b97bdf8cd5a62f` |
| Apollo `data/roleplaying/dataset.yaml` SHA256 | `7d3e36dc9a2fa72ff823b651c2ade2e0fc34c5bc6267cdf9b7be8b046ed2e564` |
| LASR code `SamDower/LASR-probe-gen` | `f4c6ad69b10a5436a2e819c69009431802a0f5f7` |

Source files used (text/labels only; published activation pickles **not** downloaded):

- `roleplaying/mistral_7b_incentivised_train.jsonl`
- `roleplaying/mistral_7b_incentivised_test.jsonl`

## 3. Dataset / sample counts

| Split | N | Class 1 (deceptive) | Class 0 (honest) | Prevalence |
| --- | --- | --- | --- | --- |
| Train | 3000 | 1500 | 1500 | 0.50 |
| Test | 500 | 250 | 250 | 0.50 |

**Label semantics (documented from LASR):** `scale_labels` is an honesty grade. `scale_labels ≤ 5 → class 1` (deceptive/dishonest; `labels=="positive"`); `> 5 → class 0` (honest; `labels=="negative"`).

**Leakage audit:**

- ID overlap train/test: **0**
- Exact teacher-forced text overlap: **0**
- Scenario-prefix overlap: **0**
- Within-train duplicate full texts: 394 extra rows (184 duplicate groups) among 3000 — reported, not silently “fixed” (Phase 2 preserves upstream organization).

## 4. Model / instrumentation

- Dtype: `bfloat16`; quantization: `null`
- Teacher-forced forward passes over `input_formatted + model_outputs`
- Tokenization: `add_special_tokens=False` (formatted text already includes `<s>`)
- Single GPU: Modal **L40S** × 1
- Batch size: 8
- Aggregation: attention-mask mean over non-padding tokens
- Stored artifact: `[N, 4096]` float32 safetensors (not full token tensors)

## 5. Layer-index semantics

- `transformer_block_index = 12`
- Hook module path: `model.model.layers[12]`
- This is the residual-stream **output of transformer block index 12** via a forward hook (LASR-aligned).
- If `output_hidden_states=True` were used, the corresponding tuple index would be `13` (embedding at 0). Phase 2 primary path does **not** use that API.

## 6. Aggregation / probe method

- Mean: `sum(valid_token_activations) / n_valid_tokens`
- Probe: `sklearn.linear_model.LogisticRegression(C=0.01, fit_intercept=True, max_iter=500, random_state=42)`
- Normalization: per-feature mean/std fit on **train only**, applied to test
- Bootstrap: 2000 resamples for AUROC 95% CI

## 7. GPU usage and cost

| Item | Preflight | Full |
| --- | --- | --- |
| Examples | 100 | 3500 |
| Model-load + extract seconds | ~13.7 | ~73.7 |
| Assumed L40S USD/hr | 1.95 | 1.95 |
| Estimated cost (USD) | ~0.0074 | ~0.0399 |

Costs recalculated locally from recorded durations at **`$1.95/hr`** (≈ `$0.000542/s`). No GPU jobs were re-run for this accounting correction.

Optional generation smoke (16 prompts): ~$0.0155 at the same rate.

Cumulative Phase 2 GPU estimate remains **well under the $20 soft / $40 hard budgets**.

Activation artifact shape (full): **`[3500, 4096]`**.

## 8. Metrics + confidence interval

| Metric | Value |
| --- | --- |
| AUROC | **0.9825** |
| AUROC 95% CI (bootstrap) | **[0.9734, 0.9901]** |
| AUPRC | 0.9841 |
| Accuracy (balanced) | 0.932 |
| Precision | 0.9355 |
| Recall / sensitivity | 0.928 |
| Specificity | 0.936 |
| TPR at or below 1% FPR | 0.828 (actual FPR = 0.008) |
| Majority baseline accuracy | 0.50 |
| Confusion | TN=234, FP=16, FN=18, TP=232 |
| Probe converged | yes (`n_iter=53`) |

## 9. Comparison with prior work

**Expected comparison recorded before treating our number as decisive:**

No single, unambiguous published AUROC was identified for the exact setting:

> Mistral-7B-Instruct-v0.2 × RoleplayDeception × incentivised × layer-12 × mean linear probe × same-distribution train/test

in the ACL 2026 Findings paper or the pinned LASR code alone. The paper emphasizes **generalization failures** for deception under distribution shift, while same-distribution / same-strategy probe fits for deception-family settings are expected to be strong when labels are reliable. Goldowsky-Dill et al. report very high AUROCs on their (mostly 70B) evaluation suites, but that is not a direct Mistral-7B RoleplayDeception number.

**Interpretation:** Our AUROC ≈ 0.98 with a tight above-chance CI is consistent with a successful **in-distribution positive-control instrumentation check**, not a claim of robust real-world deception monitoring.

## 10. Deviations

- Implementation uses `input_formatted + model_outputs` rather than LASR’s `load_jsonl_data_prompts_included` + `apply_chat_template` path, because the HF JSONL `inputs` schema (system/user/assistant/assistant list) does not match that loader’s `inputs[0]/inputs[1]` assumption, and chat-template rebuild differs from stored `input_formatted`.
- Tokenization uses `add_special_tokens=False` to avoid double-BOS.
- One Modal preflight failed on a post-hook reproducibility bug and one on safetensors API misuse; both fixed before the canonical full run.
- Assumed Modal L40S price **`$1.95/hr`** for cost accounting (corrected from an earlier `$1.10/hr` documentation constant; costs recomputed from recorded durations without re-running GPU jobs).
- Low-FPR operating-point metric reports **maximum TPR among ROC points with FPR ≤ 1%**, plus the actual FPR at that point (`tpr_at_or_below_1pct_fpr`, `actual_fpr_at_tpr_metric`). It does not select points with FPR > 1%.

## 11. Leakage / limitations

- No train/test ID, text, or scenario-prefix leakage detected.
- Within-split duplicate dialogues exist upstream; Phase 2 does not re-split.
- RoleplayDeception has explicit role/incentive structure ⇒ contextual confound risk for spontaneous-intent claims.
- Phase 2 does **not** compare against text-only baselines (reserved for later phases).
- Phase 2 does **not** test generalization, onset, or causality.

## 12. Phase 2 verdict

**PASS**

- Pipeline/provenance checks pass after validation suite.
- Test AUROC 0.982 ≥ 0.80.
- Bootstrap CI clearly above chance.
- No implementation bug or train/test leakage found that would invalidate the positive-control interpretation.

**STOP:** Do not begin Phase 3 work until this PR is independently reviewed.


## Optional generation smoke test

Ran after primary PASS: 16 Roleplay prompts, temperature 0, `max_new_tokens=32`.

- Estimated cost: ~$0.0155 (at `$1.95/hr`, from recorded wall time; no re-run)
- Purpose: validate later generation code path only
- **Not** labelled; **not** Phase 2 scientific evidence
