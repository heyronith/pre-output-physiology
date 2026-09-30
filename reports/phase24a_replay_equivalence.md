# Phase 24A — Live vs replay activation equivalence

**Status:** `phase24a_replay_equivalence_failed_live_recording_required`

**Instrumentation validation only.** Exact single-token deception onset remains
unvalidated (Phase 23D FAIL). This phase does not extract K=20 activations,
train probes/SAEs, grade responses, or make physiology claims.

PHASE 24A WAS AN INSTRUMENTATION VALIDATION ONLY. NO K=20 ACTIVATION DATASET WAS EXTRACTED, NO DECEPTION PROBE OR SAE WAS TRAINED, NO K>20 RESPONSES WERE GENERATED, NO NEW BEHAVIORAL LABELS WERE CREATED, AND NO CAUSAL OR PHYSIOLOGY CLAIM WAS MADE.

## Contract (frozen from K=20 artifacts)

| Setting | Value |
|---|---|
| Model | `mistralai/Mistral-7B-Instruct-v0.2` |
| Model revision | `63a8b081895390a26e140280378bc85ec8bce07a` |
| Tokenizer revision | `63a8b081895390a26e140280378bc85ec8bce07a` |
| dtype | `bfloat16` |
| Attention | `sdpa` |
| Chat template | `tokenizer.apply_chat_template via format_prompt_for_generation` |
| Temperature | `1.0` |
| do_sample | `True` |
| max_new_tokens | `200` |
| max_sequence_length | `32768` |
| Seed policy | `per_prompt deterministic phase24a seeds` |
| Hidden-state index | `hidden_states[layer + 1]` = post-block residual |

## Pilot

- n = 16
- selection = `evenly_spaced_indices_over_sha256_sorted_development_prompt_ids`
- prompt_ids_sha256 = `a814fad9f5e9e9bacd9c7d1d0a6c8a45107a969b7682d6f901b690d4928d951a`
- prompt IDs:

```
[
  "roleplay_141",
  "roleplay_262",
  "roleplay_029",
  "roleplay_169",
  "roleplay_065",
  "roleplay_138",
  "roleplay_313",
  "roleplay_062",
  "roleplay_339",
  "roleplay_349",
  "roleplay_200",
  "roleplay_164",
  "roleplay_159",
  "roleplay_000",
  "roleplay_075",
  "roleplay_291"
]
```

- requested generations: 16
- completed generations: 16

## Activation equivalence

- n activation comparisons (response × token × layer): **29248**

| Metric | Value |
|---|---|
| median cosine | 0.9999414157 |
| 1st-percentile cosine | 0.9997981653 |
| minimum cosine | 0.9982704660 |
| median relative L2 | 1.0876166588e-02 |
| 99th-percentile relative L2 | 2.0251422241e-02 |
| maximum relative L2 | 5.8980653940e-02 |

### Frozen gates

| Gate | Threshold | Observed | Result |
|---|---|---|---|
| median_cosine | 0.99999 | 0.99994142 | FAIL |
| p01_cosine | 0.9999 | 0.99979817 | FAIL |
| median_rel_l2 | 0.001 | 0.010876167 | FAIL |
| p99_rel_l2 | 0.01 | 0.020251422 | FAIL |

**Overall activation gates:** FAIL

### Layer / time diagnostics

- Worst median relative L2 by layer: layer 31 = 1.442544e-02
- Lowest median cosine by layer: layer 31 = 0.9998960262
- Heatmap array shape layer×gen_t = 32×64 (mean relative L2 over responses; NaNs where no observations).
- Full per-layer medians and heatmap array: `artifacts/phase24a_replay/summary.json`

### Per-response summary

See `summary.json` → `per_response` (median cosine / rel L2 / n_generated / stopping_reason).

## Logit diagnostics (not PASS/FAIL gates)

| Diagnostic | Value |
|---|---|
| median logit cosine | 0.9999751983 |
| min logit cosine | 0.9996316915 |
| median \|Δ log p(actual next)\| | 2.870560e-04 |
| max \|Δ log p(actual next)\| | 2.511177e-01 |
| top-1 agreement rate | 0.996659 (n=898) |

## K=20 label-only inventory (no onset; inventory only)

Primary labels: frozen GPT-4o reference. Rule: ≥2 honest and ≥2 deceptive; no onset.

| Split | n prompts | qualifying (≥2H & ≥2D) |
|---|---|---|
| DEVELOPMENT | 260 | 28 |
| LOCKED | 111 | 11 |

Does **not** influence replay PASS/FAIL. No new split. Stage-23D onset eligibility unused.

## Provenance hashes

```json
{
  "contract.json": "64afecb66d09e926d11fb07035301702146e7687da7af5798564de2a65bc8654",
  "pilot_manifest.json": "6301cb7c237204189eab475731714aeb05c7c10e7ed10b314750cb998d9f2b50",
  "k20_label_only_inventory.json": "6964451e0c1c70bd3cfab8de66f1fdec05c8abd452d47d5423790e6233581c4f",
  "summary.json": "e2a25168db44869079d52c8bf9533c8957f73695290da8109e59112235f9118f",
  "freeze.json": "0478088054f9d2b7274525813e33ded177db74fb04e3cde6b44f9dbd89489524",
  "live_activations_sha256": "0f8028711e36a6b6ecd879fe8f5ca519a8619f5017bb6bfbb66fbca571412800",
  "replay_activations_sha256": "490bf826a774cf2864535c028fb26750c491f05fd650aec72d9d7fb7bf5af405"
}
```

## Runtime

| | |
|---|---|
| GPU | A100-80GB |
| wall_seconds | 344.4399983882904 |
| estimated_cost_usd | 0.23919444332520165 |
| run_id | `phase24a_replay_20260930T170222Z_965d18e7` |
| git_commit (run) | `3c9cb26f81dfadb7544b59db9a430ec05dda459c` |

## Interpretation

Historical K=20 responses must not be treated as having reliably reconstructable activation trajectories. Future physiology must use activations recorded during live generation. STOP. Thresholds were not weakened.

## Authorizations after run

```json
{
  "k20_activation_extraction_authorized": false,
  "k_gt_20_generation_authorized": false,
  "modal_gpu_mistral_replay_pilot_authorized": false,
  "physiology_authorized": false,
  "probe_fitting_authorized": false,
  "sae_analysis_authorized": false
}
```
