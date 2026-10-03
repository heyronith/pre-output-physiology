# Phase 5C discovery physiology report

**Extraction run ID:** `phase5c_extract_20260925T194912Z_3947de30`  
**Selected candidate:** k1 / layer `12`  
**Selected probe SHA256:** `fa725af194eb1ca227301e6519029c818e942dab4bc130de060aa0754f4709c8`  
**Locked-test gate passed:** `True`  

## Preflight / integrity

- Repeatability min cosine: `0.9999999999851311`
- Finite activations: k0=`True` k1=`True`
- Locked families run: `False`

## Validation AUROC grid (all-pair)

| Layer | k0 | k1 |
| ---: | ---: | ---: |
| 0 | 0.5523 | 0.5031 |
| 4 | 0.9999 | 0.7296 |
| 8 | 0.9975 | 0.9857 |
| 12 | 1.0000 | 1.0000 |
| 16 | 1.0000 | 1.0000 |
| 20 | 1.0000 | 1.0000 |
| 24 | 1.0000 | 1.0000 |
| 28 | 1.0000 | 1.0000 |
| 31 | 1.0000 | 0.9910 |

## Selected k1 candidate

- AUROC: `1.0000` (95% CI `1.0000`–`1.0000`)
- Paired S3−S2 Δ: `0.9463` (95% CI `0.9381`–`0.9539`)
- Per-family AUROC: `{'gallery_wall_panel': 1.0, 'maker_bench_compartment': 1.0}`
- Behavior-valid sensitivity AUROC: `1.0000` (N_pairs=`275`)

## Baselines (validation)

- **user_visible_text_tfidf:** AUROC=`0.5000` (95% CI `0.5000`–`0.5000`)
- **full_privileged_context_tfidf:** AUROC=`0.8186` (95% CI `0.7925`–`0.8462`)
- **pretrained_semantic_embedding_logistic:** AUROC=`0.7209` (95% CI `0.7043`–`0.7418`)
- **output_logit_summary:** AUROC=`0.4059` (95% CI `0.3727`–`0.4381`)
- **first_token_identity_diagnostic:** Generated first token is constantly 12107 (Response); therefore it carries no S2/S3 class information.

## Descriptive physiology vs baselines

Selected k1 physiology AUROC=1.0000; user-visible TFIDF=0.5000; full privileged TFIDF=0.8186; semantic emb=0.7209; logit summary=0.4059. Physiology is not required to beat privileged full-context text.

## Guarantees

NO LOCKED GENERALIZATION FAMILY WAS RUN THROUGH THE MODEL OR SCORED.  
NO PHASE 5 PROMPTS OR BEHAVIOR RULES WERE CHANGED.  
NO POST-RESULT HYPERPARAMETER TUNING WAS PERFORMED.  
NO CAUSAL INTERVENTIONS WERE PERFORMED.

