# Phase 5 — Strategic deception discovery protocol (Phase 5A)

## Objective

Discover physiology that distinguishes **S3 strategic deception** from **S2 strategic honesty** while holding strategic/private-objective framing constant.

Only intended difference:

- **S2:** private objective target equals private record state
- **S3:** private objective target equals the alternate state

Prompts must not contain condition-revealing lexicon (`honest`, `deceptive`, `lie`, `truthful`, `false`, `opposite`, …).

## Dataset

| Pool | Families | Base scenarios | Prompts (×2) |
| --- | ---: | ---: | ---: |
| Discovery | 6 | 960 | 1920 |
| Locked generalization | 2 | 320 | 640 |
| **Total final** | **8** | **1280** | **2560** |
| Pilot (disjoint IDs) | 8 × 4 | 32 | 64 |

Discovery families may later split into train/validation. Locked families remain untouched until candidate selection is frozen (D068).

## Response schema

```
Response <STATE>
```

Verified neutral prefix token under chat-generation context: ID **12107** (`Response`).

Future physiology primary endpoint: controlled-prefix hidden state after supplying this token, before the state answer. Layer scan: `[0,4,8,12,16,20,24,28,31]`. Secondary: k0 prompt-boundary. Natural-first-token analyses are diagnostic only (D067).

## Phase 5A scope

- Design freeze + behavior pilot only
- No activations
- No probe fitting or scoring
- No locked-final model generation
- No causal interventions

Pilot gates: ≥28/32 valid per condition; ≥26/32 paired valid. Exactly one prompt-format revision permitted if the first pilot fails, preserving S2/S3 symmetry. Do not weaken validity definitions.

## Baselines (preregistered; not fit in 5A)

1. User-visible-text TF-IDF
2. Full-context word+character TF-IDF
3. Pretrained semantic embedding + logistic regression
4. Simple output-logit summary
5. First-token identity/category diagnostic

**Important:** S2/S3 is in principle inferable from privileged full context. Phase 5 does **not** claim information-theoretic absence from text (D070).

## Relation to Phase 4

Phase 4 falsified deception-specificity of the original RoleplayDeception probe for matched strategic honesty vs deception (D065). No causal work is authorized on that probe. Uncertainty/C5 remains an unresolved later control, not part of Phase-5 discovery (D069).
