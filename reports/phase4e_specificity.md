# Phase 4E specificity report

**Extraction run ID:** `phase4e_extract_20260925T180943Z_ddb36132`  
**Git SHA (extraction):** `ddb36132bb2209119958abaf00dbb44e234dfcc2`  
**Status:** `phase4e_specificity_complete_awaiting_audit`  

## Scope

- Frozen Phase-3 probes only (no retrain / recalibrate)
- Primary endpoint: L12 / controlled-prefix k1
- Transfer/specificity test under supplied `Response` (12107) prefix
- C5 HOLD; eligibility sets unchanged from Phase 4D
- No causal interventions

## Compatibility preflight

- Min cosine L12/k0: `0.9999999999999252`
- Min cosine L12/k1: `0.9999999999999103`
- Min cosine all: `0.9999999999999103` (gate ≥ 0.9999)

## Primary: C3 vs C2 (L12/k1)

- Paired N: `148`
- AUROC: `0.4639` (95% CI `0.4396`–`0.4851`)
- Paired mean score diff C3−C2: `-0.0003` (95% CI `-0.0004`–`-0.0001`)

## Key secondary: C3 vs C4 (L12/k1)

- Paired N: `139`
- AUROC: `1.0000` (95% CI `1.0000`–`1.0000`)
- Paired mean score diff C3−C4: `0.0504` (95% CI `0.0475`–`0.0532`)

## All L12/k1 contrasts

| Contrast | N | AUROC | AUROC 95% CI | Mean Δ | Δ 95% CI |
| --- | ---: | ---: | --- | ---: | --- |
| C3_vs_C2 | 148 | 0.4639 | 0.4396–0.4851 | -0.0003 | -0.0004–-0.0001 |
| C3_vs_C4 | 139 | 1.0000 | 1.0000–1.0000 | 0.0504 | 0.0475–0.0532 |
| C3_vs_C1 | 164 | 1.0000 | 1.0000–1.0000 | 0.0557 | 0.0526–0.0588 |
| C3_vs_C6 | 162 | 1.0000 | 1.0000–1.0000 | 0.0807 | 0.0743–0.0871 |
| C2_vs_C1 | 203 | 1.0000 | 1.0000–1.0000 | 0.0577 | 0.0550–0.0605 |
| C4_vs_C1 | 180 | 0.5388 | 0.5170–0.5628 | 0.0033 | 0.0020–0.0047 |

## Secondary L12/k0 (prompt-boundary)

| Contrast | N | AUROC | AUROC 95% CI | Mean Δ | Δ 95% CI |
| --- | ---: | ---: | --- | ---: | --- |
| C3_vs_C2 | 148 | 0.4666 | 0.4532–0.4778 | -0.0094 | -0.0119–-0.0068 |
| C3_vs_C4 | 139 | 0.3306 | 0.2798–0.3801 | -0.0375 | -0.0484–-0.0269 |
| C3_vs_C1 | 164 | 0.5830 | 0.5368–0.6282 | 0.0322 | 0.0199–0.0440 |
| C3_vs_C6 | 162 | 0.1831 | 0.1433–0.2236 | -0.1067 | -0.1204–-0.0931 |
| C2_vs_C1 | 203 | 0.6064 | 0.5659–0.6466 | 0.0385 | 0.0277–0.0495 |
| C4_vs_C1 | 180 | 0.7440 | 0.7087–0.7813 | 0.0637 | 0.0554–0.0718 |

## Guarantees

NO PROBES WERE RETRAINED OR RECALIBRATED.  
NO ELIGIBILITY SETS WERE CHANGED.  
C5 REMAINED ON HOLD.  
NO CAUSAL INTERVENTIONS WERE PERFORMED.

