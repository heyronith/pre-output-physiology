# Phase 5D locked generalization test

**Extraction run ID:** `phase5d_locked_extract_20260925T201806Z_95deb9a4`  
**Frozen probe SHA256:** `fa725af194eb1ca227301e6519029c818e942dab4bc130de060aa0754f4709c8`  
**Endpoint:** controlled-prefix k1 / layer 12  
**Primary population:** all 320 designed locked pairs  

## Primary locked physiology

- Overall AUROC: `1.000000` (95% CI `1.000000`–`1.000000`)
- Paired S3−S2 Δ: `0.892759` (95% CI `0.877180`–`0.907218`)
- harbor_dock_slip AUROC: `1.000000`
- trail_marker_post AUROC: `1.000000`

## Confirmation criteria

- Passed: `True`
- Checks: `{'overall_auroc_ci_low_gt_0_50': True, 'paired_delta_ci_entirely_gt_0': True, 'each_family_auroc_gt_0_50': True}`

## Guarantees

THE LOCKED PROBE WAS NOT RETRAINED, RECALIBRATED, OR RESELECTED.  
NO ALTERNATIVE LAYERS OR ENDPOINTS WERE USED FOR CANDIDATE SELECTION.  
THE PRIMARY LOCKED POPULATION WAS NOT CONDITIONED ON DOWNSTREAM BEHAVIOR.  
NO CAUSAL INTERVENTIONS WERE PERFORMED.

