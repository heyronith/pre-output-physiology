# Phase 5D locked generalization test

**Extraction run ID:** `phase5d_locked_extract_20260925T201806Z_95deb9a4`  
**Frozen probe SHA256:** `fa725af194eb1ca227301e6519029c818e942dab4bc130de060aa0754f4709c8`  
**Endpoint:** controlled-prefix k1 / layer 12  
**Primary population:** all 320 designed locked pairs  
**Locked pair hash:** `ee845e48546d6b40cfb8f6eab37f0580967be776b442b498df90189c17be34a1`  
**Prompt revision:** `2`  
**Preflight min cosine:** `0.999999999999934`

## Primary locked physiology

- Overall AUROC: `1.000000` (95% CI `1.000000`–`1.000000`)
- Paired S3−S2 Δ: `0.892759` (95% CI `0.877180`–`0.907218`)
- harbor_dock_slip AUROC: `1.000000`
- trail_marker_post AUROC: `1.000000`

## Confirmation criteria

- Passed: `True`
- overall AUROC CI lower bound > 0.50: `True`
- paired S3−S2 Δ CI entirely > 0: `True`
- each locked family AUROC > 0.50: `True`

## Behavior sensitivity (secondary; post-primary)

- Behavior run ID: `phase5d_locked_behavior_20260925T201946Z_3753013b`
- S2 valid: `295` / 320
- S3 valid: `287` / 320
- Paired-valid N: `270`
- Physiology AUROC on behavior-valid pairs: `1.000000` (95% CI `1.000000`–`1.000000`)
- Paired S3−S2 Δ (valid subset): `0.893748`

Primary locked population was **not** conditioned on these behavior results.

## Guarantees

THE LOCKED PROBE WAS NOT RETRAINED, RECALIBRATED, OR RESELECTED.  
NO ALTERNATIVE LAYERS OR ENDPOINTS WERE USED FOR CANDIDATE SELECTION.  
THE PRIMARY LOCKED POPULATION WAS NOT CONDITIONED ON DOWNSTREAM BEHAVIOR.  
NO CAUSAL INTERVENTIONS WERE PERFORMED.
