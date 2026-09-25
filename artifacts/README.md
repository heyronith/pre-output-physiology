# Artifacts directory

Phase 1 produces no experimental run artifacts.

## Policy

- Every future experimental run receives a unique `run_id`.
- Primary raw artifacts must **never** be overwritten.
- Derived artifacts must retain provenance back to the originating raw run (see `docs/reproducibility.md`).
- Artifact hashes should be recorded in run manifests.

## Planned (not yet created)

- `runs/<run_id>/` — raw and derived outputs for a single run (gitignored)
- Manifests may be committed only when they contain no secrets and no primary activation tensors.

## Phase 3A (committed summaries only)

- `phase3a_summaries/` — commit-safe split/onset coverage summaries and onset audit **index** (IDs/hashes only).
- Raw onset audit text excerpts live under gitignored `data/processed/phase3_roleplay/onset_audit_raw/` (dataset license: do not commit full response/explanation text).

## Phase 4A (committed summaries only)

- `phase4a_summaries/` — condition matrix + frozen probe/surface **manifests** (hashes only).
- Raw probe/surface weight files live under gitignored `artifacts/phase4_models/`.
- Scenario/prompt JSONL design payloads live under gitignored `data/processed/phase4_design/`.

## Phase 4B pilot (committed summaries only)

- `phase4b_pilot/` — revision-0 generation manifest + behavior summary (no raw texts).
- `phase4b_pilot_revision1/` — revision-1 re-pilot summaries.
- Raw pilot outputs live under gitignored `artifacts/runs/phase4b_pilot_*/`.

## Phase 4C controlled-prefix pilot (committed summaries only)

- `phase4c_controlled_prefix_pilot/` — controlled-prefix generation + behavior summaries.
- Raw outputs under gitignored `artifacts/runs/phase4c_prefix_*/`.
- Revision-1 prompts unchanged; prefix token 12107 supplied (D058/D059).

## Phase 4D final behavior (committed summaries only)

- `phase4d_final_behavior/` — final generation manifest, behavior summary, frozen contrast eligibility.
- Raw final outputs under gitignored `artifacts/runs/phase4d_final_*/`.
- Conditions C1/C2/C3/C4/C6 only; C5 HOLD (D061/D062). No activations.

## Phase 4E specificity (committed summaries only)

- `phase4e_specificity/` — B1 compat preflight, extraction manifest, probe scores, contrast metrics.
- Raw activations under gitignored `artifacts/runs/phase4e_extract_*/`.
- Frozen Phase-3 probes only; eligibility unchanged; C5 HOLD (D063).

## Phase 4F natural-token diagnostic (committed summaries only)

- `phase4f_natural_token_diagnostic/` — natural greedy first-token k1 vs controlled-prefix comparison.
- Raw activations under gitignored `artifacts/runs/phase4f_natural_*/`.
- Diagnostic only (D064); no full response regeneration.

## Phase 5A design + pilot (committed summaries only)

- `phase5a_design/` — condition matrix + hashes (no raw scenario texts required in git).
- `phase5a_pilot_revision1/` — failed revision-1 free-generation pilot summaries (D071).
- `phase5a_pilot/` — revision-2 re-pilot summaries after the behavior pilot.
- Design JSONL under gitignored `data/processed/phase5_design/`.
- Raw pilot outputs under gitignored `artifacts/runs/phase5a_pilot_*/`.

## Phase 5B discovery behavior (committed summaries only)

- `phase5b_discovery_split/` — pre-generation family train/val split + primary all-pair populations + future probe procedure (D072/D073).
- `phase5b_discovery_behavior/` — discovery generation/behavior summaries + sensitivity valid-pair freeze.
- Raw discovery outputs under gitignored `artifacts/runs/phase5b_discovery_*/`.
- Locked families never generated.

## Phase 5C discovery physiology (committed summaries only)

- `phase5c_physiology_freeze/` — pre-extraction contracts (selection rule, logit formula, embedding pin).
- `phase5c_discovery_physiology/` — extraction manifest, physiology summary, selected probe npz + SHA256.
- Raw activations under gitignored `artifacts/runs/phase5c_extract_*/`.
- Locked families never extracted or scored.
