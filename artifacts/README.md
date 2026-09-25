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
