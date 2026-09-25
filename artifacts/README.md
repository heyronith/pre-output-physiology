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
