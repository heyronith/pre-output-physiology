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
