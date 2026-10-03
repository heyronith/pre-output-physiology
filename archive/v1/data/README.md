# Data directory

Phase 1 does not populate experimental datasets.

## Policy

- Do not place primary experimental trajectories here until Phase 2+ is authorized.
- Prefer scenario/group identifiers and versioned label-generation metadata over ad-hoc dumps.
- Large raw payloads should live in versioned object storage; this tree may hold manifests and pointers.
- Never overwrite a primary raw dataset in place; write a new revision.

## Planned (not yet created)

- `raw/` — immutable primary collections (gitignored)
- `processed/` — derived splits with provenance to raw revisions (gitignored)
- `external/` — third-party dataset snapshots or pointers (gitignored)

RoleplayDeception (positive control) will be referenced by pinned upstream revision in Phase 2 configs—not vendored into this repository during Phase 1.
