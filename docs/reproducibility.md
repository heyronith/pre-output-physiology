# Reproducibility contract

This contract applies to **every experimental run after Phase 1 authorization**. Phase 1 itself produces no experimental runs.

## Required per-run metadata

Every experimental run later receives a manifest recording at least:

| Field | Description |
| --- | --- |
| `run_id` | Unique immutable identifier |
| `timestamp` | UTC timestamp for run start |
| `git_commit` | Full SHA of this repository |
| `model_id` | e.g. `mistralai/Mistral-7B-Instruct-v0.2` |
| `model_revision` | Exact Hugging Face commit (never silently `main`) |
| `dataset_revision` | Pinned dataset / label-generation revision |
| `config_file` | Path + content hash of the governing YAML |
| `random_seed` | Seed(s) for sampling and probe fitting |
| `hardware` | Machine class / provider |
| `gpu_type` | GPU model string |
| `software_versions` | Python, torch, transformers, CUDA, key libs |
| `precision` | e.g. `bfloat16`; quantization must be `null` for primary results |
| `generation_parameters` | temperature, top_p, max_new_tokens, stop rules, chat template version |
| `activation_locations_collected` | layers, positions/token regimes, hook points |
| `output_artifact_hashes` | Content hashes of primary raw outputs |

Optional but recommended:

- `label_pipeline_version`
- `judge_prompt_id` / judge model / judge settings (if used)
- `split_seed` and split membership file hash
- `parent_run_id` for derived re-analyses

## Artifact immutability

1. **Primary raw experimental artifacts must never be overwritten.**
2. Corrections create a **new** `run_id` (or a clearly marked amended raw revision) with linkage to the superseded artifact.
3. **Derived artifacts** (probes, metrics, plots) must retain provenance back to the raw run (`parent_run_id` / `source_run_id` + hashes).
4. Debugging runs that use quantized or non-canonical models must set `precision` / `quantization` fields explicitly and must be flagged `debug_only: true` so they cannot be mistaken for primary evidence.

## Scientific vs debug tracks

| Track | Allowed as primary evidence? |
| --- | --- |
| Modal (or equivalent) full-precision canonical forward pass | Yes, when authorized |
| Local quantized / approximate activations | **No** — debugging only |
| Surface-text baselines computed from the same trajectories | Yes, as baselines |

## Environment and tooling

- Python **3.11** via **uv**
- Config validation: `uv run python scripts/validate_phase1.py`
- Tests: `uv run pytest`
- Lint (if configured): `uv run ruff check .`

Phase 1 validation **must not** download the 7B model.

## Secrets

- Use `.env` locally; only `.env.example` is committed (names only).
- Never commit tokens, activation dumps with credentials, or Modal secrets.

## Upstream pins recorded in Phase 1

| Dependency | Pin |
| --- | --- |
| LASR-probe-gen (reference only; not vendored) | `f4c6ad69b10a5436a2e819c69009431802a0f5f7` |
| Primary HF model revision | `TO_BE_PINNED_BEFORE_PHASE2` |
