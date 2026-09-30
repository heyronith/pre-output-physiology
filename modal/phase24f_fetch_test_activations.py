"""Modal: fetch Phase-24D LOCKED_TEST capture NPZs for confirmatory analysis.

Usage (only after frozen provenance verification):

    uv run modal run modal/phase24f_fetch_test_activations.py
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "pre-output-physiology-phase24f-fetch"
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "artifacts/phase24f_confirmation/activations_local"
SPLIT = REPO_ROOT / "artifacts/phase24c_design/split_manifest.json"
META24D = REPO_ROOT / "artifacts/phase24d_collection/capture_meta.json"
SEALED = REPO_ROOT / "artifacts/phase24d_collection/by_split/LOCKED_TEST/sealed_summary.json"

image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy==1.26.4")
app = modal.App(APP_NAME)
mistral_vol = modal.Volume.from_name("preoutput-mistral-cache", create_if_missing=True)
MODEL_CACHE = "/vol/hf_cache"


@app.function(
    image=image,
    timeout=60 * 60,
    volumes={MODEL_CACHE: mistral_vol},
    memory=8192,
)
def fetch_batch(rows_json: str) -> dict[str, bytes]:
    rows = json.loads(rows_json)
    out: dict[str, bytes] = {}
    for r in rows:
        if r["split"] != "test":
            raise RuntimeError(f"phase24f fetch is TEST-only; got split={r['split']}")
        p = Path(r["path"])
        if not p.exists():
            raise FileNotFoundError(str(p))
        out[r["trajectory_id"]] = p.read_bytes()
    return out


@app.local_entrypoint()
def main() -> None:
    from pre_output_physiology.phase24c_design import sha256_file
    from pre_output_physiology.phase24f_confirmation import verify_frozen_provenance

    ver = verify_frozen_provenance(REPO_ROOT)
    if not ver["verified"]:
        raise SystemExit(f"STOP: provenance failed before TEST fetch\n{ver}")

    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    test_ids = set(split["test_prompt_ids"])
    sealed = json.loads(SEALED.read_text(encoding="utf-8"))
    expected_hashes = sealed["artifact_hashes"]["trajectory_artifact_sha256s"]

    meta = json.loads(META24D.read_text(encoding="utf-8"))
    rows = []
    for m in meta["meta_rows"]:
        if not m.get("completed"):
            continue
        if m["split"] != "test" or m["prompt_id"] not in test_ids:
            continue
        rows.append(
            {
                "trajectory_id": m["trajectory_id"],
                "split": "test",
                "path": m["artifact_path"],
            }
        )

    if len(rows) != 176:
        raise SystemExit(f"expected 176 TEST trajectories, got {len(rows)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest_dir = OUT_DIR / "LOCKED_TEST"
    dest_dir.mkdir(exist_ok=True)

    chunk = 40
    fetched = 0
    for i in range(0, len(rows), chunk):
        batch = rows[i : i + chunk]
        print(f"fetch TEST batch {i // chunk + 1} n={len(batch)}", flush=True)
        data = fetch_batch.remote(json.dumps(batch))
        for r in batch:
            blob = data[r["trajectory_id"]]
            dest = dest_dir / f"{r['trajectory_id']}.npz"
            dest.write_bytes(blob)
            got = sha256_file(str(dest))
            exp = expected_hashes[r["trajectory_id"]]
            if got != exp:
                raise SystemExit(
                    f"STOP: TEST NPZ hash mismatch {r['trajectory_id']}: "
                    f"{got} != {exp}"
                )
            fetched += 1

    manifest = {
        "n_fetched": fetched,
        "split": "LOCKED_TEST",
        "dest": str(dest_dir.relative_to(REPO_ROOT)),
        "hash_verified_against_sealed_summary": True,
    }
    (OUT_DIR / "fetch_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
