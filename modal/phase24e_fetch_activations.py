"""Modal: fetch Phase-24D/24B TRAIN+VALIDATION capture NPZs only (TEST forbidden).

Usage:

    uv run modal run modal/phase24e_fetch_activations.py
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "pre-output-physiology-phase24e-fetch"
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "artifacts/phase24e_discovery/activations_local"
SCHEDULE = REPO_ROOT / "artifacts/phase24d_collection/schedule.json"
SPLIT = REPO_ROOT / "artifacts/phase24c_design/split_manifest.json"
META24D = REPO_ROOT / "artifacts/phase24d_collection/capture_meta.json"
LABELS_TRAIN = REPO_ROOT / "artifacts/phase24d_collection/by_split/TRAIN/labels.json"
REUSE = REPO_ROOT / "artifacts/phase24d_collection/by_split/TRAIN/phase24b_reuse_index.json"

image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy==1.26.4")
app = modal.App(APP_NAME)
mistral_vol = modal.Volume.from_name("preoutput-mistral-cache", create_if_missing=True)
MODEL_CACHE = "/vol/hf_cache"
P24D_ROOT = "/vol/hf_cache/phase24d_captures"
P24B_ROOT = "/vol/hf_cache/phase24b_captures"


@app.function(
    image=image,
    timeout=60 * 60 * 2,
    volumes={MODEL_CACHE: mistral_vol},
    memory=8192,
)
def fetch_batch(rows_json: str) -> dict[str, bytes]:
    """Fetch NPZ bytes for TRAIN/VALIDATION only. Refuse TEST."""
    rows = json.loads(rows_json)
    out: dict[str, bytes] = {}
    for r in rows:
        if r["split"] not in ("train", "validation"):
            raise RuntimeError(f"TEST seal violation: split={r['split']}")
        path_s = str(r["path"]).replace("\\", "/")
        if (
            "LOCKED_TEST" in path_s
            or "labels_SEALED" in path_s
            or "/test/" in path_s.lower()
        ):
            raise RuntimeError(f"TEST seal violation: {path_s}")
        p = Path(r["path"])
        if not p.exists():
            raise FileNotFoundError(str(p))
        out[r["trajectory_id"]] = p.read_bytes()
    return out


@app.local_entrypoint()
def main() -> None:
    from pre_output_physiology.phase24e_discovery import (
        assert_not_test_path,
        assert_split_allowed,
    )

    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    train_ids = set(split["train_prompt_ids"])
    val_ids = set(split["validation_prompt_ids"])
    test_ids = set(split["test_prompt_ids"])
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    meta = json.loads(META24D.read_text(encoding="utf-8"))
    rows = []
    for m in meta["meta_rows"]:
        if not m.get("completed"):
            continue
        if m["prompt_id"] in test_ids or m["split"] == "test":
            continue  # never fetch TEST
        assert_split_allowed(m["split"])
        assert_not_test_path(m["artifact_path"])
        rows.append(
            {
                "trajectory_id": m["trajectory_id"],
                "split": m["split"],
                "path": m["artifact_path"],
                "source": "phase24d",
            }
        )

    reuse = json.loads(REUSE.read_text(encoding="utf-8"))
    for tid in reuse["trajectory_ids"]:
        path = f"{P24B_ROOT}/{tid}.npz"
        assert_not_test_path(path)
        rows.append(
            {
                "trajectory_id": tid,
                "split": "train",
                "path": path,
                "source": "phase24b",
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "TRAIN").mkdir(exist_ok=True)
    (OUT_DIR / "VALIDATION").mkdir(exist_ok=True)

    # chunk fetches
    chunk = 40
    fetched = 0
    for i in range(0, len(rows), chunk):
        batch = rows[i : i + chunk]
        print(f"fetch batch {i // chunk + 1} n={len(batch)}", flush=True)
        data = fetch_batch.remote(json.dumps(batch))
        for r in batch:
            blob = data[r["trajectory_id"]]
            folder = "TRAIN" if r["split"] == "train" else "VALIDATION"
            dest = OUT_DIR / folder / f"{r['trajectory_id']}.npz"
            assert_not_test_path(dest)
            dest.write_bytes(blob)
            fetched += 1
        print(f"  wrote {fetched}/{len(rows)}", flush=True)

    manifest = {
        "n_requested": len(rows),
        "n_fetched": fetched,
        "train_dir": str(OUT_DIR / "TRAIN"),
        "validation_dir": str(OUT_DIR / "VALIDATION"),
        "note": "TEST activations intentionally not fetched",
    }
    (OUT_DIR / "fetch_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
