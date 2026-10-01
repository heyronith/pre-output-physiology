"""Modal CPU: Phase 24G-R2 corrected LOPO + 500 nested full-pipeline diagnostics.

Prospective amendment to 500 nested reps recorded before corrected results.

Usage:

  # 1) Upload frozen inputs + seed prior LOPO checkpoint (laptop online once)
  uv run modal run modal/phase24g_r2_modal500.py::upload_inputs

  # 2) Detached compute (survives laptop sleep / disconnect)
  uv run modal run --detach modal/phase24g_r2_modal500.py::run_diagnostics

  # 3) Optional status / download
  uv run modal run modal/phase24g_r2_modal500.py::status
  uv run modal run modal/phase24g_r2_modal500.py::download_results
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import modal

APP_NAME = "pre-output-physiology-phase24g-r2-modal500"
VOLUME_NAME = "pre-output-physiology-phase24g-r2"
REPO_ROOT = Path(__file__).resolve().parents[1]
WORK = "/vol/work"
HF_CACHE = "/vol/hf_cache"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==1.26.4",
        "scikit-learn==1.5.2",
        "scipy==1.14.1",
        "transformers==4.44.2",
        "huggingface_hub==0.24.6",
        "tokenizers==0.19.1",
        "pyyaml==6.0.2",
        "safetensors==0.4.5",
        "filelock==3.16.1",
        "packaging==24.1",
        "regex==2024.9.11",
        "requests==2.32.3",
        "tqdm==4.66.5",
    )
    .env(
        {
            "HF_HOME": HF_CACHE,
            "TRANSFORMERS_CACHE": HF_CACHE,
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_python_source("pre_output_physiology")
    .add_local_dir("scripts", remote_path="/root/scripts")
)

app = modal.App(APP_NAME)
work_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
mistral_vol = modal.Volume.from_name("preoutput-mistral-cache", create_if_missing=True)


def _force_threads() -> None:
    for k in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[k] = "1"


def _git_commit_local() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
        .decode()
        .strip()
    )


# ---------------------------------------------------------------------------
# Worker globals (set in process initializer)
# ---------------------------------------------------------------------------
_G_CACHE: dict[str, dict] | None = None
_G_TOK = None
_G_DEV: list[dict] | None = None


def _init_pool(cache: dict, tok, dev: list[dict]) -> None:
    global _G_CACHE, _G_TOK, _G_DEV
    _force_threads()
    _G_CACHE = cache
    _G_TOK = tok
    _G_DEV = dev
    sys.path.insert(0, "/root/scripts")
    sys.path.insert(0, "/root")


def _run_one_discovery(tr_r, va_r, *, proxy: bool):
    from run_phase24g_diagnostics import discovery_delta_grid_fast

    from pre_output_physiology.phase24c_design import select_candidate_region
    from pre_output_physiology.phase24g_r2_modal500 import assert_no_proxy_in_primary

    if not proxy:
        assert_no_proxy_in_primary(False)
    grid = discovery_delta_grid_fast(
        tr_r, va_r, _G_CACHE, _G_TOK, proxy_delta=proxy
    )
    cand = select_candidate_region(grid)
    delta = float("nan")
    if cand is not None:
        delta = float(cand["layer_delta_auroc"])
    return cand, delta


def _nested_worker(payload: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    from run_phase24g_diagnostics import eval_coordinate

    from pre_output_physiology.phase24g_diagnostics import candidate_in_neighborhood
    from pre_output_physiology.phase24g_r2_modal500 import (
        assert_no_proxy_in_primary,
        seed_for_rep,
    )

    assert _G_CACHE is not None and _G_TOK is not None and _G_DEV is not None
    rep = int(payload["rep"])
    proxy = bool(payload["proxy"])
    if not proxy:
        assert_no_proxy_in_primary(False)
    tr_set, va_set, ho_set = set(payload["tr"]), set(payload["va"]), set(payload["ho"])
    tr_r = [r for r in _G_DEV if r["prompt_id"] in tr_set]
    va_r = [r for r in _G_DEV if r["prompt_id"] in va_set]
    ho_r = [r for r in _G_DEV if r["prompt_id"] in ho_set]
    cand, disc = _run_one_discovery(tr_r, va_r, proxy=proxy)
    seed = seed_for_rep(rep)
    if cand is None:
        return {
            "rep": rep,
            "rep_id": f"{rep:03d}",
            "seed": seed,
            "proxy": proxy,
            "candidate": None,
            "discovery_delta": float("nan"),
            "heldout_delta": float("nan"),
            "optimism": float("nan"),
            "split": {
                "tr": sorted(tr_set),
                "va": sorted(va_set),
                "ho": sorted(ho_set),
            },
            "in_neighborhood": False,
        }
    fit = tr_r + va_r
    hm = eval_coordinate(
        fit,
        ho_r,
        _G_CACHE,
        _G_TOK,
        t=int(cand["candidate_time"]),
        layer=int(cand["candidate_layer"]),
    )
    held = (
        float(hm["delta_auroc"])
        if not hm.get("skipped") and np.isfinite(hm.get("delta_auroc", np.nan))
        else float("nan")
    )
    opt = (
        disc - held
        if np.isfinite(disc) and np.isfinite(held)
        else float("nan")
    )
    return {
        "rep": rep,
        "rep_id": f"{rep:03d}",
        "seed": seed,
        "proxy": proxy,
        "candidate": {
            "time": cand["candidate_time"],
            "layer": cand["candidate_layer"],
            "band": cand["band"],
        },
        "discovery_delta": disc,
        "heldout_delta": held,
        "optimism": opt,
        "split": {
            "tr": sorted(tr_set),
            "va": sorted(va_set),
            "ho": sorted(ho_set),
        },
        "in_neighborhood": candidate_in_neighborhood(cand),
    }


def _lopo_worker(payload: dict[str, Any]) -> dict[str, Any]:
    from pre_output_physiology.phase24g_diagnostics import candidate_in_neighborhood
    from pre_output_physiology.phase24g_r2_modal500 import assert_no_proxy_in_primary

    assert _G_CACHE is not None and _G_TOK is not None and _G_DEV is not None
    proxy = bool(payload["proxy"])
    if not proxy:
        assert_no_proxy_in_primary(False)
    tr_set, va_set = set(payload["tr"]), set(payload["va"])
    tr_r = [r for r in _G_DEV if r["prompt_id"] in tr_set]
    va_r = [r for r in _G_DEV if r["prompt_id"] in va_set]
    cand, delta = _run_one_discovery(tr_r, va_r, proxy=proxy)
    return {
        "left_out_prompt": payload["left"],
        "left_out_split": payload["left_out_split"],
        "candidate": cand,
        "delta_auroc": delta,
        "exact_t1_l20": bool(
            cand
            and cand["candidate_time"] == 1
            and cand["candidate_layer"] == 20
        ),
        "in_neighborhood": candidate_in_neighborhood(cand),
        "proxy": proxy,
        "tr": payload["tr"],
        "va": payload["va"],
    }


def _make_nested_split(prompts: list[str], rep: int, seed: int) -> dict[str, list[str]]:
    import numpy as np

    rng = np.random.default_rng(seed)
    order = list(prompts)
    rng.shuffle(order)
    return {
        "tr": sorted(order[:14]),
        "va": sorted(order[14:21]),
        "ho": sorted(order[21:28]),
    }


def _write_progress(work: Path, manifest: dict[str, Any]) -> None:
    from pre_output_physiology.phase24g_r2_modal500 import atomic_write_json

    atomic_write_json(work / "progress_manifest.json", manifest)
    work_vol.commit()


@app.function(
    image=image,
    timeout=60 * 60 * 2,
    memory=32768,
    cpu=4,
    volumes={WORK: work_vol},
)
def upload_bundle(bundle_meta_json: str, files: dict[str, bytes]) -> dict[str, Any]:
    """Write a batch of files onto the work volume."""
    work = Path(WORK)
    meta = json.loads(bundle_meta_json)
    written = []
    for rel, blob in files.items():
        dest = work / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(blob)
        written.append(rel)
    work_vol.commit()
    return {"n_written": len(written), "meta": meta}


@app.function(
    image=image,
    timeout=60 * 60 * 8,
    memory=131072,
    cpu=16,
    volumes={WORK: work_vol, HF_CACHE: mistral_vol},
)
def run_diagnostics() -> dict[str, Any]:
    """Full LOPO 28 + nested 500 with 8 in-process workers; resumable."""
    _force_threads()
    sys.path.insert(0, "/root/scripts")
    sys.path.insert(0, "/root")

    from run_phase24g_diagnostics import (
        _load_npz,
        _load_tokenizer,
        _primary,
    )

    from pre_output_physiology.phase24g_r2_modal500 import (
        ANALYSIS_SEED,
        LOPO_N,
        N_WORKERS,
        NESTED_REPS,
        PROXY_COMPARE_NESTED,
        STATUS,
        assert_nested_reps_exact,
        assert_no_proxy_in_primary,
        assert_phase24f_immutable,
        atomic_write_json,
        config_hash,
        config_payload,
        lopo_path,
        nested_path,
        progress_manifest,
        rep_id,
        scan_completed_lopo,
        scan_completed_nested,
        seed_for_rep,
        verify_input_hashes,
    )

    assert_nested_reps_exact()
    work = Path(WORK)
    inputs = work / "inputs"
    repo_view = inputs / "repo"
    # Point diagnostics paths: we rewrite ACT_DEV via env/chdir layout
    # Mirror expected relative tree under inputs/repo
    os.chdir(str(repo_view))
    sys.path.insert(0, str(repo_view / "scripts"))
    sys.path.insert(0, str(repo_view / "src"))

    # Re-import path-sensitive modules after chdir by adjusting REPO_ROOT peers
    # Prefer volume repo_view as canonical
    if not (repo_view / "artifacts/phase24e_discovery/activations_local").exists():
        raise RuntimeError("STOP: activations missing on volume; run upload first")

    expected_hashes = json.loads(
        (work / "frozen_input_hashes.json").read_text(encoding="utf-8")
    )
    verify_input_hashes(expected_hashes, repo_view)
    assert_phase24f_immutable(repo_view)

    code_commit = (work / "code_commit.txt").read_text(encoding="utf-8").strip()
    cfg = config_payload()
    cfg_h = config_hash(cfg)
    atomic_write_json(work / "config.json", cfg)

    # Build records using volume layout: patch module-level paths
    import run_phase24g_diagnostics as diag

    diag.REPO_ROOT = repo_view
    diag.ACT_DEV = (
        repo_view / "artifacts/phase24e_discovery/activations_local"
    )
    diag.ACT_TEST = (
        repo_view
        / "artifacts/phase24f_confirmation/activations_local"
        / "LOCKED_TEST"
    )
    diag.SPLIT_PATH = repo_view / "artifacts/phase24c_design/split_manifest.json"
    diag.LABELS_TRAIN = (
        repo_view / "artifacts/phase24d_collection/by_split/TRAIN/labels.json"
    )
    diag.LABELS_VAL = (
        repo_view
        / "artifacts/phase24d_collection/by_split/VALIDATION/labels.json"
    )
    diag.LABELS_TEST = (
        repo_view
        / "artifacts/phase24d_collection/by_split/TEST/labels_SEALED.json"
    )
    diag.META24D = repo_view / "artifacts/phase24d_collection/capture_meta.json"
    diag.TRAJ24B = repo_view / "artifacts/phase24b_live/trajectories.jsonl"
    diag.REUSE = (
        repo_view
        / "artifacts/phase24d_collection/by_split/TRAIN/phase24b_reuse_index.json"
    )
    diag.SELECTED_C = repo_view / "artifacts/phase24e_discovery/selected_C.json"

    print("Loading tokenizer + DEVELOPMENT NPZs…", flush=True)
    tok_dir = work / "inputs" / "tokenizer"
    if tok_dir.exists() and (tok_dir / "tokenizer.json").exists():
        from transformers import AutoTokenizer

        print(f"  loading tokenizer from {tok_dir}", flush=True)
        tok = AutoTokenizer.from_pretrained(str(tok_dir), use_fast=True)
    else:
        print("  loading tokenizer from HF cache", flush=True)
        tok = _load_tokenizer()
    print("  tokenizer ready", flush=True)

    # DEVELOPMENT-only records (TEST NPZs not uploaded; not needed for LOPO/nested)
    def _build_dev_only(tokenizer):
        import json as _json

        from run_phase24g_diagnostics import _label_map

        split = _json.loads(diag.SPLIT_PATH.read_text(encoding="utf-8"))
        train_set = set(split["train_prompt_ids"])
        val_set = set(split["validation_prompt_ids"])
        labels = {}
        labels.update(_label_map(diag.LABELS_TRAIN, train_set))
        labels.update(_label_map(diag.LABELS_VAL, val_set))
        meta = _json.loads(diag.META24D.read_text(encoding="utf-8"))
        meta_by = {
            m["trajectory_id"]: m for m in meta["meta_rows"] if m.get("completed")
        }
        reuse_ids = set(
            _json.loads(diag.REUSE.read_text(encoding="utf-8"))["trajectory_ids"]
        )
        traj24b = {
            _json.loads(x)["trajectory_id"]: _json.loads(x)
            for x in diag.TRAJ24B.read_text(encoding="utf-8").splitlines()
            if x.strip()
        }
        by_split: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
        for tid, lab in labels.items():
            pid = lab["prompt_id"]
            if pid in train_set:
                sp, folder = "train", "TRAIN"
            elif pid in val_set:
                sp, folder = "validation", "VALIDATION"
            else:
                continue
            npz = diag.ACT_DEV / folder / f"{tid}.npz"
            if not npz.exists():
                raise FileNotFoundError(npz)
            if tid in meta_by:
                m = meta_by[tid]
                gen_ids = list(m["generated_token_ids"])
                prompt_text = tokenizer.decode(
                    m["prompt_token_ids"], skip_special_tokens=False
                )
            elif tid in reuse_ids:
                m = traj24b[tid]
                gen_ids = list(m["generated_token_ids"])
                prompt_text = tokenizer.decode(
                    m["prompt_token_ids"], skip_special_tokens=False
                )
            else:
                raise KeyError(tid)
            by_split[sp].append(
                {
                    "trajectory_id": tid,
                    "prompt_id": pid,
                    "split": sp,
                    "label": lab["label"],
                    "primary": lab["primary"],
                    "generated_token_ids": gen_ids,
                    "prompt_text": prompt_text,
                    "n_generated": len(gen_ids),
                    "npz_path": str(npz),
                }
            )
        return by_split

    by_split = _build_dev_only(tok)
    print(
        f"  records train={len(by_split['train'])} val={len(by_split['validation'])}",
        flush=True,
    )
    train, val = by_split["train"], by_split["validation"]
    # DEVELOPMENT only for LOPO/nested
    dev = train + val
    cache: dict[str, dict] = {}
    for i, r in enumerate(train + val):
        cache[r["trajectory_id"]] = _load_npz(Path(r["npz_path"]))
        if (i + 1) % 50 == 0:
            print(f"  mmap NPZ {i+1}/{len(train)+len(val)}", flush=True)
    print(f"  cache ready n={len(cache)}", flush=True)
    all_dev_prompts = sorted({r["prompt_id"] for r in _primary(dev)})
    assert len(all_dev_prompts) == LOPO_N
    print(f"  prompts={len(all_dev_prompts)}", flush=True)

    out_root = work / "results"
    out_root.mkdir(parents=True, exist_ok=True)

    def refresh_manifest(running: list[str], failed: list[dict]) -> dict[str, Any]:
        ns = scan_completed_nested(out_root)
        ls = scan_completed_lopo(out_root, all_dev_prompts, proxy=False)
        man = progress_manifest(
            code_commit=code_commit,
            config_hash=cfg_h,
            nested_scan=ns,
            lopo_scan=ls,
            failed=failed,
            running=running,
        )
        _write_progress(work, man)
        return man

    failed: list[dict[str, Any]] = []

    # ---- LOPO full ----
    print("=== LOPO full-pipeline ===", flush=True)
    lopo_scan = scan_completed_lopo(out_root, all_dev_prompts, proxy=False)
    # Seed prior 4/28 if present on volume and missing locally
    prior_seed = work / "seed_lopo_full_checkpoint.jsonl"
    if prior_seed.exists():
        from pre_output_physiology.phase24g_r2_modal500 import (
            verify_prior_checkpoint_full_pipeline,
        )

        seeded = verify_prior_checkpoint_full_pipeline(prior_seed)
        for row in seeded:
            left = row["left_out_prompt"]
            dest = lopo_path(out_root, left, proxy=False)
            if not dest.exists():
                atomic_write_json(dest, row)
                print(f"  seeded prior LOPO {left}", flush=True)
        work_vol.commit()
        lopo_scan = scan_completed_lopo(out_root, all_dev_prompts, proxy=False)

    missing_lopo = lopo_scan["missing"]
    print(f"  LOPO full: {lopo_scan['n_completed']}/28; missing={len(missing_lopo)}", flush=True)

    # Build payloads for missing LOPO
    train_pids = {r["prompt_id"] for r in train}
    lopo_payloads = []
    for i, left in enumerate(all_dev_prompts):
        if left not in missing_lopo:
            continue
        tr_r = [r for r in train if r["prompt_id"] != left]
        va_r = [r for r in val if r["prompt_id"] != left]
        if len({r["prompt_id"] for r in tr_r}) < 4 or len({r["prompt_id"] for r in va_r}) < 2:
            rem = [p for p in all_dev_prompts if p != left]
            import numpy as np

            rng = np.random.default_rng(ANALYSIS_SEED + i)
            rng.shuffle(rem)
            n_tr = max(4, int(0.7 * len(rem)))
            tr_set, va_set = set(rem[:n_tr]), set(rem[n_tr:])
        else:
            tr_set = {r["prompt_id"] for r in tr_r}
            va_set = {r["prompt_id"] for r in va_r}
        lopo_payloads.append(
            {
                "left": left,
                "left_out_split": "train" if left in train_pids else "validation",
                "tr": sorted(tr_set),
                "va": sorted(va_set),
                "proxy": False,
                "idx": i,
            }
        )

    if lopo_payloads:
        refresh_manifest([p["left"] for p in lopo_payloads], failed)
        with ProcessPoolExecutor(
            max_workers=min(N_WORKERS, len(lopo_payloads)),
            initializer=_init_pool,
            initargs=(cache, tok, dev),
        ) as pool:
            futs = {pool.submit(_lopo_worker, p): p for p in lopo_payloads}
            for fut in as_completed(futs):
                p = futs[fut]
                try:
                    row = fut.result()
                    assert_no_proxy_in_primary(False)
                    dest = lopo_path(out_root, row["left_out_prompt"], proxy=False)
                    if dest.exists():
                        # Never overwrite
                        print(f"  skip existing {dest.name}", flush=True)
                    else:
                        atomic_write_json(dest, row)
                    print(
                        f"  LOPO-full done {row['left_out_prompt']}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    failed.append({"kind": "lopo_full", "id": p["left"], "error": str(exc)})
                    print(f"  LOPO FAIL {p['left']}: {exc}", flush=True)
                refresh_manifest([], failed)

    lopo_scan = scan_completed_lopo(out_root, all_dev_prompts, proxy=False)
    if lopo_scan["n_completed"] != LOPO_N:
        refresh_manifest([], failed)
        return {
            "status": "incomplete_lopo",
            "lopo": lopo_scan["n_completed"],
            "failed": failed,
        }

    # ---- Nested 500 ----
    print("=== Nested full-pipeline 500 ===", flush=True)
    # Persist splits once
    splits_path = work / "nested_prompt_splits.json"
    if not splits_path.exists():
        splits = []
        for rep in range(NESTED_REPS):
            sp = _make_nested_split(all_dev_prompts, rep, seed_for_rep(rep))
            splits.append({"rep": rep, "rep_id": rep_id(rep), **sp})
        atomic_write_json(
            splits_path,
            {"analysis_seed": ANALYSIS_SEED, "n_reps": NESTED_REPS, "splits": splits},
        )
        work_vol.commit()
    splits_blob = json.loads(splits_path.read_text(encoding="utf-8"))
    splits_by_rep = {int(s["rep"]): s for s in splits_blob["splits"]}

    nested_scan = scan_completed_nested(out_root)
    missing_nested = nested_scan["missing"]
    print(
        f"  nested: {nested_scan['n_completed']}/{NESTED_REPS}; "
        f"missing={len(missing_nested)}",
        flush=True,
    )
    nested_payloads = []
    for rep in missing_nested:
        sp = splits_by_rep[rep]
        nested_payloads.append(
            {
                "rep": rep,
                "tr": sp["tr"],
                "va": sp["va"],
                "ho": sp["ho"],
                "proxy": False,
            }
        )

    if nested_payloads:
        refresh_manifest([rep_id(p["rep"]) for p in nested_payloads[:N_WORKERS]], failed)
        with ProcessPoolExecutor(
            max_workers=N_WORKERS,
            initializer=_init_pool,
            initargs=(cache, tok, dev),
        ) as pool:
            futs = {pool.submit(_nested_worker, p): p for p in nested_payloads}
            done_i = 0
            for fut in as_completed(futs):
                p = futs[fut]
                try:
                    row = fut.result()
                    dest = nested_path(out_root, int(row["rep"]))
                    if dest.exists():
                        print(f"  skip existing nested {dest.name}", flush=True)
                    else:
                        atomic_write_json(dest, row)
                    done_i += 1
                    if done_i % 5 == 0 or done_i == len(nested_payloads):
                        ns = scan_completed_nested(out_root)
                        print(
                            f"  nested-full {ns['n_completed']}/{NESTED_REPS}",
                            flush=True,
                        )
                        refresh_manifest([], failed)
                except Exception as exc:  # noqa: BLE001
                    failed.append(
                        {
                            "kind": "nested_full",
                            "id": rep_id(p["rep"]),
                            "error": str(exc),
                        }
                    )
                    print(f"  nested FAIL {p['rep']}: {exc}", flush=True)
                    refresh_manifest([], failed)

    nested_scan = scan_completed_nested(out_root)
    if nested_scan["n_completed"] != NESTED_REPS:
        man = refresh_manifest([], failed)
        return {
            "status": "incomplete_nested",
            "nested": nested_scan["n_completed"],
            "failed": failed,
            "manifest": man,
        }

    # ---- Proxy comparisons (secondary) ----
    print("=== Proxy LOPO comparison ===", flush=True)
    proxy_lopo_scan = scan_completed_lopo(out_root, all_dev_prompts, proxy=True)
    proxy_lopo_payloads = []
    # Use same tr/va as full LOPO artifacts
    for left in proxy_lopo_scan["missing"]:
        full_row = json.loads(
            lopo_path(out_root, left, proxy=False).read_text(encoding="utf-8")
        )
        proxy_lopo_payloads.append(
            {
                "left": left,
                "left_out_split": full_row["left_out_split"],
                "tr": full_row["tr"],
                "va": full_row["va"],
                "proxy": True,
            }
        )
    if proxy_lopo_payloads:
        with ProcessPoolExecutor(
            max_workers=min(N_WORKERS, len(proxy_lopo_payloads)),
            initializer=_init_pool,
            initargs=(cache, tok, dev),
        ) as pool:
            futs = {pool.submit(_lopo_worker, p): p for p in proxy_lopo_payloads}
            for fut in as_completed(futs):
                p = futs[fut]
                try:
                    row = fut.result()
                    dest = lopo_path(out_root, row["left_out_prompt"], proxy=True)
                    if not dest.exists():
                        atomic_write_json(dest, row)
                except Exception as exc:  # noqa: BLE001
                    failed.append({"kind": "lopo_proxy", "id": p["left"], "error": str(exc)})
                refresh_manifest([], failed)

    print("=== Proxy nested first 50 ===", flush=True)
    proxy_nested_dir = out_root / "nested_proxy"
    proxy_nested_dir.mkdir(parents=True, exist_ok=True)
    proxy_payloads = []
    for rep in range(PROXY_COMPARE_NESTED):
        dest = proxy_nested_dir / f"rep_{rep_id(rep)}.json"
        if dest.exists():
            continue
        sp = splits_by_rep[rep]
        proxy_payloads.append(
            {
                "rep": rep,
                "tr": sp["tr"],
                "va": sp["va"],
                "ho": sp["ho"],
                "proxy": True,
            }
        )
    if proxy_payloads:
        with ProcessPoolExecutor(
            max_workers=N_WORKERS,
            initializer=_init_pool,
            initargs=(cache, tok, dev),
        ) as pool:
            futs = {pool.submit(_nested_worker, p): p for p in proxy_payloads}
            for fut in as_completed(futs):
                p = futs[fut]
                try:
                    row = fut.result()
                    dest = proxy_nested_dir / f"rep_{rep_id(int(row['rep']))}.json"
                    if not dest.exists():
                        atomic_write_json(dest, row)
                except Exception as exc:  # noqa: BLE001
                    failed.append(
                        {
                            "kind": "nested_proxy",
                            "id": rep_id(p["rep"]),
                            "error": str(exc),
                        }
                    )
                refresh_manifest([], failed)

    man = refresh_manifest([], failed)
    man["status"] = STATUS
    atomic_write_json(work / "progress_manifest.json", man)
    atomic_write_json(
        work / "run_complete.json",
        {
            "status": STATUS,
            "lopo": LOPO_N,
            "nested": NESTED_REPS,
            "failed": failed,
            "code_commit": code_commit,
            "config_hash": cfg_h,
        },
    )
    work_vol.commit()
    return {"status": STATUS, "lopo": LOPO_N, "nested": NESTED_REPS, "failed": failed}


@app.function(
    image=image,
    timeout=60 * 10,
    memory=4096,
    cpu=1,
    volumes={WORK: work_vol},
)
def status() -> dict[str, Any]:
    work = Path(WORK)
    man_path = work / "progress_manifest.json"
    if not man_path.exists():
        return {"status": "no_progress_yet"}
    return json.loads(man_path.read_text(encoding="utf-8"))


@app.local_entrypoint()
def upload_inputs() -> None:
    """Upload frozen repo inputs + activations to Modal Volume."""
    from pre_output_physiology.phase24g_r2_modal500 import (
        STARTING_SHA,
        collect_input_hashes,
        config_hash,
        config_payload,
    )

    commit = _git_commit_local()
    if not commit.startswith(STARTING_SHA[:7]) and commit != STARTING_SHA:
        # Allow commits after starting SHA on this branch
        print(f"code_commit={commit} (starting={STARTING_SHA})", flush=True)

    hashes = collect_input_hashes(REPO_ROOT)
    cfg = config_payload()
    cfg_h = config_hash(cfg)

    # Prepare small metadata first
    meta_files: dict[str, bytes] = {
        "frozen_input_hashes.json": (
            json.dumps(hashes, indent=2, sort_keys=True) + "\n"
        ).encode(),
        "code_commit.txt": (commit + "\n").encode(),
        "config.json": (json.dumps(cfg, indent=2, sort_keys=True) + "\n").encode(),
        "config_hash.txt": (cfg_h + "\n").encode(),
    }
    prior = (
        REPO_ROOT
        / "artifacts/phase24g_r_correction/lopo_full_checkpoint.jsonl"
    )
    if prior.exists():
        meta_files["seed_lopo_full_checkpoint.jsonl"] = prior.read_bytes()

    print("Uploading metadata…", flush=True)
    upload_bundle.remote(json.dumps({"kind": "meta"}), meta_files)

    # Upload tree via modal CLI for large NPZs (more reliable for multi-GB)
    vol = VOLUME_NAME
    mappings = [
        (
            REPO_ROOT / "artifacts/phase24e_discovery/activations_local",
            "inputs/repo/artifacts/phase24e_discovery/activations_local",
        ),
        (
            REPO_ROOT / "artifacts/phase24e_discovery/selected_C.json",
            "inputs/repo/artifacts/phase24e_discovery/selected_C.json",
        ),
        (
            REPO_ROOT / "artifacts/phase24c_design",
            "inputs/repo/artifacts/phase24c_design",
        ),
        (
            REPO_ROOT / "artifacts/phase24d_collection",
            "inputs/repo/artifacts/phase24d_collection",
        ),
        (
            REPO_ROOT / "artifacts/phase24b_live/trajectories.jsonl",
            "inputs/repo/artifacts/phase24b_live/trajectories.jsonl",
        ),
        (
            REPO_ROOT / "artifacts/phase24f_confirmation/primary_metrics.json",
            "inputs/repo/artifacts/phase24f_confirmation/primary_metrics.json",
        ),
        (
            REPO_ROOT / "artifacts/phase24g_diagnostics",
            "inputs/repo/artifacts/phase24g_diagnostics",
        ),
        (
            REPO_ROOT / "artifacts/phase24g_r_correction",
            "inputs/repo/artifacts/phase24g_r_correction",
        ),
        (REPO_ROOT / "src", "inputs/repo/src"),
        (REPO_ROOT / "scripts", "inputs/repo/scripts"),
    ]
    # labels TEST sealed needed for _build_all_records even if we don't load TEST npz
    for local, remote in mappings:
        if not local.exists():
            print(f"SKIP missing {local}", flush=True)
            continue
        print(f"modal volume put {local} -> {remote}", flush=True)
        cmd = [
            "modal",
            "volume",
            "put",
            "--force",
            vol,
            str(local),
            remote,
        ]
        subprocess.check_call(cmd, cwd=str(REPO_ROOT))

    # Empty TEST act dir placeholder so path exists
    print("Upload complete.", flush=True)
    print(json.dumps({"volume": vol, "commit": commit, "config_hash": cfg_h}, indent=2))


@app.local_entrypoint()
def launch() -> None:
    """Detached entry: uv run modal run --detach modal/phase24g_r2_modal500.py::launch"""
    print("Launching run_diagnostics on Modal…", flush=True)
    out = run_diagnostics.remote()
    print(json.dumps(out, indent=2, default=str))


@app.local_entrypoint()
def main() -> None:
    """Default: print status. Use ::launch with --detach for compute."""
    print(json.dumps(status.remote(), indent=2))
