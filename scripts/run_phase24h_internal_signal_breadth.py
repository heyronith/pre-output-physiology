#!/usr/bin/env python3
"""Phase 24H: exploratory internal-signal-breadth representation discovery.

DEVELOPMENT (TRAIN+VALIDATION, 28 prompts) ONLY. CPU only; no model generation.
Phase-24F stays the final confirmatory FAIL.

Usage:
  run_phase24h_internal_signal_breadth.py --hash-only
  run_phase24h_internal_signal_breadth.py --smoke      # infrastructure + timing only
  run_phase24h_internal_signal_breadth.py --full       # one scientific run
  run_phase24h_internal_signal_breadth.py --report-only
"""

from __future__ import annotations

import os
import sys

# Controlled BLAS/OpenMP threads BEFORE numpy/sklearn import.
_THREADS = "1"
if "--threads" in sys.argv:
    _THREADS = sys.argv[sys.argv.index("--threads") + 1]
for _k in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_k] = _THREADS
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # tokenizer only, from local cache

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
import resource  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology import phase24h_internal_signal_breadth as H  # noqa: E402
from pre_output_physiology.phase24c_design import sha256_file  # noqa: E402
from pre_output_physiology.phase24e_discovery import (  # noqa: E402
    SealViolationError,
    prefix_token_ids,
    verify_frozen_input_hashes,
)
from pre_output_physiology.phase24g_diagnostics import (  # noqa: E402
    assert_phase24f_immutable,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

OUT = REPO_ROOT / "artifacts/phase24h_internal_signal_breadth"
REPORT = REPO_ROOT / "reports/phase24h_internal_signal_breadth.md"
CFG = REPO_ROOT / "configs/experiments/phase24h_internal_signal_breadth.yaml"
ACT_DEV = REPO_ROOT / H.ACT_DEV_REL
SPLIT_PATH = REPO_ROOT / "artifacts/phase24c_design/split_manifest.json"
LABELS_TRAIN = REPO_ROOT / "artifacts/phase24d_collection/by_split/TRAIN/labels.json"
LABELS_VAL = REPO_ROOT / "artifacts/phase24d_collection/by_split/VALIDATION/labels.json"
META24D = REPO_ROOT / "artifacts/phase24d_collection/capture_meta.json"
TRAJ24B = REPO_ROOT / "artifacts/phase24b_live/trajectories.jsonl"
REUSE = REPO_ROOT / "artifacts/phase24d_collection/by_split/TRAIN/phase24b_reuse_index.json"
PROTOCOL_FILES = (
    "configs/experiments/phase24h_internal_signal_breadth.yaml",
    "src/pre_output_physiology/phase24h_internal_signal_breadth.py",
    "scripts/run_phase24h_internal_signal_breadth.py",
    "tests/test_phase24h_internal_signal_breadth.py",
)
EXPECTED_MODEL = {
    "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
    "model_revision": "63a8b081895390a26e140280378bc85ec8bce07a",
    "tokenizer_revision": "63a8b081895390a26e140280378bc85ec8bce07a",
    "dtype": "bfloat16",
}

_G: dict[str, Any] = {}  # inherited by forked workers


# ---- config / git ----
def load_config() -> dict[str, Any]:
    return yaml.safe_load(CFG.read_text(encoding="utf-8"))


def config_matches_library(cfg: dict[str, Any]) -> list[str]:
    """Return list of mismatches between frozen YAML and library constants."""
    bad: list[str] = []

    def chk(name: str, a: Any, b: Any) -> None:
        if a != b:
            bad.append(f"{name}: config={a!r} library={b!r}")

    chk("early_times", tuple(cfg["times"]["early_times"]), H.EARLY_TIMES)
    chk("n_dev_prompts", cfg["data"]["n_dev_prompts"], H.N_DEV_PROMPTS)
    chk("salts", tuple(cfg["outer_cross_fitting"]["salts"]), H.OUTER_SALTS)
    chk("n_folds", cfg["outer_cross_fitting"]["n_folds"], H.N_OUTER_FOLDS)
    chk("inner_folds", cfg["outer_cross_fitting"]["inner_cv"]["n_folds"], H.INNER_FOLDS)
    chk("probe_C", cfg["layer_probe"]["C"], H.PROBE_C)
    chk("surface_C_text", cfg["surface"]["frozen_C_text"], H.SURFACE_C_TEXT)
    chk("surface_C_logits", cfg["surface"]["frozen_C_logits"], H.SURFACE_C_LOGITS)
    chk("surface_C_meta", cfg["surface"]["frozen_C_meta"], H.SURFACE_C_META)
    chk("combined_C", cfg["combination"]["frozen_C"], H.COMBINED_C)
    chk("meta_C_grid", tuple(cfg["families"]["C_MULTILAYER"]["meta_C_grid"]), H.META_C_GRID)
    chk("pca_dims", tuple(cfg["families"]["E_LOWDIM"]["pca_dims"]), H.PCA_DIMS)
    chk("pca_logit_C", cfg["families"]["E_LOWDIM"]["logistic_C"], H.PCA_LOGIT_C)
    chk("svm_C_grid", tuple(cfg["families"]["F_NONLINEAR"]["C_grid"]), H.SVM_C_GRID)
    chk("n_bootstrap", cfg["headline_estimand"]["n_bootstrap"], H.N_BOOTSTRAP)
    chk("n_shuffles", cfg["negative_control"]["n_shuffles"], H.N_SHUFFLES)
    chk("shuffle_salts", tuple(cfg["negative_control"]["outer_salts_used"]), H.SHUFFLE_SALTS)
    chk("min_fraction", cfg["promotion"]["min_fraction_positive_prompts"],
        H.MIN_FRACTION_POSITIVE_PROMPTS)
    chk("seed", cfg["analysis_seed"], H.ANALYSIS_SEED)
    chk("status", cfg["status_on_completion"], H.STATUS)
    chk("starting_sha", cfg["starting_sha"], H.STARTING_SHA)
    return bad


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), *args], text=True).strip()


def git_state() -> dict[str, Any]:
    dirty = _git("status", "--porcelain", "--", *PROTOCOL_FILES)
    untracked_or_modified = [ln for ln in dirty.splitlines() if ln.strip()]
    first = _git("log", "--reverse", "--format=%H", "--", PROTOCOL_FILES[0]).splitlines()
    return {
        "head": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "protocol_files_dirty": untracked_or_modified,
        "protocol_freeze_commit": first[0] if first else None,
        "protocol_file_sha256": {p: sha256_file(str(REPO_ROOT / p)) for p in PROTOCOL_FILES
                                 if (REPO_ROOT / p).exists()},
    }


# ---- records ----
def _label_map(path: Path, allowed: set[str]) -> tuple[dict[str, dict], int]:
    H.assert_dev_input_path(path)
    blob = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    n_ambiguous = 0
    for r in blob["rows"]:
        if r["prompt_id"] not in allowed:
            continue
        if not r.get("open_valid"):
            continue
        cls = r.get("open_class")
        if cls == "ambiguous":
            n_ambiguous += 1
        elif cls in ("honest", "deceptive"):
            out[r["trajectory_id"]] = {"label": cls, "prompt_id": r["prompt_id"]}
    return out, n_ambiguous


def build_records(tok) -> tuple[list[dict], dict[str, Any]]:
    """DEV honest/deceptive records with metadata verification. Never touches TEST."""
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    train_set = set(split["train_prompt_ids"])
    val_set = set(split["validation_prompt_ids"])
    lab_tr, amb_tr = _label_map(LABELS_TRAIN, train_set)
    lab_va, amb_va = _label_map(LABELS_VAL, val_set)
    labels = {**lab_tr, **lab_va}

    meta = json.loads(META24D.read_text(encoding="utf-8"))
    meta_by = {m["trajectory_id"]: m for m in meta["meta_rows"] if m.get("completed")}
    reuse_ids = set(json.loads(REUSE.read_text(encoding="utf-8"))["trajectory_ids"])
    traj24b = {
        json.loads(x)["trajectory_id"]: json.loads(x)
        for x in TRAJ24B.read_text(encoding="utf-8").splitlines()
        if x.strip()
    }
    records: list[dict] = []
    meta_problems: list[str] = []
    for tid in sorted(labels):
        lab = labels[tid]
        pid = lab["prompt_id"]
        folder = "TRAIN" if pid in train_set else "VALIDATION"
        npz = ACT_DEV / folder / f"{tid}.npz"
        H.assert_dev_activation_path(npz, REPO_ROOT)
        if not npz.exists():
            raise FileNotFoundError(npz)
        if tid in meta_by:
            m = meta_by[tid]
            src = "phase24d_new"
        elif tid in reuse_ids:
            m = traj24b[tid]
            src = "phase24b_reuse"
        else:
            raise KeyError(tid)
        for k, v in EXPECTED_MODEL.items():
            if m.get(k) != v:
                meta_problems.append(f"{tid}:{k}={m.get(k)!r}")
        if m.get("split") not in (None, "train", "validation"):
            meta_problems.append(f"{tid}:split={m.get('split')!r}")
        gen_ids = list(m["generated_token_ids"])
        records.append(
            {
                "trajectory_id": tid,
                "prompt_id": pid,
                "label": lab["label"],
                "n_generated": len(gen_ids),
                "generated_token_ids": gen_ids,
                "prompt_text": tok.decode(m["prompt_token_ids"], skip_special_tokens=False),
                "npz_path": str(npz),
                "source": src,
                "recorded_sha256": m.get("artifact_sha256"),
            }
        )
    if meta_problems:
        raise RuntimeError(f"model/tokenizer metadata mismatch: {meta_problems[:5]}")
    prompts = sorted({r["prompt_id"] for r in records})
    H.assert_dev_prompts(prompts, split)
    summary = {
        "n_records": len(records),
        "n_prompts": len(prompts),
        "n_honest": sum(r["label"] == "honest" for r in records),
        "n_deceptive": sum(r["label"] == "deceptive" for r in records),
        "n_ambiguous_excluded": amb_tr + amb_va,
        "n_train_prompts": len(train_set),
        "n_validation_prompts": len(val_set),
        "expected_model_metadata": EXPECTED_MODEL,
        "metadata_verified": True,
        "test_labels_opened": False,
    }
    return records, summary


def _load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        EXPECTED_MODEL["model_id"], revision=EXPECTED_MODEL["model_revision"], use_fast=True
    )


def hash_dev_inputs(records: list[dict], workers: int, limit: int | None = None) -> dict:
    """SHA256 of the CONTENTS of every DEV NPZ (TRAIN+VALIDATION dirs)."""
    files: list[Path] = []
    for sub in H.ALLOWED_ACT_SUBDIRS:
        for p in sorted((ACT_DEV / sub).glob("*.npz")):
            H.assert_dev_activation_path(p, REPO_ROOT)
            files.append(p)
    if limit:
        files = files[:limit]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        hashes = list(ex.map(lambda p: sha256_file(str(p)), files))
    by_name = {p.name: h for p, h in zip(files, hashes, strict=True)}
    used = {Path(r["npz_path"]).name: r for r in records}
    n_match = n_recorded = 0
    mismatches = []
    for name, r in used.items():
        if name in by_name and r.get("recorded_sha256"):
            n_recorded += 1
            if by_name[name] == r["recorded_sha256"]:
                n_match += 1
            else:
                mismatches.append(name)
    agg = hashlib.sha256(
        "\n".join(
            f"{p.parent.name}/{p.name} {h}" for p, h in zip(files, hashes, strict=True)
        ).encode()
    ).hexdigest()
    return {
        "n_files_hashed": len(files),
        "aggregate_sha256": agg,
        "n_used_with_recorded_sha256": n_recorded,
        "n_used_recorded_sha256_match": n_match,
        "recorded_sha256_mismatches": mismatches,
        "files": {f"{p.parent.name}/{p.name}": h for p, h in zip(files, hashes, strict=True)},
        "hashed_at": utc_now_iso(),
    }


def load_dev_data(records: list[dict], tok) -> H.DevData:
    n = len(records)
    first = H.load_dev_npz_arrays(records[0]["npz_path"], REPO_ROOT)
    n_layers, dim = first["h0"].shape
    h0 = np.zeros((n, n_layers, dim), dtype=np.float32)
    hs = {t: np.zeros((n, n_layers, dim), dtype=np.float32) for t in H.EARLY_TIMES}
    vocab = next(iter(first["logits"].values())).shape[0]
    lg = {t: np.zeros((n, vocab), dtype=np.float32) for t in H.EARLY_TIMES}
    texts: dict[int, list[str]] = {t: [""] * n for t in H.EARLY_TIMES}
    for i, r in enumerate(records):
        a = first if i == 0 else H.load_dev_npz_arrays(r["npz_path"], REPO_ROOT)
        if a["n_generated"] != r["n_generated"]:
            raise RuntimeError(f"n_generated mismatch for {r['trajectory_id']}")
        h0[i] = a["h0"]
        for t in H.EARLY_TIMES:
            if r["n_generated"] > t:
                hs[t][i] = a["h"][t]
                lg[t][i] = a["logits"][t]
                pref = prefix_token_ids(r["generated_token_ids"], t)
                texts[t][i] = r["prompt_text"] + (
                    tok.decode(pref, skip_special_tokens=True) if pref else ""
                )
        if (i + 1) % 100 == 0:
            print(f"  loaded {i + 1}/{n}", flush=True)
    return H.DevData(
        tids=[r["trajectory_id"] for r in records],
        prompt_ids=[r["prompt_id"] for r in records],
        labels=[r["label"] for r in records],
        n_generated=np.asarray([r["n_generated"] for r in records]),
        texts=texts,
        logits=lg,
        h0=h0,
        h=hs,
    )


# ---- workers ----
def _task_real(args: tuple[int, int]) -> tuple[int, int, H.FoldOutput, float]:
    salt, fold = args
    data: H.DevData = _G["data"]
    tr, ev = H.train_eval_split(_G["dev_prompts"], salt, fold)
    t0 = time.perf_counter()
    out = H.run_outer_fold(data, tr, ev)
    return salt, fold, out, time.perf_counter() - t0


def _task_control(args: tuple) -> tuple[int, int, int, str, dict, float]:
    rep, salt, fold, gkey, variant, times, surface = args
    data: H.DevData = _G["data"]
    tr, ev = H.train_eval_split(_G["dev_prompts"], salt, fold)
    act_map = H.shuffle_permutation(data.prompt_ids, data.survival_level(), rep)
    t0 = time.perf_counter()
    out = H.run_outer_fold(
        data, tr, ev, act_map=act_map, variants=(variant,), times=times,
        surface_cache=dict(surface),
    )
    return rep, salt, fold, gkey, out.rows, time.perf_counter() - t0


def _pool(workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("fork")
    )


def _rss_mb() -> dict[str, float]:
    div = 1024 * 1024  # macOS reports bytes
    if sys.platform != "darwin":
        div = 1024
    return {
        "self_max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / div,
        "children_max_rss_mb": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / div,
    }


# ---- preflight ----
def preflight(require_clean: bool) -> dict[str, Any]:
    cfg = load_config()
    bad = config_matches_library(cfg)
    if bad:
        raise RuntimeError(f"config/library mismatch: {bad}")
    imm = assert_phase24f_immutable(REPO_ROOT)
    tree = H.phase24f_tree_hash(REPO_ROOT)
    verify = verify_frozen_input_hashes(REPO_ROOT)
    if not verify["verified"]:
        raise RuntimeError("frozen input hash verification failed")
    # guard self-test: the primary code must raise on TEST / sealed / 24F activations
    for bad_path in (
        REPO_ROOT / "artifacts/phase24f_confirmation/activations_local/LOCKED_TEST/x.npz",
        REPO_ROOT / "artifacts/phase24d_collection/by_split/LOCKED_TEST/labels_SEALED.json",
    ):
        try:
            H.assert_dev_activation_path(bad_path, REPO_ROOT)
        except SealViolationError:
            continue
        raise RuntimeError(f"path guard failed to reject {bad_path}")
    git = git_state()
    if require_clean and git["protocol_files_dirty"]:
        raise RuntimeError(f"protocol files not committed: {git['protocol_files_dirty']}")
    if require_clean and not git["protocol_freeze_commit"]:
        raise RuntimeError("protocol config has no commit: freeze before scientific run")
    return {"immutability": imm, "phase24f_tree_hash": tree, "input_verification": verify,
            "git": git, "config_ok": True}


# ---- smoke ----
def run_smoke(workers: int) -> int:
    smoke_dir = OUT / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    pre = preflight(require_clean=True)
    print("preflight ok; freeze commit", pre["git"]["protocol_freeze_commit"], flush=True)
    t0 = time.perf_counter()
    tok = _load_tokenizer()
    records, summary = build_records(tok)
    t_rec = time.perf_counter() - t0
    t0 = time.perf_counter()
    hsh = hash_dev_inputs(records, workers=4, limit=24)
    t_hash_24 = time.perf_counter() - t0
    n_all = len(list((ACT_DEV / "TRAIN").glob("*.npz"))) + len(
        list((ACT_DEV / "VALIDATION").glob("*.npz"))
    )
    t0 = time.perf_counter()
    data = load_dev_data(records, tok)
    t_load = time.perf_counter() - t0
    dev_prompts = sorted(set(data.prompt_ids))
    _G.update(data=data, dev_prompts=dev_prompts)
    surv = {t: int(data.survivors(t).sum()) for t in H.EARLY_TIMES}

    t0 = time.perf_counter()
    salt, fold = 0, 0
    _, _, out, fold_s = _task_real((salt, fold))
    n_rows = len(out.rows)
    n_finite = int(sum(np.isfinite(r.comb).all() for r in out.rows.values()))

    # shuffle infrastructure: 2 reps on a single (variant,time) group, fold 0
    ctl_times = []
    tr, ev = H.train_eval_split(dev_prompts, salt, fold)
    for rep in range(2):
        a = (rep, salt, fold, "RAW|t1", "RAW", (1,), out.surface)
        r = _task_control(a)
        ctl_times.append(r[5])
    # tiny metric/promotion plumbing check (values are NOT reported)
    preds = H.assemble_predictions(data, [(salt, fold, out)])
    key = "C_MULTILAYER|RAW|t1"
    _ = H.row_metrics({0: preds[key][0]}, n_boot=10, seed=1)

    # extrapolation (single-process seconds -> wall with `workers`, 0.75 efficiency)
    eff = max(1.0, workers * 0.75)
    n_real = len(H.OUTER_SALTS) * H.N_OUTER_FOLDS
    est_real_h = fold_s * n_real / eff / 3600
    probes = {k: v for k, v in out.timings.items() if k.startswith("probes_RAW_t1")}
    per_group_time = float(np.mean(ctl_times))  # one variant, one time, one fold
    # per-fold cost of an all-times single-variant control group (SURFACE is cached)
    group_all_times = sum(v for k, v in out.timings.items() if k.startswith("variant_RAW_"))
    ctl_fold_all_times = group_all_times if group_all_times > 0 else per_group_time * 4
    n_ctl_tasks = H.N_SHUFFLES * H.N_OUTER_FOLDS * len(H.SHUFFLE_SALTS)
    est_control_per_allgroup_h = ctl_fold_all_times * n_ctl_tasks / eff / 3600
    est_control_per_single_h = per_group_time * n_ctl_tasks / eff / 3600
    est_control_worst_h = 2 * est_control_per_allgroup_h  # both variants, all rows promoted
    est_hash_all_min = t_hash_24 / 24 * n_all / 60
    est_total_worst_h = (
        est_real_h + est_control_worst_h + (t_rec + t_load) / 3600 + est_hash_all_min / 60
    )
    bench = {
        "analysis_label": "SMOKE_INFRASTRUCTURE_ONLY_NO_SCIENTIFIC_CONCLUSIONS",
        "n_records": summary["n_records"],
        "survivors_by_time": surv,
        "fold0_wall_s": fold_s,
        "fold0_timings_s": out.timings,
        "n_rows_produced": n_rows,
        "n_rows_all_finite": n_finite,
        "expected_rows": len(H.row_keys_all()),
        "record_build_s": t_rec,
        "data_load_s": t_load,
        "hash_24_files_s": t_hash_24,
        "n_dev_npz_files": n_all,
        "est_hash_all_min": est_hash_all_min,
        "control_task_s_single_variant_single_time": ctl_times,
        "workers_assumed": workers,
        "parallel_efficiency_assumed": 0.75,
        "est_real_stage_hours": est_real_h,
        "est_control_hours_per_single_group": est_control_per_single_h,
        "est_control_hours_per_variant_all_times_group": est_control_per_allgroup_h,
        "est_control_hours_worst_case_all_promoted": est_control_worst_h,
        "est_total_hours_worst_case": est_total_worst_h,
        "est_total_hours_no_promotion": est_real_h + (t_rec + t_load) / 3600
        + est_hash_all_min / 60,
        "memory": _rss_mb(),
        "smoke_wall_s": time.perf_counter() - t_all,
        "stop_threshold_hours": 8,
        "hash_sample_ok": hsh["recorded_sha256_mismatches"] == [],
        "probe_example_timings": probes,
        "git": pre["git"],
    }
    write_json(smoke_dir / "smoke_benchmark.json", bench)
    print(json.dumps({k: v for k, v in bench.items() if k not in ("git", "fold0_timings_s")},
                     indent=2, default=str), flush=True)
    return 0


# ---- full run ----
def run_full(workers: int, force: bool) -> int:
    t_start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "row_metrics.json").exists() and not force:
        raise RuntimeError("full results already exist; scientific run is once-only (--force)")
    pre = preflight(require_clean=True)
    write_json(OUT / "phase24f_immutability.json",
               {**pre["immutability"], "tree_hash_before": pre["phase24f_tree_hash"]})
    write_json(OUT / "frozen_input_verification.json", pre["input_verification"])
    print("preflight ok", flush=True)

    tok = _load_tokenizer()
    records, summary = build_records(tok)
    write_json(OUT / "input_summary.json", summary)
    print("hashing DEV NPZ contents…", flush=True)
    t0 = time.perf_counter()
    hsh = hash_dev_inputs(records, workers=4)
    if hsh["recorded_sha256_mismatches"]:
        bad = hsh["recorded_sha256_mismatches"][:3]
        raise RuntimeError(f"NPZ hash mismatch vs recorded: {bad}")
    write_json(OUT / "input_hashes.json", hsh)
    t_hash = time.perf_counter() - t0
    print(f"hashed {hsh['n_files_hashed']} files in {t_hash:.0f}s", flush=True)

    print("loading arrays…", flush=True)
    t0 = time.perf_counter()
    data = load_dev_data(records, tok)
    t_load = time.perf_counter() - t0
    dev_prompts = sorted(set(data.prompt_ids))
    assert len(dev_prompts) == H.N_DEV_PROMPTS
    _G.update(data=data, dev_prompts=dev_prompts)

    folds_blob = {
        str(s): [sorted(f) for f in H.outer_folds(dev_prompts, s)] for s in H.OUTER_SALTS
    }
    write_json(OUT / "outer_folds.json", folds_blob)

    # ---- real cross-fitting
    tasks = [(s, f) for s in H.OUTER_SALTS for f in range(H.N_OUTER_FOLDS)]
    fold_outputs: list[tuple[int, int, H.FoldOutput]] = []
    fold_times: dict[str, float] = {}
    t0 = time.perf_counter()
    with _pool(workers) as ex:
        for salt, fold, out, secs in ex.map(_task_real, tasks):
            fold_outputs.append((salt, fold, out))
            fold_times[f"salt{salt}_fold{fold}"] = secs
            print(f"  real salt={salt} fold={fold} {secs:.0f}s", flush=True)
    t_real = time.perf_counter() - t0
    surface_by_fold = {(s, f): o.surface for s, f, o in fold_outputs}

    preds = H.assemble_predictions(data, fold_outputs)
    # every trajectory predicted once per salt for fixed-time rows
    coverage = {
        k: {str(s): int(len(v["rows"])) for s, v in by.items()} for k, by in preds.items()
    }

    # ---- metrics
    metrics: dict[str, Any] = {}
    t0 = time.perf_counter()
    for key in H.row_keys_all():
        if key not in preds or set(preds[key]) != set(H.OUTER_SALTS):
            metrics[key] = {
                "missing": True,
                "promotion": {"promoted": False, "label": H.NOT_PROMOTED},
            }
            continue
        seed = H.derive_seed("bootstrap", key)
        metrics[key] = H.row_metrics(preds[key], H.N_BOOTSTRAP, seed)
        metrics[key]["coverage"] = coverage[key]
        metrics[key]["selection_summary"] = _selection_summary(key, fold_outputs)
    t_metrics = time.perf_counter() - t0

    # ---- negative control for promoted rows
    promoted = [k for k, m in metrics.items() if m["promotion"]["promoted"]]
    print(f"promoted rows before control: {len(promoted)}", flush=True)
    controls: dict[str, Any] = {}
    t_ctl = 0.0
    if promoted:
        t0 = time.perf_counter()
        controls = run_controls(promoted, preds, surface_by_fold, workers)
        t_ctl = time.perf_counter() - t0
    for key in promoted:
        v = controls[key]
        metrics[key]["shuffle_control"] = v
        metrics[key]["final_label"] = H.REVOKED if v["revoked"] else H.PROMOTED
    for m in metrics.values():
        m.setdefault("final_label", H.NOT_PROMOTED)

    # ---- artifacts
    write_json(OUT / "row_metrics.json", {"analysis_label": H.EXPLORATORY_LABEL,
                                          "rows": metrics})
    write_json(OUT / "promotion.json", {
        "analysis_label": H.EXPLORATORY_LABEL,
        "promoted_before_control": promoted,
        "final_promoted": [k for k, m in metrics.items() if m["final_label"] == H.PROMOTED],
        "revoked": [k for k, m in metrics.items() if m["final_label"] == H.REVOKED],
    })
    write_json(OUT / "shuffle_control.json", controls)
    write_json(OUT / "stability_by_salt.json", {
        k: {s: {"delta_auroc": v["delta_auroc"], "surface_auroc": v["surface_auroc"],
                "combined_auroc": v["combined_auroc"], "delta_auprc": v["delta_auprc"]}
            for s, v in m["per_salt"].items()}
        for k, m in metrics.items() if "per_salt" in m
    })
    write_json(OUT / "same_prompt_analysis.json", {
        k: {"summary": m["same_prompt"], "prompts": m["same_prompt_rows"]}
        for k, m in metrics.items() if "same_prompt" in m
    })
    write_json(OUT / "selection_diagnostics.json", {
        k: m.get("selection_summary") for k, m in metrics.items()
    })
    save = {}
    for key, by in preds.items():
        for s, arrs in by.items():
            for fld, arr in arrs.items():
                save[f"{key}__s{s}__{fld}"] = np.asarray(arr)
    np.savez_compressed(OUT / "cross_fitted_predictions.npz", **save)
    write_json(OUT / "record_index.json", {"tids": data.tids, "prompt_ids": data.prompt_ids})

    tree_after = H.phase24f_tree_hash(REPO_ROOT)
    imm_after = assert_phase24f_immutable(REPO_ROOT)
    if tree_after != pre["phase24f_tree_hash"]:
        raise RuntimeError("Phase-24F tree changed during Phase 24H run")
    write_json(OUT / "phase24f_immutability.json", {
        **imm_after, "tree_hash_before": pre["phase24f_tree_hash"],
        "tree_hash_after": tree_after, "unchanged": True})

    wall = time.perf_counter() - t_start
    manifest = {
        "status": H.STATUS,
        "analysis_label": H.EXPLORATORY_LABEL,
        "guarantee": H.GUARANTEE,
        "git": pre["git"],
        "starting_sha": H.STARTING_SHA,
        "finished_at": utc_now_iso(),
        "n_workers": workers,
        "threads_env": _THREADS,
        "wall_seconds": wall,
        "stage_seconds": {"hash": t_hash, "load": t_load, "real": t_real,
                          "metrics": t_metrics, "control": t_ctl},
        "fold_seconds": fold_times,
        "memory": _rss_mb(),
        "n_rows": len(metrics),
        "input_summary": summary,
        "input_aggregate_sha256": hsh["aggregate_sha256"],
        "test_accessed": False,
        "gpu_used": False,
    }
    write_json(OUT / "run_manifest.json", manifest)
    write_report(metrics, manifest, summary)
    write_json(OUT / "artifact_hashes.json", {
        p.name: sha256_file(str(p)) for p in sorted(OUT.glob("*.json")) + [
            OUT / "cross_fitted_predictions.npz", REPORT]
        if p.exists() and p.name != "artifact_hashes.json"
    })
    print(f"DONE status={H.STATUS} wall={wall / 3600:.2f}h", flush=True)
    return 0


def _selection_summary(key: str, fold_outputs: list[tuple[int, int, H.FoldOutput]]) -> list:
    out = []
    for salt, fold, fo in sorted(fold_outputs, key=lambda x: (x[0], x[1])):
        rp = fo.rows.get(key)
        if rp is not None:
            out.append({"salt": salt, "fold": fold, "surface_time": rp.surface_time,
                        "n_eval": int(len(rp.eval_rows)),
                        "meta": {k: v for k, v in rp.meta.items() if k != "inner_cv_auroc"}})
    return out


def run_controls(
    promoted: list[str],
    preds: dict,
    surface_by_fold: dict,
    workers: int,
) -> dict[str, Any]:
    groups: dict[tuple, list[str]] = {}
    for k in promoted:
        variant, times = H.required_scope(k)
        groups.setdefault((variant, times), []).append(k)
    data: H.DevData = _G["data"]
    shuffle_deltas: dict[str, list[float]] = {k: [] for k in promoted}
    for (variant, times), keys in groups.items():
        gkey = f"{variant}|{','.join(map(str, times))}"
        print(f"control group {gkey} rows={keys}", flush=True)
        tasks = [
            (rep, salt, fold, gkey, variant, times, surface_by_fold[(salt, fold)])
            for rep in range(H.N_SHUFFLES)
            for salt in H.SHUFFLE_SALTS
            for fold in range(H.N_OUTER_FOLDS)
        ]
        by_rep: dict[int, list[tuple[int, int, H.FoldOutput]]] = {}
        with _pool(workers) as ex:
            for i, (rep, salt, fold, _g, rows, _s) in enumerate(
                ex.map(_task_control, tasks, chunksize=1)
            ):
                by_rep.setdefault(rep, []).append(
                    (salt, fold, H.FoldOutput(rows=rows, surface={}, timings={}))
                )
                if (i + 1) % 70 == 0:
                    print(f"  control {gkey} {(i + 1) // 7}/{H.N_SHUFFLES} reps", flush=True)
        for rep in range(H.N_SHUFFLES):
            pr = H.assemble_predictions(data, by_rep[rep])
            for k in keys:
                if k in pr and 0 in pr[k]:
                    shuffle_deltas[k].append(H.salt_delta(pr[k], 0))
                else:
                    shuffle_deltas[k].append(float("nan"))
    out = {}
    for k in promoted:
        real = H.salt_delta(preds[k], 0)
        v = H.shuffle_control_verdict(real, shuffle_deltas[k])
        v["shuffle_deltas"] = shuffle_deltas[k]
        out[k] = v
    return out


# ---- report ----
def _k(key: str) -> str:
    """Markdown-table-safe row key (escape the pipe separator)."""
    return "`" + key.replace("|", "\\|") + "`"


def _f(x: Any, nd: int = 3) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "nan"


def write_report(
    metrics: dict[str, Any], manifest: dict[str, Any], summary: dict[str, Any]
) -> None:
    L: list[str] = []
    L.append("# Phase 24H — Internal-signal-breadth representation discovery\n")
    L.append(f"**Status:** `{H.STATUS}`  \n**Label:** `{H.EXPLORATORY_LABEL}`\n")
    L.append(f"> {H.GUARANTEE}\n")
    L.append(
        "Phase-24F remains the **final confirmatory FAIL** (t=1, L20). Nothing here changes it. "
        "All ΔAUROC values are `AUROC(SURFACE+ACTIVATION) − AUROC(SURFACE)` at the **same time** "
        "(multitime uses SURFACE@t8). Activation-alone AUROC is descriptive only and is never "
        "incremental evidence.\n"
    )
    L.append("## Data and protocol\n")
    L.append(
        f"- DEV prompts: {summary['n_prompts']} (20 TRAIN + 8 VALIDATION); "
        f"trajectories {summary['n_records']} "
        f"({summary['n_honest']} honest / {summary['n_deceptive']} deceptive; "
        f"{summary['n_ambiguous_excluded']} ambiguous excluded)\n"
        f"- Early times frozen {list(H.EARLY_TIMES)}; outer salts {list(H.OUTER_SALTS)} × "
        f"{H.N_OUTER_FOLDS} folds (4 held-out prompts each); inner prompt-group CV "
        f"{H.INNER_FOLDS}-fold\n"
        f"- Bootstrap: {H.N_BOOTSTRAP} prompt-cluster resamples; shuffle control: "
        f"{H.N_SHUFFLES} prompt-preserving activation shuffles (salt 0)\n"
        f"- Protocol freeze commit: `{manifest['git']['protocol_freeze_commit']}`; "
        f"run HEAD `{manifest['git']['head']}`\n"
        f"- Multiplicity: {len(metrics)} exploratory rows were screened; promotion is a "
        "replication-screening label, **not** confirmatory evidence.\n"
    )
    final = [k for k, m in metrics.items() if m.get("final_label") == H.PROMOTED]
    revoked = [k for k, m in metrics.items() if m.get("final_label") == H.REVOKED]
    L.append("## Outcome\n")
    L.append(f"- Rows meeting all five promotion criteria: "
             f"{len(final) + len(revoked)}; revoked by shuffle control: {len(revoked)}\n")
    L.append(f"- **Final `promising_for_independent_replication` rows: {len(final)}**"
             + (": " + ", ".join(f"`{k}`" for k in final) if final else "") + "\n")
    L.append("## All rows (pooled over outer salts)\n")
    L.append("| row | ΔAUROC | 95% CI | Δ salt0/1/2 | SURF AUROC | COMB AUROC | ΔAUPRC | "
             "n mixed | med Δ within | frac>0 | label |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for k in H.row_keys_all():
        m = metrics[k]
        if m.get("missing"):
            L.append(f"| {_k(k)} | missing | | | | | | | | | {H.NOT_PROMOTED} |")
            continue
        ci = m["bootstrap"]["ci95"]
        ds = "/".join(_f(m["per_salt"][str(s)]["delta_auroc"], 3) for s in H.OUTER_SALTS)
        sp = m["same_prompt"]
        L.append(
            f"| {_k(k)} | {_f(m['pooled_delta_auroc'])} | [{_f(ci[0])}, {_f(ci[1])}] | {ds} | "
            f"{_f(m['pooled_surface_auroc'])} | {_f(m['pooled_combined_auroc'])} | "
            f"{_f(m['pooled_delta_auprc'])} | {sp['n_qualifying']} | {_f(sp['median_delta'])} | "
            f"{_f(sp['fraction_gt_0'], 2)} | {m['final_label']} |"
        )
    L.append("\n## Same-prompt analysis (cross-fitted predictions only; no per-prompt models)\n")
    L.append("Prompts with ≥2 honest and ≥2 deceptive; within-prompt Δ averaged over salts.\n")
    L.append("| row | n qual. | median Δ | mean Δ | frac>0 | median 95% CI | mean 95% CI |")
    L.append("|---|---|---|---|---|---|---|")
    for k in H.row_keys_all():
        m = metrics[k]
        if m.get("missing"):
            continue
        sp = m["same_prompt"]
        b = sp["bootstrap"]
        L.append(
            f"| {_k(k)} | {sp['n_qualifying']} | {_f(sp['median_delta'])} | "
            f"{_f(sp['mean_delta'])} | {_f(sp['fraction_gt_0'], 2)} | "
            f"[{_f(b['median_ci95'][0])}, {_f(b['median_ci95'][1])}] | "
            f"[{_f(b['mean_ci95'][0])}, {_f(b['mean_ci95'][1])}] |"
        )
    L.append("\n## Promotion criteria detail\n")
    L.append("| row | pooled>0 | CI low>0 | all salts>0 | within median>0 | frac≥0.60 | promoted |")
    L.append("|---|---|---|---|---|---|---|")
    crit_order = (
        "pooled_delta_gt_0",
        "bootstrap_lower_gt_0",
        "delta_gt_0_in_all_salts",
        "within_prompt_median_gt_0",
        "fraction_positive_prompts_ge_0.60",
    )
    for k in H.row_keys_all():
        m = metrics[k]
        c = m["promotion"].get("criteria")
        if not c:
            continue
        flags = " | ".join("yes" if c[name] else "no" for name in crit_order)
        verdict = "yes" if m["promotion"]["promoted"] else "no"
        L.append(f"| {_k(k)} | {flags} | {verdict} |")
    L.append("\n## Negative control (prompt-preserving activation shuffle)\n")
    any_ctl = False
    for k in H.row_keys_all():
        v = metrics[k].get("shuffle_control")
        if not v:
            continue
        any_ctl = True
        L.append(
            f"- `{k}`: real salt-0 Δ={_f(v['real_delta'])}; shuffle mean={_f(v['shuffle_mean'])}, "
            f"q95={_f(v['shuffle_q95'])}, max={_f(v['shuffle_max'])}; one-sided p="
            f"{_f(v['one_sided_p'], 4)} → **{'REVOKED' if v['revoked'] else 'survives'}**"
        )
    if not any_ctl:
        L.append("No row met the promotion rule, so no shuffle control was required.")
    L.append("\n## Compute\n")
    L.append(f"- Wall: {manifest['wall_seconds'] / 3600:.2f} h; workers {manifest['n_workers']}, "
             f"BLAS threads {manifest['threads_env']}; CPU only, no GPU/model generation.\n"
             f"- Peak RSS: {manifest['memory']}\n")
    L.append(f"\nInput NPZ aggregate SHA256: `{manifest['input_aggregate_sha256']}`\n")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")


def report_only() -> int:
    metrics = json.loads((OUT / "row_metrics.json").read_text())["rows"]
    manifest = json.loads((OUT / "run_manifest.json").read_text())
    summary = json.loads((OUT / "input_summary.json").read_text())
    write_report(metrics, manifest, summary)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--hash-only", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    if args.hash_only:
        tok = _load_tokenizer()
        records, summary = build_records(tok)
        hsh = hash_dev_inputs(records, workers=4)
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / "input_hashes.json", hsh)
        write_json(OUT / "input_summary.json", summary)
        print(json.dumps({k: v for k, v in hsh.items() if k != "files"}, indent=2))
        return 0
    if args.smoke:
        return run_smoke(args.workers)
    if args.full:
        return run_full(args.workers, args.force)
    if args.report_only:
        return report_only()
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
