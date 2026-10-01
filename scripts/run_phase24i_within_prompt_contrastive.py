#!/usr/bin/env python3
"""Phase 24I: within-prompt contrastive analysis of one frozen Phase-24H lead.

DEVELOPMENT (TRAIN+VALIDATION, 28 prompts) ONLY. CPU only; no model generation.
Phase-24F stays the final confirmatory FAIL.

Usage:
  run_phase24i_within_prompt_contrastive.py --hash-only
  run_phase24i_within_prompt_contrastive.py --smoke      # infrastructure + timing only
  run_phase24i_within_prompt_contrastive.py --full       # one scientific run
  run_phase24i_within_prompt_contrastive.py --report-only
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
from pre_output_physiology import phase24i_within_prompt_contrastive as I  # noqa: E402, N812
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

OUT = REPO_ROOT / "artifacts/phase24i_within_prompt_contrastive"
REPORT = REPO_ROOT / "reports/phase24i_within_prompt_contrastive.md"
CFG = REPO_ROOT / "configs/experiments/phase24i_within_prompt_contrastive.yaml"
ACT_DEV = REPO_ROOT / H.ACT_DEV_REL
SPLIT_PATH = REPO_ROOT / "artifacts/phase24c_design/split_manifest.json"
LABELS_TRAIN = REPO_ROOT / "artifacts/phase24d_collection/by_split/TRAIN/labels.json"
LABELS_VAL = REPO_ROOT / "artifacts/phase24d_collection/by_split/VALIDATION/labels.json"
META24D = REPO_ROOT / "artifacts/phase24d_collection/capture_meta.json"
TRAJ24B = REPO_ROOT / "artifacts/phase24b_live/trajectories.jsonl"
REUSE = REPO_ROOT / "artifacts/phase24d_collection/by_split/TRAIN/phase24b_reuse_index.json"
H24_DIR = REPO_ROOT / "artifacts/phase24h_internal_signal_breadth"
PROTOCOL_FILES = (
    "configs/experiments/phase24i_within_prompt_contrastive.yaml",
    "src/pre_output_physiology/phase24i_within_prompt_contrastive.py",
    "scripts/run_phase24i_within_prompt_contrastive.py",
    "tests/test_phase24i_within_prompt_contrastive.py",
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

    chk("lead_row", cfg["lead"]["phase24h_row"], I.LEAD_KEY)
    chk("lead_time", cfg["lead"]["time"], I.LEAD_TIME)
    chk("n_layers", cfg["lead"]["n_layers"], I.N_LAYERS)
    chk("layer_probe_layers", cfg["layer_probe"]["n_layers"], I.N_LAYERS)
    chk("n_dev_prompts", cfg["data"]["n_dev_prompts"], I.N_DEV_PROMPTS)
    chk("n_records", cfg["data"]["n_records"], I.N_EXPECTED_RECORDS)
    chk("input_aggregate", cfg["data"]["input_aggregate_sha256"],
        I.PHASE24H_INPUT_AGGREGATE_SHA256)
    chk("folds_sha", cfg["outer_folds"]["sha256"], I.PHASE24H_FOLDS_SHA256)
    chk("folds_source", cfg["outer_folds"]["source"], I.PHASE24H_FOLDS_REL)
    chk("salts", tuple(cfg["outer_folds"]["salts"]), I.OUTER_SALTS)
    chk("n_folds", cfg["outer_folds"]["n_folds"], I.N_OUTER_FOLDS)
    chk("prompts_per_fold", cfg["outer_folds"]["prompts_per_fold"], I.PROMPTS_PER_EVAL_FOLD)
    chk("inner_folds", cfg["outer_folds"]["inner_cv"]["n_folds"], I.INNER_FOLDS)
    chk("layer_C", cfg["layer_probe"]["C"], I.LAYER_C)
    chk("meta_C", cfg["multilayer_meta"]["C"], I.META_C)
    chk("combined_C", cfg["combination"]["C"], I.COMBINED_C)
    chk("surface_C_text", cfg["surface"]["frozen_C_text"], I.SURFACE_C_TEXT)
    chk("surface_C_logits", cfg["surface"]["frozen_C_logits"], I.SURFACE_C_LOGITS)
    chk("surface_C_meta", cfg["surface"]["frozen_C_meta"], I.SURFACE_C_META)
    chk("surface_time", cfg["surface"]["time"], I.LEAD_TIME)
    chk("n_bootstrap", cfg["primary_estimand"]["bootstrap"]["n_bootstrap"], I.N_BOOTSTRAP)
    chk("n_shuffles", cfg["shuffle_control"]["n_shuffles"], I.N_SHUFFLES)
    chk("shuffle_salts", tuple(cfg["shuffle_control"]["outer_salts_used"]), I.SHUFFLE_SALTS)
    chk("min_fraction", cfg["promotion"]["min_fraction_positive_prompts"],
        I.MIN_FRACTION_POSITIVE_PROMPTS)
    chk("seed", cfg["analysis_seed"], I.ANALYSIS_SEED)
    chk("status", cfg["status_on_completion"], I.STATUS)
    chk("starting_sha", cfg["starting_sha"], I.STARTING_SHA)
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


# ---- records (identical cohort construction as Phase 24H; DEV only) ----
def _label_map(path: Path, allowed: set[str]) -> tuple[dict[str, dict], int]:
    I.assert_dev_input_path(path)
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
        I.assert_dev_activation_path(npz, REPO_ROOT)
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
    if len(records) != I.N_EXPECTED_RECORDS:
        raise RuntimeError(f"expected {I.N_EXPECTED_RECORDS} records, got {len(records)}")
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
            I.assert_dev_activation_path(p, REPO_ROOT)
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
    named = {f"{p.parent.name}/{p.name}": h for p, h in zip(files, hashes, strict=True)}
    return {
        "n_files_hashed": len(files),
        "aggregate_sha256": I.aggregate_input_hash(named),
        "n_used_with_recorded_sha256": n_recorded,
        "n_used_recorded_sha256_match": n_match,
        "recorded_sha256_mismatches": mismatches,
        "files": named,
        "hashed_at": utc_now_iso(),
    }


def load_dev_data(records: list[dict], tok) -> H.DevData:
    """Only h(0), h(1), logits(1) and the t=1 SURFACE prefix are loaded."""
    t = I.LEAD_TIME
    n = len(records)
    first = H.load_dev_npz_arrays(records[0]["npz_path"], REPO_ROOT, times=(t,))
    n_layers, dim = first["h0"].shape
    h0 = np.zeros((n, n_layers, dim), dtype=np.float32)
    h1 = np.zeros((n, n_layers, dim), dtype=np.float32)
    vocab = first["logits"][t].shape[0]
    lg = np.zeros((n, vocab), dtype=np.float32)
    texts: list[str] = [""] * n
    for i, r in enumerate(records):
        a = first if i == 0 else H.load_dev_npz_arrays(
            r["npz_path"], REPO_ROOT, times=(t,)
        )
        if a["n_generated"] != r["n_generated"]:
            raise RuntimeError(f"n_generated mismatch for {r['trajectory_id']}")
        h0[i] = a["h0"]
        if r["n_generated"] > t:
            h1[i] = a["h"][t]
            lg[i] = a["logits"][t]
            pref = prefix_token_ids(r["generated_token_ids"], t)
            texts[i] = r["prompt_text"] + (
                tok.decode(pref, skip_special_tokens=True) if pref else ""
            )
        if (i + 1) % 100 == 0:
            print(f"  loaded {i + 1}/{n}", flush=True)
    return H.DevData(
        tids=[r["trajectory_id"] for r in records],
        prompt_ids=[r["prompt_id"] for r in records],
        labels=[r["label"] for r in records],
        n_generated=np.asarray([r["n_generated"] for r in records]),
        texts={t: texts},
        logits={t: lg},
        h0=h0,
        h={t: h1},
    )


# ---- workers ----
def _task_real(args: tuple[int, int]) -> tuple[I.FoldResult, float]:
    salt, fold = args
    data: H.DevData = _G["data"]
    tr, ev = I.train_eval_split(_G["folds"], _G["dev_prompts"], salt, fold)
    t0 = time.perf_counter()
    out = I.run_outer_fold(data, tr, ev, salt=salt, fold=fold)
    return out, time.perf_counter() - t0


def _task_control(args: tuple) -> tuple[int, I.FoldResult, float]:
    rep, salt, fold, surface = args
    data: H.DevData = _G["data"]
    tr, ev = I.train_eval_split(_G["folds"], _G["dev_prompts"], salt, fold)
    act_map = I.shuffle_permutation(
        data.prompt_ids, data.survivors(I.LEAD_TIME).astype(int), rep
    )
    t0 = time.perf_counter()
    out = I.run_outer_fold(
        data, tr, ev, salt=salt, fold=fold, act_map=act_map, surface=surface
    )
    out.surface = None  # keep pickles small
    return rep, out, time.perf_counter() - t0


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
    tree = I.phase24f_tree_hash(REPO_ROOT)
    verify = verify_frozen_input_hashes(REPO_ROOT)
    if not verify["verified"]:
        raise RuntimeError("frozen input hash verification failed")
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    dev_prompts = sorted(set(split["train_prompt_ids"]) | set(split["validation_prompt_ids"]))
    folds = I.load_phase24h_folds(REPO_ROOT, dev_prompts)  # raises on hash mismatch
    # guard self-test: the code must raise on TEST / sealed / 24F activations
    for bad_path in (
        REPO_ROOT / "artifacts/phase24f_confirmation/activations_local/LOCKED_TEST/x.npz",
        REPO_ROOT / "artifacts/phase24d_collection/by_split/LOCKED_TEST/labels_SEALED.json",
    ):
        try:
            I.assert_dev_activation_path(bad_path, REPO_ROOT)
        except SealViolationError:
            continue
        raise RuntimeError(f"path guard failed to reject {bad_path}")
    git = git_state()
    if require_clean and git["protocol_files_dirty"]:
        raise RuntimeError(f"protocol files not committed: {git['protocol_files_dirty']}")
    if require_clean and not git["protocol_freeze_commit"]:
        raise RuntimeError("protocol config has no commit: freeze before scientific run")
    fold_digest = sha256_file(str(REPO_ROOT / I.PHASE24H_FOLDS_REL))
    return {"immutability": imm, "phase24f_tree_hash": tree, "input_verification": verify,
            "git": git, "config_ok": True, "folds": folds, "dev_prompts": dev_prompts,
            "folds_sha256": fold_digest,
            "folds_sha256_matches_frozen": fold_digest == I.PHASE24H_FOLDS_SHA256}


def check_surface_vs_phase24h(data: H.DevData, results: list[I.FoldResult]) -> dict[str, Any]:
    """Infrastructure check: recomputed SURFACE equals stored Phase-24H SURFACE@t1."""
    z = np.load(H24_DIR / "cross_fitted_predictions.npz", allow_pickle=False)
    rec = json.loads((H24_DIR / "record_index.json").read_text(encoding="utf-8"))
    same_index = rec["tids"] == data.tids and rec["prompt_ids"] == data.prompt_ids
    out: dict[str, Any] = {"record_index_identical_to_phase24h": bool(same_index)}
    diffs = []
    for salt in sorted({r.salt for r in results}):
        rows = z[f"{I.LEAD_KEY}__s{salt}__rows"]
        surf = z[f"{I.LEAD_KEY}__s{salt}__surf"]
        ref = dict(zip(rows.tolist(), surf.tolist(), strict=True))
        for r in results:
            if r.salt != salt:
                continue
            for row, v in zip(r.eval_rows.tolist(), r.surf.tolist(), strict=True):
                diffs.append(abs(ref[row] - v))
    out["n_compared"] = len(diffs)
    out["max_abs_diff_surface_vs_phase24h"] = float(max(diffs)) if diffs else float("nan")
    out["surface_identical"] = bool(diffs) and max(diffs) < 1e-9
    return out


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
    dev_prompts = pre["dev_prompts"]
    if sorted(set(data.prompt_ids)) != dev_prompts:
        raise RuntimeError("record prompts differ from DEV manifest")
    _G.update(data=data, dev_prompts=dev_prompts, folds=pre["folds"])
    n_surv = int(data.survivors(I.LEAD_TIME).sum())

    salt, fold = 0, 0
    out, fold_s = _task_real((salt, fold))
    n_finite = int(np.isfinite(out.surf).all() and np.isfinite(out.act).all()
                   and np.isfinite(out.comb).all())
    surf_check = check_surface_vs_phase24h(data, [out])
    # shuffle infrastructure: 2 reps on fold 0 (SURFACE cached)
    ctl_times = []
    for rep in range(2):
        _rep, _o, secs = _task_control((rep, salt, fold, out.surface))
        ctl_times.append(secs)
    # metric/promotion plumbing check on a single fold (values are NOT reported)
    preds = I.assemble_predictions(data, [out])
    _ = I.full_metrics(preds, n_boot=10, seed=1)

    eff = max(1.0, workers * 0.75)
    n_real = len(I.OUTER_SALTS) * I.N_OUTER_FOLDS
    est_real_h = fold_s * n_real / eff / 3600
    n_ctl_tasks = I.N_SHUFFLES * I.N_OUTER_FOLDS * len(I.SHUFFLE_SALTS)
    est_control_h = float(np.mean(ctl_times)) * n_ctl_tasks / eff / 3600
    est_hash_all_min = t_hash_24 / 24 * n_all / 60
    fixed_h = (t_rec + t_load) / 3600 + est_hash_all_min / 60
    bench = {
        "analysis_label": "SMOKE_INFRASTRUCTURE_ONLY_NO_SCIENTIFIC_CONCLUSIONS",
        "n_records": summary["n_records"],
        "n_survivors_t1": n_surv,
        "fold0_wall_s": fold_s,
        "fold0_timings_s": out.timings,
        "fold0_output_finite": bool(n_finite),
        "fold0_pair_counts": out.info["train_pairs"]["n_hxd_pairs"],
        "fold0_pair_bearing_train_prompts": out.info["train_pairs"]["n_pair_bearing_prompts"],
        "surface_vs_phase24h": surf_check,
        "record_build_s": t_rec,
        "data_load_s": t_load,
        "hash_24_files_s": t_hash_24,
        "n_dev_npz_files": n_all,
        "est_hash_all_min": est_hash_all_min,
        "control_task_s": ctl_times,
        "workers_assumed": workers,
        "parallel_efficiency_assumed": 0.75,
        "est_real_stage_hours": est_real_h,
        "est_control_stage_hours_if_promoted": est_control_h,
        "est_total_hours_no_promotion": est_real_h + fixed_h,
        "est_total_hours_worst_case_with_control": est_real_h + est_control_h + fixed_h,
        "memory": _rss_mb(),
        "smoke_wall_s": time.perf_counter() - t_all,
        "stop_threshold_hours": 4,
        "hash_sample_ok": hsh["recorded_sha256_mismatches"] == [],
        "folds_sha256": pre["folds_sha256"],
        "folds_sha256_matches_frozen": pre["folds_sha256_matches_frozen"],
        "git": pre["git"],
    }
    write_json(smoke_dir / "smoke_benchmark.json", bench)
    print(json.dumps({k: v for k, v in bench.items() if k not in ("git", "fold0_timings_s")},
                     indent=2, default=str), flush=True)
    return 0


# ---- full run ----
def run_controls(preds: dict, real_results: list[I.FoldResult], workers: int) -> dict[str, Any]:
    data: H.DevData = _G["data"]
    surfaces = {(r.salt, r.fold): r.surface for r in real_results}
    tasks = [
        (rep, salt, fold, surfaces[(salt, fold)])
        for rep in range(I.N_SHUFFLES)
        for salt in I.SHUFFLE_SALTS
        for fold in range(I.N_OUTER_FOLDS)
    ]
    by_rep: dict[int, list[I.FoldResult]] = {}
    with _pool(workers) as ex:
        for i, (rep, res, _s) in enumerate(ex.map(_task_control, tasks, chunksize=1)):
            by_rep.setdefault(rep, []).append(res)
            if (i + 1) % 70 == 0:
                print(f"  control {(i + 1) // 7}/{I.N_SHUFFLES} reps", flush=True)
    deltas: list[float] = []
    for rep in range(I.N_SHUFFLES):
        pr = I.assemble_predictions(data, by_rep[rep])
        deltas.append(I.salt_prompt_equal_delta(pr, 0))
    real = I.salt_prompt_equal_delta(preds, 0)
    v = I.shuffle_control_verdict(real, deltas)
    v["shuffle_deltas"] = deltas
    return v


def run_full(workers: int, force: bool) -> int:
    t_start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "metrics.json").exists() and not force:
        raise RuntimeError("full results already exist; scientific run is once-only (--force)")
    pre = preflight(require_clean=True)
    write_json(OUT / "phase24f_immutability.json",
               {**pre["immutability"], "tree_hash_before": pre["phase24f_tree_hash"]})
    write_json(OUT / "frozen_input_verification.json", pre["input_verification"])
    write_json(OUT / "outer_folds_verification.json", {
        "source": I.PHASE24H_FOLDS_REL, "sha256": pre["folds_sha256"],
        "expected_sha256": I.PHASE24H_FOLDS_SHA256,
        "matches": pre["folds_sha256_matches_frozen"],
        "new_fold_assignments_generated": False,
    })
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
    h_verify = I.verify_against_phase24h_hashes(hsh["files"], REPO_ROOT)
    write_json(OUT / "input_hashes.json", hsh)
    write_json(OUT / "input_hash_verification_vs_phase24h.json", h_verify)
    if not h_verify["verified"]:
        raise RuntimeError("DEV NPZ hashes differ from the Phase-24H manifest")
    t_hash = time.perf_counter() - t0
    print(f"hashed {hsh['n_files_hashed']} files in {t_hash:.0f}s; matches Phase-24H", flush=True)

    print("loading arrays…", flush=True)
    t0 = time.perf_counter()
    data = load_dev_data(records, tok)
    t_load = time.perf_counter() - t0
    dev_prompts = pre["dev_prompts"]
    assert sorted(set(data.prompt_ids)) == dev_prompts
    _G.update(data=data, dev_prompts=dev_prompts, folds=pre["folds"])
    write_json(OUT / "outer_folds_used.json", {
        str(s): [sorted(f) for f in fl] for s, fl in pre["folds"].items()
    })

    # ---- real cross-fitting
    tasks = [(s, f) for s in I.OUTER_SALTS for f in range(I.N_OUTER_FOLDS)]
    results: list[I.FoldResult] = []
    fold_times: dict[str, float] = {}
    t0 = time.perf_counter()
    with _pool(workers) as ex:
        for res, secs in ex.map(_task_real, tasks):
            results.append(res)
            fold_times[f"salt{res.salt}_fold{res.fold}"] = secs
            print(f"  real salt={res.salt} fold={res.fold} {secs:.0f}s", flush=True)
    t_real = time.perf_counter() - t0

    surf_check = check_surface_vs_phase24h(data, results)
    write_json(OUT / "surface_vs_phase24h_check.json", surf_check)
    if not (surf_check["record_index_identical_to_phase24h"] and surf_check["surface_identical"]):
        raise RuntimeError(f"SURFACE reuse check failed: {surf_check}")

    preds = I.assemble_predictions(data, results)
    for s, p in preds.items():
        if len(p["rows"]) != int(data.survivors(I.LEAD_TIME).sum()):
            raise RuntimeError(f"salt {s}: not every trajectory predicted exactly once")
    t0 = time.perf_counter()
    metrics = I.full_metrics(preds, I.N_BOOTSTRAP, I.derive_seed("bootstrap", I.LEAD_KEY))
    t_metrics = time.perf_counter() - t0
    metrics["comparison_24h_vs_24i"] = I.compare_with_phase24h(REPO_ROOT, metrics)

    # ---- shuffle control (promoted only)
    controls: dict[str, Any] = {}
    t_ctl = 0.0
    final_label = I.NOT_PROMOTED
    if metrics["promotion"]["promoted"]:
        print("promoted -> running shuffle control", flush=True)
        t0 = time.perf_counter()
        controls = run_controls(preds, results, workers)
        t_ctl = time.perf_counter() - t0
        final_label = I.REVOKED if controls["revoked"] else I.PROMOTED
    metrics["shuffle_control"] = controls or None
    metrics["final_label"] = final_label

    # ---- artifacts
    write_json(OUT / "metrics.json", {"analysis_label": I.EXPLORATORY_LABEL, **metrics})
    write_json(OUT / "promotion.json", {
        "analysis_label": I.EXPLORATORY_LABEL,
        "criteria": metrics["promotion"]["criteria"],
        "promoted_before_control": metrics["promotion"]["promoted"],
        "final_label": final_label,
    })
    write_json(OUT / "shuffle_control.json", controls)
    write_json(OUT / "within_prompt_deltas.json", {
        "summary": {k: v for k, v in metrics["primary"].items() if k != "rows"},
        "prompts": metrics["primary"]["rows"],
    })
    write_json(OUT / "comparison_24h_vs_24i.json", metrics["comparison_24h_vs_24i"])
    write_json(OUT / "pair_counts_by_outer_fold.json", [
        {"salt": r.salt, "fold": r.fold, "eval_prompts": r.info["eval_prompts"],
         **{k: v for k, v in r.info["train_pairs"].items()}}
        for r in sorted(results, key=lambda r: (r.salt, r.fold))
    ])
    write_json(OUT / "fold_diagnostics.json", [
        {k: v for k, v in r.info.items() if k != "train_pairs"} | {"timings_s": r.timings}
        for r in sorted(results, key=lambda r: (r.salt, r.fold))
    ])
    save: dict[str, np.ndarray] = {}
    for s, arrs in preds.items():
        for fld, arr in arrs.items():
            save[f"s{s}__{fld}"] = np.asarray(arr)
    np.savez_compressed(OUT / "cross_fitted_predictions.npz", **save)
    write_json(OUT / "record_index.json", {"tids": data.tids, "prompt_ids": data.prompt_ids})

    tree_after = I.phase24f_tree_hash(REPO_ROOT)
    imm_after = assert_phase24f_immutable(REPO_ROOT)
    if tree_after != pre["phase24f_tree_hash"]:
        raise RuntimeError("Phase-24F tree changed during Phase 24I run")
    write_json(OUT / "phase24f_immutability.json", {
        **imm_after, "tree_hash_before": pre["phase24f_tree_hash"],
        "tree_hash_after": tree_after, "unchanged": True})

    wall = time.perf_counter() - t_start
    manifest = {
        "status": I.STATUS,
        "analysis_label": I.EXPLORATORY_LABEL,
        "guarantee": I.GUARANTEE,
        "git": pre["git"],
        "starting_sha": I.STARTING_SHA,
        "finished_at": utc_now_iso(),
        "n_workers": workers,
        "threads_env": _THREADS,
        "wall_seconds": wall,
        "stage_seconds": {"hash": t_hash, "load": t_load, "real": t_real,
                          "metrics": t_metrics, "control": t_ctl},
        "fold_seconds": fold_times,
        "memory": _rss_mb(),
        "input_summary": summary,
        "input_aggregate_sha256": hsh["aggregate_sha256"],
        "input_matches_phase24h_manifest": h_verify["verified"],
        "outer_folds_sha256": pre["folds_sha256"],
        "outer_folds_sha256_matches_frozen": pre["folds_sha256_matches_frozen"],
        "test_accessed": False,
        "gpu_used": False,
    }
    write_json(OUT / "run_manifest.json", manifest)
    write_report(metrics, manifest, summary, results)
    write_json(OUT / "artifact_hashes.json", {
        p.name: sha256_file(str(p)) for p in sorted(OUT.glob("*.json")) + [
            OUT / "cross_fitted_predictions.npz", REPORT]
        if p.exists() and p.name != "artifact_hashes.json"
    })
    print(f"DONE status={I.STATUS} wall={wall / 3600:.2f}h final={final_label}", flush=True)
    return 0


# ---- report ----
def _f(x: Any, nd: int = 3) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "nan"


def write_report(
    metrics: dict[str, Any], manifest: dict[str, Any], summary: dict[str, Any],
    results: list[I.FoldResult] | None = None,
) -> None:
    p = metrics["primary"]
    pr = metrics["promotion"]
    pool = metrics["pooled_descriptive"]
    cmp_ = metrics["comparison_24h_vs_24i"]
    L: list[str] = []
    L.append("# Phase 24I — Within-prompt contrastive analysis (frozen Phase-24H lead)\n")
    L.append(f"**Status:** `{I.STATUS}`  \n**Label:** `{I.EXPLORATORY_LABEL}`\n")
    L.append(f"> {I.GUARANTEE}\n")
    L.append(
        "**Question (frozen).** Among multiple stochastic rollouts from the same prompt, is "
        "there a common t=1 residual-stream activation pattern that ranks deceptive "
        "rollouts above honest rollouts on entirely unseen prompts, and does that pattern "
        "add predictive value beyond the visible SURFACE baseline?\n"
    )
    L.append(
        f"**Lead (only representation evaluated):** `{I.LEAD_KEY}` — "
        "Δh(1,L)=h(1,L)−h(0,L), layers 0–31. Phase-24F remains the **final confirmatory "
        "FAIL**; nothing here changes it.\n"
    )
    L.append("## Data and protocol\n")
    L.append(
        f"- DEV prompts: {summary['n_prompts']}; rollouts {summary['n_records']} "
        f"({summary['n_honest']} honest / {summary['n_deceptive']} deceptive; "
        f"{summary['n_ambiguous_excluded']} ambiguous excluded)\n"
        f"- Exact Phase-24H outer folds reused: SHA256 `{manifest['outer_folds_sha256']}` "
        f"(matches frozen: {manifest['outer_folds_sha256_matches_frozen']}); salts "
        f"{list(I.OUTER_SALTS)} × {I.N_OUTER_FOLDS} folds × 4 held-out prompts\n"
        f"- DEV NPZ aggregate SHA256 `{manifest['input_aggregate_sha256']}` "
        f"(matches Phase-24H manifest: {manifest['input_matches_phase24h_manifest']})\n"
        f"- Pairwise within-prompt training: both orientations of every H×D pair; "
        f"prompt-equal and class-equal weights; layer C={I.LAYER_C}, meta C={I.META_C}, "
        f"combined C={I.COMBINED_C}; inner {I.INNER_FOLDS}-fold prompt-group OOF\n"
        f"- Bootstrap: {I.N_BOOTSTRAP} prompt resamples; shuffle control: "
        f"{I.N_SHUFFLES} within-prompt activation-bundle shuffles (salt 0, only if promoted)\n"
        f"- Protocol freeze commit `{manifest['git']['protocol_freeze_commit']}`; "
        f"run HEAD `{manifest['git']['head']}`\n"
    )
    L.append("## Primary result (prompt-equal within-prompt ΔAUROC)\n")
    L.append(
        "Δ = AUROC(SURFACE+ACTIVATION) − AUROC(SURFACE) **within** each qualifying "
        "held-out prompt (≥2 honest and ≥2 deceptive), averaged over the 3 salts per "
        "prompt; primary = mean over prompts. Not a pooled trajectory AUROC.\n"
    )
    ci = p["bootstrap"]["mean_ci95"]
    mci = p["bootstrap"]["median_ci95"]
    L.append(f"- Qualifying prompts: **{p['n_qualifying']}**\n")
    L.append(f"- **Primary mean Δ = {_f(p['mean_delta'], 4)}**, 95% prompt-bootstrap CI "
             f"[{_f(ci[0], 4)}, {_f(ci[1], 4)}] ({p['bootstrap']['n_reps']} reps)\n")
    L.append(f"- Median prompt Δ = {_f(p['median_delta'], 4)} "
             f"(bootstrap CI [{_f(mci[0], 4)}, {_f(mci[1], 4)}]); fraction Δ>0 = "
             f"{_f(p['fraction_gt_0'], 3)}\n")
    L.append("- Per-salt mean Δ: " + ", ".join(
        f"salt {s}: {_f(v, 4)}" for s, v in p["per_salt_mean_delta"].items()) + "\n")
    L.append(f"- Mean within-prompt AUROC: SURFACE {_f(p['mean_surface_auroc'])}, "
             f"COMBINED {_f(p['mean_combined_auroc'])}; activation alone "
             f"{_f(p['mean_activation_auroc_descriptive_only'])} "
             "(descriptive only; never incremental evidence)\n")
    L.append("## Promotion (all five required)\n")
    L.append("| criterion | pass |")
    L.append("|---|---|")
    for k, v in pr["criteria"].items():
        L.append(f"| {k} | {'yes' if v else 'no'} |")
    L.append(f"\n- Before shuffle control: **{pr['label']}**; final label: "
             f"**{metrics['final_label']}**\n")
    L.append("## Shuffle control\n")
    sc = metrics.get("shuffle_control")
    if sc:
        L.append(
            f"Real salt-0 Δ={_f(sc['real_delta'], 4)}; shuffle mean={_f(sc['shuffle_mean'], 4)}, "
            f"q95={_f(sc['shuffle_q95'], 4)}, max={_f(sc['shuffle_max'], 4)}; one-sided p="
            f"{_f(sc['one_sided_p'], 4)} → **{'REVOKED' if sc['revoked'] else 'survives'}**\n"
        )
    else:
        L.append("The row was not promoted, so the shuffle control was not run (per protocol).\n")
    L.append("## Per-prompt within-prompt Δ (cross-fitted predictions only)\n")
    L.append("| prompt | nH | nD | SURFACE | COMBINED | Δ | Δ salt0/1/2 | activation-alone |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in p["rows"]:
        ds = "/".join(_f(r["delta_by_salt"].get(str(s)), 3) for s in I.OUTER_SALTS)
        L.append(
            f"| {r['prompt_id']} | {r['n_honest']} | {r['n_deceptive']} | "
            f"{_f(r['surface_auroc'])} | {_f(r['combined_auroc'])} | {_f(r['delta_auroc'], 4)} | "
            f"{ds} | {_f(r['activation_auroc_descriptive_only'])} |"
        )
    L.append("\n## Phase 24H vs Phase 24I (descriptive only; 24H artifacts untouched)\n")
    h, i = cmp_["phase24h"], cmp_["phase24i"]
    L.append("| quantity | 24H `C_MULTILAYER|DELTA|t1` | 24I |")
    L.append("|---|---|---|")
    L.append(f"| global pooled ΔAUROC (mean over salts) | {_f(h['global_pooled_delta_auroc'], 4)} "
             f"| {_f(i['global_pooled_delta_auroc'], 4)} |")
    L.append(f"| within-prompt mean Δ | {_f(h['within_prompt_mean_delta'], 4)} "
             f"| {_f(i['within_prompt_mean_delta'], 4)} |")
    L.append(f"| within-prompt median Δ | {_f(h['within_prompt_median_delta'], 4)} "
             f"| {_f(i['within_prompt_median_delta'], 4)} |")
    L.append(f"| fraction prompts Δ>0 | {_f(h['within_prompt_fraction_gt_0'], 3)} "
             f"| {_f(i['within_prompt_fraction_gt_0'], 3)} |")
    L.append(f"| qualifying prompts | {h['n_qualifying']} | {i['n_qualifying']} |")
    L.append("\nPooled per-salt ΔAUROC (descriptive): " + "; ".join(
        f"salt {s}: {_f(pool[s]['delta_auroc'], 4)} (SURF {_f(pool[s]['surface_auroc'])}, "
        f"COMB {_f(pool[s]['combined_auroc'])}, ACT {_f(pool[s]['activation_auroc'])})"
        for s in map(str, I.OUTER_SALTS) if s in pool) + "\n")
    if results:
        L.append("## Pair-bearing training prompts and H×D pairs per outer fold\n")
        L.append("| salt | fold | pair-bearing train prompts | H×D pairs | train rollouts |")
        L.append("|---|---|---|---|---|")
        for r in sorted(results, key=lambda r: (r.salt, r.fold)):
            tp = r.info["train_pairs"]
            L.append(f"| {r.salt} | {r.fold} | {tp['n_pair_bearing_prompts']} | "
                     f"{tp['n_hxd_pairs']} | {tp['n_rollouts']} |")
    L.append("\n## Compute\n")
    L.append(f"- Wall: {manifest['wall_seconds'] / 3600:.2f} h; workers {manifest['n_workers']}, "
             f"BLAS threads {manifest['threads_env']}; CPU only, no GPU/model generation.\n"
             f"- Peak RSS: {manifest['memory']}\n")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")


def report_only() -> int:
    metrics = json.loads((OUT / "metrics.json").read_text())
    manifest = json.loads((OUT / "run_manifest.json").read_text())
    summary = json.loads((OUT / "input_summary.json").read_text())
    pairs = json.loads((OUT / "pair_counts_by_outer_fold.json").read_text())
    results = []
    for r in pairs:
        results.append(I.FoldResult(
            salt=r["salt"], fold=r["fold"], eval_rows=np.zeros(0, dtype=int),
            surf=np.zeros(0), act=np.zeros(0), comb=np.zeros(0),
            info={"train_pairs": {k: v for k, v in r.items()
                                  if k not in ("salt", "fold", "eval_prompts")}},
        ))
    write_report(metrics, manifest, summary, results)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--hash-only", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    if args.hash_only:
        tok = _load_tokenizer()
        records, _summary = build_records(tok)
        hsh = hash_dev_inputs(records, workers=4)
        ver = I.verify_against_phase24h_hashes(hsh["files"], REPO_ROOT)
        print(json.dumps(ver, indent=2))
        return 0 if ver["verified"] else 1
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
