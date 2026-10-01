#!/usr/bin/env python3
"""Phase 24G-R: correct LOPO + nested diagnostics with full stacked pipelines.

Requires exactly 1000 nested reps and forbids proxy for primary diagnostics.
Prior Phase-24G exploratory artifacts are snapshotted by hash and not overwritten.
"""

from __future__ import annotations

# Limit BLAS/OpenMP before numpy/sklearn init (critical for fork pools).
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import contextlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_phase24g_diagnostics import (  # noqa: E402
    _build_all_records,
    _load_npz,
    _load_tokenizer,
    _primary,
    discovery_delta_grid_fast,
    eval_coordinate,
)

from pre_output_physiology.phase24c_design import (  # noqa: E402
    select_candidate_region,
    sha256_file,
)
from pre_output_physiology.phase24g_diagnostics import (  # noqa: E402
    candidate_in_neighborhood,
)
from pre_output_physiology.phase24g_r_correction import (  # noqa: E402
    ANALYSIS_SEED,
    GUARANTEE,
    LOPO_N,
    NESTED_REPS,
    PROXY_COMPARE_NESTED,
    STARTING_SHA,
    STATUS,
    assert_nested_reps_exact,
    assert_no_proxy_in_primary,
    assert_phase24f_immutable,
    category_change_statement,
    classify_diagnostic_category,
    compare_proxy_vs_full,
    snapshot_prior_24g_hashes,
    summarize_lopo,
    summarize_nested,
    verify_prior_24g_unchanged,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

OUT = REPO_ROOT / "artifacts/phase24g_r_correction"
PRIOR_24G = REPO_ROOT / "artifacts/phase24g_diagnostics"
CFG = REPO_ROOT / "configs/experiments/phase24g_r_diagnostic_correction.yaml"
REPORT = REPO_ROOT / "reports/phase24g_r_diagnostic_correction.md"
AUDIT = REPO_ROOT / "reports/phase24g_r_code_audit_manifest.md"
DECISION_LOG = REPO_ROOT / "docs/decision_log.md"

# Worker globals (populated under fork)
_G_CACHE: dict[str, dict] | None = None
_G_TOK = None
_G_DEV: list[dict] | None = None


@contextlib.contextmanager
def _silence_stdout():
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        old = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old


def _init_worker(cache: dict, tok, dev: list[dict]) -> None:
    global _G_CACHE, _G_TOK, _G_DEV
    _G_CACHE = cache
    _G_TOK = tok
    _G_DEV = dev
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _make_nested_splits(
    prompts: list[str], rep: int, seed: int = ANALYSIS_SEED
) -> tuple[set[str], set[str], set[str]]:
    rng = np.random.default_rng(seed + rep)
    order = list(prompts)
    rng.shuffle(order)
    return set(order[:14]), set(order[14:21]), set(order[21:28])


def _run_one_discovery(
    train_recs: list[dict],
    val_recs: list[dict],
    cache: dict,
    tok,
    *,
    proxy: bool,
) -> tuple[dict[str, Any] | None, float]:
    if not proxy:
        assert_no_proxy_in_primary(False)
    with _silence_stdout():
        grid = discovery_delta_grid_fast(
            train_recs, val_recs, cache, tok, proxy_delta=proxy
        )
    cand = select_candidate_region(grid)
    delta = float("nan")
    if cand is not None:
        delta = float(cand["layer_delta_auroc"])
    return cand, delta


def _nested_rep_full(payload: dict[str, Any]) -> dict[str, Any]:
    """Full-pipeline nested repetition worker."""
    assert _G_CACHE is not None and _G_TOK is not None and _G_DEV is not None
    rep = int(payload["rep"])
    tr_set = set(payload["tr"])
    va_set = set(payload["va"])
    ho_set = set(payload["ho"])
    proxy = bool(payload["proxy"])
    if not proxy:
        assert_no_proxy_in_primary(False)

    tr_r = [r for r in _G_DEV if r["prompt_id"] in tr_set]
    va_r = [r for r in _G_DEV if r["prompt_id"] in va_set]
    ho_r = [r for r in _G_DEV if r["prompt_id"] in ho_set]
    cand, disc_delta = _run_one_discovery(
        tr_r, va_r, _G_CACHE, _G_TOK, proxy=proxy
    )
    if cand is None:
        return {
            "rep": rep,
            "candidate": None,
            "discovery_delta": float("nan"),
            "heldout_delta": float("nan"),
            "optimism": float("nan"),
            "proxy": proxy,
            "split": {"tr": sorted(tr_set), "va": sorted(va_set), "ho": sorted(ho_set)},
        }
    # Held-out eval with full stacked contrast at selected coordinate
    fit_recs = tr_r + va_r
    with _silence_stdout():
        hm = eval_coordinate(
            fit_recs,
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
        disc_delta - held
        if np.isfinite(disc_delta) and np.isfinite(held)
        else float("nan")
    )
    return {
        "rep": rep,
        "candidate": {
            "time": cand["candidate_time"],
            "layer": cand["candidate_layer"],
            "band": cand["band"],
        },
        "discovery_delta": disc_delta,
        "heldout_delta": held,
        "optimism": opt,
        "proxy": proxy,
        "split": {"tr": sorted(tr_set), "va": sorted(va_set), "ho": sorted(ho_set)},
    }


def _lopo_payload_row(
    left: str,
    train: list[dict],
    val: list[dict],
    dev: list[dict],
    all_dev_prompts: list[str],
    *,
    proxy: bool,
    idx: int,
) -> dict[str, Any]:
    """Build LOPO train/val prompt sets (serial, then executed in workers)."""
    train_pids = {r["prompt_id"] for r in train}
    tr_r = [r for r in train if r["prompt_id"] != left]
    va_r = [r for r in val if r["prompt_id"] != left]
    if len({r["prompt_id"] for r in tr_r}) < 4 or len({r["prompt_id"] for r in va_r}) < 2:
        rem = [p for p in all_dev_prompts if p != left]
        rng = np.random.default_rng(ANALYSIS_SEED + idx)
        rng.shuffle(rem)
        n_tr = max(4, int(0.7 * len(rem)))
        tr_set, va_set = set(rem[:n_tr]), set(rem[n_tr:])
    else:
        tr_set = {r["prompt_id"] for r in tr_r}
        va_set = {r["prompt_id"] for r in va_r}
    return {
        "left": left,
        "left_out_split": "train" if left in train_pids else "validation",
        "tr": sorted(tr_set),
        "va": sorted(va_set),
        "proxy": proxy,
        "idx": idx,
    }


def _lopo_worker(payload: dict[str, Any]) -> dict[str, Any]:
    assert _G_CACHE is not None and _G_TOK is not None and _G_DEV is not None
    proxy = bool(payload["proxy"])
    if not proxy:
        assert_no_proxy_in_primary(False)
    tr_set = set(payload["tr"])
    va_set = set(payload["va"])
    tr_r = [r for r in _G_DEV if r["prompt_id"] in tr_set]
    va_r = [r for r in _G_DEV if r["prompt_id"] in va_set]
    cand, delta = _run_one_discovery(tr_r, va_r, _G_CACHE, _G_TOK, proxy=proxy)
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


def main() -> int:
    # Limit BLAS threads so fork workers don't oversubscribe
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    t_wall0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    assert_nested_reps_exact()

    print("Asserting Phase-24F immutability…", flush=True)
    imm = assert_phase24f_immutable(REPO_ROOT)
    write_json(OUT / "phase24f_immutability.json", imm)

    print("Snapshotting prior Phase-24G artifact hashes…", flush=True)
    prior_snap = snapshot_prior_24g_hashes(REPO_ROOT)
    write_json(OUT / "prior_phase24g_artifact_hashes.json", prior_snap)
    if not prior_snap["all_present"]:
        raise SystemExit("STOP: prior Phase-24G artifacts incomplete")

    # Load runtime benchmark if present; else require it
    bench_path = OUT / "runtime_benchmark.json"
    if not bench_path.exists():
        raise SystemExit(
            "STOP: run scripts/benchmark_phase24g_r_full_grid.py first"
        )
    bench = json.loads(bench_path.read_text(encoding="utf-8"))
    if bench.get("verdict") == "PROHIBITIVE":
        raise SystemExit(
            "STOP: full-pipeline 1000-nested runtime prohibitive; "
            "do not reduce reps or use proxy without authorization"
        )

    print("Loading tokenizer + records + NPZs…", flush=True)
    tok = _load_tokenizer()
    by_split = _build_all_records(tok)
    train, val, test = by_split["train"], by_split["validation"], by_split["test"]
    del test  # LOPO/nested use DEVELOPMENT only; avoid loading TEST NPZs
    dev = train + val
    cache: dict[str, dict] = {}
    for r in train + val:
        cache[r["trajectory_id"]] = _load_npz(Path(r["npz_path"]))
    print(f"Loaded {len(cache)} DEVELOPMENT NPZs into cache", flush=True)
    all_dev_prompts = sorted({r["prompt_id"] for r in _primary(dev)})
    assert len(all_dev_prompts) == LOPO_N

    # Serial in-process execution avoids fork COW / swap thrash on this host.
    # Measured ~98s/full grid in-RAM; 1000 nested ≈ 27h wall.
    n_workers = 1
    print(
        f"Using serial in-process execution (n_workers={n_workers}); "
        "multiprocess pools swap-thrash on 24GB with full stacked grids",
        flush=True,
    )
    _init_worker(cache, tok, dev)

    def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
            fh.flush()

    def _load_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows

    # --- 1. Full-pipeline LOPO (serial) ---
    print("=== 1. Full-pipeline LOPO (28) ===", flush=True)
    lopo_full_payloads = [
        _lopo_payload_row(
            left, train, val, dev, all_dev_prompts, proxy=False, idx=i
        )
        for i, left in enumerate(all_dev_prompts)
    ]
    lopo_ckpt = OUT / "lopo_full_checkpoint.jsonl"
    lopo_full = _load_jsonl(lopo_ckpt)
    done_left = {r["left_out_prompt"] for r in lopo_full}
    remaining_lopo = [
        p for p in lopo_full_payloads if p["left"] not in done_left
    ]
    print(
        f"  LOPO-full resume: {len(done_left)} done, "
        f"{len(remaining_lopo)} remaining",
        flush=True,
    )
    for i, payload in enumerate(remaining_lopo):
        row = _lopo_worker(payload)
        lopo_full.append(row)
        _append_jsonl(lopo_ckpt, row)
        print(
            f"  LOPO-full {len(done_left)+i+1}/28 leave "
            f"{row['left_out_prompt']}",
            flush=True,
        )
    lopo_full.sort(key=lambda r: r["left_out_prompt"])
    lopo_full_summary = summarize_lopo(lopo_full)
    write_json(
        OUT / "lopo_full_pipeline.json",
        {"summary": lopo_full_summary, "rows": lopo_full},
    )

    # Proxy LOPO for comparison (same leave-outs / same tr-va prompt sets)
    print("=== Proxy LOPO comparison (28) ===", flush=True)
    lopo_proxy_payloads = [
        {**p, "proxy": True} for p in lopo_full_payloads
    ]
    lopo_proxy_ckpt = OUT / "lopo_proxy_checkpoint.jsonl"
    lopo_proxy = _load_jsonl(lopo_proxy_ckpt)
    done_proxy = {r["left_out_prompt"] for r in lopo_proxy}
    remaining_proxy = [
        p for p in lopo_proxy_payloads if p["left"] not in done_proxy
    ]
    print(
        f"  LOPO-proxy resume: {len(done_proxy)} done, "
        f"{len(remaining_proxy)} remaining",
        flush=True,
    )
    for i, payload in enumerate(remaining_proxy):
        row = _lopo_worker(payload)
        lopo_proxy.append(row)
        _append_jsonl(lopo_proxy_ckpt, row)
        print(
            f"  LOPO-proxy {len(done_proxy)+i+1}/28 leave "
            f"{row['left_out_prompt']}",
            flush=True,
        )
    lopo_proxy.sort(key=lambda r: r["left_out_prompt"])
    write_json(
        OUT / "lopo_proxy_comparison_runs.json",
        {"rows": lopo_proxy, "note": "comparison only; not primary"},
    )

    # --- 2. Nested 1000 full-pipeline (serial) ---
    print(f"=== 2. Nested full-pipeline ({NESTED_REPS} reps) ===", flush=True)
    payloads = []
    for rep in range(NESTED_REPS):
        tr, va, ho = _make_nested_splits(all_dev_prompts, rep)
        payloads.append(
            {
                "rep": rep,
                "tr": sorted(tr),
                "va": sorted(va),
                "ho": sorted(ho),
                "proxy": False,
            }
        )
    # Record splits before execution
    write_json(
        OUT / "nested_prompt_splits.json",
        {
            "analysis_seed": ANALYSIS_SEED,
            "n_reps": NESTED_REPS,
            "splits": [
                {"rep": p["rep"], "tr": p["tr"], "va": p["va"], "ho": p["ho"]}
                for p in payloads
            ],
        },
    )

    nested_ckpt = OUT / "nested_full_checkpoint.jsonl"
    nested_full = _load_jsonl(nested_ckpt)
    done_reps = {int(r["rep"]) for r in nested_full}
    remaining_nested = [p for p in payloads if p["rep"] not in done_reps]
    print(
        f"  nested-full resume: {len(done_reps)} done, "
        f"{len(remaining_nested)} remaining",
        flush=True,
    )
    nested_full_list: list[dict[str, Any]] = list(nested_full)
    t_nested0 = time.perf_counter()
    for i, payload in enumerate(remaining_nested):
        row = _nested_rep_full(payload)
        nested_full_list.append(row)
        _append_jsonl(nested_ckpt, row)
        n_done = len(done_reps) + i + 1
        if n_done % 5 == 0 or n_done == NESTED_REPS or i == 0:
            elapsed = time.perf_counter() - t_nested0
            rate = (i + 1) / max(elapsed, 1e-9)
            eta_h = (len(remaining_nested) - i - 1) / max(rate, 1e-9) / 3600
            print(
                f"  nested-full {n_done}/{NESTED_REPS} "
                f"(rate={rate*3600:.1f}/h eta={eta_h:.1f}h)",
                flush=True,
            )
    nested_full_list.sort(key=lambda r: r["rep"])
    nested_full = nested_full_list
    if len(nested_full) != NESTED_REPS:
        raise SystemExit(
            f"STOP: nested completed {len(nested_full)} != required {NESTED_REPS}"
        )
    nested_elapsed = time.perf_counter() - t_nested0
    nested_summary = summarize_nested(nested_full)
    nested_summary["wall_seconds"] = nested_elapsed
    nested_summary["n_workers"] = n_workers
    write_json(
        OUT / "nested_full_pipeline_1000.json",
        {"summary": nested_summary, "rows": nested_full},
    )

    # --- 3. Proxy nested first 50 with identical splits ---
    print(
        f"=== 3. Proxy nested comparison (first {PROXY_COMPARE_NESTED}) ===",
        flush=True,
    )
    proxy_payloads = [
        {**payloads[rep], "proxy": True} for rep in range(PROXY_COMPARE_NESTED)
    ]
    nested_proxy_ckpt = OUT / "nested_proxy_checkpoint.jsonl"
    nested_proxy = _load_jsonl(nested_proxy_ckpt)
    done_proxy_reps = {int(r["rep"]) for r in nested_proxy}
    remaining_np = [p for p in proxy_payloads if p["rep"] not in done_proxy_reps]
    print(
        f"  nested-proxy resume: {len(done_proxy_reps)} done, "
        f"{len(remaining_np)} remaining",
        flush=True,
    )
    for i, payload in enumerate(remaining_np):
        row = _nested_rep_full(payload)
        nested_proxy.append(row)
        _append_jsonl(nested_proxy_ckpt, row)
        n_done = len(done_proxy_reps) + i + 1
        if n_done % 5 == 0 or n_done == PROXY_COMPARE_NESTED:
            print(
                f"  nested-proxy {n_done}/{PROXY_COMPARE_NESTED}",
                flush=True,
            )
    nested_proxy.sort(key=lambda r: r["rep"])
    write_json(
        OUT / "nested_proxy_comparison_50.json",
        {"rows": nested_proxy, "note": "comparison only; not primary"},
    )

    proxy_vs_full = {
        "lopo": compare_proxy_vs_full(lopo_proxy, lopo_full),
        "nested_first_50": compare_proxy_vs_full(
            nested_proxy, nested_full[:PROXY_COMPARE_NESTED]
        ),
    }
    write_json(OUT / "proxy_vs_full_comparison.json", proxy_vs_full)

    # --- 5. Reclassify ---
    old_cat = json.loads(
        (PRIOR_24G / "diagnostic_category.json").read_text(encoding="utf-8")
    )["category"]
    prior_topo = json.loads(
        (PRIOR_24G / "val_test_topology.json").read_text(encoding="utf-8")
    )
    prior_same = json.loads(
        (PRIOR_24G / "same_prompt_robustness.json").read_text(encoding="utf-8")
    )
    prior_probe = json.loads(
        (PRIOR_24G / "probe_direction_stability.json").read_text(encoding="utf-8")
    )
    p24f = json.loads(
        (
            REPO_ROOT / "artifacts/phase24f_confirmation/primary_metrics.json"
        ).read_text(encoding="utf-8")
    )
    evidence = {
        "topology_pearson": prior_topo["overall"]["pearson"],
        "lopo_exact_frac": lopo_full_summary["frac_exact_t1_l20"],
        "lopo_neighborhood_frac": lopo_full_summary["frac_neighborhood"],
        "optimism_gap": nested_summary["optimism_gap_mean"],
        "test_same_prompt_frac_gt0": prior_same.get(
            "test_ge2_combined_summary", {}
        ).get("fraction_gt_0", float("nan")),
        "probe_mean_cosine": prior_probe["coordinates"]["frozen_candidate"][
            "stability"
        ]["mean_pairwise_cosine"],
        "test_delta_auroc": p24f["delta_auroc"],
        "test_ci_width": float(
            p24f["delta_auroc_bootstrap"]["ci95"][1]
            - p24f["delta_auroc_bootstrap"]["ci95"][0]
        ),
    }
    new_cat = classify_diagnostic_category(evidence)
    change = category_change_statement(old_cat, new_cat)
    category_art = {
        "previous_category_phase24g": old_cat,
        "corrected_category": new_cat,
        "category_C_status": change,
        "evidence": evidence,
        "definitions": {
            "A": "Stable region, underpowered confirmation",
            "B": "Real but prompt-specific / heterogeneous signal",
            "C": "Candidate-selection instability",
            "D": "Little robust evidence",
        },
        "phase24f_remains_fail": True,
        "pipeline": "full_stacked",
        "nested_reps": NESTED_REPS,
    }
    write_json(OUT / "corrected_diagnostic_category.json", category_art)

    # Verify prior 24G unchanged
    prior_check = verify_prior_24g_unchanged(REPO_ROOT, prior_snap)
    write_json(OUT / "prior_phase24g_unchanged_verification.json", prior_check)
    if not prior_check["unchanged"]:
        raise SystemExit(f"STOP: prior Phase-24G artifacts mutated: {prior_check}")

    wall = time.perf_counter() - t_wall0
    git_commit = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
        .decode()
        .strip()
    )
    manifest = {
        "created_at": utc_now_iso(),
        "starting_sha": STARTING_SHA,
        "git_commit": git_commit,
        "status": STATUS,
        "nested_reps": NESTED_REPS,
        "lopo_n": LOPO_N,
        "proxy_used_in_primary": False,
        "phase24f_immutable": True,
        "prior_24g_unchanged": True,
        "wall_seconds": wall,
        "nested_wall_seconds": nested_elapsed,
        "n_workers": n_workers,
        "corrected_category": new_cat,
        "category_C_status": change,
        "guarantee": GUARANTEE,
    }
    write_json(OUT / "correction_manifest.json", manifest)

    # Config + report
    CFG.write_text(
        f"""# Phase 24G-R — corrected LOPO/nested diagnostics

experiment_id: phase24g_r_diagnostic_correction
phase: phase24g_r
status: {STATUS}
starting_sha: {STARTING_SHA}
nested_reps: {NESTED_REPS}
lopo_n: {LOPO_N}
proxy_used_in_primary: false
corrected_category: {new_cat}
category_C_status: {change}
phase24f_primary_pass: false

authorizations:
  exploratory_corrected_diagnostics: true
  primary_endpoint_redefinition: false
  sae_analysis_authorized: false
  causal_intervention_authorized: false
""",
        encoding="utf-8",
    )

    ns = nested_summary
    report = f"""# Phase 24G-R — Corrected LOPO / nested diagnostics

**Status:** `{STATUS}`

{GUARANTEE}

## Phase-24F immutability

Verified: **{imm['immutable']}** (ΔAUROC={imm['delta_auroc']:.4f}, CI={imm['ci95']})

## Prior Phase-24G artifacts

Snapshotted and verified unchanged: **{prior_check['unchanged']}**

## 1. Full-pipeline LOPO (n={LOPO_N})

Exact t=1/L20: **{lopo_full_summary['frac_exact_t1_l20']:.4f}**
Frac t=1: **{lopo_full_summary['frac_time_1']:.4f}**
Frac L20: **{lopo_full_summary['frac_layer_20']:.4f}**
Neighborhood: **{lopo_full_summary['frac_neighborhood']:.4f}**
Time hist: `{json.dumps(lopo_full_summary['time_hist'])}`
Layer hist: `{json.dumps(lopo_full_summary['layer_hist'])}`

Pipeline: full SURFACE+ACTIVATION − SURFACE (proxy forbidden).

## 2. Nested optimism ({NESTED_REPS} reps)

Completed: **{ns['n_reps_completed']} / {NESTED_REPS}**
Candidate production: **{ns['frac_candidate_exists']:.4f}**
Discovery ΔAUROC mean/median:
**{ns['expected_discovery_delta_mean']:.4f}** /
**{ns['expected_discovery_delta_median']:.4f}**
Held-out ΔAUROC mean/median:
**{ns['expected_heldout_delta_mean']:.4f}** /
**{ns['expected_heldout_delta_median']:.4f}**
Optimism gap mean/median:
**{ns['optimism_gap_mean']:.4f}** / **{ns['optimism_gap_median']:.4f}**
Optimism empirical 2.5–97.5% interval (of nested gaps; not a CI on the mean):
**{ns['optimism_gap_empirical_percentile_025_975']}**
Frac exact t=1/L20: **{ns['frac_exact_t1_l20']:.4f}**
Time hist: `{json.dumps(ns['time_hist'])}`
Layer hist: `{json.dumps(ns['layer_hist'])}`
Nested wall seconds: **{nested_elapsed:.1f}** (workers={n_workers})

## 3. Proxy vs full

`{json.dumps(proxy_vs_full, indent=2)}`

## 5. Corrected category

Previous (Phase 24G): **{old_cat}**
Corrected: **{new_cat}** ({category_art['definitions'][new_cat]})
Category C status: **{change}**

Phase-24F FAIL unchanged.
"""
    REPORT.write_text(report, encoding="utf-8")

    # Audit manifest
    diff_stat = subprocess.check_output(
        ["git", "diff", "--stat", STARTING_SHA], cwd=REPO_ROOT
    ).decode()
    audit = f"""# Phase 24G-R code audit manifest

Starting SHA: `{STARTING_SHA}`
Working tree at report generation: `{git_commit}`
Final status: `{STATUS}`

## Changed files (purpose)

| File | Purpose |
|---|---|
| `src/pre_output_physiology/phase24g_r_correction.py` | Constants, summaries, compare |
| `scripts/run_phase24g_r_correction.py` | Full-pipeline LOPO + nested runner |
| `scripts/benchmark_phase24g_r_full_grid.py` | Runtime gate before 1000 nested |
| `tests/test_phase24g_r_correction.py` | Critical invariant tests |
| `configs/experiments/phase24g_r_diagnostic_correction.yaml` | Config freeze |
| `reports/phase24g_r_diagnostic_correction.md` | Correction report |
| `reports/phase24g_r_code_audit_manifest.md` | This audit package |
| `artifacts/phase24g_r_correction/*` | Corrected outputs only |
| `docs/decision_log.md` | D165 entry |

## Critical functions

### LOPO
- `scripts/run_phase24g_r_correction.py::_lopo_worker`
- `scripts/run_phase24g_r_correction.py::_lopo_payload_row`
- `scripts/run_phase24g_r_correction.py::_run_one_discovery`
  with `proxy=False`

### Nested resampling
- `scripts/run_phase24g_r_correction.py::_make_nested_splits`
- `scripts/run_phase24g_r_correction.py::_nested_rep_full`
- constant `NESTED_REPS = 1000` in `phase24g_r_correction.py`

### Candidate selection
- `pre_output_physiology.phase24c_design.select_candidate_region`
  (frozen Phase-24E rule)

### Stacking / OOF
- `run_phase24g_diagnostics.discovery_maps_fast(..., proxy_delta=False)`
- `run_phase24g_diagnostics.eval_coordinate`
  (full TEXT/LOGITS/SURFACE/ACTIVATION stack)
- `phase24e_discovery.oof_log_odds`, `fit_predict_log_odds`

### Tests
- `tests/test_phase24g_r_correction.py`

## Confirmations

- Prior Phase-24G artifacts **not modified**
  (hash verification: {prior_check['unchanged']})
- Phase-24F primary artifacts **not overwritten**
  (immutability: {imm['immutable']})
- Primary LOPO/nested used **full stacked pipelines**
  (`proxy_used_in_primary=false`)
- Nested repetitions completed:
  **{ns['n_reps_completed']}** (required 1000)

## git diff --stat from starting SHA

```
{diff_stat}
```

## Final commit SHA

To be filled after freeze commit (see completion report ending SHA).
"""
    AUDIT.write_text(audit, encoding="utf-8")

    # hashes
    hash_files = [p for p in OUT.iterdir() if p.is_file() and p.name != "artifact_hashes.json"]
    hash_files.extend([CFG, REPORT, AUDIT])
    hashes = {p.name: sha256_file(str(p)) for p in sorted(set(hash_files), key=lambda x: x.name)}
    write_json(OUT / "artifact_hashes.json", hashes)

    dlog = DECISION_LOG.read_text(encoding="utf-8")
    if "### D165 — Phase 24G-R" not in dlog:
        entry = (
            "\n### D165 — Phase 24G-R corrected full-pipeline diagnostics\n\n"
            "- **Type:** **OUR RESEARCH DECISION**\n"
            f"- **Decision:** Starting from Phase-24G `{STARTING_SHA[:7]}…`, "
            "Phase 24G-R re-ran LOPO (28) and nested optimism (exactly 1000) "
            "using full frozen stacked SURFACE+ACTIVATION pipelines (no proxy). "
            f"Corrected category **{new_cat}** ({change} prior Category C). "
            "Phase-24F FAIL unchanged. Prior 24G exploratory artifacts retained "
            "by hash. Status awaiting independent code audit.\n"
            "- **Date:** 2026-10-01\n"
        )
        DECISION_LOG.write_text(dlog.rstrip() + "\n" + entry, encoding="utf-8")

    print(
        json.dumps(
            {
                "status": STATUS,
                "category": new_cat,
                "category_C_status": change,
                "lopo_exact": lopo_full_summary["frac_exact_t1_l20"],
                "nested_completed": nested_summary["n_reps_completed"],
                "optimism_gap": nested_summary["optimism_gap_mean"],
                "wall_seconds": wall,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
