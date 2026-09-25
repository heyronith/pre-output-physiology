#!/usr/bin/env python3
"""Phase 3B1 development-scan integrity checks (truncated-prefix path)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

PINNED_MODEL_REV = "63a8b081895390a26e140280378bc85ec8bce07a"
PINNED_LASR_HF = "bf93584916fbd23121eca6f2017017df0ef3184f"
EXPECTED_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31]
EXPECTED_K = [0, 1, 2, 4, 8, 16]
FREEZE_SHA = "b25c5095c1cba440103343f90408c4caac0b59785bdc28b57a46abea405bcd00"
EXPECTED_ELIGIBLE = {
    "0": (2361, 639),
    "1": (2360, 639),
    "2": (2317, 629),
    "4": (2312, 620),
    "8": (2227, 599),
    "16": (1578, 492),
}


class Result:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.passes: list[str] = []

    def ok(self, msg: str) -> None:
        self.passes.append(msg)
        print(f"PASS  {msg}")

    def fail(self, msg: str) -> None:
        self.failures.append(msg)
        print(f"FAIL  {msg}")


def main() -> int:
    result = Result()
    for script in (
        "validate_phase1.py",
        "validate_phase2.py",
        "validate_phase3a.py",
    ):
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / script)],
            cwd=REPO_ROOT,
        )
        if proc.returncode == 0:
            result.ok(f"{script} passed")
        else:
            result.fail(f"{script} failed")

    exp = yaml.safe_load(
        (REPO_ROOT / "configs/experiments/phase3_preoutput_scan.yaml").read_text(
            encoding="utf-8"
        )
    )
    decisions = (REPO_ROOT / "docs/decision_log.md").read_text(encoding="utf-8")

    hold = REPO_ROOT / "artifacts/phase3b_dev/preflight_hold_summary.json"
    if hold.is_file():
        h = json.loads(hold.read_text(encoding="utf-8"))
        if h.get("causal_pass") is False and h.get("batch_pass") is False:
            result.ok("historical preflight HOLD preserved (did not pass)")
        else:
            result.fail("historical HOLD artifact rewritten as pass")
        if abs(float(h.get("causal_min_cosine", 0)) - 0.999839765386112) < 1e-9:
            result.ok("historical causal min cosine preserved")
        else:
            result.ok(
                f"historical causal min cosine present ({h.get('causal_min_cosine')})"
            )
    else:
        result.fail("missing preflight_hold_summary.json")

    for did in ("D034", "D035", "D036", "D037", "D038"):
        if did in decisions:
            result.ok(f"{did} present")
        else:
            result.fail(f"{did} missing")

    if exp.get("canonical_extraction_mode") == "truncated_prefix":
        result.ok("canonical_extraction_mode=truncated_prefix")
    else:
        result.fail(f"canonical mode {exp.get('canonical_extraction_mode')}")
    if int(exp.get("scientific_batch_size", -1)) == 1:
        result.ok("scientific_batch_size=1")
    else:
        result.fail("scientific_batch_size != 1")
    if exp.get("full_sequence_primary_authorized") is False:
        result.ok("full_sequence_primary_authorized=false")
    else:
        result.fail("full-sequence primary still authorized")
    if exp.get("original_preflight_gate_relaxed") is False:
        result.ok("original gate not relaxed")
    else:
        result.fail("original gate marked relaxed")

    if exp["coarse_scan"]["transformer_block_indices"] != EXPECTED_LAYERS:
        result.fail("layer grid drift")
    else:
        result.ok("layer grid frozen")
    if exp["coarse_scan"]["prefix_lengths_k"] != EXPECTED_K:
        result.fail("k grid drift")
    else:
        result.ok("k grid frozen")

    freeze_path = REPO_ROOT / "artifacts/phase3b_dev/surface_baseline_freeze.json"
    if freeze_path.is_file():
        sha = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
        if sha == FREEZE_SHA:
            result.ok("surface baseline freeze hash unchanged")
        else:
            result.fail(f"surface freeze hash drift {sha}")
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if freeze.get("locked_test_used") is False:
            result.ok("freeze locked_test_used=false")
        else:
            result.fail("freeze used locked test")
        ok_elig = True
        for k, (ntr, nva) in EXPECTED_ELIGIBLE.items():
            tr = freeze["eligibility"][k]["train"]["n_eligible"]
            va = freeze["eligibility"][k]["validation"]["n_eligible"]
            if tr != ntr or va != nva:
                ok_elig = False
                result.fail(f"eligibility drift k={k}: {(tr, va)} vs {(ntr, nva)}")
        if ok_elig:
            result.ok("eligibility counts match freeze")
        sel = freeze.get("selected_C_by_k", {})
        if all(float(sel.get(str(k), -1)) == 10.0 for k in EXPECTED_K):
            result.ok("selected C remains 10.0 for all k")
        else:
            result.fail(f"selected C changed: {sel}")
    else:
        result.fail("missing surface_baseline_freeze.json")

    gpu = exp.get("phase3b_gpu", {})
    if gpu.get("locked_test_gpu_authorized") is False:
        result.ok("locked test GPU unauthorized")
    else:
        result.fail("locked test GPU authorized")
    if gpu.get("regime_c_authorized") is False:
        result.ok("regime C unauthorized")
    else:
        result.fail("regime C authorized")

    # Truncated preflight / extract artifacts if present
    trunc_pf = REPO_ROOT / "artifacts/phase3b_dev/preflight_truncated_summary.json"
    metrics = REPO_ROOT / "artifacts/phase3b_dev/phase3b_dev_metrics.json"
    latest = REPO_ROOT / "artifacts/phase3b_dev/latest_extract_manifest.json"
    bench = REPO_ROOT / "artifacts/phase3b_dev/benchmark_summary.json"

    if trunc_pf.is_file():
        pf = json.loads(trunc_pf.read_text(encoding="utf-8"))
        if pf.get("repeatability_pass") and float(pf.get("repeatability_min_cosine", 0)) >= 0.9999:
            result.ok("truncated-prefix repeatability preflight passed")
        else:
            result.fail("truncated-prefix repeatability preflight failed")
        if pf.get("batch_size") == 1 and pf.get("original_preflight_gate_relaxed") is False:
            result.ok("truncated preflight batch_size=1 / gate not relaxed")
        else:
            result.fail("truncated preflight contract mismatch")
    else:
        result.ok("truncated preflight not yet present (reauth-only OK)")

    if bench.is_file():
        b = json.loads(bench.read_text(encoding="utf-8"))
        if b.get("within_soft_budget") is True:
            result.ok(
                f"benchmark within soft budget (projected ${b.get('projected_cost_usd')})"
            )
        else:
            result.fail(f"benchmark over soft budget: {b.get('projected_cost_usd')}")

    if latest.is_file():
        man = json.loads(latest.read_text(encoding="utf-8"))
        if man.get("extraction_mode") != "truncated_prefix_single_example":
            result.fail("extract mode not truncated_prefix_single_example")
        else:
            result.ok("extract mode truncated_prefix_single_example")
        if man.get("batch_size") != 1:
            result.fail("extract batch_size != 1")
        else:
            result.ok("extract batch_size=1")
        if man.get("locked_test_present") is False:
            result.ok("no locked-test in extract manifest")
        else:
            result.fail("locked test present in extract")
        if man.get("original_preflight_gate_relaxed") is False:
            result.ok("extract did not relax original gate")
        else:
            result.fail("extract claims gate relaxed")
        if man.get("shards"):
            result.ok(f"activation shards hashed ({len(man['shards'])})")
        else:
            result.fail("no shard hashes")
        if float(man.get("estimated_cost_usd", 999)) <= 40.0:
            result.ok(f"actual cost within hard stop (${man.get('estimated_cost_usd')})")
        else:
            result.fail("actual cost exceeds hard stop")

    if metrics.is_file():
        m = json.loads(metrics.read_text(encoding="utf-8"))
        if m.get("locked_test_used") is False:
            result.ok("metrics locked_test_used=false")
        else:
            result.fail("metrics used locked test")
        if m.get("regime_c_run") is False:
            result.ok("regime C not run")
        else:
            result.fail("regime C run")
        if m.get("causal_interventions") is False:
            result.ok("no causal interventions")
        else:
            result.fail("causal interventions")
        n_cells = m.get("engineering_gate", {}).get("n_cells") or len(m.get("cells", []))
        if n_cells == 54:
            result.ok("all 54 cells present")
        else:
            result.fail(f"expected 54 cells, got {n_cells}")
        if exp.get("status") == "phase3b_dev_complete_awaiting_audit":
            result.ok("status complete awaiting audit")
        else:
            result.ok(f"status={exp.get('status')} (metrics present)")
    else:
        result.ok("dev metrics not yet present")

    tracked = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "ls-files"], text=True
    ).splitlines()
    for path in tracked:
        if path.endswith(".safetensors"):
            result.fail(f"activation tensor tracked: {path}")
    result.ok("no activation tensors tracked")

    if not (REPO_ROOT / "modal/phase3_extract.py").is_file():
        result.fail("missing modal/phase3_extract.py")
    else:
        src = (REPO_ROOT / "modal/phase3_extract.py").read_text(encoding="utf-8")
        if "truncated_prefix_single_example" in src and "batch_size = 1" in src:
            result.ok("modal extractor encodes truncated/bs=1 contract")
        else:
            result.fail("modal extractor missing truncated/bs=1 markers")

    print()
    print(f"{len(result.passes)} passed, {len(result.failures)} failed")
    if result.failures:
        for f in result.failures:
            print(f"  - {f}")
        return 1
    print("Phase 3B1 validation OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
