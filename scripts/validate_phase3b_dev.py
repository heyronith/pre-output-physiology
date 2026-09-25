#!/usr/bin/env python3
"""Phase 3B1 development-scan integrity checks."""

from __future__ import annotations

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


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


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

    exp = load_yaml(REPO_ROOT / "configs/experiments/phase3_preoutput_scan.yaml")
    model = load_yaml(REPO_ROOT / "configs/models/mistral_7b_instruct_v02.yaml")
    ds = load_yaml(REPO_ROOT / "configs/datasets/roleplay_deception.yaml")

    if model.get("revision") == PINNED_MODEL_REV:
        result.ok("model revision pinned")
    else:
        result.fail("model revision mismatch")
    if ds.get("lasr_hf_revision") == PINNED_LASR_HF:
        result.ok("dataset revision pinned")
    else:
        result.fail("dataset revision mismatch")

    layers = exp["coarse_scan"]["transformer_block_indices"]
    ks = exp["coarse_scan"]["prefix_lengths_k"]
    if layers == EXPECTED_LAYERS and ks == EXPECTED_K:
        result.ok("frozen layer/k grid")
    else:
        result.fail(f"grid drift layers={layers} ks={ks}")

    pe = exp.get("primary_endpoints", {})
    if pe.get("regime_a", {}).get("layer") == 12 and pe.get("regime_a", {}).get("k") == 0:
        result.ok("primary Regime A frozen at 12/k0")
    else:
        result.fail(f"regime A endpoints {pe.get('regime_a')}")
    if pe.get("regime_b", {}).get("layer") == 12 and pe.get("regime_b", {}).get("k") == 1:
        result.ok("primary Regime B frozen at 12/k1")
    else:
        result.fail(f"regime B endpoints {pe.get('regime_b')}")

    elig = exp.get("eligibility", {})
    if elig.get("k_gt_0") == "canonical_response_token_length_strictly_greater_than_k":
        result.ok("eligibility rule response_length > k")
    else:
        result.fail(f"eligibility {elig}")

    freeze_path = REPO_ROOT / "artifacts/phase3b_dev/surface_baseline_freeze.json"
    if freeze_path.is_file():
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if freeze.get("locked_test_used") is False:
            result.ok("surface baseline freeze: locked_test_used=false")
        else:
            result.fail("surface baseline freeze used locked test")
        if set(freeze.get("selected_C_by_k", {})) == {str(k) for k in EXPECTED_K}:
            result.ok("surface baseline C frozen for all k")
        else:
            result.fail("missing C for some k")
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
    if gpu.get("regime_c_status") == "hold_insufficient_onset_resolution":
        result.ok("regime C HOLD recorded")
    else:
        result.fail("regime C status missing")

    # Preflight artifact if present
    preflight_paths = list(
        (REPO_ROOT / "artifacts" / "runs").glob("phase3b1_preflight_*/preflight.json")
    )
    metrics_path = REPO_ROOT / "artifacts/phase3b_dev/phase3b_dev_metrics.json"
    if preflight_paths:
        latest = max(preflight_paths, key=lambda p: p.stat().st_mtime)
        pf = json.loads(latest.read_text(encoding="utf-8"))
        if pf.get("locked_test_used") is False:
            result.ok("preflight locked_test_used=false")
        else:
            result.fail("preflight used locked test")
        if pf.get("causal_pass") and pf.get("batch_pass"):
            result.ok("causal+batch preflight passed")
            if not metrics_path.is_file():
                result.fail(
                    "preflight passed but phase3b_dev_metrics.json missing "
                    "(run analyze after extract)"
                )
            else:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                if metrics.get("locked_test_used") is False:
                    result.ok("dev metrics: locked_test_used=false")
                else:
                    result.fail("dev metrics used locked test")
                if metrics.get("regime_c_run") is False:
                    result.ok("regime C not run")
                else:
                    result.fail("regime C was run")
                if metrics.get("causal_interventions") is False:
                    result.ok("no causal interventions")
                else:
                    result.fail("causal interventions present")
                if metrics.get("engineering_gate", {}).get("all_54_cells_present"):
                    result.ok("all 54 cells present")
                else:
                    result.fail("missing cells in metrics")
        else:
            result.ok(
                "preflight HOLD recorded "
                f"(causal_pass={pf.get('causal_pass')} "
                f"batch_pass={pf.get('batch_pass')} "
                f"causal_min={pf.get('causal_min_cosine')} "
                f"batch_min={pf.get('batch_min_cosine')})"
            )
            if metrics_path.is_file():
                result.fail(
                    "dev metrics exist despite failed preflight "
                    "(full extraction must not proceed)"
                )
            else:
                result.ok("no full-extract metrics after preflight HOLD")
    else:
        result.ok("no preflight artifact yet (authorization/freeze-only state OK)")

    # Ensure no locked-test activation artifact names tracked
    tracked = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "ls-files"], text=True
    ).splitlines()
    for path in tracked:
        if "locked_test" in path and path.endswith(".safetensors"):
            result.fail(f"locked-test activation tracked: {path}")
        if "phase3" in path and path.endswith(".safetensors"):
            result.fail(f"phase3 activation tracked: {path}")
    result.ok("no phase3/locked-test activation tensors tracked")

    if not (REPO_ROOT / "modal/phase3_extract.py").is_file():
        result.fail("missing modal/phase3_extract.py")
    else:
        result.ok("phase3 modal extractor present")

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
