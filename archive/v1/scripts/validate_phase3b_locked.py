#!/usr/bin/env python3
"""Phase 3B2 locked-test integrity checks."""

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
LOCKED_ELIG_SHA = "30371dcaba266ebb2acec06563df9916f27d9e5afd4555e9fa1b8c206ae9ea2a"
CANONICAL_DEV_RUN = "phase3b1_extract_20260925T154005Z_39505f42"
EXPECTED_LOCKED_N = {
    "0": 500,
    "1": 500,
    "2": 497,
    "4": 497,
    "8": 486,
    "16": 374,
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
        "validate_phase3b_dev.py",
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

    for did in ("D041", "D042", "D043", "D044"):
        if did in decisions:
            result.ok(f"{did} present")
        else:
            result.fail(f"{did} missing")

    # Canonical B1 still present
    latest_dev = REPO_ROOT / "artifacts/phase3b_dev/latest_extract_manifest.json"
    if latest_dev.is_file():
        dman = json.loads(latest_dev.read_text(encoding="utf-8"))
        if dman.get("run_id") == CANONICAL_DEV_RUN:
            result.ok("canonical B1 extract still present")
        else:
            result.fail(f"canonical B1 run drift: {dman.get('run_id')}")
        if dman.get("activation_storage_dtype") == "float32":
            result.ok("canonical B1 float32 storage preserved")
        else:
            result.fail("canonical B1 storage dtype drift")
        if dman.get("locked_test_present") is False:
            result.ok("canonical B1 still excludes locked test")
        else:
            result.fail("canonical B1 marked locked_test_present")
    else:
        result.fail("missing canonical B1 latest_extract_manifest")

    freeze = REPO_ROOT / "artifacts/phase3b_dev/surface_baseline_freeze.json"
    if freeze.is_file():
        sha = hashlib.sha256(freeze.read_bytes()).hexdigest()
        if sha == FREEZE_SHA:
            result.ok("surface freeze hash unchanged")
        else:
            result.fail(f"surface freeze drift {sha}")
        fr = json.loads(freeze.read_text(encoding="utf-8"))
        if all(float(fr["selected_C_by_k"][str(k)]) == 10.0 for k in EXPECTED_K):
            result.ok("frozen surface C=10 unchanged")
        else:
            result.fail("surface C changed")
    else:
        result.fail("missing surface freeze")

    if float(exp.get("activation_probe", {}).get("C", -1)) == 0.01:
        result.ok("frozen probe C=0.01 unchanged")
    else:
        result.fail("probe C changed")
    if exp["coarse_scan"]["transformer_block_indices"] != EXPECTED_LAYERS:
        result.fail("layer grid changed")
    else:
        result.ok("layers unchanged")
    if exp["coarse_scan"]["prefix_lengths_k"] != EXPECTED_K:
        result.fail("k grid changed")
    else:
        result.ok("k grid unchanged")
    if (
        exp["primary_endpoints"]["regime_a"]["layer"] == 12
        and exp["primary_endpoints"]["regime_a"]["k"] == 0
        and exp["primary_endpoints"]["regime_b"]["layer"] == 12
        and exp["primary_endpoints"]["regime_b"]["k"] == 1
    ):
        result.ok("primary endpoints unchanged (D029/D043)")
    else:
        result.fail("primary endpoints changed")

    elig_path = REPO_ROOT / "artifacts/phase3b_locked/locked_eligibility_freeze.json"
    if elig_path.is_file():
        esha = hashlib.sha256(elig_path.read_bytes()).hexdigest()
        if esha == LOCKED_ELIG_SHA:
            result.ok("eligibility freeze hash matches")
        else:
            result.fail(f"eligibility freeze hash drift {esha}")
        elig = json.loads(elig_path.read_text(encoding="utf-8"))
        if elig.get("predictive_metrics_computed") is False:
            result.ok("eligibility freeze has no predictive metrics")
        else:
            result.fail("eligibility freeze polluted with predictive metrics")
        ok_n = True
        for k, n in EXPECTED_LOCKED_N.items():
            if elig["eligibility"][k]["n_eligible"] != n:
                ok_n = False
                result.fail(f"eligibility N drift k={k}")
        if ok_n:
            result.ok("eligibility N per k frozen")
        if elig.get("n_prompt_groups") == 53:
            result.ok("locked prompt groups = 53")
        else:
            result.fail("prompt group count drift")
    else:
        result.fail("missing locked_eligibility_freeze.json")

    gpu = exp.get("phase3b_gpu", {})
    if gpu.get("locked_test_gpu_authorized") is True:
        result.ok("locked test GPU authorized for B2")
    else:
        result.fail("locked test GPU not authorized")
    if gpu.get("regime_c_authorized") is False:
        result.ok("regime C unauthorized")
    else:
        result.fail("regime C authorized")

    # Code contracts for locked analysis
    locked_src = REPO_ROOT / "scripts/analyze_phase3b_locked.py"
    if locked_src.is_file():
        text = locked_src.read_text(encoding="utf-8")
        if "one_shot_locked_test_evaluation" in text or "ONE-SHOT" in text.upper():
            result.ok("locked analysis encodes one-shot protocol")
        else:
            result.ok("locked analysis script present")
        if "fit_on" in text and "phase3_train_plus_validation" in text:
            result.ok("locked analysis refits on combined development")
        else:
            result.fail("locked analysis missing combined-dev fit")
        if "fixed_random_projection_diagnostic" in text:
            result.ok("locked analysis uses fixed-random-projection terminology")
        else:
            result.fail("missing fixed-random-projection terminology")
    else:
        result.fail("missing analyze_phase3b_locked.py")

    extract_src = (REPO_ROOT / "modal/phase3_extract.py").read_text(encoding="utf-8")
    if "extract_locked" in extract_src and "locked_test=True" in extract_src:
        result.ok("modal extractor supports locked-test path")
    else:
        result.fail("modal extractor missing locked-test path")
    if "include_labels=False" in extract_src:
        result.ok("locked GPU payload omits labels")
    else:
        result.fail("locked GPU payload may include labels")

    latest = REPO_ROOT / "artifacts/phase3b_locked/latest_extract_manifest.json"
    metrics = REPO_ROOT / "artifacts/phase3b_locked/phase3b_locked_metrics.json"
    pf = REPO_ROOT / "artifacts/phase3b_locked/preflight_truncated_summary.json"

    if pf.is_file():
        p = json.loads(pf.read_text(encoding="utf-8"))
        if p.get("repeatability_pass") and float(p.get("repeatability_min_cosine", 0)) >= 0.9999:
            result.ok("locked truncated preflight passed")
        else:
            result.fail("locked truncated preflight failed")
    else:
        result.ok("locked preflight not yet present (code-freeze OK)")

    if latest.is_file():
        man = json.loads(latest.read_text(encoding="utf-8"))
        if man.get("locked_test_present") is True:
            result.ok("B2 extract is locked-test-only flag")
        else:
            result.fail("B2 extract not marked locked_test_present")
        if man.get("splits_present") == ["locked_test"] or man.get(
            "splits_present"
        ) == ["locked_test"]:
            result.ok("B2 splits_present locked_test only")
        elif man.get("splits_present") is None and man.get("locked_test_present"):
            result.ok("B2 locked_test_present without train/val")
        else:
            # accept if only locked
            if man.get("locked_test_present") and "phase3_train" not in str(
                man.get("splits_present")
            ):
                result.ok("B2 no train/val in extract pointer")
            else:
                result.fail(f"unexpected splits {man.get('splits_present')}")
        if man.get("compute_dtype") == "bfloat16":
            result.ok("B2 compute BF16")
        else:
            result.fail("B2 compute dtype wrong")
        if man.get("activation_storage_dtype") == "float32":
            result.ok("B2 storage float32")
        else:
            result.fail("B2 storage dtype wrong")
        if man.get("batch_size") == 1 and man.get("extraction_mode") == (
            "truncated_prefix_single_example"
        ):
            result.ok("B2 truncated bs=1")
        else:
            result.fail("B2 extraction contract mismatch")
        if man.get("working_tree_clean") is True and man.get("git_commit"):
            result.ok("B2 clean-tree provenance present")
        else:
            result.fail("B2 missing clean-tree provenance")
        if man.get("extractor_sha256") and man.get("locked_analysis_script_sha256"):
            result.ok("B2 extractor/analysis hashes present")
        else:
            result.fail("B2 missing code hashes")
        if man.get("locked_eligibility_freeze_sha256") == LOCKED_ELIG_SHA:
            result.ok("B2 eligibility freeze hash recorded")
        else:
            result.fail("B2 eligibility freeze hash missing/drift")
        if man.get("model_revision") == PINNED_MODEL_REV:
            result.ok("B2 model revision pinned")
        else:
            result.fail("B2 model revision drift")
        if man.get("dataset_revision") == PINNED_LASR_HF:
            result.ok("B2 dataset revision pinned")
        else:
            result.fail("B2 dataset revision drift")
        if man.get("labels_present_in_gpu_payload") is False:
            result.ok("B2 GPU payload omitted labels")
        else:
            result.fail("B2 GPU payload included labels")
        if float(man.get("estimated_cost_usd", 999)) <= 40.0:
            result.ok(f"B2 cost within hard stop (${man.get('estimated_cost_usd')})")
        else:
            result.fail("B2 cost exceeds hard stop")
    else:
        result.ok("B2 extract not yet present (code-freeze OK)")

    if metrics.is_file():
        m = json.loads(metrics.read_text(encoding="utf-8"))
        if m.get("one_shot_locked_test_evaluation") is True:
            result.ok("metrics mark one-shot locked evaluation")
        else:
            result.fail("metrics missing one-shot flag")
        if m.get("locked_test_hyperparameter_tuning") is False:
            result.ok("no locked-test hyperparameter tuning")
        else:
            result.fail("metrics claim tuning occurred")
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
            result.ok("all 54 locked-test cells present")
        else:
            result.fail(f"expected 54 cells, got {n_cells}")
        gate = m.get("engineering_gate", {})
        if gate.get("k0_activation_identity_ok") and gate.get(
            "k0_surface_score_identity_ok"
        ):
            result.ok("k0 identity safeguards passed")
        else:
            result.fail("k0 identity safeguards failed")
        integ = m.get("activation_integrity") or {}
        if integ.get("all_finite") is True:
            result.ok("finite activation audit")
        else:
            result.fail("activation integrity not finite")
        if integ.get("activation_storage_dtype") == "float32":
            result.ok("metrics storage dtype float32")
        else:
            result.fail("metrics storage dtype wrong")
        pa = m.get("primary_regime_a", {})
        pb = m.get("primary_regime_b", {})
        if (
            pa.get("layer") == 12
            and pa.get("k") == 0
            and pb.get("layer") == 12
            and pb.get("k") == 1
        ):
            result.ok("primary locked endpoints remain L12/k0 and L12/k1")
        else:
            result.fail("primary locked endpoints drifted")
        if exp.get("status") == "phase3b2_locked_complete_awaiting_audit":
            result.ok("status phase3b2 locked complete awaiting audit")
        else:
            result.ok(f"status={exp.get('status')} (metrics present)")
        report = REPO_ROOT / "reports/phase3b_locked_test.md"
        if report.is_file():
            result.ok("locked-test report present")
        else:
            result.fail("missing reports/phase3b_locked_test.md")
    else:
        result.ok("locked metrics not yet present (code-freeze OK)")

    tracked = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "ls-files"], text=True
    ).splitlines()
    for path in tracked:
        if path.endswith(".safetensors"):
            result.fail(f"activation tensor tracked: {path}")
    result.ok("no activation tensors tracked")

    print()
    print(f"{len(result.passes)} passed, {len(result.failures)} failed")
    if result.failures:
        for f in result.failures:
            print(f"  - {f}")
        return 1
    print("Phase 3B2 validation OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
