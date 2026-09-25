#!/usr/bin/env python3
"""Phase 3A integrity checks (local only; no Modal / activations)."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.baselines import SurfaceBaselineContract  # noqa: E402
from pre_output_physiology.trajectory import (  # noqa: E402
    COARSE_TRANSFORMER_BLOCKS,
    PREFIX_LENGTHS_K,
    PRIMARY_ANCHOR_LAYER,
)

PINNED_MODEL_REV = "63a8b081895390a26e140280378bc85ec8bce07a"
PINNED_LASR_HF = "bf93584916fbd23121eca6f2017017df0ef3184f"
EXPECTED_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31]
EXPECTED_K = [0, 1, 2, 4, 8, 16]
PHASE3_STATUS = "prepared_awaiting_gpu_authorization"


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
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(path)
    return data


def tracked_files() -> list[str]:
    out = subprocess.check_output(["git", "-C", str(REPO_ROOT), "ls-files"], text=True)
    return [line for line in out.splitlines() if line]


def main() -> int:
    result = Result()

    for script in ("validate_phase1.py", "validate_phase2.py"):
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / script)],
            cwd=REPO_ROOT,
        )
        if proc.returncode == 0:
            result.ok(f"{script} passed")
        else:
            result.fail(f"{script} failed")

    model = load_yaml(REPO_ROOT / "configs/models/mistral_7b_instruct_v02.yaml")
    ds = load_yaml(REPO_ROOT / "configs/datasets/roleplay_deception.yaml")
    exp = load_yaml(REPO_ROOT / "configs/experiments/phase3_preoutput_scan.yaml")

    if model.get("revision") == PINNED_MODEL_REV:
        result.ok("exact model revision remains pinned")
    else:
        result.fail(f"model revision mismatch: {model.get('revision')}")

    if ds.get("lasr_hf_revision") == PINNED_LASR_HF:
        result.ok("exact dataset revision remains pinned")
    else:
        result.fail(f"dataset revision mismatch: {ds.get('lasr_hf_revision')}")

    if exp.get("status") == PHASE3_STATUS:
        result.ok(f"phase3 status={PHASE3_STATUS}")
    else:
        result.fail(f"phase3 status={exp.get('status')!r} (GPU not yet authorized)")

    if exp.get("phase3b_gpu", {}).get("authorized") is False:
        result.ok("phase3b_gpu.authorized is false")
    else:
        result.fail("phase3b_gpu must not be authorized in Phase 3A")

    layers = list(exp.get("coarse_scan", {}).get("transformer_block_indices", []))
    ks = list(exp.get("coarse_scan", {}).get("prefix_lengths_k", []))
    anchor = exp.get("coarse_scan", {}).get("primary_anchor_layer")
    if layers == EXPECTED_LAYERS == COARSE_TRANSFORMER_BLOCKS:
        result.ok("coarse layers frozen")
    else:
        result.fail(f"coarse layers {layers}")
    if ks == EXPECTED_K == PREFIX_LENGTHS_K:
        result.ok("prefix lengths frozen")
    else:
        result.fail(f"prefix lengths {ks}")
    if anchor == PRIMARY_ANCHOR_LAYER == 12:
        result.ok("primary anchor layer = 12")
    else:
        result.fail(f"primary anchor {anchor}")

    probe_c = float(exp.get("activation_probe", {}).get("C", -1))
    if probe_c == 0.01:
        result.ok("activation probe C = 0.01")
    else:
        result.fail(f"activation probe C = {probe_c}")

    stats = exp.get("statistics", {})
    if (
        int(stats.get("bootstrap_resamples", 0)) >= 2000
        and stats.get("bootstrap_unit") == "prompt_group"
    ):
        result.ok("group-aware bootstrap required (>=2000)")
    else:
        result.fail(f"bootstrap contract missing: {stats}")

    onset = exp.get("onset_annotation", {})
    if onset.get("activation_dependent") is False and onset.get("llm_api_allowed") is False:
        result.ok("onset rules activation-independent / no LLM API")
    else:
        result.fail(f"onset rules unsafe: {onset}")

    contract = SurfaceBaselineContract()
    if contract.prompt_plus_prefix and contract.select_C_on == "phase3_validation_only":
        result.ok("surface baseline contract present")
    else:
        result.fail("surface baseline contract incomplete")

    if not (REPO_ROOT / "docs/phase3_protocol.md").is_file():
        result.fail("missing docs/phase3_protocol.md")
    else:
        result.ok("phase3 protocol present")
    if not (REPO_ROOT / "docs/phase4_control_plan.md").is_file():
        result.fail("missing docs/phase4_control_plan.md")
    else:
        result.ok("phase4 control plan present")

    # Split / grouping checks if prepare artifacts exist
    summary_path = REPO_ROOT / "artifacts/phase3a_summaries/phase3a_prepare_summary.json"
    manifest_path = (
        REPO_ROOT / "data/processed/phase3_roleplay/manifests/phase3a_prepare_manifest.json"
    )
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        splits = summary.get("splits", {})
        if splits.get("locked_test", {}).get("n_rows") == 500:
            result.ok("locked test row count = 500")
        else:
            result.fail(f"locked test size unexpected: {splits.get('locked_test')}")
        for name in ("phase3_train", "phase3_validation", "locked_test"):
            s = splits.get(name, {})
            if s.get("n_class0", 0) > 0 and s.get("n_class1", 0) > 0:
                result.ok(f"{name} has both classes")
            else:
                result.fail(f"{name} missing a class: {s}")
        if summary.get("activations_collected") is False:
            result.ok("summary records activations_collected=false")
        else:
            result.fail("summary claims activations collected")
    else:
        result.fail(
            "missing artifacts/phase3a_summaries/phase3a_prepare_summary.json "
            "(run scripts/prepare_phase3.py)"
        )

    if manifest_path.is_file():
        man = json.loads(manifest_path.read_text(encoding="utf-8"))
        ov = man.get("group_overlap", {})
        if ov.get("train_val") == 0 and ov.get("train_test") == 0 and ov.get("val_test") == 0:
            result.ok("manifest records zero prompt-group overlap")
        else:
            result.fail(f"group overlap: {ov}")
        if man.get("gpu_authorized") is False and man.get("activations_collected") is False:
            result.ok("manifest: GPU unauthorized / no activations")
        else:
            result.fail("manifest incorrectly authorizes GPU or activations")
    else:
        result.fail("missing phase3a prepare manifest under data/processed/")

    # No Phase 3 activation / primary result artifacts tracked or present in expected dirs
    tracked = tracked_files()
    secret_pat = re.compile(
        r"(^|/)\.env($|\.)|credentials|secrets/|\.pem$|\.key$", re.I
    )
    for path in tracked:
        if path.endswith(".env.example") or path == ".env.example":
            continue
        if secret_pat.search(path):
            result.fail(f"secret-like tracked file: {path}")
    result.ok("no secret-like files tracked")

    # Explicit path checks (no accidental Phase 3B outputs)
    bad_paths = [
        REPO_ROOT / "artifacts/runs/phase3",
        REPO_ROOT / "artifacts/derived/phase3_activations",
        REPO_ROOT / "reports/phase3_results.md",
        REPO_ROOT / "reports/phase3_preoutput_scan.md",
    ]
    bad_found = False
    for p in bad_paths:
        if p.exists():
            bad_found = True
            result.fail(f"Phase 3 results/activation artifact exists: {p}")
    if not bad_found:
        result.ok("no primary Phase 3 results / activation artifacts present")

    # Code must not claim Phase 3 GPU authorized
    phase3_py = (REPO_ROOT / "scripts/prepare_phase3.py").read_text(encoding="utf-8")
    if "activations_collected\": False" in phase3_py or "activations_collected=False" in phase3_py:
        result.ok("prepare_phase3 records no activations")
    else:
        # softer check
        if "activations_collected" in phase3_py and "False" in phase3_py:
            result.ok("prepare_phase3 mentions activations_collected False")
        else:
            result.fail("prepare_phase3 missing activations_collected=False")

    # Token-boundary audit (required before Phase 3B A/B)
    audit_path = REPO_ROOT / "artifacts/phase3a_summaries/token_boundary_audit.json"
    if not audit_path.is_file():
        result.fail(
            "missing artifacts/phase3a_summaries/token_boundary_audit.json "
            "(run scripts/audit_phase3_token_boundary.py)"
        )
    else:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        g = audit.get("global", {})
        if int(g.get("prompt_prefix_fail", -1)) == 0:
            result.ok("token boundary: zero prompt-prefix failures")
        else:
            result.fail(
                f"token boundary prompt-prefix failures: {g.get('prompt_prefix_fail')}"
            )
        if int(g.get("boundary_straddle", -1)) == 0:
            result.ok("token boundary: zero straddling examples")
        else:
            result.fail(
                f"token boundary straddling examples: {g.get('boundary_straddle')}"
            )
        if g.get("n_rows") == 3500:
            result.ok("token boundary audited all 3500 examples")
        else:
            result.fail(f"token boundary row count {g.get('n_rows')} != 3500")
        if audit.get("activations_collected") is False:
            result.ok("token boundary audit records no activations")
        else:
            result.fail("token boundary audit claims activations collected")

    print()
    print(f"{len(result.passes)} passed, {len(result.failures)} failed")
    if result.failures:
        for f in result.failures:
            print(f"  - {f}")
        return 1
    print("Phase 3A validation OK — GPU still unauthorized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
