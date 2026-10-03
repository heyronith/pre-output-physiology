#!/usr/bin/env python3
"""Phase 6B frozen-probe factorial physiology integrity checks."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology import phase6b_freeze as fz  # noqa: E402
from pre_output_physiology.phase6b_report import REQUIRED_STATEMENT  # noqa: E402

SCIENTIFIC_PATHS = [
    "src/pre_output_physiology/phase6_analysis.py",
    "src/pre_output_physiology/phase6b_freeze.py",
    "src/pre_output_physiology/phase6b_report.py",
    "src/pre_output_physiology/phase5_locked.py",
    "scripts/score_phase6b.py",
    "modal/phase6b_extract.py",
]


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), *args], text=True).strip()


def main() -> int:
    fails: list[str] = []
    n_pass = 0

    def check(cond: bool, msg: str) -> None:
        nonlocal n_pass
        if cond:
            n_pass += 1
            print(f"PASS  {msg}")
        else:
            fails.append(msg)
            print(f"FAIL  {msg}")

    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/validate_phase6a.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    check(proc.returncode == 0, "validate_phase6a.py passed (6A HOLD record intact)")

    cfg = yaml.safe_load(fz.CONFIG_PATH.read_text(encoding="utf-8"))
    status = cfg.get("status")
    check(status in {fz.STATUS_AUTHORIZED, fz.STATUS_PRIMARY_COMPLETE}, f"status {status}")
    check(cfg.get("phase6a_outcome") == fz.PHASE6A_OUTCOME, "phase6a_outcome HOLD recorded")
    check(cfg.get("phase6a_original_exact_one_line_gate_passed") is False, "6A exact gate failed")
    check(
        cfg["design"]["prompt_template_revision"] == 2, "prompt template revision 2 (no revision)"
    )
    p6b = cfg.get("phase6b", {})
    check(p6b.get("general_deception_intent_test") is False, "not framed as deception test")
    check(p6b.get("prospective_amendment_before_final_model_call") is True, "prospective amendment")
    check(p6b.get("starting_head") == fz.PHASE6A_HOLD_HEAD, "starting head pinned")
    auth = cfg["authorizations"]
    for key in (
        "final_generation_authorized",
        "pilot_generation_authorized",
        "probe_fitting_authorized",
        "causal_intervention_authorized",
    ):
        check(auth.get(key) is False, f"{key}=false")
    decisions = (REPO_ROOT / "docs/decision_log.md").read_text(encoding="utf-8")
    check("### D083" in decisions and "not** a general deception-intent test" in decisions, "D083")

    try:
        _p, _s, hashes = fz.load_verified_corpus()
        check(True, "corpus hashes verified")
    except SystemExit as exc:
        check(False, f"corpus hashes: {exc}")
        hashes = {}
    for k, v in hashes.items():
        check(p6b.get("frozen_corpus", {}).get(k) == v == cfg["design"].get(k, v), f"config {k}")
    check(fz.sha_bytes(fz.PROBE_PATH.read_bytes()) == fz.EXPECTED_PROBE_SHA256, "probe sha")
    diff = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "diff",
            "--quiet",
            cfg["phase5_merge_sha"],
            "--",
            str(fz.PROBE_PATH.relative_to(REPO_ROOT)),
        ],
        capture_output=True,
    )
    check(diff.returncode == 0, "probe unchanged since Phase-5 merge")

    runs = REPO_ROOT / "artifacts/runs"
    gen_runs = (
        [
            d.name
            for d in runs.glob("phase6*")
            if d.is_dir() and not d.name.startswith(("phase6a_pilot_", "phase6b_extract_"))
        ]
        if runs.is_dir()
        else []
    )
    check(not gen_runs, "no Phase-6 final generation runs")
    tracked = _git("ls-files").splitlines()
    check(not any(t.endswith(".safetensors") for t in tracked), "no activations tracked")

    if status == fz.STATUS_PRIMARY_COMPLETE:
        out = REPO_ROOT / "artifacts/phase6b_primary"
        summary = json.loads((out / "phase6b_primary_summary.json").read_text(encoding="utf-8"))
        pre = summary["pre_run_commit"]
        changed = _git("diff", "--name-only", pre, "HEAD", "--", *SCIENTIFIC_PATHS)
        check(not changed, f"no scientific-code changes since pre-run freeze {pre[:8]}")
        check(
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(REPO_ROOT),
                    "merge-base",
                    "--is-ancestor",
                    fz.PHASE6A_HOLD_HEAD,
                    pre,
                ]
            ).returncode
            == 0,
            "pre-run freeze descends from 6A HOLD head",
        )
        ex = summary["extraction"]
        for k, v in hashes.items():
            check(ex[k] == v, f"extraction {k}")
        check(ex["preflight_repeatability_min_cosine"] >= 0.9999, "preflight cosine ≥ 0.9999")
        check(summary["probe_sha256"] == fz.EXPECTED_PROBE_SHA256, "summary probe sha")
        check(summary["activation_integrity"]["passed"] is True, "activation integrity")
        a = summary["analysis_probability"]
        check(a["n_scenarios"] == 720 and a["n_bootstrap"] == 5000, "720 scenarios / 5000 boots")
        check(len(a["effects_by_family"]) == 6, "six family breakdowns")
        for flag in (
            "behavior_conditioned",
            "generation_performed",
            "probe_retrained",
            "probe_recalibrated",
            "layer_reselected",
            "causal_interventions_performed",
        ):
            check(summary[flag] is False, f"{flag}=false")
        check(summary["exclusions"] == 0, "no exclusions")
        with (out / "frozen_probe_scores.csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        check(len(rows) == 2880, "2880 scored rows")
        report = (REPO_ROOT / "reports/phase6b_primary.md").read_text(encoding="utf-8")
        check(REQUIRED_STATEMENT in report, "required statement in report")
        run_dir = runs / ex["run_id"]
        if (run_dir / "activations_l12_k1.safetensors").is_file():
            check(
                fz.sha_bytes((run_dir / "activations_l12_k1.safetensors").read_bytes())
                == ex["activations_sha256"],
                "local activations match manifest sha",
            )

    print()
    print(f"{n_pass} passed, {len(fails)} failed")
    if fails:
        for f in fails:
            print(f"  - {f}")
        return 1
    print("Phase 6B validation OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
