#!/usr/bin/env python3
"""Phase 14A same-prompt trajectory calibration integrity checks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from prepare_phase8_design import sha_ids, sha_prompt_texts, sha_scenarios  # noqa: E402
from prepare_phase14_design import (  # noqa: E402
    build_design,
    calibration_prompts,
    prior_families,
    prior_topic_overlap,
    sampling_schedule,
)

from pre_output_physiology.phase14_design import (  # noqa: E402
    CALIBRATION_FAMILIES,
    FINAL_FAMILIES,
    FIXED_K,
    TEMPERATURE_GRID,
    family_split,
    rates_by_temperature,
    select_temperature,
)

STATUSES = {
    "phase14a_same_prompt_trajectory_calibration_authorized",
    "phase14a_same_prompt_trajectory_calibration_pass_awaiting_audit",
    "phase14a_same_prompt_trajectory_calibration_hold",
}
CFG = REPO_ROOT / "configs/experiments/phase14_same_prompt_trajectory.yaml"
DATA = REPO_ROOT / "data/processed/phase14_design"
DESIGN = REPO_ROOT / "artifacts/phase14a_design"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text("utf-8").splitlines() if x.strip()]


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

    cfg = yaml.safe_load(CFG.read_text("utf-8"))
    status = cfg["status"]
    check(status in STATUSES, f"status {status}")
    auth = cfg["authorizations"]
    for k in (
        "final_model_calls_authorized",
        "locked_family_model_calls_authorized",
        "activation_extraction_authorized",
        "probe_fitting_authorized",
        "phase5_probe_scoring_authorized",
        "causal_intervention_authorized",
        "sampling_grid_extension_authorized",
        "post_result_prompt_changes_authorized",
    ):
        check(auth.get(k) is False, f"{k}=false")
    pre = status == "phase14a_same_prompt_trajectory_calibration_authorized"
    check(auth["calibration_trajectory_calls_authorized"] is pre, "calib auth matches")
    check(cfg["phase13_merge_sha"] == "abc3dd9c4c5d8b7b3ba6fd3a99e84d7ebac71783", "phase13 merge")
    dec = (REPO_ROOT / "docs/decision_log.md").read_text("utf-8")
    check(all(f"### {d}" in dec for d in ("D117", "D118", "D119", "D120")), "D117-D120")
    check(
        list(cfg["design"]["temperature_grid"]) == list(TEMPERATURE_GRID) == [0.7, 0.9, 1.1],
        "temperature grid frozen",
    )
    check(cfg["design"]["fixed_k"] == FIXED_K == 10, "fixed K=10")
    fams = set(CALIBRATION_FAMILIES) | set(FINAL_FAMILIES)
    check(len(fams) == 12 and not fams & prior_families(), "12 fresh families")

    matrix = json.loads((DESIGN / "design_matrix.json").read_text("utf-8"))
    calib = _jsonl(DATA / "calibration_base_scenarios.jsonl")
    final = _jsonl(DATA / "final_base_scenarios.jsonl")
    cal_pr = _jsonl(DATA / "calibration_prompts.jsonl")
    fin_pr = _jsonl(DATA / "final_prompts.jsonl")
    schedule = _jsonl(DATA / "calibration_sampling_schedule.jsonl")
    check(len(calib) == 32 and len(final) == 320 and len(cal_pr) == 32, "counts 32/320/32")
    check(len(schedule) == 1536, "1536 scheduled continuations")
    got = {
        "calibration_scenario_text_sha256": sha_scenarios(calib),
        "calibration_prompt_group_ids_sha256": sha_ids([p["prompt_group_id"] for p in cal_pr]),
        "calibration_prompt_text_sha256": sha_prompt_texts(cal_pr),
        "final_scenario_text_sha256": sha_scenarios(final),
        "final_prompt_group_ids_sha256": sha_ids([p["prompt_group_id"] for p in fin_pr]),
        "final_prompt_text_sha256": sha_prompt_texts(fin_pr),
        "sampling_schedule_sha256": sha_ids([r["continuation_id"] for r in schedule]),
    }
    for k, v in got.items():
        check(v == cfg["design"][k] == matrix[k], f"{k} frozen")
    regen = build_design()
    check(
        sha_prompt_texts(calibration_prompts(regen["calibration"]))
        == got["calibration_prompt_text_sha256"]
        and sha_scenarios(regen["final"]) == got["final_scenario_text_sha256"],
        "design regenerable",
    )
    check(
        sha_ids([r["continuation_id"] for r in sampling_schedule(cal_pr)])
        == got["sampling_schedule_sha256"],
        "schedule regenerable",
    )
    check(
        not prior_topic_overlap({s["topic_sentence"] for s in calib + final}),
        "no prior-phase text reuse",
    )
    check(
        all(
            "-10 points" in p["prompt_text"] and "Consideration" in p["prompt_text"]
            for p in cal_pr
        ),
        "K=10 and Consideration instruction present",
    )
    tk = matrix["tokenization"]
    check(
        all(
            tk[k]["response_prefix_token_id"] == 12107
            and tk[k]["response_12107_after_consideration_verified"]
            and tk[k]["all_lengths_matched"]
            for k in ("calibration", "final")
        ),
        "tokenization / Response-12107 after Consideration",
    )
    split = json.loads((DESIGN / "family_split.json").read_text("utf-8"))
    fresh = family_split()
    check(
        all(
            split[k] == fresh[k] == cfg["family_split"][k]
            for k in ("discovery_train", "discovery_validation", "locked_generalization")
        ),
        "family split frozen",
    )
    plan = json.loads((DESIGN / "future_phase14b_plan.json").read_text("utf-8"))
    check(
        plan["executed"] is False
        and plan["physiology"]["k0_role"].startswith("negative control"),
        "Phase 14B plan frozen (k0 negative control)",
    )

    art = REPO_ROOT / "artifacts"
    check(
        not [
            p
            for p in art.rglob("*")
            if p.is_file()
            and "phase14" in str(p)
            and p.suffix in {".safetensors", ".npz", ".pt", ".pth"}
        ],
        "no activations",
    )
    runs = art / "runs"
    run_dirs = [d for d in runs.glob("phase14*") if d.is_dir()] if runs.is_dir() else []
    check(
        all(d.name.startswith("phase14a_calibration_") for d in run_dirs),
        "only calibration runs",
    )
    final_ids = {s["base_scenario_id"] for s in final}
    locked = set(cfg["family_split"]["locked_generalization"])
    bad = 0
    for d in run_dirs:
        for r in _jsonl(d / "calibration_continuations.jsonl"):
            if r["prompt_group_id"] in final_ids or r["family"] in locked:
                bad += 1
            if not r["prompt_group_id"].startswith("calib_"):
                bad += 1
    check(bad == 0, "zero final/locked model calls")
    tracked = subprocess.check_output(["git", "-C", str(REPO_ROOT), "ls-files"], text=True)
    check(
        not any(
            t.startswith("data/processed/phase14_design") or "calibration_continuations" in t
            for t in tracked.splitlines()
        ),
        "raw data not tracked",
    )

    if not pre:
        s = json.loads(
            (art / "phase14a_calibration/calibration_summary.json").read_text("utf-8")
        )
        check(len(s["labeled_rows"]) == 1536, "1536 labeled continuations")
        check(s["run_id"] == cfg["calibration_result"]["run_id"], "run id in config")
        rates = rates_by_temperature(s["labeled_rows"])
        # Drop by_prompt for equality of selection inputs — recompute selection
        sel = select_temperature(rates)
        check(sel["t_star"] == s["selection"]["t_star"], "T* recomputes")
        check(
            sorted(sel["eligible_temperatures"])
            == sorted(s["selection"]["eligible_temperatures"]),
            "eligible temperatures recompute",
        )
        if status.endswith("pass_awaiting_audit"):
            check(
                sel["t_star"] is not None
                and cfg["calibration_result"]["t_star"] == sel["t_star"],
                "T* recorded",
            )
        else:
            check(sel["t_star"] is None, "hold ⇔ no eligible T")
        check(
            s["final_model_calls"] == 0 and s["locked_model_calls"] == 0,
            "zero final/locked",
        )
        check((REPO_ROOT / "reports/phase14a_calibration.md").is_file(), "report present")

    print()
    print(f"{n_pass} passed, {len(fails)} failed")
    for f in fails:
        print(f"  - {f}")
    if fails:
        return 1
    print("Phase 14A validation OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
