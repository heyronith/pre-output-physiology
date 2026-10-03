#!/usr/bin/env python3
"""Phase 8A policy-frontier calibration integrity checks."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from prepare_phase8_design import (  # noqa: E402
    build_design,
    calibration_prompts,
    final_prompts,
    prior_topic_overlap,
    sha_ids,
    sha_prompt_texts,
    sha_scenarios,
)

from pre_output_physiology.phase8_design import (  # noqa: E402
    CALIBRATION_FAMILIES,
    FINAL_FAMILIES,
    FORBIDDEN_PROMPT_SUBSTRINGS,
    PRIOR_PHASE_FAMILIES,
    W_GRID,
    family_split,
    rates_by_w,
    select_w_star,
)

STATUSES = {
    "phase8a_policy_frontier_calibration_authorized",
    "phase8a_policy_frontier_frozen_awaiting_audit",
    "phase8a_policy_frontier_hold",
}
CFG = REPO_ROOT / "configs/experiments/phase8_policy_frontier.yaml"
DATA = REPO_ROOT / "data/processed/phase8_design"
DESIGN = REPO_ROOT / "artifacts/phase8a_design"


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
        "activation_extraction_authorized",
        "probe_fitting_authorized",
        "phase5_probe_scoring_authorized",
        "causal_intervention_authorized",
    ):
        check(auth.get(k) is False, f"{k}=false")
    calib_auth = status == "phase8a_policy_frontier_calibration_authorized"
    check(auth["calibration_policy_choice_authorized"] is calib_auth, "calibration auth matches")
    check(cfg["phase7_merge_sha"] == "bba799969a8b3af923ff6a2e760c410e5d77a11a", "phase7 merge")
    dec = (REPO_ROOT / "docs/decision_log.md").read_text("utf-8")
    check(all(f"### {d}" in dec for d in ("D090", "D091", "D092", "D093")), "D090-D093")
    check(
        list(cfg["design"]["w_grid"]) == list(W_GRID) == [20, 30, 40, 45, 50, 55, 60, 70, 80],
        "W grid frozen, not extended",
    )
    fams = set(CALIBRATION_FAMILIES) | set(FINAL_FAMILIES)
    check(len(fams) == 12 and not fams & set(PRIOR_PHASE_FAMILIES), "12 fresh families")

    matrix = json.loads((DESIGN / "design_matrix.json").read_text("utf-8"))
    calib = _jsonl(DATA / "calibration_base_scenarios.jsonl")
    final = _jsonl(DATA / "final_base_scenarios.jsonl")
    cal_pr = _jsonl(DATA / "calibration_prompts.jsonl")
    check(len(calib) == 32 and len(final) == 960 and len(cal_pr) == 288, "counts 32/960/288")
    got = {
        "calibration_scenario_text_sha256": sha_scenarios(calib),
        "calibration_base_scenario_ids_sha256": sha_ids([s["base_scenario_id"] for s in calib]),
        "calibration_prompt_text_sha256": sha_prompt_texts(cal_pr),
        "final_scenario_text_sha256": sha_scenarios(final),
        "final_base_scenario_ids_sha256": sha_ids([s["base_scenario_id"] for s in final]),
    }
    for k, v in got.items():
        check(v == cfg["design"][k] == matrix[k], f"{k} frozen")
    regen = build_design()
    check(
        sha_scenarios(regen["final"]) == got["final_scenario_text_sha256"]
        and sha_prompt_texts(calibration_prompts(regen["calibration"]))
        == got["calibration_prompt_text_sha256"],
        "design regenerable from committed code",
    )
    check(
        all(v["balanced"] for part in matrix["counterbalance"].values() for v in part.values()),
        "record/alternate values and candidate order counterbalanced",
    )
    check(
        not ({s["topic_sentence"] for s in calib} & {s["topic_sentence"] for s in final}),
        "calibration/final disjoint",
    )
    check(
        not prior_topic_overlap({s["topic_sentence"] for s in calib + final}),
        "no Phase 4-7 scenario text reuse",
    )
    leaks = 0
    for p in cal_pr:
        low = p["prompt_text"].lower()
        if any(re.search(rf"\b{re.escape(t)}\b", low) for t in FORBIDDEN_PROMPT_SUBSTRINGS):
            leaks += 1
        if "_policy_choice" in low or f"takes place at {p['alternate_state']}." not in low:
            leaks += 1
        if f"Private record state: {p['record_state']}\n" not in p["prompt_text"]:
            leaks += 1
    check(leaks == 0, "conflict present; no forbidden lexicon or labels")
    audit = json.loads((DESIGN / "candidate_tokenization_audit.json").read_text("utf-8"))
    check(
        all(
            audit[k]["no_strict_prefix_pairs"]
            and audit[k]["all_candidates_nonempty"]
            and audit[k]["candidate_tokenization_invariant_to_w"]
            and audit[k]["response_prefix_token_id"] == 12107
            for k in ("calibration", "final")
        )
        and audit["calibration"]["n_prompt_contexts_checked"] == 288
        and audit["final"]["n_prompt_contexts_checked"] == 960 * len(W_GRID),
        "candidate tokenization integrity (every base x W)",
    )
    split = json.loads((DESIGN / "family_split.json").read_text("utf-8"))
    fresh = family_split()
    check(
        all(
            split[k] == fresh[k] == cfg["family_split"][k]
            for k in ("discovery_train", "discovery_validation", "locked_generalization")
        )
        and split["digests"] == fresh["digests"],
        "family split deterministic and frozen",
    )
    plan = json.loads((DESIGN / "future_phase8b_plan.json").read_text("utf-8"))
    check(
        plan["executed"] is False
        and plan["discovery_gates"]["discovery_validation_min_per_family_per_class"] == 15,
        "Phase 8B plan frozen",
    )

    art = REPO_ROOT / "artifacts"
    check(
        not [
            p
            for p in art.rglob("*")
            if p.is_file()
            and "phase8" in str(p)
            and p.suffix in {".safetensors", ".npz", ".pt", ".pth"}
        ],
        "no Phase-8 activations",
    )
    runs = art / "runs"
    bad = (
        [
            d.name
            for d in runs.glob("phase8*")
            if d.is_dir() and not d.name.startswith("phase8a_calibration_")
        ]
        if runs.is_dir()
        else []
    )
    check(not bad, "only calibration runs")
    final_ids = {s["base_scenario_id"] for s in final}
    hits = sum(
        r["base_scenario_id"] in final_ids or not r["example_id"].startswith("calib_")
        for path in (
            runs.glob("phase8a_calibration_*/calibration_choices.jsonl") if runs.is_dir() else []
        )
        for r in _jsonl(path)
    )
    check(hits == 0, "zero final-scenario model calls")
    tracked = subprocess.check_output(["git", "-C", str(REPO_ROOT), "ls-files"], text=True)
    check(
        not any(
            t.startswith("data/processed/phase8_design") or "calibration_choices" in t
            for t in tracked.splitlines()
        ),
        "raw data not tracked",
    )

    if status != "phase8a_policy_frontier_calibration_authorized":
        s = json.loads((art / "phase8a_calibration/calibration_summary.json").read_text("utf-8"))
        rates = rates_by_w(s["labeled_rows"])
        sel = select_w_star(rates)
        check(json.loads(json.dumps(sel)) == s["selection"], "selection recomputes exactly")
        check(len(s["labeled_rows"]) == 288, "288 labeled calibration choices")
        if status.endswith("frozen_awaiting_audit"):
            w = sel["w_star"]
            check(w is not None and cfg["w_star"] == w, "W* recorded in config")
            fp = final_prompts(final, w)
            h = sha_prompt_texts(fp)
            check(
                h == s["final_prompt_text_sha256"] == cfg["final_prompt_text_sha256"],
                "final prompts = frozen template instantiated with W*",
            )
            check(
                all(
                    f"(weight {w})" in p["prompt_text"]
                    and f"(weight {100 - w})" in p["prompt_text"]
                    for p in fp
                ),
                "all 960 final prompts use the single global W*",
            )
        else:
            check(sel["w_star"] is None, "hold ⇔ no eligible W")
        check((REPO_ROOT / "reports/phase8a_calibration.md").is_file(), "report present")

    print()
    print(f"{n_pass} passed, {len(fails)} failed")
    for f in fails:
        print(f"  - {f}")
    if fails:
        return 1
    print("Phase 8A validation OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
