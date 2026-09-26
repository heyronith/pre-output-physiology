#!/usr/bin/env python3
"""Phase 7A policy-choice design + pilot integrity checks."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from prepare_phase7_design import (  # noqa: E402
    build_design,
    expand_prompts,
    prior_topic_overlap,
    sha_ids,
    sha_prompt_texts,
    sha_scenarios,
)

from pre_output_physiology.phase7_design import (  # noqa: E402
    CONTEXT_ORDER,
    FORBIDDEN_PROMPT_SUBSTRINGS,
    N_FINAL_PER_FAMILY,
    N_FINAL_PROMPTS,
    N_PILOT_PROMPTS,
    PHASE5_PROBE_REFERENCE,
    PRIOR_PHASE_FAMILIES,
    SCENARIO_FAMILIES,
    assert_context_symmetry,
    family_split,
    summarize_pilot,
)

STATUSES = {
    "phase7a_policy_choice_design_frozen_pilot_authorized",
    "phase7a_policy_choice_pilot_pass_awaiting_audit",
    "phase7a_policy_choice_pilot_hold",
}
CFG_PATH = REPO_ROOT / "configs/experiments/phase7_policy_choice.yaml"
DATA = REPO_ROOT / "data/processed/phase7_design"
DESIGN = REPO_ROOT / "artifacts/phase7a_design"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


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

    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    status = cfg.get("status")
    check(status in STATUSES, f"status {status}")
    auth = cfg["authorizations"]
    for key in (
        "final_generation_authorized",
        "activation_extraction_authorized",
        "probe_fitting_authorized",
        "phase5_probe_scoring_authorized",
        "causal_intervention_authorized",
    ):
        check(auth.get(key) is False, f"{key}=false")
    pilot_auth = status == "phase7a_policy_choice_design_frozen_pilot_authorized"
    check(auth.get("pilot_generation_authorized") is pilot_auth, "pilot auth matches status")
    check(cfg["phase6_merge_sha"] == "f2b494ee41ee430782d82ac1e671b7efc5c980dd", "phase6 merge")
    decisions = (REPO_ROOT / "docs/decision_log.md").read_text(encoding="utf-8")
    check(all(f"### {d}" in decisions for d in ("D084", "D085", "D086", "D087")), "D084-D087")

    check(
        len(set(SCENARIO_FAMILIES)) == 8 and not set(SCENARIO_FAMILIES) & set(PRIOR_PHASE_FAMILIES),
        "eight fresh families disjoint from Phases 4-6",
    )
    check(list(cfg["design"]["families"]) == list(SCENARIO_FAMILIES), "config families match")

    matrix = json.loads((DESIGN / "condition_matrix.json").read_text(encoding="utf-8"))
    final_sc = _jsonl(DATA / "final_base_scenarios.jsonl")
    pilot_sc = _jsonl(DATA / "pilot_base_scenarios.jsonl")
    final_pr = _jsonl(DATA / "final_candidate_prompts.jsonl")
    pilot_pr = _jsonl(DATA / "pilot_candidate_prompts.jsonl")
    check(
        len(final_sc) == 640
        and len(final_pr) == N_FINAL_PROMPTS
        and len(pilot_sc) == 32
        and len(pilot_pr) == N_PILOT_PROMPTS,
        "counts 640/2560 final, 32/128 pilot",
    )
    fam = Counter(s["family"] for s in final_sc)
    check(all(fam[f] == N_FINAL_PER_FAMILY for f in SCENARIO_FAMILIES), "80 per family")
    design = cfg["design"]
    got = {
        "final_prompt_text_sha256": sha_prompt_texts(final_pr),
        "final_scenario_text_sha256": sha_scenarios(final_sc),
        "final_base_scenario_ids_sha256": sha_ids([s["base_scenario_id"] for s in final_sc]),
        "pilot_prompt_text_sha256": sha_prompt_texts(pilot_pr),
        "pilot_scenario_text_sha256": sha_scenarios(pilot_sc),
    }
    for k, v in got.items():
        check(v == design[k] == matrix[k], f"{k} frozen")

    regen = build_design()
    check(
        sha_prompt_texts(expand_prompts(regen["final"], split="final"))
        == design["final_prompt_text_sha256"],
        "final corpus deterministically regenerable from committed code",
    )
    check(
        sha_prompt_texts(expand_prompts(regen["pilot"], split="pilot"))
        == design["pilot_prompt_text_sha256"],
        "pilot corpus deterministically regenerable from committed code",
    )

    sym = 0
    for sc in final_sc + pilot_sc:
        try:
            assert_context_symmetry(sc)
        except AssertionError:
            sym += 1
    check(sym == 0, "four-context symmetry on all bases")
    by_base: dict[str, set[str]] = {}
    for p in final_pr + pilot_pr:
        by_base.setdefault(p["base_scenario_id"], set()).add(p["context_id"])
    check(all(v == set(CONTEXT_ORDER) for v in by_base.values()), "four contexts per base")
    conflict = all(
        p["record_state"] != p["alternate_state"]
        and p["operational_objective_state"] == p["alternate_state"]
        and f"Private record state: {p['record_state']}\n" in p["prompt_text"]
        and f"takes place at {p['alternate_state']}." in p["prompt_text"]
        for p in final_pr + pilot_pr
    )
    check(conflict, "goal-record conflict in 100% of prompts")
    leaks = 0
    for p in final_pr + pilot_pr:
        low = p["prompt_text"].lower()
        if any(re.search(rf"\b{re.escape(t)}\b", low) for t in FORBIDDEN_PROMPT_SUBSTRINGS):
            leaks += 1
        if any(c in p["prompt_text"] for c in CONTEXT_ORDER) or "_choice" in low:
            leaks += 1
        if p.get("behavior_label") is not None:
            leaks += 1
    check(leaks == 0, "no forbidden lexicon, context IDs, or preassigned labels")
    check(
        all(v["balanced"] for rep in matrix["counterbalance"].values() for v in rep.values()),
        "state values and candidate order counterbalanced",
    )
    check(
        not ({s["base_scenario_id"] for s in final_sc} & {s["base_scenario_id"] for s in pilot_sc})
        and not ({s["topic_sentence"] for s in final_sc} & {s["topic_sentence"] for s in pilot_sc}),
        "pilot/final disjoint",
    )
    check(
        not prior_topic_overlap({s["topic_sentence"] for s in final_sc + pilot_sc}),
        "no Phase 4-6 scenario text reuse",
    )
    check(
        matrix["neutral_prefix_token_id"] == 12107
        and matrix["neutral_prefix_verified_on_n_prompts"] == N_FINAL_PROMPTS + N_PILOT_PROMPTS,
        "Response token 12107 verified on all contexts",
    )

    split = json.loads((DESIGN / "family_split.json").read_text(encoding="utf-8"))
    fresh = family_split()
    check(
        all(
            split[k] == fresh[k] == cfg["family_split"][k]
            for k in ("discovery_train", "discovery_validation", "locked_generalization")
        )
        and split["digests"] == fresh["digests"],
        "family split deterministic and frozen",
    )
    check(
        sorted(
            fresh["discovery_train"]
            + fresh["discovery_validation"]
            + fresh["locked_generalization"]
        )
        == sorted(SCENARIO_FAMILIES),
        "family split partitions all families",
    )
    plan = json.loads((DESIGN / "future_physiology_plan.json").read_text(encoding="utf-8"))
    check(
        plan["executed"] is False
        and plan["phase5_probe"] == PHASE5_PROBE_REFERENCE
        and plan["primary_endpoint"] == "controlled_prefix_k1_after_token_12107",
        "future physiology plan frozen, not executed",
    )

    art = REPO_ROOT / "artifacts"
    act = [
        str(p.relative_to(REPO_ROOT))
        for p in art.rglob("*")
        if p.is_file()
        and "phase7" in str(p.relative_to(REPO_ROOT))
        and p.suffix in {".safetensors", ".npz", ".pt", ".pth"}
    ]
    check(not act, "no Phase-7 activations or probe artifacts")
    runs = art / "runs"
    bad_runs = (
        [
            d.name
            for d in runs.glob("phase7*")
            if d.is_dir() and not d.name.startswith("phase7a_pilot_")
        ]
        if runs.is_dir()
        else []
    )
    check(not bad_runs, "only Phase-7A pilot runs (no final generation)")
    final_ids = {s["base_scenario_id"] for s in final_sc}
    hits = 0
    for path in runs.glob("phase7a_pilot_*/pilot_outputs.jsonl") if runs.is_dir() else []:
        for r in _jsonl(path):
            if r.get("split") != "pilot" or r["base_scenario_id"] in final_ids:
                hits += 1
            if r.get("activation_extracted") or r.get("probe_scored"):
                hits += 1
    check(hits == 0, "pilot outputs contain no final scenarios / activation / probe flags")
    tracked = subprocess.check_output(["git", "-C", str(REPO_ROOT), "ls-files"], text=True)
    check(
        not any(
            t.startswith("data/processed/phase7_design") or t.endswith("pilot_outputs.jsonl")
            for t in tracked.splitlines()
        ),
        "raw outputs / design payload not tracked",
    )

    if status != "phase7a_policy_choice_design_frozen_pilot_authorized":
        s = json.loads((art / "phase7a_pilot/pilot_behavior_summary.json").read_text("utf-8"))
        recomputed = summarize_pilot(s["eval_rows"])
        check(recomputed == s["summary"], "pilot summary recomputes from eval rows")
        passed = s["summary"]["all_gates_pass"]
        check(
            (status.endswith("pass_awaiting_audit") and passed)
            or (status.endswith("hold") and not passed),
            "status matches pilot gates",
        )
        man = json.loads((art / "phase7a_pilot/pilot_generation_manifest.json").read_text("utf-8"))
        check(
            man["pilot_prompt_text_sha256"] == design["pilot_prompt_text_sha256"],
            "pilot run used frozen pilot prompts",
        )
        check((REPO_ROOT / "reports/phase7a_pilot.md").is_file(), "pilot report present")

    print()
    print(f"{n_pass} passed, {len(fails)} failed")
    if fails:
        for f in fails:
            print(f"  - {f}")
        return 1
    print("Phase 7A validation OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
