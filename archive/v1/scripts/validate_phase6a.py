#!/usr/bin/env python3
"""Phase 6A factorial specificity design + pilot integrity checks."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase6_conditions import (  # noqa: E402
    CONDITION_ORDER,
    EXPECTED_NEUTRAL_PREFIX_TOKEN_ID,
    FORBIDDEN_PROMPT_SUBSTRINGS,
    FROZEN_PHASE5_PROBE,
    N_FINAL_BASE_SCENARIOS,
    N_FINAL_PER_FAMILY,
    N_FINAL_PROMPTS,
    N_PILOT_BASE_SCENARIOS,
    N_PILOT_PROMPTS,
    PHASE4_FAMILIES,
    PHASE5_FAMILIES,
    SCENARIO_FAMILIES,
    assert_factorial_symmetry,
    build_condition_prompt,
    targets_for_condition,
)

STATUSES = {
    "phase6a_specificity_design_frozen_pilot_authorized",
    "phase6a_specificity_pilot_pass_awaiting_audit",
    "phase6a_specificity_pilot_hold",
    "phase6b_factorial_frozen_probe_authorized",
    "phase6b_factorial_primary_complete_awaiting_audit",
}
PHASE6A_HOLD_OUTCOME = "phase6a_specificity_pilot_hold_operational_format_failure"


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


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _sha_prompt_texts(rows: list[dict]) -> str:
    ordered = sorted(rows, key=lambda r: r["example_id"])
    payload = "\n".join(f"{r['example_id']}\t{r['prompt_text']}" for r in ordered)
    return hashlib.sha256(payload.encode()).hexdigest()


def _sha_scenarios(rows: list[dict]) -> str:
    ordered = sorted(rows, key=lambda r: r["base_scenario_id"])
    payload = "\n".join(
        "\t".join(
            [
                r["base_scenario_id"],
                r["topic_sentence"],
                r["user_question"],
                r["record_state"],
                r["alternate_state"],
            ]
        )
        for r in ordered
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def main() -> int:
    result = Result()

    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/validate_phase5d.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        result.ok("validate_phase5d.py passed (Phase 5 frozen)")
    else:
        result.fail("validate_phase5d.py failed")

    cfg = yaml.safe_load(
        (REPO_ROOT / "configs/experiments/phase6_intent_specificity.yaml").read_text(
            encoding="utf-8"
        )
    )
    status = cfg.get("status")
    if status in STATUSES:
        result.ok(f"status {status}")
    else:
        result.fail(f"unexpected status {status}")

    is_6b = str(status).startswith("phase6b")
    auth = cfg.get("authorizations", {})
    must_be_false = [
        "final_generation_authorized",
        "probe_fitting_authorized",
        "causal_intervention_authorized",
    ]
    if not is_6b:
        must_be_false += ["activation_extraction_authorized", "probe_scoring_authorized"]
    for key in must_be_false:
        if auth.get(key) is False:
            result.ok(f"{key}=false")
        else:
            result.fail(f"{key} must be false")
    expected_pilot_auth = status == "phase6a_specificity_design_frozen_pilot_authorized"
    if auth.get("pilot_generation_authorized") is expected_pilot_auth:
        result.ok(f"pilot_generation_authorized={expected_pilot_auth}")
    else:
        result.fail("pilot auth inconsistent with status")

    decisions = (REPO_ROOT / "docs/decision_log.md").read_text(encoding="utf-8")
    for did in ("D077", "D078", "D079", "D080"):
        if did in decisions:
            result.ok(f"{did} present")
        else:
            result.fail(f"{did} missing")

    if len(SCENARIO_FAMILIES) == 6 and len(set(SCENARIO_FAMILIES)) == 6:
        result.ok("six unique families")
    else:
        result.fail("family count wrong")
    if set(SCENARIO_FAMILIES) & (set(PHASE4_FAMILIES) | set(PHASE5_FAMILIES)):
        result.fail("family name overlap with Phase 4/5")
    else:
        result.ok("no Phase-4/5 family name overlap")
    if list(cfg["design"]["families"]) == list(SCENARIO_FAMILIES):
        result.ok("config families match code")
    else:
        result.fail("config/code family mismatch")

    probe_path = REPO_ROOT / FROZEN_PHASE5_PROBE["probe_artifact"]
    digest = hashlib.sha256(probe_path.read_bytes()).hexdigest()
    if digest == FROZEN_PHASE5_PROBE["probe_sha256"] == cfg["frozen_phase5_probe"][
        "probe_sha256"
    ]:
        result.ok("frozen Phase-5 probe SHA256 exact")
    else:
        result.fail(f"probe sha mismatch {digest}")
    tracked = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "ls-files"], text=True
    ).splitlines()
    probe_rel = FROZEN_PHASE5_PROBE["probe_artifact"]
    diff = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "diff", "--quiet", cfg["phase5_merge_sha"], "--", probe_rel],
        capture_output=True,
    )
    if diff.returncode == 0:
        result.ok("probe artifact unchanged since Phase-5 merge")
    else:
        result.fail("probe artifact modified since Phase-5 merge")

    matrix = json.loads(
        (REPO_ROOT / "artifacts/phase6a_design/condition_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    if matrix["neutral_prefix_token_id"] == EXPECTED_NEUTRAL_PREFIX_TOKEN_ID and matrix[
        "neutral_prefix_verified_on_n_prompts"
    ] == N_FINAL_PROMPTS + N_PILOT_PROMPTS:
        result.ok("Response token 12107 verified on all final+pilot prompt contexts")
    else:
        result.fail("Response token verification missing")
    if matrix.get("F10_is_deception") is False:
        result.ok("F10 recorded as non-deceptive control")
    else:
        result.fail("F10 interpretation flag wrong")

    data = REPO_ROOT / "data/processed/phase6_design"
    final_sc = _load_jsonl(data / "final_base_scenarios.jsonl")
    pilot_sc = _load_jsonl(data / "pilot_base_scenarios.jsonl")
    final_pr = _load_jsonl(data / "final_candidate_prompts.jsonl")
    pilot_pr = _load_jsonl(data / "pilot_candidate_prompts.jsonl")
    if (
        len(final_sc) == N_FINAL_BASE_SCENARIOS
        and len(pilot_sc) == N_PILOT_BASE_SCENARIOS
        and len(final_pr) == N_FINAL_PROMPTS
        and len(pilot_pr) == N_PILOT_PROMPTS
    ):
        result.ok("counts 720/2880 final, 24/96 pilot")
    else:
        result.fail("count drift")
    fam_counts = Counter(s["family"] for s in final_sc)
    if all(fam_counts[f] == N_FINAL_PER_FAMILY for f in SCENARIO_FAMILIES):
        result.ok("120 base scenarios per family")
    else:
        result.fail(f"per-family counts {dict(fam_counts)}")

    design = cfg["design"]
    checks = {
        "final_prompt_text_sha256": _sha_prompt_texts(final_pr),
        "pilot_prompt_text_sha256": _sha_prompt_texts(pilot_pr),
        "final_scenario_text_sha256": _sha_scenarios(final_sc),
        "pilot_scenario_text_sha256": _sha_scenarios(pilot_sc),
    }
    for key, value in checks.items():
        if value == design[key] == matrix[key]:
            result.ok(f"{key} frozen")
        else:
            result.fail(f"{key} drift")
    if int(design["prompt_template_revision"]) == int(matrix["prompt_template_revision"]):
        result.ok(f"prompt_template_revision={design['prompt_template_revision']}")
    else:
        result.fail("revision mismatch")

    sym_fail = 0
    by_base: dict[str, dict[str, dict]] = {}
    for p in final_pr + pilot_pr:
        by_base.setdefault(p["base_scenario_id"], {})[p["condition_id"]] = p
    for sc in final_sc + pilot_sc:
        try:
            assert_factorial_symmetry(sc)
            prompts = by_base[sc["base_scenario_id"]]
            if set(prompts) != set(CONDITION_ORDER):
                raise AssertionError("missing condition")
            for cid, p in prompts.items():
                if p["prompt_text"] != build_condition_prompt(condition_id=cid, scenario=sc):
                    raise AssertionError("stored prompt differs from template")
                op, comm = targets_for_condition(
                    cid,
                    record_state=sc["record_state"],
                    alternate_state=sc["alternate_state"],
                )
                if p["operational_target"] != op or p["communication_target"] != comm:
                    raise AssertionError("target metadata wrong")
                if sc["topic_sentence"] not in p["prompt_text"] or sc[
                    "user_question"
                ] not in p["prompt_text"]:
                    raise AssertionError("scenario text missing")
        except AssertionError:
            sym_fail += 1
    if sym_fail == 0:
        result.ok(f"factorial structural symmetry on all {len(final_sc) + len(pilot_sc)} bases")
    else:
        result.fail(f"factorial symmetry failures: {sym_fail}")

    lex_hits = 0
    for p in final_pr + pilot_pr:
        low = p["prompt_text"].lower()
        if any(re.search(rf"\b{re.escape(t)}\b", low) for t in FORBIDDEN_PROMPT_SUBSTRINGS):
            lex_hits += 1
        if any(re.search(rf"\b{c}\b", p["prompt_text"]) for c in CONDITION_ORDER):
            lex_hits += 1
    if lex_hits == 0:
        result.ok("no forbidden lexicon or condition IDs in model-visible text")
    else:
        result.fail(f"lexicon/condition leaks: {lex_hits}")

    cb_ok = True
    for split_rows in (final_sc, pilot_sc):
        for fam in SCENARIO_FAMILIES:
            rows = [s for s in split_rows if s["family"] == fam]
            rec = Counter(s["record_state"] for s in rows)
            alt = Counter(s["alternate_state"] for s in rows)
            if any(rec[v] != alt[v] for v in set(rec) | set(alt)):
                cb_ok = False
            if any(s["record_state"] == s["alternate_state"] for s in rows):
                cb_ok = False
    if cb_ok:
        result.ok("record-state values exactly counterbalanced in every family (final+pilot)")
    else:
        result.fail("counterbalance failure")

    final_ids = {s["base_scenario_id"] for s in final_sc}
    pilot_ids = {s["base_scenario_id"] for s in pilot_sc}
    if not final_ids & pilot_ids:
        result.ok("pilot/final IDs disjoint")
    else:
        result.fail("pilot/final ID overlap")
    if not {s["entity"] for s in final_sc} & {s["entity"] for s in pilot_sc}:
        result.ok("pilot/final entities disjoint")
    else:
        result.fail("pilot/final entity overlap")
    if not {s["topic_sentence"] for s in final_sc} & {s["topic_sentence"] for s in pilot_sc}:
        result.ok("pilot/final scenario text disjoint")
    else:
        result.fail("pilot/final scenario text overlap")
    keys = [(s["family"], s["entity"], s["record_state"], s["alternate_state"]) for s in final_sc]
    if len(set(keys)) == len(keys):
        result.ok("no duplicate final scenarios")
    else:
        result.fail("duplicate final scenarios")

    revision = int(design["prompt_template_revision"])
    if revision == 1:
        result.ok("revision 1 (no format revision used)")
    elif revision == 2:
        r1_path = REPO_ROOT / "artifacts/phase6a_pilot_revision1/pilot_behavior_summary.json"
        if r1_path.is_file() and (REPO_ROOT / "reports/phase6a_pilot_revision1.md").is_file():
            r1 = json.loads(r1_path.read_text(encoding="utf-8"))
            if r1["summary"]["gates"]["all_operational_gates_pass"] is False:
                result.ok("revision 2 justified by failed revision-1 pilot")
            else:
                result.fail("revision 2 used although revision-1 pilot passed")
            r1_ids = {r["base_scenario_id"] for r in r1["eval_rows"]}
            if not r1_ids & (pilot_ids | final_ids):
                result.ok("revision-2 pilot + finals disjoint from revision-1 pilot IDs")
            else:
                result.fail("revision-1 pilot IDs reused")
            if matrix.get("pilot_seed") != matrix.get("pilot_seed_revision1"):
                result.ok("fresh pilot seed for revision 2")
            else:
                result.fail("pilot seed not refreshed")
            if matrix.get("pilot_revision1_scenario_text_sha256") != design[
                "pilot_scenario_text_sha256"
            ]:
                result.ok("revision-2 pilot scenario text differs from revision 1")
            else:
                result.fail("revision-2 pilot reuses revision-1 scenarios")
        else:
            result.fail("missing revision-1 pilot artifacts")
    else:
        result.fail(f"more than one format revision: {revision}")

    prior_text = set()
    for rel in (
        "data/processed/phase5_design/final_base_scenarios.jsonl",
        "data/processed/phase5_design/pilot_base_scenarios.jsonl",
    ):
        path = REPO_ROOT / rel
        if path.is_file():
            prior_text |= {s["topic_sentence"] for s in _load_jsonl(path)}
    if prior_text and not prior_text & {s["topic_sentence"] for s in final_sc + pilot_sc}:
        result.ok("no Phase-5 scenario text reuse")
    elif not prior_text:
        result.fail("Phase-5 design payload unavailable for overlap check")
    else:
        result.fail("Phase-5 scenario text reused")

    future = json.loads(
        (REPO_ROOT / "artifacts/phase6a_design/future_frozen_probe_analysis.json").read_text(
            encoding="utf-8"
        )
    )
    if future.get("executed") is False and future.get("new_probe_training") is False:
        result.ok("future frozen-probe analysis preregistered, not executed")
    else:
        result.fail("future analysis flags wrong")

    act_hits = [
        str(p.relative_to(REPO_ROOT))
        for p in (REPO_ROOT / "artifacts").rglob("*")
        if p.is_file()
        and "phase6" in str(p.relative_to(REPO_ROOT))
        and p.suffix in {".safetensors", ".npz", ".pt", ".pth"}
        and not (
            is_6b
            and p.suffix == ".safetensors"
            and p.relative_to(REPO_ROOT).parts[:2] == ("artifacts", "runs")
            and p.relative_to(REPO_ROOT).parts[2].startswith("phase6b_extract_")
        )
    ]
    if act_hits:
        result.fail(f"phase6 activation/weight artifacts present: {act_hits[:3]}")
    else:
        result.ok("no Phase-6 activations or probe artifacts")
    runs = REPO_ROOT / "artifacts/runs"
    non_pilot = [
        d.name
        for d in runs.glob("phase6*")
        if d.is_dir()
        and not d.name.startswith("phase6a_pilot_")
        and not (is_6b and d.name.startswith("phase6b_extract_"))
    ] if runs.is_dir() else []
    if non_pilot:
        result.fail(f"non-pilot Phase-6 runs present: {non_pilot}")
    else:
        result.ok("only Phase-6A pilot / Phase-6B extraction runs exist (no final generation)")
    final_output_hits = 0
    if runs.is_dir():
        for path in runs.glob("phase6a_pilot_*/pilot_outputs.jsonl"):
            for row in _load_jsonl(path):
                if row.get("split") != "pilot" or row["base_scenario_id"] in final_ids:
                    final_output_hits += 1
                if row.get("activation_extracted") or row.get("probe_scored"):
                    final_output_hits += 1
    if final_output_hits == 0:
        result.ok("pilot outputs contain no final scenarios / activation / probe flags")
    else:
        result.fail(f"pilot output integrity hits: {final_output_hits}")

    if status != "phase6a_specificity_design_frozen_pilot_authorized":
        summary_path = REPO_ROOT / "artifacts/phase6a_pilot/pilot_behavior_summary.json"
        report = REPO_ROOT / "reports/phase6a_pilot.md"
        if summary_path.is_file() and report.is_file():
            result.ok("pilot summary/report present")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))["summary"]
            gates = summary["gates"]["all_operational_gates_pass"]
            if (
                (status.endswith("pass_awaiting_audit") and gates)
                or (status.endswith("hold") and not gates)
                or (is_6b and not gates and cfg.get("phase6a_outcome") == PHASE6A_HOLD_OUTCOME)
            ):
                result.ok("status matches pilot gates")
            else:
                result.fail(f"status/gates mismatch {status}/{gates}")
            man = json.loads(
                (REPO_ROOT / "artifacts/phase6a_pilot/pilot_generation_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            if man.get("pilot_prompt_text_sha256") == design["pilot_prompt_text_sha256"]:
                result.ok("pilot run used frozen pilot prompts")
            else:
                result.fail("pilot run prompt hash mismatch")
        else:
            result.fail("missing pilot summary/report")

    for path in tracked:
        if "phase6" in path and path.endswith("pilot_outputs.jsonl"):
            result.fail(f"raw pilot outputs tracked: {path}")
        if path.startswith("data/processed/phase6_design"):
            result.fail(f"design payload tracked: {path}")
    result.ok("raw outputs / design payloads not tracked")

    print()
    print(f"{len(result.passes)} passed, {len(result.failures)} failed")
    if result.failures:
        for f in result.failures:
            print(f"  - {f}")
        return 1
    print("Phase 6A validation OK.")
    if is_6b:
        print(f"PHASE 6A OUTCOME REMAINS {PHASE6A_HOLD_OUTCOME}.")
        print("THE FROZEN PHASE-5 PROBE WAS NOT RETRAINED, RECALIBRATED, OR MODIFIED.")
        print("NO FINAL PHASE-6 RESPONSES WERE GENERATED.")
    else:
        print("NO PHASE 6 ACTIVATIONS WERE COLLECTED.")
        print("THE FROZEN PHASE-5 PROBE WAS NOT SCORED, RETRAINED, RECALIBRATED, OR MODIFIED.")
        print("NO FINAL PHASE-6 SCENARIOS WERE RUN THROUGH THE MODEL.")
    print("NO CAUSAL INTERVENTIONS WERE PERFORMED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
