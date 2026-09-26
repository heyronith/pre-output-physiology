#!/usr/bin/env python3
"""Phase 11A order-robust behavior integrity checks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from prepare_phase8_design import sha_ids, sha_prompt_texts  # noqa: E402
from prepare_phase11_design import discovery_prompts, load_phase10_final  # noqa: E402

from pre_output_physiology.phase10_design import family_split  # noqa: E402
from pre_output_physiology.phase11_design import (  # noqa: E402
    DISCOVERY_TRAIN,
    DISCOVERY_VALIDATION,
    LOCKED,
    mask_candidate_line,
    order_robust_label,
    split_gate,
)

STATUSES = {
    "phase11a_order_robust_behavior_authorized",
    "phase11a_order_robust_behavior_pass_awaiting_audit",
    "phase11a_order_robust_behavior_hold",
}
CFG = REPO_ROOT / "configs/experiments/phase11_order_robust_behavior.yaml"
DATA = REPO_ROOT / "data/processed/phase11_design"
DESIGN = REPO_ROOT / "artifacts/phase11a_design"


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
        "locked_family_model_calls_authorized",
        "activation_extraction_authorized",
        "probe_fitting_authorized",
        "phase5_probe_scoring_authorized",
        "causal_intervention_authorized",
        "k_search_authorized",
        "prompt_revision_authorized",
        "decoder_revision_authorized",
    ):
        check(auth.get(k) is False, f"{k}=false")
    pre = status == "phase11a_order_robust_behavior_authorized"
    check(auth["discovery_behavior_calls_authorized"] is pre, "discovery auth matches status")
    check(cfg["phase10_merge_sha"] == "57e2cec72e124741c08ae4a4d94d62eaa177746a", "phase10 merge")
    p10 = yaml.safe_load((REPO_ROOT / "configs/experiments/phase10_risk_frontier.yaml")
                         .read_text("utf-8"))
    check(p10["status"] == cfg["phase10_status"] == "phase10a_risk_frontier_sanity_hold",
          "Phase 10A remains sanity hold")
    dec = (REPO_ROOT / "docs/decision_log.md").read_text("utf-8")
    check(all(f"### {d}" in dec for d in ("D104", "D105", "D106", "D107")), "D104-D107")
    check(cfg["design"]["fixed_k"] == 10, "fixed K=10")
    split = family_split()
    check(
        tuple(split["discovery_train"]) == DISCOVERY_TRAIN
        and tuple(split["discovery_validation"]) == DISCOVERY_VALIDATION
        and tuple(split["locked_generalization"]) == LOCKED
        and all(cfg["design"]["family_split"][k] == split[k] for k in (
            "discovery_train", "discovery_validation", "locked_generalization")),
        "family split unchanged from Phase 10",
    )
    final = load_phase10_final()
    check(len(final) == 960, "reused Phase-10 final bases verified (hashes)")
    matrix = json.loads((DESIGN / "design_matrix.json").read_text("utf-8"))
    pr = _jsonl(DATA / "discovery_order_prompts.jsonl")
    rf = [p for p in pr if p["order"] == "RF"]
    af = [p for p in pr if p["order"] == "AF"]
    check(len(rf) == len(af) == 720 and not any(p["family"] in LOCKED for p in pr),
          "720 RF + 720 AF discovery prompts; no locked")
    for k, rows in (("rf_prompt_text_sha256", rf), ("af_prompt_text_sha256", af)):
        check(sha_prompt_texts(rows) == cfg["design"][k] == matrix[k], f"{k} frozen")
    regen = discovery_prompts(final)
    check(sha_prompt_texts(regen["RF"]) == cfg["design"]["rf_prompt_text_sha256"]
          and sha_prompt_texts(regen["AF"]) == cfg["design"]["af_prompt_text_sha256"],
          "prompts regenerable from committed code")
    check(sha_ids([p["base_scenario_id"] for p in rf]) == cfg["design"][
        "discovery_base_ids_sha256"], "discovery base IDs frozen")
    rf_by = {p["base_scenario_id"]: p for p in rf}
    ok = all(
        mask_candidate_line(rf_by[a["base_scenario_id"]]["prompt_text"])
        == mask_candidate_line(a["prompt_text"])
        and f"Candidate states: {a['alternate_state']} | {a['record_state']}\n" in a[
            "prompt_text"]
        and f"Candidate states: {a['record_state']} | {a['alternate_state']}\n" in rf_by[
            a["base_scenario_id"]]["prompt_text"]
        and "-10 points" in a["prompt_text"]
        for a in af
    )
    check(ok, "RF/AF byte-identical outside candidate line; K=10 wording")
    tk = matrix["tokenization"]
    check(tk["response_prefix_token_id"] == 12107 and tk["n_prompt_contexts_checked"] == 1440
          and tk["candidate_tokenization_invariant_to_order"] and tk["all_lengths_matched"]
          and tk["no_strict_prefix_pairs"], "tokenization integrity")
    plan = json.loads((DESIGN / "future_phase11b_plan.json").read_text("utf-8"))
    check(plan["executed"] is False and plan["primary_endpoint"].startswith("k0")
          and plan["layer_grid"] == [0, 4, 8, 12, 16, 20, 24, 28, 31], "Phase 11B plan frozen")
    check(order_robust_label("record", "alternate") == "order_sensitive_unlabeled",
          "label rule")

    art = REPO_ROOT / "artifacts"
    check(not [p for p in art.rglob("*") if p.is_file() and "phase11" in str(p)
               and p.suffix in {".safetensors", ".npz", ".pt", ".pth"}], "no activations")
    runs = art / "runs"
    run_dirs = [d for d in runs.glob("phase11*") if d.is_dir()] if runs.is_dir() else []
    check(all(d.name.startswith("phase11a_behavior_") for d in run_dirs), "only behavior runs")
    locked_ids = {r["base_scenario_id"] for r in final if r["family"] in LOCKED}
    locked_calls = sum(r["base_scenario_id"] in locked_ids or r["family"] in LOCKED
                       for d in run_dirs for r in _jsonl(d / "behavior_choices.jsonl"))
    check(locked_calls == 0, "zero locked-family model calls")
    tracked = subprocess.check_output(["git", "-C", str(REPO_ROOT), "ls-files"], text=True)
    check(not any(t.startswith("data/processed/phase11_design") or "behavior_choices" in t
                  for t in tracked.splitlines()), "raw data not tracked")

    if not pre:
        s = json.loads((art / "phase11a_behavior/behavior_summary.json").read_text("utf-8"))
        bases = json.loads((art / "phase11a_behavior/base_labels.json").read_text("utf-8"))
        check(len(bases) == 720 and s["n_evaluations"] == 1440, "720 bases / 1440 choices")
        check(s["run_id"] == cfg["behavior_result"]["run_id"], "run id recorded in config")
        check(all(b["label"] == order_robust_label(b["rf_choice"], b["af_choice"])
                  for b in bases), "labels recompute")
        gates = {sp: split_gate([b for b in bases if b["split"] == sp], sp)
                 for sp in ("discovery_train", "discovery_validation")}
        check(json.loads(json.dumps(gates)) == s["gates"], "gates recompute exactly")
        passed = all(g["pass"] for g in gates.values())
        check((status == "phase11a_order_robust_behavior_pass_awaiting_audit") == passed,
              "status matches gate outcome")
        check((REPO_ROOT / "reports/phase11a_behavior.md").is_file(), "report present")

    print()
    print(f"{n_pass} passed, {len(fails)} failed")
    for f in fails:
        print(f"  - {f}")
    if fails:
        return 1
    print("Phase 11A validation OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
