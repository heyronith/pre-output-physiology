#!/usr/bin/env python3
"""Phase 12A family-bias source diagnostic integrity checks."""

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
from prepare_phase11_design import load_phase10_final  # noqa: E402
from prepare_phase12_design import build_prompts, cell_hashes  # noqa: E402

from pre_output_physiology.phase11_design import LOCKED  # noqa: E402
from pre_output_physiology.phase12_diagnostic import (  # noqa: E402
    CELLS,
    family_dependence,
    interpretation,
    neutralization_violations,
    select_bases,
)

STATUSES = {
    "phase12a_family_bias_source_diagnostic_authorized",
    "phase12a_family_bias_source_diagnostic_complete_awaiting_audit",
    "phase12a_family_bias_source_diagnostic_hold",
}
CFG = REPO_ROOT / "configs/experiments/phase12_family_bias_source.yaml"
DATA = REPO_ROOT / "data/processed/phase12_design"
DESIGN = REPO_ROOT / "artifacts/phase12a_design"


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
        "locked_family_model_calls_authorized", "activation_extraction_authorized",
        "probe_fitting_authorized", "phase5_probe_scoring_authorized",
        "causal_intervention_authorized", "k_tuning_authorized", "payoff_changes_authorized",
        "decoder_changes_authorized", "post_result_prompt_tuning_authorized",
    ):
        check(auth.get(k) is False, f"{k}=false")
    pre = status == "phase12a_family_bias_source_diagnostic_authorized"
    check(auth["diagnostic_model_calls_authorized"] is pre, "diagnostic auth matches status")
    check(cfg["phase11_merge_sha"] == "2ea8b23fa669ddbe7aa59d040785683e299d4c0d", "phase11 merge")
    p11 = yaml.safe_load((REPO_ROOT / "configs/experiments/phase11_order_robust_behavior.yaml")
                         .read_text("utf-8"))
    check(p11["status"] == "phase11a_order_robust_behavior_hold", "Phase 11A remains hold")
    dec = (REPO_ROOT / "docs/decision_log.md").read_text("utf-8")
    check(all(f"### {d}" in dec for d in ("D109", "D110", "D111")), "D109-D111")

    final = load_phase10_final()
    sel = select_bases(final)
    check(sha_ids([b["base_scenario_id"] for b in sel]) == cfg["design"][
        "selected_base_ids_sha256"], "deterministic label-independent selection reproduces")
    check(len(sel) == 144 and not {b["family"] for b in sel} & set(LOCKED),
          "144 bases, 24/family, no locked")
    matrix = json.loads((DESIGN / "design_matrix.json").read_text("utf-8"))
    rows = build_prompts(sel)
    got = cell_hashes(rows)
    for k, v in got.items():
        check(v == cfg["design"][k] == matrix[k], f"{k} frozen and regenerable")
    new = _jsonl(DATA / "new_cell_prompts.jsonl")
    check(sha_prompt_texts(new) == cfg["design"]["new_prompt_text_sha256"] and len(new) == 864,
          "864 new B/C/D prompts frozen")
    by_base = {b["base_scenario_id"]: b for b in sel}
    n_viol = sum(bool(neutralization_violations(r["prompt_text"], by_base[
        r["base_scenario_id"]], r["cell"])) for r in rows)
    check(n_viol == 0, "neutralization rules hold for every prompt")
    tk = matrix["tokenization"]
    check(tk["response_prefix_token_id"] == 12107 and tk["n_prompt_contexts_checked"] == 1152
          and tk["all_lengths_matched"] and tk["no_strict_prefix_pairs"], "tokenization gates")
    check(matrix["cell_a_reuse"]["n_cell_a_variants_reused"] == 288
          and matrix["cell_a_reuse"]["new_model_calls_for_cell_a"] == 0, "cell A reuse verified")

    art = REPO_ROOT / "artifacts"
    check(not [p for p in art.rglob("*") if p.is_file() and "phase12" in str(p)
               and p.suffix in {".safetensors", ".npz", ".pt", ".pth"}], "no activations")
    runs = art / "runs"
    run_dirs = [d for d in runs.glob("phase12*") if d.is_dir()] if runs.is_dir() else []
    check(all(d.name.startswith("phase12a_diagnostic_") for d in run_dirs),
          "only diagnostic runs")
    locked_calls = sum(r["family"] in LOCKED or r.get("cell") == "A"
                       for d in run_dirs for r in _jsonl(d / "diagnostic_choices.jsonl"))
    check(locked_calls == 0, "zero locked-family calls and zero new cell-A calls")
    tracked = subprocess.check_output(["git", "-C", str(REPO_ROOT), "ls-files"], text=True)
    check(not any(t.startswith("data/processed/phase12_design") or "diagnostic_choices" in t
                  for t in tracked.splitlines()), "raw data not tracked")

    if not pre:
        s = json.loads((art / "phase12a_diagnostic/diagnostic_summary.json").read_text("utf-8"))
        bases = json.loads((art / "phase12a_diagnostic/base_labels_by_cell.json")
                           .read_text("utf-8"))
        check(s["run_id"] == cfg["diagnostic_result"]["run_id"], "run id recorded in config")
        check(all(len(bases[c]) == 144 for c in CELLS), "144 bases per cell")
        dep = {c: family_dependence(bases[c]) for c in CELLS}
        check(json.loads(json.dumps(dep)) == s["family_dependence_by_cell"],
              "family dependence recomputes")
        it = interpretation(dep)
        check(json.loads(json.dumps(it)) == s["interpretation"], "interpretation recomputes")
        ok = (status == "phase12a_family_bias_source_diagnostic_complete_awaiting_audit") == it[
            "replication_passed"]
        check(ok, "status matches replication gate")
        check((REPO_ROOT / "reports/phase12a_diagnostic.md").is_file(), "report present")

    print()
    print(f"{n_pass} passed, {len(fails)} failed")
    for f in fails:
        print(f"  - {f}")
    if fails:
        return 1
    print("Phase 12A validation OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
