#!/usr/bin/env python3
"""Phase 13A semantic-component diagnostic integrity checks."""

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
from prepare_phase13_design import build_prompts, cell_hashes, load_selected  # noqa: E402

from pre_output_physiology.phase11_design import LOCKED  # noqa: E402
from pre_output_physiology.phase13_diagnostic import (  # noqa: E402
    CELLS,
    NEW_CELLS,
    REUSE_CELLS,
    SELECTED_BASE_IDS_SHA256,
    family_dependence,
    interpretation,
    neutralization_violations,
)

STATUSES = {
    "phase13a_semantic_component_diagnostic_authorized",
    "phase13a_semantic_component_diagnostic_complete_awaiting_audit",
    "phase13a_semantic_component_diagnostic_hold",
}
CFG = REPO_ROOT / "configs/experiments/phase13_semantic_component.yaml"
DATA = REPO_ROOT / "data/processed/phase13_design"
DESIGN = REPO_ROOT / "artifacts/phase13a_design"


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
        "k_tuning_authorized",
        "payoff_changes_authorized",
        "decoder_changes_authorized",
        "post_result_prompt_tuning_authorized",
    ):
        check(auth.get(k) is False, f"{k}=false")
    pre = status == "phase13a_semantic_component_diagnostic_authorized"
    check(auth["diagnostic_model_calls_authorized"] is pre, "diagnostic auth matches status")
    check(cfg["phase12_merge_sha"] == "fa41d9245f42136e5650d5eca20cb431926b72e9", "phase12 merge")
    check(cfg["phase11_status"] == "phase11a_order_robust_behavior_hold", "Phase 11A remains hold")
    dec = (REPO_ROOT / "docs/decision_log.md").read_text("utf-8")
    check(all(f"### {d}" in dec for d in ("D113", "D114", "D115")), "D113-D115")

    sel = load_selected()
    check(
        sha_ids([b["base_scenario_id"] for b in sel]) == SELECTED_BASE_IDS_SHA256
        == cfg["design"]["selected_base_ids_sha256"],
        "reuses exact Phase-12 selected bases",
    )
    check(len(sel) == 144 and not {b["family"] for b in sel} & set(LOCKED), "144, no locked")
    matrix = json.loads((DESIGN / "design_matrix.json").read_text("utf-8"))
    rows = build_prompts(sel)
    got = cell_hashes(rows)
    for k, v in got.items():
        check(v == cfg["design"][k] == matrix[k], f"{k} frozen and regenerable")
    new = _jsonl(DATA / "new_cell_prompts.jsonl")
    check(
        sha_prompt_texts(new) == cfg["design"]["new_prompt_text_sha256"] and len(new) == 1728,
        "1728 new prompts frozen",
    )
    check(set(cfg["design"]["new_cells"]) == set(NEW_CELLS), "new_cells list")
    by_base = {b["base_scenario_id"]: b for b in sel}
    n_viol = sum(
        bool(
            neutralization_violations(
                r["prompt_text"], by_base[r["base_scenario_id"]], r["cell"]
            )
        )
        for r in rows
    )
    check(n_viol == 0, "neutralization rules hold for every prompt")
    tk = matrix["tokenization"]
    check(
        tk["response_prefix_token_id"] == 12107
        and tk["n_prompt_contexts_checked"] == 2304
        and tk["all_lengths_matched"]
        and tk["all_states_are_slot"]
        and tk["no_strict_prefix_pairs"],
        "tokenization gates",
    )
    check(
        matrix["cell_reuse"]["n_prompt_variants_byte_identical"] == 576
        and matrix["cell_reuse"]["new_model_calls_for_reused_cells"] == 0
        and matrix["cell_reuse"]["reused_cells"] == REUSE_CELLS,
        "000/111 reuse verified byte-identical to Phase-12 D/B",
    )

    art = REPO_ROOT / "artifacts"
    check(
        not [
            p
            for p in art.rglob("*")
            if p.is_file()
            and "phase13" in str(p)
            and p.suffix in {".safetensors", ".npz", ".pt", ".pth"}
        ],
        "no activations",
    )
    runs = art / "runs"
    run_dirs = [d for d in runs.glob("phase13*") if d.is_dir()] if runs.is_dir() else []
    check(all(d.name.startswith("phase13a_diagnostic_") for d in run_dirs), "only diagnostic runs")
    locked_calls = sum(
        r["family"] in LOCKED or r.get("cell") in REUSE_CELLS
        for d in run_dirs
        for r in _jsonl(d / "diagnostic_choices.jsonl")
    )
    check(locked_calls == 0, "zero locked-family calls and zero new reused-cell calls")
    tracked = subprocess.check_output(["git", "-C", str(REPO_ROOT), "ls-files"], text=True)
    check(
        not any(
            t.startswith("data/processed/phase13_design") or "diagnostic_choices" in t
            for t in tracked.splitlines()
        ),
        "raw data not tracked",
    )

    if not pre:
        s = json.loads((art / "phase13a_diagnostic/diagnostic_summary.json").read_text("utf-8"))
        bases = json.loads(
            (art / "phase13a_diagnostic/base_labels_by_cell.json").read_text("utf-8")
        )
        check(s["run_id"] == cfg["diagnostic_result"]["run_id"], "run id recorded in config")
        check(all(len(bases[c]) == 144 for c in CELLS), "144 bases per cell")
        dep = {c: family_dependence(bases[c]) for c in CELLS}
        check(
            json.loads(json.dumps(dep)) == s["family_dependence_by_cell"],
            "family dependence recomputes",
        )
        it = interpretation(dep)
        check(json.loads(json.dumps(it)) == s["interpretation"], "interpretation recomputes")
        ok = (
            status == "phase13a_semantic_component_diagnostic_complete_awaiting_audit"
        ) == (it["replication_111_passed"] and s["cell_000_exact_match_to_phase12_D"])
        check(ok, "status matches replication gates")
        check((REPO_ROOT / "reports/phase13a_diagnostic.md").is_file(), "report present")

    print()
    print(f"{n_pass} passed, {len(fails)} failed")
    for f in fails:
        print(f"  - {f}")
    if fails:
        return 1
    print("Phase 13A validation OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
