#!/usr/bin/env python3
"""Phase 9A policy-flip assay diagnostic integrity checks."""

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

from prepare_phase8_design import sha_ids, sha_prompt_texts, sha_scenarios  # noqa: E402
from prepare_phase9_design import build_prompts, build_scenarios, prior_topic_overlap  # noqa: E402

from pre_output_physiology.phase8_design import (  # noqa: E402
    CALIBRATION_FAMILIES,
    FINAL_FAMILIES,
    PRIOR_PHASE_FAMILIES,
)
from pre_output_physiology.phase9_diagnostic import (  # noqa: E402
    FAMILIES,
    FORBIDDEN_PROMPT_SUBSTRINGS,
    interpretation,
    paired_bootstrap_mean,
)

STATUSES = {
    "phase9a_policy_flip_diagnostic_authorized",
    "phase9a_policy_flip_diagnostic_complete_awaiting_audit",
}
CFG = REPO_ROOT / "configs/experiments/phase9_policy_flip_diagnostic.yaml"
DATA = REPO_ROOT / "data/processed/phase9_design"
DESIGN = REPO_ROOT / "artifacts/phase9a_design"


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
        "activation_extraction_authorized",
        "probe_fitting_authorized",
        "phase5_probe_scoring_authorized",
        "causal_intervention_authorized",
        "final_dataset_construction_authorized",
        "prompt_tuning_after_results_authorized",
        "additional_objective_weights_authorized",
        "additional_decoder_variants_authorized",
    ):
        check(auth.get(k) is False, f"{k}=false")
    check(cfg["phase8_merge_sha"] == "35e6d503641ef09c06efa4493ca92c554fc13d10", "phase8 merge")
    dec = (REPO_ROOT / "docs/decision_log.md").read_text("utf-8")
    check(all(f"### {d}" in dec for d in ("D095", "D096", "D097")), "D095-D097")
    prior = set(PRIOR_PHASE_FAMILIES) | set(CALIBRATION_FAMILIES) | set(FINAL_FAMILIES)
    check(len(set(FAMILIES)) == 4 and not set(FAMILIES) & prior, "4 fresh families")

    matrix = json.loads((DESIGN / "design_matrix.json").read_text("utf-8"))
    sc = _jsonl(DATA / "diagnostic_base_scenarios.jsonl")
    pr = _jsonl(DATA / "diagnostic_prompts.jsonl")
    check(len(sc) == 32 and len(pr) == 64, "counts 32 bases / 64 prompts")
    p7 = [p for p in pr if p["framing"] == "P7"]
    p8 = [p for p in pr if p["framing"] == "P8"]
    got = {
        "scenario_text_sha256": sha_scenarios(sc),
        "base_scenario_ids_sha256": sha_ids([s["base_scenario_id"] for s in sc]),
        "p7_prompt_text_sha256": sha_prompt_texts(p7),
        "p8_prompt_text_sha256": sha_prompt_texts(p8),
    }
    for k, v in got.items():
        check(v == cfg["design"][k] == matrix[k], f"{k} frozen")
    regen = build_scenarios()
    rp = build_prompts(regen)
    check(
        sha_scenarios(regen) == got["scenario_text_sha256"]
        and sha_prompt_texts(rp["P7"]) == got["p7_prompt_text_sha256"]
        and sha_prompt_texts(rp["P8"]) == got["p8_prompt_text_sha256"],
        "design regenerable from committed code",
    )
    check(all(v["balanced"] for v in matrix["counterbalance"].values()), "counterbalanced")
    check(not prior_topic_overlap({s["topic_sentence"] for s in sc}), "no Phase 4-8 text reuse")
    leaks = 0
    for p in pr:
        t = p["prompt_text"]
        low = t.lower()
        if any(re.search(rf"\b{re.escape(w)}\b", low) for w in FORBIDDEN_PROMPT_SUBSTRINGS):
            leaks += 1
        if f"takes place at {p['alternate_state']}." not in t:
            leaks += 1
        if f"Private record state: {p['record_state']}\n" not in t:
            leaks += 1
        if p["framing"] == "P8" and not ("(weight 80)" in t and "(weight 20)" in t):
            leaks += 1
        if p["framing"] == "P7" and "weight" in low:
            leaks += 1
    check(leaks == 0, "conflict present; framing wording correct; no forbidden lexicon")
    tk = matrix["tokenization"]
    check(
        tk["response_prefix_token_id"] == 12107
        and tk["n_prompt_contexts_checked"] == 64
        and tk["no_strict_prefix_pairs"]
        and tk["all_candidates_nonempty"]
        and tk["all_lengths_matched"],
        "candidate tokenization integrity (every prompt)",
    )
    check(matrix["p7_p8_differ_only_in_objective_block"] is True, "P7/P8 differ only in block")

    art = REPO_ROOT / "artifacts"
    check(
        not [
            p for p in art.rglob("*")
            if p.is_file() and "phase9" in str(p)
            and p.suffix in {".safetensors", ".npz", ".pt", ".pth"}
        ],
        "no Phase-9 activations",
    )
    runs = art / "runs"
    bad = [d.name for d in runs.glob("phase9*") if d.is_dir()
           and not d.name.startswith("phase9a_diagnostic_")] if runs.is_dir() else []
    check(not bad, "only diagnostic runs")
    tracked = subprocess.check_output(["git", "-C", str(REPO_ROOT), "ls-files"], text=True)
    check(
        not any(t.startswith("data/processed/phase9_design") or "diagnostic_outputs" in t
                for t in tracked.splitlines()),
        "raw data not tracked",
    )

    if status == "phase9a_policy_flip_diagnostic_complete_awaiting_audit":
        s = json.loads((art / "phase9a_diagnostic/diagnostic_summary.json").read_text("utf-8"))
        rows = json.loads((art / "phase9a_diagnostic/labeled_rows.json").read_text("utf-8"))
        check(len(rows) == 64 and s["n_evaluations"] == 128, "128 evaluations")
        check(s["run_id"] == cfg["diagnostic_result"]["run_id"], "run id recorded in config")
        m = {(r["base_scenario_id"], r["framing"]): r["record_minus_alternate_logit"]
             for r in rows}
        diffs = [m[(b, "P8")] - m[(b, "P7")] for b in sorted({r["base_scenario_id"]
                                                               for r in rows})]
        check(paired_bootstrap_mean(diffs) == s["paired_shift_p8_minus_p7"], "shift recomputes")
        check(
            interpretation(s["cell_choice_counts"], s["margins_by_cell"],
                           s["paired_shift_p8_minus_p7"]) == s["interpretation"],
            "interpretation recomputes",
        )
        check(
            all(s["margins_by_cell"][f"{f}_FREE"] == s["margins_by_cell"][f"{f}_CONSTRAINED"]
                for f in ("P7", "P8")),
            "margins shared across decoders within framing",
        )
        check(all(sum(v.values()) == 32 for v in s["cell_choice_counts"].values()),
              "32 per cell")
        check((REPO_ROOT / "reports/phase9a_diagnostic.md").is_file(), "report present")

    print()
    print(f"{n_pass} passed, {len(fails)} failed")
    for f in fails:
        print(f"  - {f}")
    if fails:
        return 1
    print("Phase 9A validation OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
