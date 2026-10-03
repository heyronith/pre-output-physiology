#!/usr/bin/env python3
"""Phase 19A unseen validation family screen integrity checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from prepare_phase8_design import sha_ids, sha_prompt_texts, sha_scenarios  # noqa: E402
from prepare_phase14_design import prior_families  # noqa: E402

from pre_output_physiology.phase19_screen import (  # noqa: E402
    GUARANTEE,
    N_PER_FAMILY,
    N_S1_CONTINUATIONS,
    N_SCREEN_PROMPTS,
    NEW_FAMILIES,
    PHASE15_RULE_HASH,
    PHASE18_STATUS_REQUIRED,
    STATUS_AUTHORIZED,
    STATUS_COHORT_HOLD,
    STATUS_FAMILY_HOLD,
    STATUS_PASS,
    STATUS_S1_DONE,
    STATUS_S2_DONE_AWAITING_FRESH,
    TEMPERATURE,
    THRESHOLD_HASH,
    is_fresh_usable,
    is_s1_candidate,
    is_s2_confirmed,
    threshold_manifest,
)

STATUSES = {
    STATUS_AUTHORIZED,
    STATUS_S1_DONE,
    STATUS_S2_DONE_AWAITING_FRESH,
    STATUS_FAMILY_HOLD,
    STATUS_COHORT_HOLD,
    STATUS_PASS,
}
CFG = REPO_ROOT / "configs/experiments/phase19_unseen_validation.yaml"
DESIGN = REPO_ROOT / "artifacts/phase19a_design"
DATA = REPO_ROOT / "data/processed/phase19_design"


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
        "temperature_search_authorized",
        "prompt_changes_authorized",
        "payoff_changes_authorized",
        "decoder_changes_authorized",
        "onset_rule_changes_authorized",
        "threshold_changes_authorized",
        "selection_rule_changes_authorized",
        "phase18_train_model_calls_authorized",
        "causal_intervention_authorized",
    ):
        check(auth.get(k) is False, f"{k}=false")

    check(
        cfg["phase18_merge_sha"] == "9b975e28e08f0f9ad2a3a315dd475c92a68cb346",
        "phase18 merge",
    )
    check(float(cfg["design"]["temperature"]) == TEMPERATURE == 0.9, "T=0.9")
    check(cfg["design"]["phase15_rule_hash"] == PHASE15_RULE_HASH, "phase15 rule")
    check(threshold_manifest()["threshold_hash"] == THRESHOLD_HASH, "threshold hash")

    p18 = yaml.safe_load(
        (
            REPO_ROOT / "configs/experiments/phase18_enriched_cohort.yaml"
        ).read_text()
    )
    check(p18["status"] == PHASE18_STATUS_REQUIRED, "Phase 18 remains HOLD")

    dec = (REPO_ROOT / "docs/decision_log.md").read_text("utf-8")
    check("### D130" in dec, "D130")

    check(set(NEW_FAMILIES).isdisjoint(prior_families()), "families novel")
    check(len(NEW_FAMILIES) == 6, "6 families")

    matrix = json.loads((DESIGN / "design_matrix.json").read_text("utf-8"))
    prompts = [
        json.loads(x)
        for x in (DATA / "screen_prompts.jsonl").read_text().splitlines()
        if x.strip()
    ]
    scenarios = [
        json.loads(x)
        for x in (DATA / "screen_base_scenarios.jsonl").read_text().splitlines()
        if x.strip()
    ]
    check(len(prompts) == N_SCREEN_PROMPTS == 144, "144 prompts")
    check(
        sha_ids([p["prompt_group_id"] for p in prompts])
        == matrix["screen_prompt_group_ids_sha256"],
        "prompt ids hash",
    )
    check(
        sha_prompt_texts(prompts) == matrix["screen_prompt_text_sha256"],
        "prompt text hash",
    )
    check(
        sha_scenarios(scenarios) == matrix["screen_scenario_text_sha256"],
        "scenario hash",
    )
    for fam in NEW_FAMILIES:
        check(
            sum(1 for p in prompts if p["family"] == fam) == N_PER_FAMILY,
            f"24×{fam}",
        )

    s1 = [
        json.loads(x)
        for x in (DATA / "s1_sampling_schedule.jsonl").read_text().splitlines()
        if x.strip()
    ]
    check(len(s1) == N_S1_CONTINUATIONS == 2304, "2304 S1")
    check(
        sha_ids([r["continuation_id"] for r in s1]) == matrix["s1_schedule_sha256"],
        "S1 schedule hash",
    )
    check(all(r["sample_seed"] >= 19_000_000 for r in s1), "S1 seeds ≥19M")
    check(
        is_s1_candidate({"n_valid": 14, "n_record": 2, "n_alternate": 2})
        and not is_s1_candidate({"n_valid": 13, "n_record": 2, "n_alternate": 2}),
        "S1 threshold frozen",
    )
    check(
        is_s2_confirmed({"n_valid": 14, "n_record": 3, "n_alternate": 3})
        and not is_s2_confirmed({"n_valid": 14, "n_record": 2, "n_alternate": 3}),
        "S2 threshold frozen",
    )
    check(
        is_fresh_usable({"n_valid": 42, "n_record": 6, "n_alternate": 6})
        and not is_fresh_usable({"n_valid": 41, "n_record": 6, "n_alternate": 6}),
        "fresh usability frozen",
    )

    if status == STATUS_AUTHORIZED:
        check(auth.get("s1_model_calls_authorized") is True, "S1 open")
        check(matrix.get("model_calls_performed") is False, "no calls yet")
    if status == STATUS_S1_DONE:
        check(auth.get("s1_model_calls_authorized") is False, "S1 closed")
        check(auth.get("s2_model_calls_authorized") is True, "S2 open")
        check(matrix.get("s2_schedule_frozen") is True, "S2 frozen")
    if status == STATUS_S2_DONE_AWAITING_FRESH:
        check(auth.get("s2_model_calls_authorized") is False, "S2 closed")
        check(auth.get("fresh_confirmation_model_calls_authorized") is True, "fresh open")
        check(matrix.get("fresh_schedule_frozen") is True, "fresh frozen")
    if status in (
        STATUS_FAMILY_HOLD,
        STATUS_COHORT_HOLD,
        STATUS_PASS,
    ):
        check(auth.get("s1_model_calls_authorized") is False, "S1 closed final")
        check(auth.get("s2_model_calls_authorized") is False, "S2 closed final")
        check(
            auth.get("fresh_confirmation_model_calls_authorized") is False,
            "fresh closed final",
        )
        if status == STATUS_FAMILY_HOLD:
            screen = json.loads(
                (REPO_ROOT / "artifacts/phase19a_screen/screen_summary.json").read_text()
            )
            check(screen["status"] == STATUS_FAMILY_HOLD, "family hold status")
            check(screen["n_qualifying_families"] < 2, "<2 qualifying")
        if status in (STATUS_COHORT_HOLD, STATUS_PASS):
            conf = json.loads(
                (
                    REPO_ROOT
                    / "artifacts/phase19a_confirmation/confirmation_summary.json"
                ).read_text()
            )
            check(conf["status"] == status, "confirmation status")
            check(conf["activations_collected"] is False, "no activations")
            check(conf["phase18_train_model_calls"] == 0, "no P18 train calls")
            check(GUARANTEE in conf["guarantee"], "guarantee")

    check(
        not [
            p
            for p in (REPO_ROOT / "artifacts").rglob("*")
            if p.is_file()
            and "phase19" in str(p)
            and p.suffix in {".safetensors", ".npz", ".pt", ".pth"}
        ],
        "no activation files",
    )

    print()
    print(f"{n_pass} passed, {len(fails)} failed")
    for f in fails:
        print(f"  - {f}")
    if fails:
        return 1
    print("Phase 19A validation OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
