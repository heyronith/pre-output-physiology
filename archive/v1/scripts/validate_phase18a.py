#!/usr/bin/env python3
"""Phase 18A enriched-cohort fresh confirmation integrity checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from prepare_phase8_design import sha_ids, sha_prompt_texts, sha_scenarios  # noqa: E402

from pre_output_physiology.phase17_screen import (  # noqa: E402
    FINAL_PROMPT_GROUP_IDS_SHA256,
    FINAL_PROMPT_TEXT_SHA256,
    FINAL_SCENARIO_TEXT_SHA256,
)
from pre_output_physiology.phase18_cohort import (  # noqa: E402
    EXCLUDED_FAMILIES,
    GUARANTEE,
    LOCKED_FAMILIES,
    N_COHORT,
    N_CONTINUATIONS,
    PHASE15_RULE_HASH,
    PHASE17_CONFIRMED_IDS_SHA256,
    PHASE17_STATUS,
    QUALIFYING_FAMILIES,
    SEED_BASE,
    STATUS_AUTHORIZED,
    STATUS_HOLD,
    STATUS_PASS,
    TEMPERATURE,
    THRESHOLD_HASH,
    is_fresh_usable,
    load_phase17_confirmed,
    select_enriched_cohort,
    threshold_manifest,
)

STATUSES = {STATUS_AUTHORIZED, STATUS_PASS, STATUS_HOLD}
CFG = REPO_ROOT / "configs/experiments/phase18_enriched_cohort.yaml"
DESIGN = REPO_ROOT / "artifacts/phase18a_design"
DATA = REPO_ROOT / "data/processed/phase18_design"


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
        "replacement_of_failed_prompts_authorized",
        "causal_intervention_authorized",
        "phase18b_authorized",
    ):
        check(auth.get(k) is False, f"{k}=false")

    check(
        cfg["phase17_merge_sha"] == "8353b8bafafbc554ac8ff8056d029ba4de10f384",
        "phase17 merge",
    )
    check(float(cfg["design"]["temperature"]) == TEMPERATURE == 0.9, "T=0.9")
    check(
        cfg["design"]["phase15_rule_hash"] == PHASE15_RULE_HASH,
        "phase15 rule hash",
    )
    check(threshold_manifest()["threshold_hash"] == THRESHOLD_HASH, "threshold hash")

    p17 = yaml.safe_load(
        (
            REPO_ROOT / "configs/experiments/phase17_policy_unstable_screen.yaml"
        ).read_text()
    )
    check(p17["status"] == PHASE17_STATUS, "Phase 17 remains HOLD")

    dec = (REPO_ROOT / "docs/decision_log.md").read_text("utf-8")
    if status == STATUS_AUTHORIZED:
        check("### D128" in dec, "D128")
    else:
        check(all(f"### {d}" in dec for d in ("D128", "D129")), "D128-D129")

    final = [
        json.loads(x)
        for x in (REPO_ROOT / "data/processed/phase14_design/final_prompts.jsonl")
        .read_text()
        .splitlines()
        if x.strip()
    ]
    scen = [
        json.loads(x)
        for x in (
            REPO_ROOT / "data/processed/phase14_design/final_base_scenarios.jsonl"
        )
        .read_text()
        .splitlines()
        if x.strip()
    ]
    check(sha_scenarios(scen) == FINAL_SCENARIO_TEXT_SHA256, "final scenario hash")
    check(
        sha_ids([p["prompt_group_id"] for p in final]) == FINAL_PROMPT_GROUP_IDS_SHA256,
        "final group hash",
    )
    check(sha_prompt_texts(final) == FINAL_PROMPT_TEXT_SHA256, "final prompt hash")

    screen = json.loads(
        (REPO_ROOT / "artifacts/phase17a_screen/screen_summary.json").read_text()
    )
    check(screen["status"] == PHASE17_STATUS, "screen HOLD")
    confirmed = load_phase17_confirmed(screen)
    check(len(confirmed) == 40, "40 confirmed source")
    check(
        sha_ids(sorted(c["prompt_group_id"] for c in confirmed))
        == PHASE17_CONFIRMED_IDS_SHA256,
        "confirmed IDs hash",
    )
    regen = select_enriched_cohort(confirmed)
    matrix = json.loads((DESIGN / "design_matrix.json").read_text("utf-8"))
    check(regen["cohort_sha256"] == matrix["cohort_sha256"], "cohort regenerable")
    check(matrix["n_cohort"] == N_COHORT == 24, "24 cohort")
    check(matrix["n_continuations"] == N_CONTINUATIONS == 1152, "1152 schedule size")
    check(matrix["seed_base"] == SEED_BASE == 18_000_000, "seed base 18M")
    check(
        matrix["phase17_confirmed_ids_sha256"] == PHASE17_CONFIRMED_IDS_SHA256,
        "matrix confirmed hash",
    )

    prompts = [
        json.loads(x)
        for x in (DATA / "cohort_prompts.jsonl").read_text().splitlines()
        if x.strip()
    ]
    check(len(prompts) == N_COHORT, "24 prompts file")
    check(
        sha_ids([p["prompt_group_id"] for p in prompts]) == matrix["cohort_sha256"],
        "prompts hash",
    )
    check(
        set(p["family"] for p in prompts).isdisjoint(set(LOCKED_FAMILIES)),
        "no locked in cohort",
    )
    check(
        set(p["family"] for p in prompts).isdisjoint(set(EXCLUDED_FAMILIES)),
        "no excluded in cohort",
    )
    for fam in QUALIFYING_FAMILIES:
        check(
            sum(1 for p in prompts if p["family"] == fam) == 6,
            f"6×{fam}",
        )
    check(matrix["n_train"] == 18, "18 train")
    check(matrix["n_validation"] == 6, "6 validation")

    schedule = [
        json.loads(x)
        for x in (DATA / "fresh_sampling_schedule.jsonl").read_text().splitlines()
        if x.strip()
    ]
    check(len(schedule) == N_CONTINUATIONS, "1152 schedule rows")
    check(
        sha_ids([r["continuation_id"] for r in schedule])
        == matrix["fresh_schedule_sha256"],
        "schedule hash",
    )
    seeds = [r["sample_seed"] for r in schedule]
    check(min(seeds) >= SEED_BASE, "seeds ≥ 18M")
    check(all(s >= 18_000_000 for s in seeds), "disjoint from P14/P17")
    check(matrix["threshold_hash"] == THRESHOLD_HASH, "matrix threshold")
    check(matrix["temperature"] == 0.9, "matrix T")
    check(
        is_fresh_usable(
            {"n_valid": 42, "n_record": 6, "n_alternate": 6}
        )
        and not is_fresh_usable(
            {"n_valid": 41, "n_record": 6, "n_alternate": 6}
        ),
        "fresh usability frozen",
    )

    if status == STATUS_AUTHORIZED:
        check(
            auth.get("fresh_confirmation_model_calls_authorized") is True,
            "fresh calls open",
        )
        check(auth.get("modal_gpu_authorized") is True, "modal open")
        check(matrix.get("model_calls_performed") is False, "no calls yet")

    if status in (STATUS_PASS, STATUS_HOLD):
        check(
            auth.get("fresh_confirmation_model_calls_authorized") is False,
            "fresh calls closed",
        )
        conf = json.loads(
            (
                REPO_ROOT
                / "artifacts/phase18a_confirmation/confirmation_summary.json"
            ).read_text()
        )
        check(conf["status"] == status, "confirmation status")
        check(conf["locked_model_calls"] == 0, "locked=0")
        check(conf["activations_collected"] is False, "no activations")
        check(conf["n_continuations"] == 1152, "1152 evaluated")
        check(
            (REPO_ROOT / "reports/phase18a_enriched_cohort_fresh_confirmation.md").is_file(),
            "report",
        )
        check(
            GUARANTEE
            in (
                REPO_ROOT / "reports/phase18a_enriched_cohort_fresh_confirmation.md"
            ).read_text(),
            "guarantee",
        )
        if status == STATUS_PASS:
            check(conf["gates"]["passed"] is True, "gates passed")
            check(conf.get("fresh_usable_prompt_group_ids") is not None, "usable frozen")
        else:
            check(conf["gates"]["passed"] is False, "gates failed")

    check(
        not [
            p
            for p in (REPO_ROOT / "artifacts").rglob("*")
            if p.is_file()
            and "phase18" in str(p)
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
    print("Phase 18A validation OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
