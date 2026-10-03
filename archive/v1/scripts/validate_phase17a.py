#!/usr/bin/env python3
"""Phase 17A policy-unstable screen integrity checks."""

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
    GUARANTEE,
    N_DISCOVERY_PROMPTS,
    N_S1_CONTINUATIONS,
    PHASE15_RULE_HASH,
    STATUS_AUTHORIZED,
    STATUS_HOLD,
    STATUS_PASS,
    TEMPERATURE,
    THRESHOLD_HASH,
    discovery_prompts,
    family_split,
    is_s1_candidate,
    is_s2_confirmed,
    threshold_manifest,
)

STATUS_S1_DONE = "phase17a_policy_unstable_screen_s1_complete_awaiting_s2"
STATUSES = {STATUS_AUTHORIZED, STATUS_S1_DONE, STATUS_PASS, STATUS_HOLD}
CFG = REPO_ROOT / "configs/experiments/phase17_policy_unstable_screen.yaml"
DESIGN = REPO_ROOT / "artifacts/phase17a_design"
DATA = REPO_ROOT / "data/processed/phase17_design"


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
        "causal_intervention_authorized",
    ):
        check(auth.get(k) is False, f"{k}=false")

    check(
        cfg["phase16_merge_sha"] == "bfb3bccc42e0812a0459e316e428e75f6ac7edda",
        "phase16 merge",
    )
    check(float(cfg["design"]["temperature"]) == TEMPERATURE == 0.9, "T=0.9")
    check(
        cfg["design"]["phase15_rule_hash"] == PHASE15_RULE_HASH,
        "phase15 rule hash",
    )
    check(threshold_manifest()["threshold_hash"] == THRESHOLD_HASH, "threshold hash")

    # Prior phases remain HOLDs / unsupported
    p14 = yaml.safe_load(
        (REPO_ROOT / "configs/experiments/phase14_same_prompt_trajectory.yaml").read_text()
    )
    p15 = yaml.safe_load(
        (
            REPO_ROOT / "configs/experiments/phase15_same_prompt_onset_reanalysis.yaml"
        ).read_text()
    )
    check(
        p14["status"] == "phase14a_same_prompt_trajectory_calibration_hold",
        "P14 HOLD",
    )
    check(
        p15["status"] == "phase15a_same_prompt_onset_reanalysis_hold",
        "P15 HOLD",
    )

    dec = (REPO_ROOT / "docs/decision_log.md").read_text("utf-8")
    if status == STATUS_AUTHORIZED:
        check("### D125" in dec, "D125")
    else:
        check(all(f"### {d}" in dec for d in ("D125", "D126")), "D125-D126")
        if status in (STATUS_PASS, STATUS_HOLD):
            check("### D127" in dec, "D127")

    # Frozen Phase-14 finals
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
    split = family_split()
    for k in ("discovery_train", "discovery_validation", "locked_generalization"):
        check(cfg["family_split"][k] == split[k], f"split {k}")

    matrix = json.loads((DESIGN / "design_matrix.json").read_text("utf-8"))
    disc = [
        json.loads(x)
        for x in (DATA / "discovery_prompts.jsonl").read_text().splitlines()
        if x.strip()
    ]
    check(len(disc) == N_DISCOVERY_PROMPTS, "240 discovery")
    check(
        set(p["family"] for p in disc).isdisjoint(set(split["locked_generalization"])),
        "no locked in discovery file",
    )
    check(
        sha_ids([p["prompt_group_id"] for p in disc])
        == matrix["discovery_prompt_group_ids_sha256"],
        "discovery ids hash",
    )
    s1 = [
        json.loads(x)
        for x in (DATA / "s1_sampling_schedule.jsonl").read_text().splitlines()
        if x.strip()
    ]
    check(len(s1) == N_S1_CONTINUATIONS, "3840 S1 schedule")
    check(
        sha_ids([r["continuation_id"] for r in s1]) == matrix["s1_schedule_sha256"],
        "S1 schedule hash",
    )
    check(matrix["threshold_hash"] == THRESHOLD_HASH, "matrix threshold")
    check(matrix["temperature"] == 0.9, "matrix T")

    # Threshold immutability smoke
    check(
        is_s1_candidate(
            {"n_valid": 14, "n_record": 2, "n_alternate": 2}
        )
        and not is_s1_candidate(
            {"n_valid": 13, "n_record": 2, "n_alternate": 2}
        ),
        "S1 threshold frozen",
    )
    check(
        is_s2_confirmed(
            {"n_valid": 14, "n_record": 3, "n_alternate": 3}
        )
        and not is_s2_confirmed(
            {"n_valid": 14, "n_record": 2, "n_alternate": 3}
        ),
        "S2 threshold frozen",
    )

    if status != STATUS_AUTHORIZED:
        # Auth flags by stage
        if status == STATUS_S1_DONE:
            check(auth.get("s1_model_calls_authorized") is False, "S1 closed")
            check(auth.get("s2_model_calls_authorized") is True, "S2 open")
            check(matrix.get("s2_schedule_frozen") is True, "S2 schedule frozen")
            s2 = [
                json.loads(x)
                for x in (DATA / "s2_sampling_schedule.jsonl").read_text().splitlines()
                if x.strip()
            ]
            check(
                sha_ids([r["continuation_id"] for r in s2])
                == matrix["s2_schedule_sha256"],
                "S2 schedule hash",
            )
            s1s = json.loads(
                (REPO_ROOT / "artifacts/phase17a_s1/s1_summary.json").read_text()
            )
            check(s1s["n_continuations"] == 3840, "S1 labeled 3840")
            check(s1s["locked_model_calls"] == 0, "S1 locked=0")
        if status in (STATUS_PASS, STATUS_HOLD):
            check(auth.get("s1_model_calls_authorized") is False, "S1 closed final")
            check(auth.get("s2_model_calls_authorized") is False, "S2 closed final")
            screen = json.loads(
                (REPO_ROOT / "artifacts/phase17a_screen/screen_summary.json").read_text()
            )
            check(screen["status"] == status, "screen status")
            check(screen["locked_model_calls"] == 0, "screen locked=0")
            check(screen["activations_collected"] is False, "no activations")
            check(
                (REPO_ROOT / "reports/phase17a_policy_unstable_screen.md").is_file(),
                "report",
            )
            check(
                GUARANTEE
                in (REPO_ROOT / "reports/phase17a_policy_unstable_screen.md").read_text(),
                "guarantee",
            )
            if status == STATUS_PASS:
                check(screen["gates"]["passed"] is True, "gates passed")
                check(screen.get("selection") is not None, "selection present")
            else:
                check(screen["gates"]["passed"] is False, "gates failed")

    # No activation artifacts for phase17
    check(
        not [
            p
            for p in (REPO_ROOT / "artifacts").rglob("*")
            if p.is_file()
            and "phase17" in str(p)
            and p.suffix in {".safetensors", ".npz", ".pt", ".pth"}
        ],
        "no activation files",
    )
    # Regenerability of discovery slice
    check(
        [p["prompt_group_id"] for p in discovery_prompts(final)]
        == [p["prompt_group_id"] for p in disc],
        "discovery regenerable",
    )

    print()
    print(f"{n_pass} passed, {len(fails)} failed")
    for f in fails:
        print(f"  - {f}")
    if fails:
        return 1
    print("Phase 17A validation OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
