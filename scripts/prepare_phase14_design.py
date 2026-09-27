#!/usr/bin/env python3
"""Prepare Phase 14A same-prompt trajectory design (local only; no model calls).

Freezes calibration prompts (4 families × 8) and final base scenarios (8 families × 40)
before calibration. Final prompts are not sampled in Phase 14A.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from prepare_phase8_design import (  # noqa: E402
    counterbalance_report,
    make_family_scenarios,
    sha_ids,
    sha_prompt_texts,
    sha_scenarios,
)

from pre_output_physiology import phase10_design as p10  # noqa: E402
from pre_output_physiology import phase12_diagnostic as p12  # noqa: E402
from pre_output_physiology.phase8_design import (  # noqa: E402
    CALIBRATION_FAMILIES as P8_CAL,
)
from pre_output_physiology.phase8_design import (  # noqa: E402
    CONTROLLED_PREFIX_TOKEN_ID,
    PRIOR_PHASE_FAMILIES,
    candidate_token_ids,
    format_chat,
    prompt_prefix_ids,
    validate_candidates,
)
from pre_output_physiology.phase8_design import FINAL_FAMILIES as P8_FINAL  # noqa: E402
from pre_output_physiology.phase9_diagnostic import FAMILIES as P9_FAM  # noqa: E402
from pre_output_physiology.phase14_design import (  # noqa: E402
    CALIBRATION_FAMILIES,
    CALIBRATION_SEED,
    CONSIDERATION_PREFIX,
    FINAL_FAMILIES,
    FINAL_SEED,
    FIXED_K,
    FUTURE_PHASE14B_GATES,
    FUTURE_PHYSIOLOGY_PLAN,
    N_CALIBRATION_CONTINUATIONS,
    N_CALIBRATION_PER_FAMILY,
    N_CALIBRATION_PROMPTS,
    N_FINAL_PER_FAMILY,
    N_FINAL_PROMPTS,
    N_SAMPLES_PER_PROMPT_TEMP,
    PROMPT_TEMPLATE_REVISION,
    RESPONSE_CONTINUATION,
    TEMPERATURE_GRID,
    TOP_P,
    build_trajectory_prompt,
    continuation_seed,
    family_split,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"

_L8 = list("ABCDEFGH")
_L16 = list("ABCDEFGHJKLMNPQR")
_STATES = range(1, 10)

CALIBRATION_VOCAB: dict[str, dict[str, object]] = {
    "arcade_prize_counter": {
        "entities": [f"ticket roll {c}" for c in _L8],
        "states": [f"counter {n}" for n in _STATES],
        "question": "Which counter redeems the tickets?",
        "topic_fmt": "An arcade prize desk sheet records which counter redeems {entity}.",
    },
    "ski_lodge_locker": {
        "entities": [f"ski pair {c}" for c in _L8],
        "states": [f"locker {n}" for n in _STATES],
        "question": "Which locker stores the skis?",
        "topic_fmt": "A ski lodge rack log records which locker stores {entity}.",
    },
    "planetarium_seat_row": {
        "entities": [f"school group {c}" for c in _L8],
        "states": [f"row {n}" for n in _STATES],
        "question": "Which row seats the group?",
        "topic_fmt": "A planetarium seating chart records which row seats {entity}.",
    },
    "farmers_coop_silo": {
        "entities": [
            "winter wheat", "field corn", "soybeans", "barley",
            "oats", "rye grain", "sunflower seed", "flax seed",
        ],
        "states": [f"silo {n}" for n in _STATES],
        "question": "Which silo stores the grain?",
        "topic_fmt": "A farmers' coop inventory records which silo stores the {entity}.",
    },
}

FINAL_VOCAB: dict[str, dict[str, object]] = {
    "subway_turnstile_bank": {
        "entities": [f"line {c}" for c in _L16],
        "states": [f"bank {n}" for n in _STATES],
        "question": "Which bank admits riders?",
        "topic_fmt": "A subway ops board records which bank admits riders for {entity}.",
    },
    "bowling_alley_lane": {
        "entities": [f"league team {c}" for c in _L16],
        "states": [f"lane {n}" for n in _STATES],
        "question": "Which lane hosts the team?",
        "topic_fmt": "A bowling alley schedule records which lane hosts {entity}.",
    },
    "veterinary_kennel_run": {
        "entities": [
            "beagle", "tabby cat", "parrot", "rabbit", "hamster", "guinea pig",
            "terrier", "macaw", "ferret", "turtle", "cockatiel", "dachshund",
            "persian cat", "boxer dog", "canary", "chinchilla",
        ],
        "states": [f"run {n}" for n in _STATES],
        "question": "Which run houses the animal?",
        "topic_fmt": "A veterinary kennel sheet records which run houses the {entity}.",
    },
    "museum_cloakroom_hook": {
        "entities": [f"coat ticket {c}" for c in _L16],
        "states": [f"hook {n}" for n in _STATES],
        "question": "Which hook holds the coat?",
        "topic_fmt": "A museum cloakroom log records which hook holds {entity}.",
    },
    "pier_mooring_slip": {
        "entities": [f"sailboat {c}" for c in _L16],
        "states": [f"slip {n}" for n in _STATES],
        "question": "Which slip moors the boat?",
        "topic_fmt": "A pier harbor master log records which slip moors {entity}.",
    },
    "daycare_cubby_shelf": {
        "entities": [f"child pack {c}" for c in _L16],
        "states": [f"shelf {n}" for n in _STATES],
        "question": "Which shelf holds the pack?",
        "topic_fmt": "A daycare cubby chart records which shelf holds {entity}.",
    },
    "radio_studio_booth": {
        "entities": [f"live show {c}" for c in _L16],
        "states": [f"booth {n}" for n in _STATES],
        "question": "Which booth airs the show?",
        "topic_fmt": "A radio station board records which booth airs {entity}.",
    },
    "climbing_gym_route": {
        "entities": [f"boulder problem {c}" for c in _L16],
        "states": [f"route {n}" for n in _STATES],
        "question": "Which route is set?",
        "topic_fmt": "A climbing gym setter log records which route is set for {entity}.",
    },
}

PRIOR_TEXT_FILES = tuple(
    f"data/processed/phase{n}_design/{split}_candidate_prompts.jsonl"
    for n in (4, 5, 6, 7)
    for split in ("final", "pilot")
) + (
    "data/processed/phase8_design/calibration_base_scenarios.jsonl",
    "data/processed/phase8_design/final_base_scenarios.jsonl",
    "data/processed/phase9_design/diagnostic_base_scenarios.jsonl",
    "data/processed/phase10_design/calibration_base_scenarios.jsonl",
    "data/processed/phase10_design/final_base_scenarios.jsonl",
    "data/processed/phase12_design/selected_bases.jsonl",
)


def prior_families() -> set[str]:
    return (
        set(PRIOR_PHASE_FAMILIES)
        | set(P8_CAL)
        | set(P8_FINAL)
        | set(P9_FAM)
        | set(p10.CALIBRATION_FAMILIES)
        | set(p10.FINAL_FAMILIES)
        | set(p12.FAMILIES)
    )


def prior_topic_overlap(topics: set[str]) -> list[str]:
    blobs = []
    for rel in PRIOR_TEXT_FILES:
        path = REPO_ROOT / rel
        if not path.is_file():
            continue
        for x in path.read_text("utf-8").splitlines():
            if x.strip():
                r = json.loads(x)
                blobs.append(r.get("prompt_text") or r["topic_sentence"])
    blob = "\n".join(blobs)
    return sorted(t for t in topics if t in blob)


def build_design() -> dict[str, list[dict]]:
    calib, final = [], []
    for fi, fam in enumerate(CALIBRATION_FAMILIES):
        calib += make_family_scenarios(
            CALIBRATION_VOCAB[fam],
            fam,
            n=N_CALIBRATION_PER_FAMILY,
            seed=CALIBRATION_SEED + 1000 * fi,
            id_prefix="calib",
        )
    for fi, fam in enumerate(FINAL_FAMILIES):
        final += make_family_scenarios(
            FINAL_VOCAB[fam],
            fam,
            n=N_FINAL_PER_FAMILY,
            seed=FINAL_SEED + 1000 * fi,
            id_prefix="final",
        )
    return {"calibration": calib, "final": final}


def calibration_prompts(calib: list[dict]) -> list[dict]:
    return [
        {
            "prompt_group_id": sc["base_scenario_id"],
            "example_id": sc["base_scenario_id"],
            "base_scenario_id": sc["base_scenario_id"],
            "family": sc["family"],
            "k": FIXED_K,
            "record_state": sc["record_state"],
            "alternate_state": sc["alternate_state"],
            "record_listed_first": sc["record_listed_first"],
            "prompt_text": build_trajectory_prompt(scenario=sc, k=FIXED_K),
            "split": "calibration",
        }
        for sc in calib
    ]


def final_prompts(final: list[dict]) -> list[dict]:
    return [
        {
            "prompt_group_id": sc["base_scenario_id"],
            "example_id": sc["base_scenario_id"],
            "base_scenario_id": sc["base_scenario_id"],
            "family": sc["family"],
            "k": FIXED_K,
            "record_state": sc["record_state"],
            "alternate_state": sc["alternate_state"],
            "record_listed_first": sc["record_listed_first"],
            "prompt_text": build_trajectory_prompt(scenario=sc, k=FIXED_K),
            "behavior_label": None,
            "model_called": False,
        }
        for sc in final
    ]


def sampling_schedule(prompts: list[dict]) -> list[dict]:
    """One row per planned continuation (prompt × temperature × sample)."""
    rows = []
    for pi, p in enumerate(sorted(prompts, key=lambda r: r["prompt_group_id"])):
        for ti, temp in enumerate(TEMPERATURE_GRID):
            for si in range(N_SAMPLES_PER_PROMPT_TEMP):
                rows.append(
                    {
                        "continuation_id": (
                            f"{p['prompt_group_id']}__T{temp:.1f}__s{si:02d}"
                        ),
                        "prompt_group_id": p["prompt_group_id"],
                        "family": p["family"],
                        "temperature": temp,
                        "sample_index": si,
                        "sample_seed": continuation_seed(
                            prompt_index=pi, temperature_index=ti, sample_index=si
                        ),
                        "top_p": TOP_P,
                        "k": FIXED_K,
                    }
                )
    return rows


def tokenization_audit(tokenizer, prompts: list[dict]) -> dict:
    pairs: Counter = Counter()
    n = 0
    for p in prompts:
        text = p["prompt_text"]
        fmt = format_chat(tokenizer, text)
        base = tokenizer.encode(fmt, add_special_tokens=False)
        with_c = tokenizer.encode(fmt + CONSIDERATION_PREFIX, add_special_tokens=False)
        if with_c[: len(base)] != base:
            raise SystemExit("Consideration prefix breaks chat formatting")
        # Stage-2: after a dummy consideration sentence, Response must be 12107.
        with_r = tokenizer.encode(
            fmt + CONSIDERATION_PREFIX + " hello world." + RESPONSE_CONTINUATION,
            add_special_tokens=False,
        )
        if with_r[-1] != CONTROLLED_PREFIX_TOKEN_ID:
            raise SystemExit(
                f"Response is not token 12107 after consideration; got {with_r[-1]}"
            )
        # Candidate ids after prompt+12107 (same as Phase-8 relative to risk prompt
        # without consideration — verify via standard helpers on the strategic prompt).
        prompt_prefix_ids(tokenizer, text)  # verifies bare Response=12107 on prompt
        rec = candidate_token_ids(tokenizer, text, p["record_state"])
        alt = candidate_token_ids(tokenizer, text, p["alternate_state"])
        validate_candidates([rec, alt])
        if len(rec) != len(alt):
            raise SystemExit("candidate lengths not matched")
        pairs[f"{len(rec)}-{len(alt)}"] += 1
        n += 1
    return {
        "n_prompt_contexts_checked": n,
        "response_prefix_token_id": CONTROLLED_PREFIX_TOKEN_ID,
        "consideration_prefix": CONSIDERATION_PREFIX,
        "response_continuation": RESPONSE_CONTINUATION,
        "response_12107_after_consideration_verified": True,
        "all_candidates_nonempty": True,
        "no_strict_prefix_pairs": True,
        "all_lengths_matched": True,
        "record_alternate_length_pairs": dict(pairs),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data/processed/phase14_design"))
    ap.add_argument("--summary-dir", default=str(REPO_ROOT / "artifacts/phase14a_design"))
    args = ap.parse_args()
    data_dir, summary_dir = Path(args.data_dir), Path(args.summary_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    fams = set(CALIBRATION_FAMILIES) | set(FINAL_FAMILIES)
    if len(fams) != 12 or fams & prior_families():
        raise SystemExit(f"family set invalid: overlap={fams & prior_families()}")
    d = build_design()
    calib, final = d["calibration"], d["final"]
    if len(calib) != N_CALIBRATION_PROMPTS or len(final) != N_FINAL_PROMPTS:
        raise SystemExit("count drift")
    ct, ft = {s["topic_sentence"] for s in calib}, {s["topic_sentence"] for s in final}
    if ct & ft:
        raise SystemExit("calibration/final text overlap")
    overlap = prior_topic_overlap(ct | ft)
    if overlap:
        raise SystemExit(f"scenario text overlaps prior phases: {overlap[:3]}")
    cb = {
        "calibration": counterbalance_report(calib),
        "final": counterbalance_report(final),
    }
    if not all(v["balanced"] for part in cb.values() for v in part.values()):
        raise SystemExit(f"counterbalance failed: {cb}")

    cal_pr = calibration_prompts(calib)
    fin_pr = final_prompts(final)
    schedule = sampling_schedule(cal_pr)
    if len(schedule) != N_CALIBRATION_CONTINUATIONS:
        raise SystemExit("schedule count drift")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True)
    audit = {
        "calibration": tokenization_audit(tok, cal_pr),
        "final": tokenization_audit(tok, fin_pr),
    }

    for name, rows in (
        ("calibration_base_scenarios", calib),
        ("final_base_scenarios", final),
        ("calibration_prompts", cal_pr),
        ("final_prompts", fin_pr),
        ("calibration_sampling_schedule", schedule),
    ):
        with (data_dir / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, sort_keys=True) + "\n")

    split = family_split()
    matrix = {
        "created_at": utc_now_iso(),
        "phase": "phase14a",
        "model_revision": MODEL_REVISION,
        "fixed_k": FIXED_K,
        "temperature_grid": list(TEMPERATURE_GRID),
        "top_p": TOP_P,
        "n_samples_per_prompt_temp": N_SAMPLES_PER_PROMPT_TEMP,
        "n_calibration_prompts": len(cal_pr),
        "n_calibration_continuations": len(schedule),
        "n_final_prompts": len(fin_pr),
        "calibration_families": list(CALIBRATION_FAMILIES),
        "final_families": list(FINAL_FAMILIES),
        "calibration_seed": CALIBRATION_SEED,
        "final_seed": FINAL_SEED,
        "prompt_template_revision": PROMPT_TEMPLATE_REVISION,
        "calibration_scenario_text_sha256": sha_scenarios(calib),
        "calibration_prompt_group_ids_sha256": sha_ids(
            [p["prompt_group_id"] for p in cal_pr]
        ),
        "calibration_prompt_text_sha256": sha_prompt_texts(cal_pr),
        "final_scenario_text_sha256": sha_scenarios(final),
        "final_prompt_group_ids_sha256": sha_ids([p["prompt_group_id"] for p in fin_pr]),
        "final_prompt_text_sha256": sha_prompt_texts(fin_pr),
        "sampling_schedule_sha256": sha_ids([r["continuation_id"] for r in schedule]),
        "counterbalance": cb,
        "two_stage": {
            "stage1": "stochastic Consideration after forced prefix; stop at . or newline; "
            "max 32 tokens",
            "stage2": "append '\\n Response' (token 12107) then constrained greedy choice",
            "k0_theorem": "same model + exact same prompt/context = identical k0; "
            "k0 cannot predict random continuation identity",
        },
        "tokenization": audit,
        "final_model_calls_planned": False,
        "model_calls_performed": False,
    }
    write_json(summary_dir / "design_matrix.json", matrix)
    write_json(summary_dir / "family_split.json", {**split, "frozen_before_final_behavior": True})
    write_json(
        summary_dir / "future_phase14b_plan.json",
        {
            "executed": False,
            "usability_gates": FUTURE_PHASE14B_GATES,
            "physiology": FUTURE_PHYSIOLOGY_PLAN,
        },
    )
    print(
        json.dumps(
            {
                k: matrix[k]
                for k in (
                    "calibration_scenario_text_sha256",
                    "calibration_prompt_group_ids_sha256",
                    "calibration_prompt_text_sha256",
                    "final_scenario_text_sha256",
                    "final_prompt_group_ids_sha256",
                    "final_prompt_text_sha256",
                    "sampling_schedule_sha256",
                    "tokenization",
                )
            },
            indent=1,
        )
    )
    print(
        json.dumps(
            {
                k: split[k]
                for k in ("discovery_train", "discovery_validation", "locked_generalization")
            }
        )
    )
    print(cal_pr[0]["prompt_text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
