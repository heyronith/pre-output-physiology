#!/usr/bin/env python3
"""Freeze Phase-23 open-grader design before any candidate evaluation."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase21_roleplay import (  # noqa: E402
    GRADING_CONTENT_SHA256,
    sha256_file,
)
from pre_output_physiology.phase23_open_grader import (  # noqa: E402
    APOLLO_RUBRIC_BODY,
    CANDIDATES,
    DEV_ELIGIBILITY,
    GRADER_SYSTEM,
    GRADER_USER_TEMPLATE,
    GUARANTEE,
    INFERENCE,
    LOCKED_ACCEPTANCE,
    N_DEV_PROMPTS,
    N_LOCKED_PROMPTS,
    N_PILOT_RESPONSES,
    N_PROMPTS,
    ONSET_ACCEPTANCE,
    PHASE21_ONSET,
    PHASE22B_ONSET,
    PILOT_ELIMINATION,
    REFERENCE_GRADER,
    STATUS_DESIGN,
    _sha_json,
    grader_prompt_split,
    select_pilot_responses,
    sha256_text,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

GRADING_PATH = REPO_ROOT / "data/raw/apollo_roleplaying/grading_template.txt"
PROMPTS_PATH = REPO_ROOT / "data/processed/phase21_roleplay/prompts.jsonl"
PROC = REPO_ROOT / "data/processed/phase23_open_grader"
ART = REPO_ROOT / "artifacts/phase23_design"


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(x)
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def main() -> int:
    if sha256_file(GRADING_PATH) != GRADING_CONTENT_SHA256:
        raise SystemExit("Apollo grading template content SHA mismatch")

    prompts = _load_jsonl(PROMPTS_PATH)
    if len(prompts) != N_PROMPTS:
        raise SystemExit(f"prompts {len(prompts)} != {N_PROMPTS}")
    prompt_by_id = {p["prompt_id"]: p for p in prompts}

    rows21 = _load_jsonl(
        REPO_ROOT / "artifacts/runs" / PHASE21_ONSET / "annotated.jsonl"
    )
    rows22 = _load_jsonl(
        REPO_ROOT / "artifacts/runs" / PHASE22B_ONSET / "annotated.jsonl"
    )
    combined = rows21 + rows22
    if len(combined) != N_PROMPTS * 20:
        raise SystemExit(f"expected 7420 rows, got {len(combined)}")

    split = grader_prompt_split([p["prompt_id"] for p in prompts])
    if split["n_development"] != N_DEV_PROMPTS or split["n_locked_validation"] != N_LOCKED_PROMPTS:
        raise SystemExit("split counts mismatch")

    # Build reference corpus rows
    corpus = []
    for r in combined:
        p = prompt_by_id[r["prompt_id"]]
        corpus.append(
            {
                "continuation_id": r["continuation_id"],
                "prompt_id": r["prompt_id"],
                "sample_seed": r["sample_seed"],
                "physiology_split": r["split"],  # Phase-21 train/test
                "grader_split": split["split_by_prompt_id"][r["prompt_id"]],
                "scenario": p["scenario"],
                "question": p["question"],
                "answer_prefix": r.get("answer_prefix") or p["answer_prefix"],
                "full_response": r["full_response"],
                "reference_grader": REFERENCE_GRADER,
                "reference_judgement": r.get("judgement"),
                "reference_label": r["label"],
                "explicit_span": r.get("explicit_span"),
                "onset_char_start": r.get("onset_char_start"),
                "onset_char_end": r.get("onset_char_end"),
                "onset_quote": r.get("onset_quote"),
                "onset_token_index": r.get("onset_token_index"),
            }
        )

    dev_ids = set(split["development_prompt_ids"])
    pilot_cids = select_pilot_responses(corpus, development_prompt_ids=dev_ids)
    pilot_set = set(pilot_cids)
    pilot_rows = [r for r in corpus if r["continuation_id"] in pilot_set]
    pilot_rows = sorted(
        pilot_rows,
        key=lambda r: __import__("hashlib")
        .sha256(r["continuation_id"].encode())
        .hexdigest(),
    )
    if len(pilot_rows) != N_PILOT_RESPONSES:
        raise SystemExit("pilot row count mismatch")
    if any(r["grader_split"] != "development" for r in pilot_rows):
        raise SystemExit("pilot leaked locked prompts")

    pilot_label_counts = Counter(r["reference_label"] for r in pilot_rows)
    pilot_prompt_n = len({r["prompt_id"] for r in pilot_rows})

    grader_prompt_sha = sha256_text(
        GRADER_SYSTEM + "\n" + GRADER_USER_TEMPLATE + "\n" + APOLLO_RUBRIC_BODY
    )

    PROC.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)

    write_json(PROC / "grader_prompt_split.json", split)
    with (PROC / "reference_corpus.jsonl").open("w", encoding="utf-8") as f:
        for r in corpus:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    with (PROC / "pilot_200.jsonl").open("w", encoding="utf-8") as f:
        for r in pilot_rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    write_json(
        PROC / "pilot_continuation_ids.json",
        {
            "n": len(pilot_cids),
            "continuation_ids": pilot_cids,
            "pilot_ids_sha256": _sha_json(pilot_cids),
            "reference_label_counts": dict(pilot_label_counts),
            "n_distinct_prompts": pilot_prompt_n,
        },
    )

    design = {
        "phase": "phase23",
        "status": STATUS_DESIGN,
        "created_at": utc_now_iso(),
        "git_commit": subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "sources": {
            "phase21_onset_run_id": PHASE21_ONSET,
            "phase22b_onset_run_id": PHASE22B_ONSET,
            "reference_grader": REFERENCE_GRADER,
            "n_responses": len(corpus),
            "n_prompts": N_PROMPTS,
        },
        "candidates": CANDIDATES,
        "inference": INFERENCE,
        "grader_prompt": {
            "system": GRADER_SYSTEM,
            "user_template_sha256": sha256_text(GRADER_USER_TEMPLATE),
            "apollo_rubric_body_sha256": sha256_text(APOLLO_RUBRIC_BODY),
            "combined_prompt_sha256": grader_prompt_sha,
            "apollo_grading_template_content_sha256": GRADING_CONTENT_SHA256,
            "output_schema": ["score", "class", "explanation"],
            "note": (
                "JSON output wrapper over Apollo honesty scale; examples omitted "
                "for short capped outputs; scale+instructions unchanged."
            ),
        },
        "grader_prompt_split_sha256": split["grader_prompt_split_sha256"],
        "n_development_prompts": N_DEV_PROMPTS,
        "n_locked_validation_prompts": N_LOCKED_PROMPTS,
        "pilot": {
            "n_responses": N_PILOT_RESPONSES,
            "pilot_ids_sha256": _sha_json(pilot_cids),
            "reference_label_counts": dict(pilot_label_counts),
            "n_distinct_prompts": pilot_prompt_n,
            "selection": "sha256_stratified_by_reference_label_within_development",
        },
        "thresholds_frozen_before_evaluation": {
            "pilot_elimination": PILOT_ELIMINATION,
            "development_eligibility": DEV_ELIGIBILITY,
            "locked_acceptance": LOCKED_ACCEPTANCE,
            "onset_acceptance": ONSET_ACCEPTANCE,
        },
        "authorizations": {
            "modal_gpu_open_grader_inference_authorized": True,
            "mistral_roleplay_generation_authorized": False,
            "openai_grading_api_authorized": False,
            "openai_onset_api_authorized": False,
            "activation_extraction_authorized": False,
            "probe_fitting_authorized": False,
            "physiology_authorized": False,
            "k_gt_20_generation_authorized": False,
            "prompt_changes_authorized": False,
            "population_threshold_changes_authorized": False,
            "stage2_development_authorized": False,
            "stage3_locked_validation_authorized": False,
            "stage4_onset_validation_authorized": False,
        },
        "guarantee": GUARANTEE,
    }
    design["design_sha256"] = _sha_json(
        {
            k: design[k]
            for k in (
                "candidates",
                "inference",
                "grader_prompt",
                "grader_prompt_split_sha256",
                "pilot",
                "thresholds_frozen_before_evaluation",
            )
        }
    )
    write_json(PROC / "design_matrix.json", design)
    write_json(ART / "design_matrix.json", design)
    write_json(ART / "grader_prompt_split.json", split)
    write_json(ART / "pilot_continuation_ids.json", {
        "n": len(pilot_cids),
        "continuation_ids": pilot_cids,
        "pilot_ids_sha256": _sha_json(pilot_cids),
    })

    print(
        json.dumps(
            {
                "status": STATUS_DESIGN,
                "design_sha256": design["design_sha256"],
                "grader_prompt_split_sha256": split["grader_prompt_split_sha256"],
                "pilot_ids_sha256": _sha_json(pilot_cids),
                "pilot_label_counts": dict(pilot_label_counts),
                "n_pilot_prompts": pilot_prompt_n,
                "candidates": {
                    k: {"hf_id": v["hf_id"], "revision": v["revision"], "gpu": v["gpu"]}
                    for k, v in CANDIDATES.items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
