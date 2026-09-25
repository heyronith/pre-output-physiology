#!/usr/bin/env python3
"""Audit Phase 3 prompt/response token boundary across all examples.

Local only. No Modal. No activations. No predictive fitting.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.onset import (  # noqa: E402
    annotate_onset,
    onset_coverage_summary,
)
from pre_output_physiology.provenance import (  # noqa: E402
    sha256_text,
    utc_now_iso,
    write_json,
    write_jsonl,
)
from pre_output_physiology.trajectory import (  # noqa: E402
    analyze_prompt_response_boundary,
)

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
SPLIT_FILES = (
    ("phase3_train", "phase3_train_metadata.jsonl"),
    ("phase3_validation", "phase3_validation_metadata.jsonl"),
    ("locked_test", "locked_test_metadata.jsonl"),
)


def _length_stats(values: list[int]) -> dict:
    if not values:
        return {"min": None, "median": None, "max": None}
    return {
        "min": int(min(values)),
        "median": float(statistics.median(values)),
        "max": int(max(values)),
    }


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=str(REPO_ROOT / "data/processed/phase3_roleplay"),
    )
    parser.add_argument(
        "--out",
        default=str(
            REPO_ROOT / "artifacts/phase3a_summaries/token_boundary_audit.json"
        ),
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Capture prior onset coverage for comparison
    prior_summary_path = (
        REPO_ROOT / "artifacts/phase3a_summaries/phase3a_prepare_summary.json"
    )
    prior_onset = None
    if prior_summary_path.is_file():
        prior_onset = json.loads(prior_summary_path.read_text(encoding="utf-8")).get(
            "onset_coverage"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, use_fast=True
    )

    split_reports: dict[str, dict] = {}
    all_anns = []
    failure_records: list[dict] = []
    global_prompt_lens: list[int] = []
    global_resp_lens: list[int] = []
    onset_token_changed = 0
    onset_token_compared = 0

    for split_name, filename in SPLIT_FILES:
        path = data_dir / filename
        if not path.is_file():
            raise SystemExit(
                f"Missing {path}. Run scripts/prepare_phase3.py first."
            )
        rows = _load_jsonl(path)
        prefix_pass = 0
        prefix_fail = 0
        straddle = 0
        standalone_match = 0
        standalone_mismatch = 0
        prompt_lens: list[int] = []
        resp_lens: list[int] = []
        anns = []

        for row in rows:
            prompt = row["input_formatted"]
            response = row["model_outputs"]
            tok = analyze_prompt_response_boundary(prompt, response, tokenizer)
            prompt_lens.append(len(tok.prompt_ids))
            resp_lens.append(len(tok.response_suffix_ids))
            global_prompt_lens.append(len(tok.prompt_ids))
            global_resp_lens.append(len(tok.response_suffix_ids))

            if tok.prompt_prefix_exact:
                prefix_pass += 1
            else:
                prefix_fail += 1
                failure_records.append(
                    {
                        "example_id": row["example_id"],
                        "split": split_name,
                        "prompt_sha256": row.get("prompt_sha256")
                        or sha256_text(prompt),
                        "model_output_sha256": sha256_text(response),
                        "failure": "prompt_prefix_mismatch",
                        "n_prompt_ids": len(tok.prompt_ids),
                        "n_full_ids": len(tok.full_ids),
                        "response_start_token_index": tok.response_start_token_index,
                    }
                )

            if tok.boundary_token_straddle:
                straddle += 1
                failure_records.append(
                    {
                        "example_id": row["example_id"],
                        "split": split_name,
                        "prompt_sha256": row.get("prompt_sha256")
                        or sha256_text(prompt),
                        "model_output_sha256": sha256_text(response),
                        "failure": "boundary_token_straddle",
                        "straddling_token_index": tok.straddling_token_index,
                        "boundary_char": tok.boundary_char,
                    }
                )

            if tok.standalone_equals_suffix:
                standalone_match += 1
            else:
                standalone_mismatch += 1

            prev_tok = row.get("conservative_deceptive_sentence_onset_token")
            ann = annotate_onset(
                example_id=row["example_id"],
                split=split_name,
                prompt_sha256=row.get("prompt_sha256") or sha256_text(prompt),
                model_output=response,
                binary_label=int(row["eventual_deception"]),
                scale_label_explanation=row.get("scale_label_explanation"),
                label_explanation=row.get("label_explanation") or None,
                tokenizer=tokenizer,
                input_formatted=prompt,
            )
            anns.append(ann)
            if (
                ann.onset_status == "high_confidence"
                and prev_tok is not None
                and ann.conservative_deceptive_sentence_onset_token is not None
            ):
                onset_token_compared += 1
                if prev_tok != ann.conservative_deceptive_sentence_onset_token:
                    onset_token_changed += 1

            row["onset_status"] = ann.onset_status
            row["conservative_deceptive_sentence_onset_char"] = (
                ann.conservative_deceptive_sentence_onset_char
            )
            row["conservative_deceptive_sentence_onset_token"] = (
                ann.conservative_deceptive_sentence_onset_token
            )
            row["evidence_span_sha256"] = ann.evidence_span_sha256
            row["canonical_response_n_tokens"] = len(tok.response_suffix_ids)
            row["standalone_equals_suffix"] = tok.standalone_equals_suffix
            row["prompt_prefix_exact"] = tok.prompt_prefix_exact
            row["boundary_token_straddle"] = tok.boundary_token_straddle

        write_jsonl(path, rows)
        all_anns.extend(anns)
        split_reports[split_name] = {
            "n_rows": len(rows),
            "prompt_prefix_pass": prefix_pass,
            "prompt_prefix_fail": prefix_fail,
            "boundary_straddle": straddle,
            "standalone_equals_suffix": standalone_match,
            "standalone_vs_suffix_mismatch": standalone_mismatch,
            "prompt_token_length": _length_stats(prompt_lens),
            "response_token_length": _length_stats(resp_lens),
            "onset_coverage": onset_coverage_summary(anns),
        }

    # Regenerate onset annotations + commit-safe audit index
    write_jsonl(
        data_dir / "onset_annotations.jsonl",
        [a.to_dict() for a in all_anns],
    )

    import numpy as np

    rng = np.random.default_rng(42)

    def sample_anns(pool, n):
        pool = list(pool)
        if len(pool) <= n:
            return pool
        idx = rng.choice(len(pool), size=n, replace=False)
        return [pool[i] for i in sorted(idx.tolist())]

    by_id = {}
    for _split_name, filename in SPLIT_FILES:
        for row in _load_jsonl(data_dir / filename):
            by_id[row["example_id"]] = row

    hi_tv = [
        a
        for a in all_anns
        if a.onset_status == "high_confidence"
        and a.split in {"phase3_train", "phase3_validation"}
    ]
    hi_te = [
        a for a in all_anns if a.onset_status == "high_confidence" and a.split == "locked_test"
    ]
    unresolved = [a for a in all_anns if a.onset_status == "unresolved"]
    audit_anns = sample_anns(hi_tv, 30) + sample_anns(hi_te, 20) + sample_anns(
        unresolved, 20
    )
    audit_raw_dir = data_dir / "onset_audit_raw"
    audit_raw_dir.mkdir(parents=True, exist_ok=True)
    audit_raw = []
    audit_index = []
    for a in audit_anns:
        row = by_id[a.example_id]
        audit_raw.append(
            {
                "example_id": a.example_id,
                "split": a.split,
                "prompt_sha256": a.prompt_sha256,
                "model_outputs": row["model_outputs"],
                "scale_label_explanation": row.get("scale_label_explanation"),
                "evidence_span": a.evidence_span,
                "conservative_deceptive_sentence_onset_char": (
                    a.conservative_deceptive_sentence_onset_char
                ),
                "conservative_deceptive_sentence_onset_token": (
                    a.conservative_deceptive_sentence_onset_token
                ),
                "onset_status": a.onset_status,
            }
        )
        audit_index.append(
            {
                "example_id": a.example_id,
                "split": a.split,
                "prompt_sha256": a.prompt_sha256,
                "onset_status": a.onset_status,
                "evidence_span_sha256": a.evidence_span_sha256,
                "onset_char": a.conservative_deceptive_sentence_onset_char,
                "onset_token": a.conservative_deceptive_sentence_onset_token,
                "model_output_sha256": sha256_text(row["model_outputs"]),
                "explanation_sha256": sha256_text(
                    str(row.get("scale_label_explanation") or "")
                ),
            }
        )
    write_jsonl(audit_raw_dir / "onset_audit_sample.jsonl", audit_raw)
    write_jsonl(
        REPO_ROOT / "artifacts/phase3a_summaries/onset_audit_index.jsonl",
        audit_index,
    )

    onset_coverage = {k: v["onset_coverage"] for k, v in split_reports.items()}
    total_prefix_fail = sum(v["prompt_prefix_fail"] for v in split_reports.values())
    total_straddle = sum(v["boundary_straddle"] for v in split_reports.values())
    verdict = (
        "TOKEN BOUNDARY PASS — PHASE 3B A/B ELIGIBLE"
        if total_prefix_fail == 0 and total_straddle == 0
        else "TOKEN BOUNDARY HOLD"
    )

    audit = {
        "created_at": utc_now_iso(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "add_special_tokens": False,
        "canonical_response_definition": (
            "suffix of tokenizer(input_formatted + model_outputs) after "
            "verifying exact prompt-id prefix and no boundary-straddling token"
        ),
        "splits": {
            k: {kk: vv for kk, vv in v.items() if kk != "onset_coverage"}
            for k, v in split_reports.items()
        },
        "global": {
            "n_rows": sum(v["n_rows"] for v in split_reports.values()),
            "prompt_prefix_pass": sum(
                v["prompt_prefix_pass"] for v in split_reports.values()
            ),
            "prompt_prefix_fail": total_prefix_fail,
            "boundary_straddle": total_straddle,
            "standalone_equals_suffix": sum(
                v["standalone_equals_suffix"] for v in split_reports.values()
            ),
            "standalone_vs_suffix_mismatch": sum(
                v["standalone_vs_suffix_mismatch"] for v in split_reports.values()
            ),
            "prompt_token_length": _length_stats(global_prompt_lens),
            "response_token_length": _length_stats(global_resp_lens),
        },
        "failures": failure_records[:200],
        "n_failures_recorded": len(failure_records),
        "onset_token_realignment": {
            "high_confidence_compared_to_prior_metadata": onset_token_compared,
            "onset_token_index_changed": onset_token_changed,
            "prior_onset_coverage": prior_onset,
            "updated_onset_coverage": onset_coverage,
        },
        "verdict": verdict,
        "activations_collected": False,
        "gpu_jobs_run": False,
    }
    write_json(out_path, audit)

    # Update prepare summary onset coverage
    if prior_summary_path.is_file():
        summary = json.loads(prior_summary_path.read_text(encoding="utf-8"))
        summary["onset_coverage"] = onset_coverage
        summary["token_boundary_audit"] = {
            "path": str(out_path.relative_to(REPO_ROOT)),
            "verdict": verdict,
            "prompt_prefix_fail": total_prefix_fail,
            "boundary_straddle": total_straddle,
            "standalone_vs_suffix_mismatch": audit["global"][
                "standalone_vs_suffix_mismatch"
            ],
        }
        write_json(prior_summary_path, summary)

    print(
        json.dumps(
            {
                "verdict": verdict,
                "global": audit["global"],
                "onset_token_index_changed": onset_token_changed,
                "out": str(out_path),
            },
            indent=2,
        )
    )
    return 0 if verdict.startswith("TOKEN BOUNDARY PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
