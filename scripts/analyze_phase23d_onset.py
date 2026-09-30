#!/usr/bin/env python3
"""Analyze Phase-23D onset validation against frozen GPT-4o onset annotations.

Token error uses the frozen Mistral response-token indexing rule
(``char_onset_to_response_token_index`` / Phase 21–22B). The deprecated
``abs(char_diff) / 4`` proxy is never used as the primary metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase23_open_grader import (  # noqa: E402
    GUARANTEE,
    ONSET_ACCEPTANCE,
    ONSET_TOKENIZER_HF_ID,
    ONSET_TOKENIZER_REVISION,
    STATUS_23D_FAIL,
    STATUS_23D_PASS,
    absolute_token_index_error,
    load_frozen_onset_tokenizer,
    response_char_to_onset_token_index,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

OUT = REPO_ROOT / "artifacts/phase23d_onset"
REPORT = REPO_ROOT / "reports/phase23d_onset.md"
CFG = REPO_ROOT / "configs/experiments/phase23_open_grader_validation.yaml"
SAMPLE_MANIFEST = REPO_ROOT / "artifacts/phase23_onset_sample/manifest.json"
EXPECTED_SAMPLE_SHA = "5a439094f23cbb22ec121a7f2c074466a112a77bac737f36ce7dc6e7ddd357ca"
EXPECTED_WINNER = "gemma4_31b_it"
EXPECTED_REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"

AUTH_FALSE = {
    "modal_gpu_open_grader_inference_authorized": False,
    "stage2_development_authorized": False,
    "stage3_locked_validation_authorized": False,
    "stage4_onset_validation_authorized": False,
    "mistral_roleplay_generation_authorized": False,
    "openai_grading_api_authorized": False,
    "openai_onset_api_authorized": False,
    "activation_extraction_authorized": False,
    "probe_fitting_authorized": False,
    "physiology_authorized": False,
    "k_gt_20_generation_authorized": False,
    "prompt_changes_authorized": False,
    "population_threshold_changes_authorized": False,
    "open_grader_for_k60_authorized": False,
}


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(x)
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--authorization-commit", required=True)
    ap.add_argument("--metric-repair-commit", required=True)
    args = ap.parse_args()

    sample_man = json.loads(SAMPLE_MANIFEST.read_text(encoding="utf-8"))
    if sample_man.get("onset_sample_sha256") != EXPECTED_SAMPLE_SHA:
        raise SystemExit("STOP: frozen onset sample SHA mismatch")
    if int(sample_man.get("n", -1)) != 80:
        raise SystemExit("STOP: onset sample n != 80")

    run_dir = REPO_ROOT / "artifacts/runs" / args.run_id
    man = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    judg_path = run_dir / "onset_judgments.jsonl"
    rows = _load_jsonl(judg_path)
    judgments_sha = _sha_file(judg_path)

    if man.get("candidate") != EXPECTED_WINNER:
        raise SystemExit(f"STOP: candidate != {EXPECTED_WINNER}")
    if man.get("revision") != EXPECTED_REVISION:
        raise SystemExit("STOP: revision mismatch")
    if len(rows) != 80:
        raise SystemExit(f"STOP: n judgments {len(rows)} != 80")

    # Enrich sample rows with answer_prefix for token mapping
    sample_by = {
        r["continuation_id"]: r
        for r in _load_jsonl(
            REPO_ROOT / "artifacts/phase23_onset_sample/onset_validation.jsonl"
        )
    }

    tokenizer = load_frozen_onset_tokenizer()
    n = len(rows)
    n_invalid = 0
    explicit_agree = 0
    quote_exact = 0
    n_ref_quote = 0
    offset_within = 0
    n_both_char = 0
    char_abs_errs: list[int] = []
    token_abs_errs: list[int] = []
    token_detail: list[dict] = []
    major: list[dict] = []

    for r in rows:
        sid = r["continuation_id"]
        s = sample_by.get(sid) or {}
        answer_prefix = s.get("answer_prefix") or r.get("answer_prefix") or ""
        full_response = s.get("full_response") or r.get("full_response") or ""

        # Invalid open JSON / unusable annotation
        open_ex_raw = r.get("open_explicit_span")
        if open_ex_raw is None and not (r.get("open_onset_quote") or "").strip():
            # treat missing span fields with empty quote as valid false-span if raw present
            if not (r.get("open_raw") or "").strip():
                n_invalid += 1

        ref_ex = bool(r.get("reference_explicit_span"))
        open_ex = bool(open_ex_raw)
        if ref_ex == open_ex:
            explicit_agree += 1

        ref_q = (r.get("reference_onset_quote") or "").strip()
        open_q = (r.get("open_onset_quote") or "").strip()
        if ref_q:
            n_ref_quote += 1
            if open_q and ref_q == open_q:
                quote_exact += 1

        rs = r.get("reference_onset_char_start")
        os_ = r.get("open_onset_char_start")
        if rs is not None and os_ is not None:
            n_both_char += 1
            char_err = abs(int(rs) - int(os_))
            char_abs_errs.append(char_err)
            if char_err <= ONSET_ACCEPTANCE["char_offset_tolerance"]:
                offset_within += 1

        # True token indices via frozen Mistral rule
        ref_tok = r.get("reference_onset_token_index")
        if ref_tok is None and rs is not None and full_response:
            ref_tok = response_char_to_onset_token_index(
                tokenizer,
                answer_prefix=answer_prefix,
                full_response=full_response,
                char_start=int(rs),
            )
        pred_tok = None
        if os_ is not None and full_response:
            pred_tok = response_char_to_onset_token_index(
                tokenizer,
                answer_prefix=answer_prefix,
                full_response=full_response,
                char_start=int(os_),
            )
        # Persist computed predicted token index on the analysis row copy
        tok_err = absolute_token_index_error(
            int(ref_tok) if ref_tok is not None else None,
            pred_tok,
        )
        if tok_err is not None:
            token_abs_errs.append(int(tok_err))
            token_detail.append(
                {
                    "continuation_id": sid,
                    "reference_char_start": rs,
                    "predicted_char_start": os_,
                    "reference_token_index": int(ref_tok) if ref_tok is not None else None,
                    "predicted_token_index": pred_tok,
                    "signed_token_error": (
                        int(pred_tok) - int(ref_tok)
                        if ref_tok is not None and pred_tok is not None
                        else None
                    ),
                    "absolute_token_error": int(tok_err),
                }
            )

        if ref_ex != open_ex or (ref_q and open_q and ref_q != open_q):
            major.append(
                {
                    "continuation_id": sid,
                    "reference_explicit_span": ref_ex,
                    "open_explicit_span": open_ex,
                    "reference_quote": ref_q[:200],
                    "open_quote": open_q[:200],
                    "reference_token_index": ref_tok,
                    "predicted_token_index": pred_tok,
                    "absolute_token_error": tok_err,
                }
            )

    explicit_rate = explicit_agree / n if n else float("nan")
    quote_rate = quote_exact / max(1, n_ref_quote)
    med_tok = (
        float(statistics.median(token_abs_errs)) if token_abs_errs else float("nan")
    )
    mean_tok = (
        float(statistics.mean(token_abs_errs)) if token_abs_errs else float("nan")
    )
    med_char = (
        float(statistics.median(char_abs_errs)) if char_abs_errs else float("nan")
    )
    dist = {
        "eq_0": sum(1 for e in token_abs_errs if e == 0),
        "le_1": sum(1 for e in token_abs_errs if e <= 1),
        "le_2": sum(1 for e in token_abs_errs if e <= 2),
        "gt_2": sum(1 for e in token_abs_errs if e > 2),
        "n": len(token_abs_errs),
    }

    reasons = []
    if explicit_rate < ONSET_ACCEPTANCE["min_explicit_span_agreement"]:
        reasons.append(
            f"explicit_span_agreement {explicit_rate:.4f} < "
            f"{ONSET_ACCEPTANCE['min_explicit_span_agreement']}"
        )
    if quote_rate < ONSET_ACCEPTANCE["min_exact_quote_match_rate"]:
        reasons.append(
            f"exact_quote_match {quote_rate:.4f} < "
            f"{ONSET_ACCEPTANCE['min_exact_quote_match_rate']}"
        )
    if med_tok > ONSET_ACCEPTANCE["max_median_abs_token_error"]:
        reasons.append(
            f"median_abs_token_error {med_tok:.4f} > "
            f"{ONSET_ACCEPTANCE['max_median_abs_token_error']}"
        )

    status = STATUS_23D_PASS if not reasons else STATUS_23D_FAIL
    git_commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()

    gates = [
        {
            "metric": "explicit_span_agreement",
            "threshold": ONSET_ACCEPTANCE["min_explicit_span_agreement"],
            "observed": explicit_rate,
            "pass": explicit_rate >= ONSET_ACCEPTANCE["min_explicit_span_agreement"],
        },
        {
            "metric": "exact_quote_match",
            "threshold": ONSET_ACCEPTANCE["min_exact_quote_match_rate"],
            "observed": quote_rate,
            "pass": quote_rate >= ONSET_ACCEPTANCE["min_exact_quote_match_rate"],
        },
        {
            "metric": "median_abs_token_error",
            "threshold": ONSET_ACCEPTANCE["max_median_abs_token_error"],
            "observed": med_tok,
            "pass": (
                (not token_abs_errs)
                or med_tok <= ONSET_ACCEPTANCE["max_median_abs_token_error"]
            ),
        },
    ]

    summary = {
        "created_at": utc_now_iso(),
        "status": status,
        "authorization_commit": args.authorization_commit,
        "metric_repair_commit": args.metric_repair_commit,
        "analysis_git_commit": git_commit,
        "inference_git_commit": man.get("git_commit"),
        "run_id": args.run_id,
        "candidate": EXPECTED_WINNER,
        "hf_id": man.get("hf_id"),
        "revision": man.get("revision"),
        "onset_sample_sha256": EXPECTED_SAMPLE_SHA,
        "judgments_sha256": judgments_sha,
        "tokenizer": {
            "hf_id": ONSET_TOKENIZER_HF_ID,
            "revision": ONSET_TOKENIZER_REVISION,
            "rule": (
                "Phase-21 char_onset_to_response_token_index: "
                "len(encode(full_response[:char_start], add_special_tokens=False)) "
                "with answer_prefix+completion coordinate system"
            ),
            "deprecated_proxy_not_used": "abs(char_diff)/4",
        },
        "n": n,
        "n_valid": n - n_invalid,
        "n_invalid": n_invalid,
        "invalid_rate": n_invalid / n if n else float("nan"),
        "explicit_span_agreement": explicit_rate,
        "exact_quote_match_rate": quote_rate,
        "n_reference_quotes": n_ref_quote,
        "char_offset_tolerance": ONSET_ACCEPTANCE["char_offset_tolerance"],
        "offset_within_tolerance_rate": offset_within / n if n else float("nan"),
        "n_both_char_starts": n_both_char,
        "median_abs_char_start_error": med_char,
        "median_abs_token_error": med_tok,
        "mean_abs_token_error": mean_tok,
        "token_error_distribution": dist,
        "n_token_error_samples": len(token_abs_errs),
        "gates": gates,
        "thresholds": ONSET_ACCEPTANCE,
        "failure_reasons": reasons,
        "token_error_details": token_detail,
        "major_disagreements_sample": sorted(
            major,
            key=lambda x: (
                -(x.get("absolute_token_error") or -1),
                x["continuation_id"],
            ),
        )[:25],
        "cost": man.get("cost"),
        "guarantee": GUARANTEE,
        "authorizations_after_freeze": dict(AUTH_FALSE),
        "open_grader_for_k60_authorized": False,
        "k_gt_20_generation_authorized": False,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "summary.json", summary)
    write_json(
        OUT / "freeze.json",
        {
            "status": status,
            "authorization_commit": args.authorization_commit,
            "metric_repair_commit": args.metric_repair_commit,
            "analysis_git_commit": git_commit,
            "inference_git_commit": man.get("git_commit"),
            "run_id": args.run_id,
            "candidate": EXPECTED_WINNER,
            "revision": EXPECTED_REVISION,
            "onset_sample_sha256": EXPECTED_SAMPLE_SHA,
            "judgments_sha256": judgments_sha,
            "tokenizer_hf_id": ONSET_TOKENIZER_HF_ID,
            "tokenizer_revision": ONSET_TOKENIZER_REVISION,
            "gates": gates,
            "explicit_span_agreement": explicit_rate,
            "exact_quote_match_rate": quote_rate,
            "median_abs_token_error": med_tok,
            "offset_within_tolerance_rate": offset_within / n if n else float("nan"),
            "token_error_distribution": dist,
            "authorizations_after_freeze": dict(AUTH_FALSE),
            "open_grader_for_k60_authorized": False,
            "k_gt_20_generation_authorized": False,
        },
    )

    # Consume Stage-23D authorization in config
    text = CFG.read_text(encoding="utf-8")
    text = re.sub(r"^status:.*$", f"status: {status}", text, count=1, flags=re.M)
    for key in (
        "modal_gpu_open_grader_inference_authorized",
        "stage4_onset_validation_authorized",
        "stage3_locked_validation_authorized",
        "k_gt_20_generation_authorized",
    ):
        text = re.sub(rf"^(\s*{key}:\s*).*$", r"\1false", text, flags=re.M)
    block = (
        "phase23d_onset:\n"
        "  freeze_artifact: artifacts/phase23d_onset/freeze.json\n"
        f"  status: {status}\n"
        f"  run_id: {args.run_id}\n"
        f"  candidate: {EXPECTED_WINNER}\n"
        f"  judgments_sha256: {judgments_sha}\n"
        f"  onset_sample_sha256: {EXPECTED_SAMPLE_SHA}\n"
        f"  tokenizer_hf_id: {ONSET_TOKENIZER_HF_ID}\n"
        f"  tokenizer_revision: {ONSET_TOKENIZER_REVISION}\n"
        f"  authorization_commit: {args.authorization_commit}\n"
        f"  metric_repair_commit: {args.metric_repair_commit}\n"
        "  open_grader_for_k60_authorized: false\n"
        "  k_gt_20_generation_authorized: false\n"
        "\n"
    )
    if "phase23d_onset:" in text:
        text = re.sub(
            r"phase23d_onset:.*?(\nauthorizations:)",
            block + r"\1",
            text,
            count=1,
            flags=re.S,
        )
    else:
        text = text.replace("\nauthorizations:", "\n" + block + "authorizations:", 1)
    text = re.sub(
        r"^notes: >$.*",
        "notes: >\n"
        f"  Stage 23D complete ({status}). Authorization consumed. "
        "K>20 / physiology remain unauthorized.\n",
        text,
        count=1,
        flags=re.M | re.S,
    )
    CFG.write_text(text, encoding="utf-8")

    lines = [
        "# Phase 23D — onset annotation validation",
        "",
        f"**Status:** `{status}`",
        "",
        f"**Decision:** `{'PASS' if not reasons else 'FAIL'}`",
        "",
        "## Provenance",
        "",
        f"- Metric-repair commit: `{args.metric_repair_commit}`",
        f"- Authorization commit: `{args.authorization_commit}`",
        f"- Inference commit: `{man.get('git_commit')}`",
        f"- Analysis commit: `{git_commit}`",
        f"- Run ID: `{args.run_id}`",
        f"- Frozen sample SHA: `{EXPECTED_SAMPLE_SHA}`",
        f"- Candidate: `{EXPECTED_WINNER}` @ `{EXPECTED_REVISION}`",
        f"- Tokenizer: `{ONSET_TOKENIZER_HF_ID}` @ `{ONSET_TOKENIZER_REVISION}`",
        f"- Judgments SHA256: `{judgments_sha}`",
        f"- n: {n} (invalid {n_invalid})",
        f"- Cost USD: {man.get('cost', {}).get('estimated_cost_usd')}",
        f"- Wall seconds: {man.get('cost', {}).get('wall_seconds')}",
        "",
        "## Metric repair",
        "",
        "The former `abs(char_offset_difference) / 4` proxy is **not** used.",
        "",
        "True token index uses the frozen Phase-21/22B rule "
        "`char_onset_to_response_token_index` with the pinned Mistral tokenizer:",
        "",
        "1. `completion = full_response[len(answer_prefix):]`",
        "2. `token_index = len(encode(full_response[:char_start], "
        "add_special_tokens=False))`",
        "3. `token_error = abs(predicted_token_index - reference_token_index)` "
        "(integer token distance)",
        "",
        "No whitespace/Unicode normalization or text rewriting.",
        "",
        "## Frozen results",
        "",
        "| Metric | Frozen threshold | Observed | Pass |",
        "|---|---:|---:|---|",
    ]
    for g in gates:
        thr = g["threshold"]
        obs = g["observed"]
        lines.append(
            f"| {g['metric']} | {thr} | {obs:.6f} | "
            f"{'PASS' if g['pass'] else 'FAIL'} |"
        )
    lines += [
        "",
        "### Diagnostics (not rejection gates)",
        "",
        f"- 8-character offset agreement rate: "
        f"{offset_within / n if n else float('nan'):.6f}",
        f"- Median |char start error|: {med_char}",
        f"- Mean |token error|: {mean_tok}",
        f"- Token-error distribution: {dist}",
        "",
        "## Decision",
        "",
        f"`{status}`",
        "",
        (
            "Gemma onset localization validated under the frozen contract. "
            "K>20 / K=60 sampling remains a separate unauthorized PI decision."
            if not reasons
            else "Gemma onset validation failed under the frozen contract. STOP. "
            "No prompt/sample/threshold/tokenizer retuning."
        ),
        "",
        "## Authorization state (consumed)",
        "",
    ]
    for k, v in AUTH_FALSE.items():
        lines.append(f"- `{k}`: **{str(v).lower()}**")
    lines += [
        "",
        "## Guarantee",
        "",
        GUARANTEE,
        "",
        "STAGE 23D USED THE FROZEN 80-RESPONSE ONSET SAMPLE AND FROZEN "
        "GEMMA-4-31B-IT CONFIGURATION. TOKEN ERROR WAS COMPUTED FROM ACTUAL "
        "TOKENIZER TOKEN INDICES, NOT A CHARACTER-DISTANCE PROXY. NO OPENAI "
        "API CALLS, MISTRAL GENERATION, K>20 GENERATION, ACTIVATION EXTRACTION, "
        "PROBE FITTING, PHYSIOLOGY, PROMPT TUNING, THRESHOLD TUNING, SAMPLE "
        "CHANGES, OR POST-HOC MODEL SELECTION WAS PERFORMED.",
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": status,
                "passed": not bool(reasons),
                "median_abs_token_error": med_tok,
                "judgments_sha256": judgments_sha,
                "report": str(REPORT.relative_to(REPO_ROOT)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
