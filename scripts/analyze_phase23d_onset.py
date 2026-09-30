#!/usr/bin/env python3
"""Analyze Phase-23D onset validation against frozen GPT-4o onset annotations.

IMPORTANT (pre-authorization note — do NOT treat current proxy as accepted):
The token-error estimate that uses ``abs(char_offset_difference) / 4`` is a
character-heuristic proxy, not an actual token-index error. Before Stage 23D is
ever authorized, token error must be computed with the exact frozen tokenizer
and true token positions. This file is not executable under current
authorizations; Stage 23D remains unauthorized.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase23_open_grader import (  # noqa: E402
    GUARANTEE,
    ONSET_ACCEPTANCE,
    STATUS_23D_FAIL,
    STATUS_23D_PASS,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()

    run_dir = REPO_ROOT / "artifacts/runs" / args.run_id
    man = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(x)
        for x in (run_dir / "onset_judgments.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]

    n = len(rows)
    explicit_agree = 0
    quote_exact = 0
    offset_within = 0
    token_errs: list[float] = []
    major: list[dict] = []

    for r in rows:
        ref_ex = bool(r.get("reference_explicit_span"))
        open_ex = bool(r.get("open_explicit_span"))
        if ref_ex == open_ex:
            explicit_agree += 1
        ref_q = (r.get("reference_onset_quote") or "").strip()
        open_q = (r.get("open_onset_quote") or "").strip()
        if ref_q and open_q and ref_q == open_q:
            quote_exact += 1
        rs = r.get("reference_onset_char_start")
        os_ = r.get("open_onset_char_start")
        if rs is not None and os_ is not None:
            if abs(int(rs) - int(os_)) <= ONSET_ACCEPTANCE["char_offset_tolerance"]:
                offset_within += 1
        # token onset: recompute if both quotes map; else use reference token vs None
        rt = r.get("reference_onset_token_index")
        # Without tokenizer here, approximate char->token via opening offset
        # ratio if both spans exist
        if rt is not None and os_ is not None and rs is not None:
            # Use char-offset delta as proxy token error estimate when token
            # indices unavailable. NOTE: abs(delta)/4 is NOT real token error;
            # fix before Stage 23D authorization.
            # Prefer exact: if quotes equal, token error 0
            if ref_q == open_q:
                token_errs.append(0.0)
            else:
                token_errs.append(abs(int(rs) - int(os_)) / 4.0)  # ~4 chars/token heuristic
        if ref_ex != open_ex or (ref_q and open_q and ref_q != open_q):
            major.append(
                {
                    "continuation_id": r["continuation_id"],
                    "reference_explicit_span": ref_ex,
                    "open_explicit_span": open_ex,
                    "reference_quote": ref_q[:200],
                    "open_quote": open_q[:200],
                }
            )

    explicit_rate = explicit_agree / n if n else float("nan")
    quote_rate = quote_exact / max(1, sum(1 for r in rows if r.get("reference_onset_quote")))
    med_tok = statistics.median(token_errs) if token_errs else float("nan")

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
    summary = {
        "created_at": utc_now_iso(),
        "git_commit": subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "run_id": args.run_id,
        "candidate": man["candidate"],
        "status": status,
        "n": n,
        "explicit_span_agreement": explicit_rate,
        "exact_quote_match_rate": quote_rate,
        "offset_within_tolerance_rate": offset_within / n if n else float("nan"),
        "median_abs_token_error_proxy": med_tok,
        "n_token_error_samples": len(token_errs),
        "thresholds": ONSET_ACCEPTANCE,
        "failure_reasons": reasons,
        "major_disagreements_sample": major[:25],
        "cost": man.get("cost"),
        "guarantee": GUARANTEE,
        "authorizations": {
            "k_gt_20_generation_authorized": False,
            "open_grader_for_k60_authorized": status == STATUS_23D_PASS,
        },
    }
    out = REPO_ROOT / "artifacts/phase23d_onset"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "summary.json", summary)
    report = REPO_ROOT / "reports/phase23d_onset.md"
    report.write_text(
        "\n".join(
            [
                "# Phase 23D — onset annotation validation",
                "",
                f"**Status:** `{status}`",
                "",
                f"- Explicit-span agreement: {explicit_rate:.4f}",
                f"- Exact quote match rate: {quote_rate:.4f}",
                f"- Median |token error| (proxy): {med_tok}",
                f"- Reasons: {reasons or 'none'}",
                "",
                "## Guarantee",
                "",
                GUARANTEE,
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "report": str(report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
