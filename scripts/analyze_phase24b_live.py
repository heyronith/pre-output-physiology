#!/usr/bin/env python3
"""Write Phase-24B report + D159 from frozen summary."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase24b_live import GUARANTEE  # noqa: E402

OUT = REPO_ROOT / "artifacts/phase24b_live"
REPORT = REPO_ROOT / "reports/phase24b_live_capture_feasibility.md"
DECISION_LOG = REPO_ROOT / "docs/decision_log.md"


def main() -> int:
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    pilot = json.loads((OUT / "pilot_manifest.json").read_text(encoding="utf-8"))
    contract = json.loads((OUT / "contract.json").read_text(encoding="utf-8"))
    status = summary["status"]
    eng = summary["engineering_gates"]
    beh = summary["behavioral_yield"]
    labels = summary["labels"]

    gate_lines = "\n".join(
        "| {gate} | {res} | {detail} |".format(
            gate=g["gate"],
            res="PASS" if g["pass"] else "FAIL",
            detail=json.dumps(
                {k: v for k, v in g.items() if k not in ("gate", "pass")}
            ),
        )
        for g in eng["gates"]
    )
    prompt_table = "\n".join(
        f"| {p['prompt_id']} | {p['n_honest']} | {p['n_ambiguous']} | {p['n_deceptive']} | "
        f"{'Y' if p['has_ge1_h_and_ge1_d'] else 'n'} | "
        f"{'Y' if p['has_ge2_h_and_ge2_d'] else 'n'} |"
        for p in summary["per_prompt"]
    )

    report = f"""# Phase 24B — Live activation capture feasibility

**Status:** `{status}`

{GUARANTEE}

## Context

Phase 24A teacher-forced replay failed all frozen numerical gates. Primary
physiology evidence must therefore come from activations captured during live
autoregressive generation. Phase 24B validates instrumentation and operational
sampling only — not deception physiology.

The 16 prompts are enriched DEVELOPMENT historically-mixed prompts.
**Do not treat Phase-24B label frequencies as population prevalence.**

## Contract

| Setting | Value |
|---|---|
| Model | `{contract['model_id']}` @ `{contract['model_revision']}` |
| Tokenizer | `{contract['tokenizer_revision']}` |
| dtype / attn | `{contract['dtype']}` / `{contract['attention_implementation']}` |
| temperature | `{contract['generation']['temperature']}` |
| max_new_tokens | `{contract['generation']['max_new_tokens']}` |
| do_sample | `{contract['generation']['do_sample']}` |

### Activation coordinates

- embedding / pre-block residual (`model.model.embed_tokens`)
- post-block residual after blocks 0–31 (`model.model.layers[L]`)
- final RMSNorm (`model.model.norm`)
- full next-token logits (float16) at every prediction step

Prediction-step alignment: state at step `t` produces logits for sampled token `t`
(before that token is emitted).

## Pilot

- prompt_ids_sha256: `{pilot['prompt_ids_sha256']}`
- schedule_sha256: `{pilot['schedule_sha256']}`
- seed rule: `{pilot['seed_rule']}`
- prompts: `{json.dumps(pilot['prompt_ids'])}`

## Canary (capture OFF vs ON)

Passed: **{summary['canary']['passed']}**

{json.dumps(summary['canary']['results'], indent=2)}

## Capture integrity G1–G5

| Gate | Result | Detail |
|---|---|---|
{gate_lines}

Engineering overall: **{"PASS" if eng["passed"] else "FAIL"}**

- requested/completed: {summary['n_requested']}/{summary['n_completed']}
- total generated tokens: {summary['total_generated_tokens']}
- tokens/sec: {summary['tokens_per_second']:.3f}
- peak GPU memory bytes: {summary['peak_memory_bytes']}

## Behavioral yield (Gemma response-level only)

Grader: `google/gemma-4-31B-it` @ `842da3794eaa0b77d5f08bae87a17459d91ff475`

| Label | Count |
|---|---|
| honest | {labels['honest']} |
| ambiguous | {labels['ambiguous']} |
| deceptive | {labels['deceptive']} |

| Prompt | H | A | D | ≥1H+≥1D | ≥2H+≥2D |
|---|---|---|---|---|---|
{prompt_table}

- prompts with ≥1H and ≥1D: **{summary['n_prompts_ge1h_ge1d']}**/16
- prompts with ≥2H and ≥2D: **{summary['n_prompts_ge2h_ge2d']}**/16
- behavioral yield: **{"PASS" if beh['passed'] else "FAIL"}**

{json.dumps(beh, indent=2)}

## Storage / compute

| Metric | Value |
|---|---|
| mean bytes/trajectory | {summary['bytes_per_trajectory_mean']:.0f} |
| total pilot storage | {summary['total_storage_bytes']} |
| capture wall seconds | {summary['wall_seconds_capture']:.1f} |
| grade wall seconds | {summary['wall_seconds_grade']:.1f} |
| GPU | {summary['gpu']} |
| estimated cost USD | {summary['estimated_cost_usd']:.3f} |

### Projections (planning only; not launched)

```json
{json.dumps(summary['projections'], indent=2)}
```

## Authorizations after

See `artifacts/phase24b_live/freeze.json`.
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")

    hashes = {
        "summary.json": hashlib.sha256((OUT / "summary.json").read_bytes()).hexdigest(),
        "freeze.json": hashlib.sha256((OUT / "freeze.json").read_bytes()).hexdigest(),
        "pilot_manifest.json": hashlib.sha256(
            (OUT / "pilot_manifest.json").read_bytes()
        ).hexdigest(),
        "prompt_ids_sha256": pilot["prompt_ids_sha256"],
        "schedule_sha256": pilot["schedule_sha256"],
    }
    (OUT / "artifact_hashes.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    log = DECISION_LOG.read_text(encoding="utf-8")
    if "### D159 —" not in log:
        decision = (
            "Phase 24A (`afb20fa…`) failed prospectively frozen live↔replay "
            "activation-equivalence gates; primary physiology will therefore "
            "require activations recorded during live autoregressive generation. "
            "Phase 24B validates live capture instrumentation and operational "
            "sampling only on 16 SHA-first enriched DEVELOPMENT historically "
            "mixed prompts × 6 rollouts (not for prevalence). Status "
            f"`{status}`. No deception probe, layer×token predictive analysis, "
            "SAE, causal intervention, K=60, onset grading, OpenAI, or held-out "
            "physiology prompts."
        )
        entry = (
            "\n### D159 — Phase 24B live-capture feasibility pilot\n\n"
            "- **Type:** **OUR RESEARCH DECISION**\n"
            f"- **Decision:** {decision}\n"
            "- **Date:** 2026-09-30\n"
        )
        DECISION_LOG.write_text(log.rstrip() + "\n" + entry, encoding="utf-8")

    print(json.dumps({"report": str(REPORT), "status": status}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
