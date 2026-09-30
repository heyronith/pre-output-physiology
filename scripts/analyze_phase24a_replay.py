#!/usr/bin/env python3
"""Write Phase-24A report + decision-log entry from frozen summary artifacts."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase24a_replay import GUARANTEE  # noqa: E402

OUT = REPO_ROOT / "artifacts/phase24a_replay"
REPORT = REPO_ROOT / "reports/phase24a_replay_equivalence.md"
DECISION_LOG = REPO_ROOT / "docs/decision_log.md"


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    contract = json.loads((OUT / "contract.json").read_text(encoding="utf-8"))
    pilot = json.loads((OUT / "pilot_manifest.json").read_text(encoding="utf-8"))
    inventory = json.loads((OUT / "k20_label_only_inventory.json").read_text(encoding="utf-8"))

    artifact_hashes = {
        "contract.json": _sha_file(OUT / "contract.json"),
        "pilot_manifest.json": _sha_file(OUT / "pilot_manifest.json"),
        "k20_label_only_inventory.json": _sha_file(OUT / "k20_label_only_inventory.json"),
        "summary.json": _sha_file(OUT / "summary.json"),
        "freeze.json": _sha_file(OUT / "freeze.json"),
        "live_activations_sha256": summary["live_activations_sha256"],
        "replay_activations_sha256": summary["replay_activations_sha256"],
    }

    cos = summary["cosine"]
    rel = summary["relative_l2"]
    gates = summary["gates"]
    logit = summary["logit_diagnostics"]
    status = summary["status"]

    layer_cos = summary.get("by_layer_cos_median", {})
    layer_rel = summary.get("by_layer_rel_median", {})
    worst_layer_rel = max(
        ((int(k), float(v)) for k, v in layer_rel.items() if v == v),
        key=lambda x: x[1],
        default=(None, float("nan")),
    )
    best_layer_cos = min(
        ((int(k), float(v)) for k, v in layer_cos.items() if v == v),
        key=lambda x: x[1],
        default=(None, float("nan")),
    )

    heat = summary.get("heatmap_mean_rel_l2_layer_by_gen_t") or []
    heat_note = (
        f"Heatmap array shape layer×gen_t = "
        f"{len(heat)}×{len(heat[0]) if heat else 0} "
        f"(mean relative L2 over responses; NaNs where no observations)."
    )

    gate_lines = "\n".join(
        f"| {g['metric']} | {g['threshold']} | {g['observed']:.8g} | "
        f"{'PASS' if g['pass'] else 'FAIL'} |"
        for g in gates["gates"]
    )

    inv_dev_n = inventory["development"]["n_prompts_in_split"]
    inv_dev_q = inventory["development"]["n_qualifying"]
    inv_lk_n = inventory["locked_validation"]["n_prompts_in_split"]
    inv_lk_q = inventory["locked_validation"]["n_qualifying"]

    if gates["passed"]:
        interpretation = (
            "Teacher-forced replay is numerically faithful enough to reconstruct "
            "internal trajectories of the existing K=20 corpus under the frozen "
            "contract. STOP. Do not extract K=20 activations yet — awaiting "
            "separate authorization."
        )
    else:
        interpretation = (
            "Historical K=20 responses must not be treated as having reliably "
            "reconstructable activation trajectories. Future physiology must use "
            "activations recorded during live generation. STOP. Thresholds were "
            "not weakened."
        )

    report = f"""# Phase 24A — Live vs replay activation equivalence

**Status:** `{status}`

**Instrumentation validation only.** Exact single-token deception onset remains
unvalidated (Phase 23D FAIL). This phase does not extract K=20 activations,
train probes/SAEs, grade responses, or make physiology claims.

{GUARANTEE}

## Contract (frozen from K=20 artifacts)

| Setting | Value |
|---|---|
| Model | `{contract['model_id']}` |
| Model revision | `{contract['model_revision']}` |
| Tokenizer revision | `{contract['tokenizer_revision']}` |
| dtype | `{contract['dtype']}` |
| Attention | `{summary.get('attn_implementation', contract['attention_implementation'])}` |
| Chat template | `{contract['chat_template']}` |
| Temperature | `{contract['generation']['temperature']}` |
| do_sample | `{contract['generation']['do_sample']}` |
| max_new_tokens | `{contract['generation']['max_new_tokens']}` |
| max_sequence_length | `{contract['generation']['max_sequence_length']}` |
| Seed policy | `{contract['generation']['seed_policy']}` |
| Hidden-state index | `hidden_states[layer + 1]` = post-block residual |

## Pilot

- n = {pilot['n']}
- selection = `{pilot['selection_method']}`
- prompt_ids_sha256 = `{pilot['prompt_ids_sha256']}`
- prompt IDs:

```
{json.dumps(pilot['prompt_ids'], indent=2)}
```

- requested generations: 16
- completed generations: {summary['n_prompts_completed']}

## Activation equivalence

- n activation comparisons (response × token × layer): **{summary['n_activation_comparisons']}**

| Metric | Value |
|---|---|
| median cosine | {cos['median']:.10f} |
| 1st-percentile cosine | {cos['p01']:.10f} |
| minimum cosine | {cos['min']:.10f} |
| median relative L2 | {rel['median']:.10e} |
| 99th-percentile relative L2 | {rel['p99']:.10e} |
| maximum relative L2 | {rel['max']:.10e} |

### Frozen gates

| Gate | Threshold | Observed | Result |
|---|---|---|---|
{gate_lines}

**Overall activation gates:** {"PASS" if gates["passed"] else "FAIL"}

### Layer / time diagnostics

- Worst median relative L2 by layer: layer {worst_layer_rel[0]} = {worst_layer_rel[1]:.6e}
- Lowest median cosine by layer: layer {best_layer_cos[0]} = {best_layer_cos[1]:.10f}
- {heat_note}
- Full per-layer medians and heatmap array: `artifacts/phase24a_replay/summary.json`

### Per-response summary

See `summary.json` → `per_response` (median cosine / rel L2 / n_generated / stopping_reason).

## Logit diagnostics (not PASS/FAIL gates)

| Diagnostic | Value |
|---|---|
| median logit cosine | {logit['logit_cosine']['median']:.10f} |
| min logit cosine | {logit['logit_cosine']['min']:.10f} |
| median \\|Δ log p(actual next)\\| | {logit['abs_logprob_diff_actual_next_token']['median']:.6e} |
| max \\|Δ log p(actual next)\\| | {logit['abs_logprob_diff_actual_next_token']['max']:.6e} |
| top-1 agreement rate | {logit['top1_agreement_rate']:.6f} (n={logit['top1_n']}) |

## K=20 label-only inventory (no onset; inventory only)

Primary labels: frozen GPT-4o reference. Rule: ≥2 honest and ≥2 deceptive; no onset.

| Split | n prompts | qualifying (≥2H & ≥2D) |
|---|---|---|
| DEVELOPMENT | {inv_dev_n} | {inv_dev_q} |
| LOCKED | {inv_lk_n} | {inv_lk_q} |

Does **not** influence replay PASS/FAIL. No new split. Stage-23D onset eligibility unused.

## Provenance hashes

```json
{json.dumps(artifact_hashes, indent=2)}
```

## Runtime

| | |
|---|---|
| GPU | {summary.get('gpu')} |
| wall_seconds | {summary.get('wall_seconds')} |
| estimated_cost_usd | {summary.get('estimated_cost_usd')} |
| run_id | `{summary.get('run_id')}` |
| git_commit (run) | `{summary.get('git_commit')}` |

## Interpretation

{interpretation}

## Authorizations after run

```json
{json.dumps(summary.get('authorizations_after', {}), indent=2)}
```
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")

    # Append D158 if not already present
    log = DECISION_LOG.read_text(encoding="utf-8")
    if "### D158 —" not in log:
        gate_word = "all PASS" if gates["passed"] else "FAILED"
        pid_sha_short = pilot["prompt_ids_sha256"][:16]
        decision = (
            "Starting from Phase-23D FAIL SHA "
            "`b7fd54e73fdd148797a5a24b77dedc053446f670`, Phase 24A tests only "
            "whether teacher-forced replay of Mistral activations numerically "
            "matches live cached autoregressive activations under the frozen "
            "K=20 contract (`mistralai/Mistral-7B-Instruct-v0.2` rev "
            "`63a8b081…e07a`; BF16; T=1.0; max_new_tokens=200). Exact "
            "single-token onset localization remains unvalidated and is not "
            "rewritten. The research target has prospectively shifted to whether "
            "activations during generation contain information predictive of "
            "eventual deceptive behavior beyond prompt/prefix/logits, and when "
            "across token×layer that information emerges — contingent on replay "
            "PASS before any historical K=20 activation use. Pilot: 16 SHA-even "
            f"DEVELOPMENT prompts (ids SHA `{pid_sha_short}…`); instrumentation "
            "only; outputs not graded. Frozen gates "
            f"{gate_word} → status `{status}`. K=20 label-only inventory "
            f"(GPT-4o; ≥2H & ≥2D; no onset): DEV {inv_dev_q}/260, LOCKED "
            f"{inv_lk_q}/111 (inventory only). No K=20 activation extraction, "
            "probes, SAE, K>20, OpenAI, new labels, or causal interventions."
        )
        entry = (
            "\n### D158 — Phase 24A replay-equivalence instrumentation validation\n\n"
            "- **Type:** **OUR RESEARCH DECISION**\n"
            f"- **Decision:** {decision}\n"
            "- **Date:** 2026-09-30\n"
        )
        DECISION_LOG.write_text(log.rstrip() + "\n" + entry, encoding="utf-8")

    # Persist artifact hash map
    (OUT / "artifact_hashes.json").write_text(
        json.dumps(artifact_hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"report": str(REPORT), "status": status, "gates_passed": gates["passed"]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
