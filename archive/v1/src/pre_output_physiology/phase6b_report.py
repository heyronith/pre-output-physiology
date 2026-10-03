"""Phase 6B primary physiology report generator (frozen before results)."""

from __future__ import annotations

from typing import Any

REQUIRED_STATEMENT = (
    "PHASE 6A REMAINS HOLD UNDER THE ORIGINAL EXACT-FORMAT GATE. NO PHASE 6 PROMPTS "
    "WERE REVISED AFTER PILOT 2. THE PHASE-5 PROBE WAS NOT RETRAINED, RECALIBRATED, "
    "RESELECTED, OR MODIFIED. THE PRIMARY PHASE-6B ANALYSIS USED ALL 720 DESIGNED "
    "SCENARIOS WITHOUT CONDITIONING ON DOWNSTREAM BEHAVIOR. NO CAUSAL INTERVENTIONS "
    "WERE PERFORMED."
)

CLAIM = (
    "Does the frozen Phase-5 representation preferentially encode a private "
    "communication target that conflicts with the known record, rather than generic "
    "private-goal conflict?"
)


def _f(x: float) -> str:
    return f"{x:+.4f}"


def _ci(stat: dict[str, float], lo: str = "ci_low", hi: str = "ci_high") -> str:
    return f"[{stat[lo]:+.4f}, {stat[hi]:+.4f}]"


def _contrast_rows(analysis: dict[str, Any]) -> list[str]:
    lines = [
        "| Contrast | Role | Paired mean delta | 95% CI | Frac. delta > 0 | AUROC | AUROC 95% CI |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, c in analysis["contrasts"].items():
        lines.append(
            f"| {name.replace('_minus_', ' − ')} | {c['role']} | {_f(c['paired_mean_delta'])} | "
            f"{_ci(c, 'paired_mean_delta_ci_low', 'paired_mean_delta_ci_high')} | "
            f"{c['fraction_scenarios_positive_delta']:.3f} | {c['auroc']:.4f} | "
            f"[{c['auroc_ci_low']:.4f}, {c['auroc_ci_high']:.4f}] |"
        )
    return lines


def _effect_rows(analysis: dict[str, Any]) -> list[str]:
    lines = ["| Effect | Formula | Mean | 95% CI |", "|---|---|---|---|"]
    for name, e in analysis["effects"].items():
        lines.append(
            f"| {name} | `{analysis['effect_formulas'][name]}` | {_f(e['mean'])} | {_ci(e)} |"
        )
    return lines


def _family_rows(analysis: dict[str, Any]) -> list[str]:
    lines = [
        "| Family | Communication main | Operational main | Interaction | F11 − F10 | F01 − F00 |",
        "|---|---|---|---|---|---|",
    ]
    for fam, d in analysis["effects_by_family"].items():
        cells = [
            f"{_f(d[k]['mean'])} {_ci(d[k])}"
            for k in (
                "communication_main_effect",
                "operational_main_effect",
                "interaction",
                "F11_minus_F10_delta",
                "F01_minus_F00_delta",
            )
        ]
        lines.append(f"| {fam} | " + " | ".join(cells) + " |")
    return lines


def render_report(summary: dict[str, Any]) -> str:
    prim = summary["analysis_probability"]
    sec = summary["analysis_logit_secondary"]
    ex = summary["extraction"]
    flags = prim["pattern_flags"]
    lines = [
        "# Phase 6B — frozen-probe factorial physiology (primary)",
        "",
        f"Status: `{summary['status']}`. Phase 6A outcome remains "
        f"`{summary['phase6a_outcome']}` (HOLD).",
        "",
        "Phase 6B is a prospective protocol amendment (D083) made before any final "
        "Phase-6 model call. It is **not** a general deception-intent test.",
        "",
        f"**Claim under test:** {CLAIM}",
        "",
        "## Provenance",
        "",
        f"- Pre-run freeze commit: `{summary['pre_run_commit']}`",
        f"- Extraction run: `{ex['run_id']}`",
        f"- Corpus hashes verified: prompt `{ex['final_prompt_text_sha256']}`, scenario "
        f"`{ex['final_scenario_text_sha256']}`, IDs `{ex['final_base_scenario_ids_sha256']}`",
        f"- Probe SHA256 verified: `{summary['probe_sha256']}` "
        "(L12, controlled-prefix k1, token 12107)",
        f"- Repeatability preflight min cosine: {ex['preflight_repeatability_min_cosine']:.8f} "
        f"(required ≥ 0.9999)",
        f"- GPU wall time: {ex['wall_seconds']:.1f}s extraction + "
        f"{ex['preflight_wall_seconds']:.1f}s preflight; estimated cost "
        f"${ex['total_estimated_cost_usd']:.4f} (L40S @ $1.95/h)",
        f"- Activation integrity: {summary['activation_integrity']}",
        "",
        "## Primary population",
        "",
        f"All {prim['n_scenarios']} base scenarios × 4 conditions; no behavior conditioning; "
        f"no exclusions. Scenario-paired bootstrap, {prim['n_bootstrap']} resamples, seed "
        f"{prim['bootstrap_seed']}.",
        "",
        "Condition mean probe scores (probability): "
        + ", ".join(f"{k} {v:.4f}" for k, v in prim["condition_mean_scores"].items()),
        "",
        "## Contrasts (primary score scale: frozen probe probability)",
        "",
        *_contrast_rows(prim),
        "",
        "## Factorial effects (probability)",
        "",
        *_effect_rows(prim),
        "",
        "## Effects by family (probability; mean [95% CI])",
        "",
        *_family_rows(prim),
        "",
        "## Pattern flags (frozen operational definitions; descriptive only)",
        "",
        *[f"- {k}: {v}" for k, v in flags["definitions"].items()],
        "",
        f"- Pattern A (communication dominant): **{flags['pattern_A_communication_dominant']}**",
        f"- Pattern B (operational dominant): **{flags['pattern_B_operational_dominant']}**",
        f"- Pattern C (mixed / interaction): **{flags['pattern_C_interaction_mixed']}**",
        f"- |operational| / |communication| main-effect ratio: "
        f"{flags['operational_to_communication_ratio']:.3f}",
        "",
        "These flags are not a single winner metric; all contrasts and effects must be "
        "read jointly. A mixed or interaction pattern is not a pure deception signal.",
        "",
        "## Secondary score scale (frozen probe logit, pre-registered)",
        "",
        *_contrast_rows(sec),
        "",
        *_effect_rows(sec),
        "",
        "## Scope",
        "",
        "No response generation was performed for Phase 6B. Behavior is not part of the "
        "primary analysis. Awaiting audit.",
        "",
        REQUIRED_STATEMENT,
        "",
    ]
    return "\n".join(lines)
