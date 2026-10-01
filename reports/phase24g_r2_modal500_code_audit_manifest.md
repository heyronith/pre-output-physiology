# Phase 24G-R2 — independent code-audit checklist

Status under audit:
`phase24g_r2_modal500_corrected_diagnostics_complete_awaiting_code_audit`

Proposed corrected category: **B** (not accepted until independent audit).

## Scope for auditor

1. GitHub diff on branch `phase24/diagnostic-correction-modal500` from
   `be8bbdd8012ed0869d2b4b32c0000a417e45e0b7` → freeze HEAD.
2. Critical analysis code:
   - `src/pre_output_physiology/phase24g_r2_modal500.py`
   - `modal/phase24g_r2_modal500.py`
   - `scripts/aggregate_phase24g_r2_modal500.py`
   - frozen candidate rule via Phase-24E `select_candidate_region`
   - full stacked path only (`proxy_delta=False` primary)
3. Resumability / aggregation:
   - deterministic `rep_id` / `seed_for_rep`
   - atomic write + never overwrite valid completed reps
   - aggregate only after LOPO=28 and nested=500
4. Result artifacts under `artifacts/phase24g_r2_modal500/` and Volume
   `pre-output-physiology-phase24g-r2`.
5. Phase-24F immutability:
   `artifacts/phase24g_r2_modal500/phase24f_immutability.json`

## Do not authorize next scientific phase until

- Category **B** is independently accepted or revised
- Diff + hashes audited
- No primary-endpoint redefinition found
