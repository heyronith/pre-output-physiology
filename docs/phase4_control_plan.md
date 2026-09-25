# Phase 4 control-plan (design only — not executed)

This document freezes the conceptual control battery from Phase 1 **before** Phase 3 activation results exist, so Phase 4 cannot be designed opportunistically.

**Status:** plan only. Do not generate these datasets in Phase 3A/3B.

---

## Shared design principles

- Match as many surface factors as possible across conditions (prompt length, domain, role language, token budget).
- Prefer scenario-level pairing of honest vs deceptive counterparts with the **same underlying truth**.
- Use group-level splits; version label pipelines exactly.
- Keep RoleplayDeception as positive control (condition 6), not as spontaneous-intent proof.

---

## 1. KNOWN-TRUTH / HONEST

| Field | Content |
| --- | --- |
| Purpose | Model has relevant true information and is incentivized to report it honestly |
| Matched factors | Same facts / scenario family as deceptive counterpart; similar dialogue length |
| Candidate source | Constructed QA / dialogue with private true state revealed in system context; honest incentive |
| Confound tested | Separates “has the truth” from “deceives about the truth” |

## 2. KNOWN-TRUTH / DECEPTIVE

| Field | Content |
| --- | --- |
| Purpose | Same underlying truth as (1), but deception benefits the objective |
| Matched factors | Identical private truth to (1); only incentive / objective differs |
| Candidate source | Same scenarios as (1) with deception-beneficial framing (instrumental lying) |
| Confound tested | Isolates intentional deception given knowledge |

## 3. UNCERTAINTY

| Field | Content |
| --- | --- |
| Purpose | Model lacks sufficient evidence to know the answer |
| Matched factors | Similar question form; withhold decisive facts rather than plant false ones |
| Candidate source | Underspecified prompts; optional “I don’t know” targets |
| Confound tested | Whether probes track uncertainty / hedging rather than deception |

## 4. FALSE-BELIEF

| Field | Content |
| --- | --- |
| Purpose | Incorrect private information + instruction to answer honestly |
| Matched factors | Parallel to (1)/(2) but private state is false; honesty incentive |
| Candidate source | System provides wrong fact; ask for honest report of that private state |
| Confound tested | Separates intentional deception from sincerely expressing a false belief |

## 5. STRATEGIC-NONDECEPTIVE

| Field | Content |
| --- | --- |
| Purpose | Planning / optimization without needing to mislead anyone |
| Matched factors | Similar “strategic” language length/complexity without false claims to a listener |
| Candidate source | Tool-use / planning / bargaining tasks where honesty is compatible with goals |
| Confound tested | Generic strategic reasoning vs deception-specific representations |

## 6. EXPLICIT-DECEPTION / ROLEPLAY (positive control)

| Field | Content |
| --- | --- |
| Purpose | Deception directly induced or role-conditioned |
| Matched factors | Already frozen: RoleplayDeception + Mistral-7B Phase 2 precedent |
| Candidate source | Existing LASR RoleplayDeception incentivised trajectories |
| Confound tested | Instrumentation validity; **not** spontaneous hidden intent |

---

## Phase 4 authorization gate

Controls are finalized and generated only after Phase 3 review. Exact dataset construction, seeds, and judge prompts will be versioned before any Phase 4 data collection.
