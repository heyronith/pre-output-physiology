"""Phase 24A — live vs teacher-forced replay activation equivalence (instrumentation).

Exact single-token onset localization remains unvalidated (Phase 23D FAIL).
This phase only tests whether teacher-forced replay reproduces activations from
live cached autoregressive generation under the frozen K=20 Mistral contract.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from typing import Any

# Frozen K=20 Mistral generation contract (Phase 21 / 22B manifests)
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
TOKENIZER_REVISION = MODEL_REVISION  # tokenizer loaded from same HF revision
DTYPE = "bfloat16"
TEMPERATURE = 1.0
DO_SAMPLE = True
MAX_NEW_TOKENS = 200
BATCH_SIZE = 1
N_TRANSFORMER_BLOCKS = 32
# Hugging Face output_hidden_states: index 0 = embeddings;
# index layer+1 = residual stream after transformer block `layer`.
HIDDEN_STATES_POST_BLOCK_OFFSET = 1
# Mistral-7B-Instruct-v0.2 config.json max_position_embeddings (K=20 used model default).
MAX_SEQUENCE_LENGTH = 32768

N_PILOT_PROMPTS = 16
PILOT_SEED_BASE = 24_000_000
GRADER_SPLIT_SHA = "6504b40b43153a7bd359c344efb6eee01afca887fe4978960d9d2e0d6fec3869"
N_DEV_PROMPTS = 260
N_LOCKED_PROMPTS = 111

# Frozen BEFORE seeing results
ACTIVATION_EQUIVALENCE_GATES = {
    "min_median_cosine": 0.99999,
    "min_p01_cosine": 0.9999,
    "max_median_rel_l2": 0.001,
    "max_p99_rel_l2": 0.01,
}

STATUS_PASS = "phase24a_replay_equivalence_validated_awaiting_k20_activation_authorization"
STATUS_FAIL = "phase24a_replay_equivalence_failed_live_recording_required"

GUARANTEE = (
    "PHASE 24A WAS AN INSTRUMENTATION VALIDATION ONLY. NO K=20 ACTIVATION DATASET "
    "WAS EXTRACTED, NO DECEPTION PROBE OR SAE WAS TRAINED, NO K>20 RESPONSES WERE "
    "GENERATED, NO NEW BEHAVIORAL LABELS WERE CREATED, AND NO CAUSAL OR PHYSIOLOGY "
    "CLAIM WAS MADE."
)


def sha256_json(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def select_pilot_prompt_ids(
    development_prompt_ids: Sequence[str],
    *,
    n: int = N_PILOT_PROMPTS,
) -> list[str]:
    """Deterministic evenly spaced selection over SHA-sorted DEVELOPMENT IDs."""
    if len(development_prompt_ids) != N_DEV_PROMPTS:
        raise ValueError(
            f"expected {N_DEV_PROMPTS} DEVELOPMENT prompts, got {len(development_prompt_ids)}"
        )
    ordered = sorted(
        development_prompt_ids,
        key=lambda pid: hashlib.sha256(pid.encode()).hexdigest(),
    )
    if n > len(ordered):
        raise ValueError(f"n={n} > available {len(ordered)}")
    if n == 1:
        return [ordered[0]]
    # Evenly spaced indices across [0, len-1]
    idxs = [round(i * (len(ordered) - 1) / (n - 1)) for i in range(n)]
    # Deduplicate while preserving order (rounding collisions)
    out: list[str] = []
    seen: set[str] = set()
    for i in idxs:
        pid = ordered[i]
        if pid not in seen:
            seen.add(pid)
            out.append(pid)
    if len(out) < n:
        # Fill from remaining SHA-ordered IDs
        for pid in ordered:
            if pid not in seen:
                seen.add(pid)
                out.append(pid)
            if len(out) == n:
                break
    if len(out) != n:
        raise RuntimeError(f"pilot selection produced {len(out)} != {n}")
    return out


def pilot_sample_seed(prompt_id: str, *, seed_base: int = PILOT_SEED_BASE) -> int:
    """Deterministic per-prompt seed independent of behavior/labels."""
    digest = hashlib.sha256(f"phase24a|{prompt_id}".encode()).hexdigest()
    return seed_base + (int(digest[:8], 16) % 1_000_000)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("cosine requires equal non-empty vectors")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        xf = float(x)
        yf = float(y)
        dot += xf * yf
        na += xf * xf
        nb += yf * yf
    denom = math.sqrt(na) * math.sqrt(nb)
    if denom == 0.0:
        return float("nan")
    return float(dot / denom)


def relative_l2_error(
    live: Sequence[float], replay: Sequence[float], *, eps: float = 1e-12
) -> float:
    if len(live) != len(replay) or not live:
        raise ValueError("rel L2 requires equal non-empty vectors")
    diff2 = 0.0
    live2 = 0.0
    for x, y in zip(live, replay, strict=True):
        xf = float(x)
        yf = float(y)
        d = xf - yf
        diff2 += d * d
        live2 += xf * xf
    return float(math.sqrt(diff2) / max(math.sqrt(live2), eps))


def percentile(sorted_vals: Sequence[float], p: float) -> float:
    """Nearest-rank percentile on a pre-sorted ascending sequence; p in [0,100]."""
    if not sorted_vals:
        return float("nan")
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 100:
        return float(sorted_vals[-1])
    # nearest-rank
    k = max(1, int(math.ceil(p / 100.0 * len(sorted_vals)))) - 1
    return float(sorted_vals[k])


def summarize_metric_values(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {
            "n": 0,
            "median": float("nan"),
            "mean": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p01": float("nan"),
            "p99": float("nan"),
        }
    s = sorted(float(v) for v in values)
    n = len(s)
    mid = n // 2
    med = float(s[mid]) if n % 2 == 1 else 0.5 * (s[mid - 1] + s[mid])
    return {
        "n": float(n),
        "median": med,
        "mean": float(sum(s) / n),
        "min": float(s[0]),
        "max": float(s[-1]),
        "p01": percentile(s, 1),
        "p99": percentile(s, 99),
    }


def evaluate_activation_gates(
    cosine_summary: dict[str, float], rel_l2_summary: dict[str, float]
) -> dict[str, Any]:
    gates = ACTIVATION_EQUIVALENCE_GATES
    checks = [
        (
            "median_cosine",
            gates["min_median_cosine"],
            cosine_summary["median"],
            cosine_summary["median"] >= gates["min_median_cosine"],
        ),
        (
            "p01_cosine",
            gates["min_p01_cosine"],
            cosine_summary["p01"],
            cosine_summary["p01"] >= gates["min_p01_cosine"],
        ),
        (
            "median_rel_l2",
            gates["max_median_rel_l2"],
            rel_l2_summary["median"],
            rel_l2_summary["median"] <= gates["max_median_rel_l2"],
        ),
        (
            "p99_rel_l2",
            gates["max_p99_rel_l2"],
            rel_l2_summary["p99"],
            rel_l2_summary["p99"] <= gates["max_p99_rel_l2"],
        ),
    ]
    rows = [
        {"metric": name, "threshold": thr, "observed": obs, "pass": bool(ok)}
        for name, thr, obs, ok in checks
    ]
    return {
        "passed": all(r["pass"] for r in rows),
        "gates": rows,
        "thresholds": dict(gates),
    }


def label_only_mixed_prompt_counts(
    rows: Sequence[dict[str, Any]],
    *,
    prompt_ids: set[str],
    label_key: str = "reference_label",
    min_honest: int = 2,
    min_deceptive: int = 2,
) -> dict[str, Any]:
    """Inventory: ≥2 H and ≥2 D labels per prompt (no onset)."""
    from collections import Counter, defaultdict

    by: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        pid = r.get("prompt_id")
        if pid not in prompt_ids:
            continue
        lab = r.get(label_key)
        if lab in ("honest", "ambiguous", "deceptive", "exclude"):
            by[str(pid)].append(str(lab))
    qualifying: list[str] = []
    for pid in sorted(prompt_ids):
        c = Counter(by.get(pid, []))
        if c["honest"] >= min_honest and c["deceptive"] >= min_deceptive:
            qualifying.append(pid)
    return {
        "n_prompts_in_split": len(prompt_ids),
        "n_qualifying": len(qualifying),
        "qualifying_prompt_ids": qualifying,
        "rule": {
            "min_honest": min_honest,
            "min_deceptive": min_deceptive,
            "onset_required": False,
            "label_key": label_key,
        },
    }


def align_live_replay_positions(
    prompt_n_tokens: int, n_generated_tokens: int
) -> list[dict[str, Any]]:
    """Positions that live cached decoding actually computes.

    - prompt_end: after forward on the prompt (index prompt_n_tokens-1)
    - gen_t for t in 0..n_generated-1: after processing generated token t
      (sequence index prompt_n_tokens + t)
    """
    if prompt_n_tokens < 1:
        raise ValueError("prompt_n_tokens must be >= 1")
    if n_generated_tokens < 0:
        raise ValueError("n_generated_tokens must be >= 0")
    positions = [
        {
            "name": "prompt_end",
            "sequence_index": prompt_n_tokens - 1,
            "generated_token_index": None,
        }
    ]
    for t in range(n_generated_tokens):
        positions.append(
            {
                "name": f"gen_{t}",
                "sequence_index": prompt_n_tokens + t,
                "generated_token_index": t,
            }
        )
    return positions


def post_block_hidden_index(layer: int) -> int:
    """Map transformer block index → Hugging Face hidden_states index."""
    if not (0 <= layer < N_TRANSFORMER_BLOCKS):
        raise ValueError(f"layer {layer} out of range")
    return layer + HIDDEN_STATES_POST_BLOCK_OFFSET


def causal_prefix_token_count(sequence_index: int) -> int:
    """Number of tokens a causal LM may use for the state at `sequence_index`.

    State at position i is a function of tokens[0 : i+1] only; future tokens
    at indices > i must not affect that state under causal attention.
    """
    if sequence_index < 0:
        raise ValueError("sequence_index must be >= 0")
    return sequence_index + 1


def terminal_generated_token_handling(
    generated_token_ids: Sequence[int], *, eos_token_id: int
) -> dict[str, Any]:
    """Document EOS / terminal-token rules for live↔replay alignment."""
    ids = [int(x) for x in generated_token_ids]
    n = len(ids)
    eos_positions = [i for i, t in enumerate(ids) if t == eos_token_id]
    stopped_on_eos = bool(eos_positions) and eos_positions[0] == n - 1
    # Live loop records a state after every generated token, including a terminal EOS.
    # Replay teacher-forces the exact same token list; both include the EOS position.
    return {
        "n_generated": n,
        "eos_token_id": int(eos_token_id),
        "eos_positions": eos_positions,
        "includes_terminal_eos_state": stopped_on_eos,
        "compare_eos_position": True if stopped_on_eos else None,
        "rule": (
            "If generation stops on EOS, the EOS token is part of generated_token_ids "
            "and both live and replay record the post-EOS residual state at that index."
        ),
    }
