"""Phase 24C — prospective full-trajectory design freeze (no GPU / no model calls)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

import numpy as np

# Dataset provenance (Phases 21–24)
APOLLO_REPO = "ApolloResearch/deception-detection"
APOLLO_COMMIT = "f8ec4010e74927394709dffa22b97bdf8cd5a62f"
DATASET_RELPATH = "data/roleplaying/dataset.yaml"
DATASET_BLOB_SHA = "b938599a955946f1c96fa8de79dda6aeab72452d"
DATASET_CONTENT_SHA256 = "7d3e36dc9a2fa72ff823b651c2ade2e0fc34c5bc6267cdf9b7be8b046ed2e564"
N_PROMPTS_TOTAL = 371
LOCAL_RAW_DATASET = "data/raw/apollo_roleplaying/dataset.yaml"
LOCAL_PROMPTS_JSONL = "data/processed/phase21_roleplay/prompts.jsonl"

N_ELIGIBLE = 39
N_DEV_ELIGIBLE = 28
N_LOCKED_ELIGIBLE = 11
N_PILOT_24B = 16
N_TRAIN = 20
N_VALIDATION = 8
N_TEST = 11

K_GRID = (6, 8, 10, 12, 16, 20)
FORECAST_SEED = 2403
N_MC = 100_000
DIRICHLET_ALPHA = 0.5
N_OBSERVED_24B = 6

# Primary K selection gates
TRAIN_MIN_QUALIFYING = 10  # of 20
TRAIN_P_MIN = 0.90
VAL_MIN_QUALIFYING = 4  # of 8
VAL_P_MIN = 0.80
TEST_MIN_QUALIFYING = 5  # of 11
TEST_P_MIN = 0.80
TEST_MEDIAN_H_MIN = 20
TEST_MEDIAN_D_MIN = 20
HISTORICAL_DISAGREE_P = 0.50

# Analysis freeze
TEMPORAL_LANDMARKS = (1, 2, 4, 8, 16, 32, 64)
N_LAYERS = 32
LOGISTIC_C_GRID = (0.01, 0.1, 1.0, 10.0)
VAL_BOOTSTRAP_REPS = 2_000
TEST_BOOTSTRAP_REPS = 10_000
CANDIDATE_MIN_CONSEC_LAYERS = 3
CANDIDATE_MIN_MEDIAN_DELTA = 0.05

BYTES_PER_TRAJ_24B = 20_944_420.5
GPU_SEC_PER_TRAJ_24B = 571.6561906337738 / 96.0
A100_USD_PER_HOUR = 2.50

STATUS_FROZEN = "phase24c_full_trajectory_design_frozen_awaiting_live_generation_authorization"
STATUS_K_NOT_SUPPORTED = "phase24c_k_le20_not_supported_requires_review"
STATUS_FORECAST_DISAGREE = "phase24c_k_forecasts_disagree_requires_review"

GUARANTEE = (
    "PHASE 24C WAS A PROSPECTIVE DESIGN, YIELD-FORECAST, AND STATISTICAL-FREEZE "
    "PHASE. NO NEW MODEL GENERATION, GRADING, ACTIVATION EXTRACTION, DECEPTION "
    "PROBING, SAE ANALYSIS, CAUSAL INTERVENTION, OR PRIMARY PHYSIOLOGY ANALYSIS "
    "WAS PERFORMED."
)


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_key(prompt_id: str) -> str:
    return hashlib.sha256(f"phase24c|split|{prompt_id}".encode()).hexdigest()


def build_train_val_test_split(
    *,
    pilot_24b_ids: Sequence[str],
    development_eligible: Sequence[str],
    locked_eligible: Sequence[str],
) -> dict[str, Any]:
    """Freeze TRAIN=20 / VAL=8 / TEST=11 with 24B pilots permanently TRAIN-only."""
    pilot = list(pilot_24b_ids)
    if len(pilot) != N_PILOT_24B:
        raise ValueError(f"expected {N_PILOT_24B} pilot prompts, got {len(pilot)}")
    if len(development_eligible) != N_DEV_ELIGIBLE:
        raise ValueError(f"expected {N_DEV_ELIGIBLE} DEV eligible")
    if len(locked_eligible) != N_LOCKED_ELIGIBLE:
        raise ValueError(f"expected {N_LOCKED_ELIGIBLE} LOCKED eligible")

    pilot_set = set(pilot)
    if not pilot_set.issubset(set(development_eligible)):
        raise ValueError("pilot prompts must be subset of DEVELOPMENT eligible")
    if pilot_set & set(locked_eligible):
        raise ValueError("pilot prompts must not appear in LOCKED eligible")

    remaining_dev = [p for p in development_eligible if p not in pilot_set]
    if len(remaining_dev) != 12:
        raise ValueError(f"expected 12 remaining DEV, got {len(remaining_dev)}")
    ordered = sorted(remaining_dev, key=split_key)
    train_extra = ordered[:4]
    validation = ordered[4:]
    train = list(pilot) + train_extra
    test = list(locked_eligible)

    if len(train) != N_TRAIN or len(validation) != N_VALIDATION or len(test) != N_TEST:
        raise RuntimeError("split size mismatch")
    if set(validation) & pilot_set:
        raise RuntimeError("pilot leaked into VALIDATION")
    if set(test) & (set(train) | set(validation)):
        raise RuntimeError("TEST overlap with TRAIN/VAL")
    if set(validation) & set(train):
        raise RuntimeError("VAL/TRAIN overlap")

    return {
        "train_prompt_ids": train,
        "validation_prompt_ids": validation,
        "test_prompt_ids": test,
        "pilot_24b_prompt_ids": pilot,
        "train_extra_from_remaining_dev": train_extra,
        "n_train": N_TRAIN,
        "n_validation": N_VALIDATION,
        "n_test": N_TEST,
        "train_sha256": sha256_json(train),
        "validation_sha256": sha256_json(validation),
        "test_sha256": sha256_json(test),
        "rule": (
            "all 16 Phase-24B pilots → TRAIN; remaining 12 DEV sorted by "
            "sha256(phase24c|split|{id}): first 4 → TRAIN, last 8 → VALIDATION; "
            "11 historically LOCKED eligible → TEST"
        ),
    }


def dirichlet_multinomial_draw(
    rng: np.random.Generator,
    counts_had: tuple[int, int, int],
    *,
    n_draw: int,
    alpha: float = DIRICHLET_ALPHA,
) -> tuple[int, int, int]:
    """Draw n_draw outcomes from Dirichlet-Multinomial with Jeffreys prior."""
    if n_draw < 0:
        raise ValueError("n_draw must be >= 0")
    if n_draw == 0:
        return (0, 0, 0)
    a = np.asarray(counts_had, dtype=np.float64) + alpha
    theta = rng.dirichlet(a)
    return tuple(int(x) for x in rng.multinomial(n_draw, theta))  # type: ignore[return-value]


def _dirichlet_multinomial_batch(
    rng: np.random.Generator,
    alphas: np.ndarray,
    *,
    n_draw: int,
) -> np.ndarray:
    """Vectorized Dirichlet-Multinomial.

    alphas: (N, 3) concentration parameters.
    Returns counts shape (N, 3) summing to n_draw.
    """
    alphas = np.asarray(alphas, dtype=np.float64)
    if alphas.ndim != 2 or alphas.shape[1] != 3:
        raise ValueError("alphas must be (N, 3)")
    n = alphas.shape[0]
    if n_draw < 0:
        raise ValueError("n_draw must be >= 0")
    if n_draw == 0:
        return np.zeros((n, 3), dtype=np.int64)
    # theta_i ~ Dirichlet(alpha_i) via independent Gammas
    g = rng.gamma(alphas, 1.0)
    theta = g / g.sum(axis=1, keepdims=True)
    # Multinomial via sequential categorical draws (vectorized over rows)
    cdf = np.cumsum(theta, axis=1)
    cdf[:, -1] = 1.0
    u = rng.random((n, n_draw))
    cats = (u[..., None] < cdf[:, None, :]).argmax(axis=-1)  # (N, n_draw)
    out = np.zeros((n, 3), dtype=np.int64)
    for c in range(3):
        out[:, c] = (cats == c).sum(axis=1)
    return out


def simulate_prompt_total_k(
    rng: np.random.Generator,
    base_had: tuple[int, int, int],
    *,
    k: int,
    retain_observed: int | None,
) -> tuple[int, int, int]:
    """Simulate (H,A,D) totals at budget K.

    If retain_observed is an int, keep that many observed trajectories and
    simulate only K - retain_observed additional ones (Phase-24B reuse).
    If None, simulate all K from the posterior (historical sensitivity).
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if retain_observed is None:
        return dirichlet_multinomial_draw(rng, base_had, n_draw=k)
    if retain_observed > k:
        raise ValueError("retain_observed cannot exceed k")
    h0, a0, d0 = base_had
    if retain_observed != h0 + a0 + d0:
        raise ValueError("retain_observed must equal sum(base_had)")
    add = k - retain_observed
    h1, a1, d1 = dirichlet_multinomial_draw(rng, base_had, n_draw=add)
    return (h0 + h1, a0 + a1, d0 + d1)


def qualifies_ge2(had: tuple[int, int, int]) -> bool:
    h, _a, d = had
    return h >= 2 and d >= 2


def qualifies_ge1(had: tuple[int, int, int]) -> bool:
    h, _a, d = had
    return h >= 1 and d >= 1


def _gate_row_from_arrays(
    *,
    k: int,
    train_qual: np.ndarray,
    val_qual: np.ndarray,
    test_qual: np.ndarray,
    test_h: np.ndarray,
    test_d: np.ndarray,
    train_ge1: np.ndarray | None = None,
    val_ge1: np.ndarray | None = None,
    test_ge1: np.ndarray | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "k": k,
        "p_train_ge10_of_20_ge2": float(np.mean(train_qual >= TRAIN_MIN_QUALIFYING)),
        "p_val_ge4_of_8_ge2": float(np.mean(val_qual >= VAL_MIN_QUALIFYING)),
        "p_test_ge5_of_11_ge2": float(np.mean(test_qual >= TEST_MIN_QUALIFYING)),
        "median_test_honest": float(np.median(test_h)),
        "median_test_deceptive": float(np.median(test_d)),
        "mean_train_ge2": float(np.mean(train_qual)),
        "mean_val_ge2": float(np.mean(val_qual)),
        "mean_test_ge2": float(np.mean(test_qual)),
    }
    if train_ge1 is not None:
        row["mean_train_ge1"] = float(np.mean(train_ge1))
        row["mean_val_ge1"] = float(np.mean(val_ge1))  # type: ignore[arg-type]
        row["mean_test_ge1"] = float(np.mean(test_ge1))  # type: ignore[arg-type]
    row["gates"] = {
        "train": row["p_train_ge10_of_20_ge2"] >= TRAIN_P_MIN,
        "validation": row["p_val_ge4_of_8_ge2"] >= VAL_P_MIN,
        "test_prompts": row["p_test_ge5_of_11_ge2"] >= TEST_P_MIN,
        "test_median_h": row["median_test_honest"] >= TEST_MEDIAN_H_MIN,
        "test_median_d": row["median_test_deceptive"] >= TEST_MEDIAN_D_MIN,
    }
    row["all_gates_pass"] = all(row["gates"].values())
    return row


def run_primary_live_forecast(
    *,
    pilot_counts: dict[str, tuple[int, int, int]],
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    test_ids: Sequence[str],
    k_grid: Sequence[int] = K_GRID,
    n_mc: int = N_MC,
    seed: int = FORECAST_SEED,
) -> dict[str, Any]:
    """Primary Phase-24B live forecast with prompt-heterogeneity bootstrap."""
    if len(pilot_counts) != N_PILOT_24B:
        raise ValueError("expected 16 pilot count entries")
    pilot_id_list = list(pilot_counts.keys())
    pilot_arr = np.asarray([pilot_counts[p] for p in pilot_id_list], dtype=np.float64)
    pilot_set = set(pilot_id_list)

    train_ids_l = list(train_ids)
    val_ids_l = list(val_ids)
    test_ids_l = list(test_ids)

    # Partition IDs: pilot vs non-pilot within each split
    def _part(ids: list[str]) -> tuple[list[str], list[str]]:
        p = [x for x in ids if x in pilot_set]
        o = [x for x in ids if x not in pilot_set]
        return p, o

    train_p, train_o = _part(train_ids_l)
    val_p, val_o = _part(val_ids_l)
    test_p, test_o = _part(test_ids_l)
    # Spec: all pilots are TRAIN-only
    if val_p or test_p:
        raise ValueError("pilot prompts must not appear in VAL/TEST")

    pilot_train_idx = [pilot_id_list.index(p) for p in train_p]
    n_non = len(train_o) + len(val_o) + len(test_o)

    results: dict[str, Any] = {"seed": seed, "n_mc": n_mc, "by_k": {}}
    rng = np.random.default_rng(seed)

    for k in k_grid:
        # Pilot TRAIN: retain observed 6 + simulate K-6
        n_pilot_train = len(pilot_train_idx)
        pilot_base = pilot_arr[pilot_train_idx]  # (n_pilot_train, 3)
        if k == N_OBSERVED_24B:
            pilot_totals = np.broadcast_to(
                pilot_base.astype(np.int64), (n_mc, n_pilot_train, 3)
            ).copy()
        else:
            add = np.zeros((n_mc, n_pilot_train, 3), dtype=np.int64)
            for i in range(n_pilot_train):
                add[:, i, :] = _dirichlet_multinomial_batch(
                    rng,
                    np.tile(pilot_base[i] + DIRICHLET_ALPHA, (n_mc, 1)),
                    n_draw=k - N_OBSERVED_24B,
                )
            pilot_totals = pilot_base.astype(np.int64)[None, :, :] + add

        # Non-pilot: bootstrap a pilot posterior then simulate full K
        boot = rng.integers(0, N_PILOT_24B, size=(n_mc, n_non))
        non_totals = np.zeros((n_mc, n_non, 3), dtype=np.int64)
        for j in range(N_PILOT_24B):
            mask = boot == j
            n_sel = int(mask.sum())
            if n_sel == 0:
                continue
            drawn = _dirichlet_multinomial_batch(
                rng,
                np.tile(pilot_arr[j] + DIRICHLET_ALPHA, (n_sel, 1)),
                n_draw=k,
            )
            non_totals[mask] = drawn

        # Assemble split metrics
        # train: all pilots + train_o non-pilots (first len(train_o) of non)
        i0 = 0
        i1 = len(train_o)
        i2 = i1 + len(val_o)
        i3 = i2 + len(test_o)
        assert i3 == n_non

        train_had = np.concatenate(
            [pilot_totals, non_totals[:, i0:i1, :]], axis=1
        )  # (n_mc, 20, 3)
        val_had = non_totals[:, i1:i2, :]
        test_had = non_totals[:, i2:i3, :]

        train_qual = ((train_had[:, :, 0] >= 2) & (train_had[:, :, 2] >= 2)).sum(axis=1)
        val_qual = ((val_had[:, :, 0] >= 2) & (val_had[:, :, 2] >= 2)).sum(axis=1)
        test_qual = ((test_had[:, :, 0] >= 2) & (test_had[:, :, 2] >= 2)).sum(axis=1)
        train_ge1 = ((train_had[:, :, 0] >= 1) & (train_had[:, :, 2] >= 1)).sum(axis=1)
        val_ge1 = ((val_had[:, :, 0] >= 1) & (val_had[:, :, 2] >= 1)).sum(axis=1)
        test_ge1 = ((test_had[:, :, 0] >= 1) & (test_had[:, :, 2] >= 1)).sum(axis=1)
        test_h = test_had[:, :, 0].sum(axis=1)
        test_d = test_had[:, :, 2].sum(axis=1)

        results["by_k"][str(k)] = _gate_row_from_arrays(
            k=k,
            train_qual=train_qual,
            val_qual=val_qual,
            test_qual=test_qual,
            test_h=test_h,
            test_d=test_d,
            train_ge1=train_ge1,
            val_ge1=val_ge1,
            test_ge1=test_ge1,
        )

    return results


def run_historical_sensitivity_forecast(
    *,
    historical_counts: dict[str, tuple[int, int, int]],
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    test_ids: Sequence[str],
    k_grid: Sequence[int] = K_GRID,
    n_mc: int = N_MC,
    seed: int = FORECAST_SEED,
) -> dict[str, Any]:
    """Historical GPT K=20 label sensitivity (labeled; not primary)."""
    if len(historical_counts) != N_ELIGIBLE:
        raise ValueError(f"expected {N_ELIGIBLE} historical count entries")
    train_ids_l = list(train_ids)
    val_ids_l = list(val_ids)
    test_ids_l = list(test_ids)

    results: dict[str, Any] = {
        "seed": seed,
        "n_mc": n_mc,
        "label": "historical_label_sensitivity",
        "by_k": {},
    }
    rng = np.random.default_rng(seed + 1)  # distinct stream from primary

    train_base = np.asarray(
        [historical_counts[p] for p in train_ids_l], dtype=np.float64
    )
    val_base = np.asarray([historical_counts[p] for p in val_ids_l], dtype=np.float64)
    test_base = np.asarray(
        [historical_counts[p] for p in test_ids_l], dtype=np.float64
    )

    for k in k_grid:
        k_draw = int(k)

        def _sim_split(base: np.ndarray, *, n_draw: int = k_draw) -> np.ndarray:
            n_p = base.shape[0]
            out = np.zeros((n_mc, n_p, 3), dtype=np.int64)
            for i in range(n_p):
                out[:, i, :] = _dirichlet_multinomial_batch(
                    rng,
                    np.tile(base[i] + DIRICHLET_ALPHA, (n_mc, 1)),
                    n_draw=n_draw,
                )
            return out

        train_had = _sim_split(train_base)
        val_had = _sim_split(val_base)
        test_had = _sim_split(test_base)

        train_qual = ((train_had[:, :, 0] >= 2) & (train_had[:, :, 2] >= 2)).sum(axis=1)
        val_qual = ((val_had[:, :, 0] >= 2) & (val_had[:, :, 2] >= 2)).sum(axis=1)
        test_qual = ((test_had[:, :, 0] >= 2) & (test_had[:, :, 2] >= 2)).sum(axis=1)
        test_h = test_had[:, :, 0].sum(axis=1)
        test_d = test_had[:, :, 2].sum(axis=1)

        results["by_k"][str(k)] = _gate_row_from_arrays(
            k=k,
            train_qual=train_qual,
            val_qual=val_qual,
            test_qual=test_qual,
            test_h=test_h,
            test_d=test_d,
        )
    return results


def assert_oof_stacking_no_leakage(
    *,
    fold_of: Sequence[int],
    base_pred_source_fold: Sequence[int],
) -> bool:
    """Cross-fitted stacking leakage check.

    `base_pred_source_fold[i]` is the fold held out when producing the base
    prediction for sample i. For OOF stacking this must equal `fold_of[i]`
    (prediction produced while sample i's fold was held out of the base fit).
    In-fold base predictions used for meta training constitute leakage.
    """
    if len(fold_of) != len(base_pred_source_fold):
        raise ValueError("length mismatch")
    return all(int(a) == int(b) for a, b in zip(fold_of, base_pred_source_fold, strict=True))


def select_k_from_forecasts(
    primary: dict[str, Any], historical: dict[str, Any]
) -> dict[str, Any]:
    """Smallest K≤20 meeting primary gates; check historical disagreement."""
    selected = None
    for k in K_GRID:
        if primary["by_k"][str(k)]["all_gates_pass"]:
            selected = k
            break
    if selected is None:
        return {
            "status": STATUS_K_NOT_SUPPORTED,
            "selected_k": None,
            "reason": "no_k_le_20_passes_primary_gates",
        }

    hist = historical["by_k"][str(selected)]
    # Disagreement: historical clearly fails any gate with P < 0.50
    disagree_details = []
    mapping = [
        ("train", "p_train_ge10_of_20_ge2", TRAIN_P_MIN),
        ("validation", "p_val_ge4_of_8_ge2", VAL_P_MIN),
        ("test_prompts", "p_test_ge5_of_11_ge2", TEST_P_MIN),
    ]
    for name, key, thr in mapping:
        p = hist[key]
        # "clearly fails" means P(success) < 0.50
        if p < HISTORICAL_DISAGREE_P:
            disagree_details.append(
                {"gate": name, "historical_p": p, "required_threshold": thr}
            )
    # Also class-count medians as hard fails in historical
    if hist["median_test_honest"] < TEST_MEDIAN_H_MIN:
        disagree_details.append(
            {
                "gate": "test_median_h",
                "historical_median": hist["median_test_honest"],
                "required": TEST_MEDIAN_H_MIN,
            }
        )
    if hist["median_test_deceptive"] < TEST_MEDIAN_D_MIN:
        disagree_details.append(
            {
                "gate": "test_median_d",
                "historical_median": hist["median_test_deceptive"],
                "required": TEST_MEDIAN_D_MIN,
            }
        )

    if disagree_details:
        return {
            "status": STATUS_FORECAST_DISAGREE,
            "selected_k_primary": selected,
            "selected_k": None,
            "disagreement": disagree_details,
            "primary_row": primary["by_k"][str(selected)],
            "historical_row": hist,
        }

    return {
        "status": STATUS_FROZEN,
        "selected_k": selected,
        "primary_row": primary["by_k"][str(selected)],
        "historical_row": hist,
        "disagreement": [],
    }


def project_generation_load(
    *,
    selected_k: int,
    n_pilot: int = N_PILOT_24B,
    n_other: int = N_ELIGIBLE - N_PILOT_24B,
) -> dict[str, Any]:
    if selected_k < N_OBSERVED_24B:
        raise ValueError("selected_k < observed 24B replicates")
    new_pilot = n_pilot * (selected_k - N_OBSERVED_24B)
    new_other = n_other * selected_k
    new_total = new_pilot + new_other
    reused = n_pilot * N_OBSERVED_24B
    return {
        "selected_k": selected_k,
        "reused_phase24b_trajectories": reused,
        "new_trajectories_pilot_tail": new_pilot,
        "new_trajectories_other_prompts": new_other,
        "new_trajectories_total": new_total,
        "total_trajectories_after": reused + new_total,
        "projected_new_bytes": new_total * BYTES_PER_TRAJ_24B,
        "projected_new_gpu_seconds": new_total * GPU_SEC_PER_TRAJ_24B,
        "projected_new_cost_usd": (new_total * GPU_SEC_PER_TRAJ_24B)
        / 3600.0
        * A100_USD_PER_HOUR,
    }


def delta_h(h_t: Sequence[float], h_0: Sequence[float]) -> list[float]:
    if len(h_t) != len(h_0):
        raise ValueError("Δh requires equal lengths")
    return [float(a) - float(b) for a, b in zip(h_t, h_0, strict=True)]


def prediction_step_surface_prefix_tokens(
    generated_token_ids: Sequence[int], prediction_step: int
) -> list[int]:
    """Tokens available as surface prefix before sampling token at step t: 0..t-1."""
    if prediction_step < 0:
        raise ValueError("prediction_step must be >= 0")
    if prediction_step > len(generated_token_ids):
        raise ValueError("prediction_step exceeds generated length")
    return list(generated_token_ids[:prediction_step])


def select_candidate_region(
    delta_auroc_by_time_layer: dict[int, dict[int, float]],
) -> dict[str, Any] | None:
    """Frozen VALIDATION candidate rule over times × layers."""
    times = sorted(TEMPORAL_LANDMARKS)
    for t in times:
        layer_map = delta_auroc_by_time_layer.get(t, {})
        # find contiguous bands of length >= 3 with Δ>0
        bands: list[tuple[int, int, float]] = []  # start, end inclusive, median
        start = None
        for L in range(N_LAYERS):
            d = float(layer_map.get(L, float("-inf")))
            if d > 0:
                if start is None:
                    start = L
            else:
                if start is not None:
                    end = L - 1
                    if end - start + 1 >= CANDIDATE_MIN_CONSEC_LAYERS:
                        vals = [float(layer_map[i]) for i in range(start, end + 1)]
                        med = float(np.median(vals))
                        if med >= CANDIDATE_MIN_MEDIAN_DELTA:
                            bands.append((start, end, med))
                    start = None
        if start is not None:
            end = N_LAYERS - 1
            if end - start + 1 >= CANDIDATE_MIN_CONSEC_LAYERS:
                vals = [float(layer_map[i]) for i in range(start, end + 1)]
                med = float(np.median(vals))
                if med >= CANDIDATE_MIN_MEDIAN_DELTA:
                    bands.append((start, end, med))
        if not bands:
            continue
        # highest median ΔAUROC; tie → lower start layer
        bands.sort(key=lambda b: (-b[2], b[0]))
        b0, b1, med = bands[0]
        # within band, single layer with highest Δ; tie → lower layer
        best_L = min(
            range(b0, b1 + 1),
            key=lambda L: (-float(layer_map[L]), L),
        )
        return {
            "candidate_time": t,
            "candidate_layer": best_L,
            "band": [b0, b1],
            "band_median_delta_auroc": med,
            "layer_delta_auroc": float(layer_map[best_L]),
        }
    return None


def prompt_group_folds(
    prompt_ids: Sequence[str], n_folds: int = 5
) -> list[list[str]]:
    """Deterministic prompt-grouped folds (no trajectory leakage across folds)."""
    ordered = sorted(prompt_ids)
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    for i, pid in enumerate(ordered):
        folds[i % n_folds].append(pid)
    return folds


def prompt_equal_balanced_weights(
    labels: Sequence[str], prompt_ids: Sequence[str]
) -> list[float]:
    """Balanced class weights × equal total weight per prompt."""
    if len(labels) != len(prompt_ids):
        raise ValueError("labels/prompt_ids length mismatch")
    from collections import Counter

    # class-balanced base
    classes = [y for y in labels if y in ("honest", "deceptive")]
    n = len(classes)
    if n == 0:
        return [0.0] * len(labels)
    cc = Counter(classes)
    class_w = {c: n / (len(cc) * cc[c]) for c in cc}
    # prompt totals of usable rows
    from collections import defaultdict

    per_prompt = defaultdict(int)
    for y, p in zip(labels, prompt_ids, strict=True):
        if y in ("honest", "deceptive"):
            per_prompt[p] += 1
    out: list[float] = []
    for y, p in zip(labels, prompt_ids, strict=True):
        if y not in ("honest", "deceptive"):
            out.append(0.0)
            continue
        pw = 1.0 / per_prompt[p]
        out.append(float(class_w[y] * pw))
    return out


def confirmatory_support(delta_auroc: float, ci_low: float, ci_high: float) -> bool:
    return delta_auroc > 0.0 and ci_low > 0.0


def locked_test_requires_frozen_candidate(candidate: dict[str, Any] | None) -> bool:
    if candidate is None:
        return False
    return "candidate_time" in candidate and "candidate_layer" in candidate


def cluster_bootstrap_indices(
    n_prompts: int, n_reps: int, seed: int
) -> np.ndarray:
    """Resample prompt indices with replacement (not trajectories)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_prompts, size=(n_reps, n_prompts))
