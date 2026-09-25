#!/usr/bin/env python3
"""Freeze Phase 3B1 surface-baseline hyperparameters on train/val only.

Must run before Modal activation extraction. Locked test is never used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.baselines import (  # noqa: E402
    TEXT_BASELINE_C_GRID,
    make_text_baseline_pipeline,
    prefix_structural_features,
    visible_prompt_plus_prefix_text,
)
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402
from pre_output_physiology.trajectory import (  # noqa: E402
    PREFIX_LENGTHS_K,
    analyze_prompt_response_boundary,
    eligible_for_k,
)

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _ids_hash(ids: list[str]) -> str:
    payload = "\n".join(sorted(ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _enrich(rows: list[dict], tokenizer) -> list[dict]:
    out = []
    for r in rows:
        tok = analyze_prompt_response_boundary(
            r["input_formatted"], r["model_outputs"], tokenizer
        )
        if not tok.prompt_prefix_exact or tok.boundary_token_straddle:
            raise SystemExit(
                f"Boundary failure for {r['example_id']}: "
                f"prefix_exact={tok.prompt_prefix_exact} "
                f"straddle={tok.boundary_token_straddle}"
            )
        rr = dict(r)
        rr["canonical_response_n_tokens"] = len(tok.response_suffix_ids)
        rr["_tok"] = tok
        out.append(rr)
    return out


def _eligible(rows: list[dict], k: int) -> list[dict]:
    return [r for r in rows if eligible_for_k(r["canonical_response_n_tokens"], k)]


def _visible(row: dict, k: int) -> str:
    return visible_prompt_plus_prefix_text(
        row["input_formatted"],
        row["model_outputs"],
        k,
        row["_tok"].response_offsets_in_model_outputs,
    )


def _eligibility_report(rows: list[dict], k: int) -> dict:
    elig = _eligible(rows, k)
    labels = Counter(int(r["eventual_deception"]) for r in elig)
    groups = {r["prompt_sha256"] for r in elig}
    return {
        "n_eligible": len(elig),
        "n_excluded_short": len(rows) - len(elig),
        "n_class1": labels.get(1, 0),
        "n_class0": labels.get(0, 0),
        "class1_prevalence": (labels.get(1, 0) / len(elig)) if elig else None,
        "n_prompt_groups": len(groups),
        "eligible_ids_sha256": _ids_hash([r["example_id"] for r in elig]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=str(REPO_ROOT / "data/processed/phase3_roleplay"),
    )
    parser.add_argument(
        "--out",
        default=str(
            REPO_ROOT / "artifacts/phase3b_dev/surface_baseline_freeze.json"
        ),
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, use_fast=True
    )
    train = _enrich(
        _load_jsonl(data_dir / "phase3_train_metadata.jsonl"), tokenizer
    )
    val = _enrich(
        _load_jsonl(data_dir / "phase3_validation_metadata.jsonl"), tokenizer
    )

    # Refuse to touch locked test for predictive work
    locked = data_dir / "locked_test_metadata.jsonl"
    if not locked.is_file():
        raise SystemExit("locked_test metadata missing (expected present but unused)")

    selected: dict[str, float] = {}
    candidate_aurocs: dict[str, dict[str, float]] = {}
    eligibility: dict[str, dict] = {}

    for k in PREFIX_LENGTHS_K:
        tr = _eligible(train, k)
        va = _eligible(val, k)
        eligibility[str(k)] = {
            "train": _eligibility_report(train, k),
            "validation": _eligibility_report(val, k),
        }
        if len(tr) < 10 or len(va) < 10:
            raise SystemExit(f"Too few eligible rows at k={k}")
        if len({r["eventual_deception"] for r in tr}) < 2:
            raise SystemExit(f"Train missing a class at k={k}")
        if len({r["eventual_deception"] for r in va}) < 2:
            raise SystemExit(f"Validation missing a class at k={k}")

        x_tr = [_visible(r, k) for r in tr]
        y_tr = [int(r["eventual_deception"]) for r in tr]
        x_va = [_visible(r, k) for r in va]
        y_va = [int(r["eventual_deception"]) for r in va]

        scores: dict[str, float] = {}
        for c in TEXT_BASELINE_C_GRID:
            pipe = make_text_baseline_pipeline(C=c, random_state=42)
            pipe.fit(x_tr, y_tr)
            proba = pipe.predict_proba(x_va)[:, 1]
            scores[str(c)] = float(roc_auc_score(y_va, proba))
        # highest AUROC; tie-break toward smaller C
        best_c = sorted(
            TEXT_BASELINE_C_GRID,
            key=lambda c: (-scores[str(c)], c),
        )[0]
        selected[str(k)] = float(best_c)
        candidate_aurocs[str(k)] = scores

    # Structural feature sanity (no tuning) — record only that contract is frozen
    _ = prefix_structural_features("x", 1)

    artifact = {
        "created_at": utc_now_iso(),
        "git_sha": _git_sha(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "locked_test_used": False,
        "C_grid": list(TEXT_BASELINE_C_GRID),
        "selected_C_by_k": selected,
        "validation_auroc_by_C": candidate_aurocs,
        "eligibility": eligibility,
        "prompt_only_uses_C_at_k0": selected["0"],
        "structural_lr_C": 1.0,
        "logit_summary_lr_C": 1.0,
        "selection_rule": (
            "max validation AUROC; ties broken toward smaller C; "
            "fit on eligible train only"
        ),
        "n_train_rows": len(train),
        "n_validation_rows": len(val),
        "train_ids_sha256": _ids_hash([r["example_id"] for r in train]),
        "validation_ids_sha256": _ids_hash([r["example_id"] for r in val]),
    }
    write_json(out_path, artifact)
    print(json.dumps({
        "out": str(out_path),
        "selected_C_by_k": selected,
        "locked_test_used": False,
        "eligibility_train": {k: v["train"]["n_eligible"] for k, v in eligibility.items()},
        "eligibility_val": {
            k: v["validation"]["n_eligible"] for k, v in eligibility.items()
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
