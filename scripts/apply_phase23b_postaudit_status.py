#!/usr/bin/env python3
"""Apply Phase-23B post-audit status/provenance updates (zero model calls)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pre_output_physiology.phase23_open_grader import STATUS_23B_POSTAUDIT  # noqa: E402
from pre_output_physiology.provenance import utc_now_iso, write_json  # noqa: E402

OUT = REPO_ROOT / "artifacts/phase23b_development"
AUDIT = OUT / "disagreement_audit"

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
}


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    git_commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    man = json.loads((AUDIT / "manifest.json").read_text(encoding="utf-8"))

    # winner_selection
    ws = json.loads((OUT / "winner_selection.json").read_text(encoding="utf-8"))
    ws["status"] = STATUS_23B_POSTAUDIT
    ws["stage3_scientifically_eligible"] = True
    ws["stage3_locked_validation_authorized"] = False
    ws["stage3_authorized"] = False
    ws["postaudit_git_commit"] = git_commit
    ws["postaudit_at"] = utc_now_iso()
    ws["disagreement_audit"] = {
        "blinded_items_sha256": man["blinded_items_sha256"],
        "ab_mapping_sha256": man["ab_mapping_sha256"],
        "n": man["n"],
        "explanation_omitted_from_both": True,
    }
    ws["reason"] = (
        "sole numeric gate passer: gemma4_31b_it; label-only population is canonical "
        "primary; onset-gated retained as sensitivity; disagreement audit re-blinded; "
        "Stage 23C scientifically eligible but unauthorized"
    )
    write_json(OUT / "winner_selection.json", ws)

    # summary (large): patch key fields only
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    summary["status"] = STATUS_23B_POSTAUDIT
    summary["postaudit_git_commit"] = git_commit
    summary["postaudit_at"] = utc_now_iso()
    summary["winner_selection"] = ws
    summary["disagreement_audit"] = {
        "n": man["n"],
        "path": "artifacts/phase23b_development/disagreement_audit",
        "blinded_items_sha256": man["blinded_items_sha256"],
        "ab_mapping_sha256": man["ab_mapping_sha256"],
        "blinding": man["blinding"],
        "note": man["note"],
    }
    summary["authorizations_after_freeze"] = dict(AUTH_FALSE)
    summary["stage3"] = {
        "scientifically_eligible": True,
        "authorized": False,
        "proposed_winner": "gemma4_31b_it",
    }
    write_json(OUT / "summary.json", summary)

    # freeze
    freeze = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    freeze["status"] = STATUS_23B_POSTAUDIT
    freeze["postaudit_git_commit"] = git_commit
    freeze["postaudit_at"] = utc_now_iso()
    freeze["stage3_locked_validation_authorized"] = False
    freeze["stage3_scientifically_eligible"] = True
    freeze["stage3"] = {
        "scientifically_eligible": True,
        "authorized": False,
        "proposed_winner": "gemma4_31b_it",
    }
    freeze["authorizations_after_freeze"] = dict(AUTH_FALSE)
    freeze["population_metric_primary"] = "label_only_mixed_behavior_population"
    freeze["population_metric_secondary"] = "reference_onset_gated_sensitivity_analysis"
    freeze["winner"] = "gemma4_31b_it"
    freeze["winner_reason"] = ws["reason"]
    freeze["disagreement_audit"] = {
        "blinded_items_sha256": man["blinded_items_sha256"],
        "ab_mapping_sha256": man["ab_mapping_sha256"],
        "n": man["n"],
        "explanation_omitted_from_both": True,
        "sample_preserved": True,
    }
    # Clarify legacy top-level onset-gated fields are secondary
    for cand in ("gemma4_31b_it", "qwen35_27b"):
        if cand in freeze:
            freeze[cand]["population_jaccard_legacy_onset_gated"] = freeze[cand].get(
                "population_jaccard"
            )
            freeze[cand]["n_status_changed_legacy_onset_gated"] = freeze[cand].get(
                "n_status_changed"
            )
            # Point primary counts at label-only
            lo = freeze[cand]["population_label_only"]
            freeze[cand]["population_jaccard_primary_label_only"] = lo["jaccard"]
            freeze[cand]["n_status_changed_primary_label_only"] = lo["n_status_changed"]
    freeze["winner_selection"] = ws
    write_json(OUT / "freeze.json", freeze)

    print(
        json.dumps(
            {
                "status": STATUS_23B_POSTAUDIT,
                "blinded_items_sha256": man["blinded_items_sha256"],
                "ab_mapping_sha256": man["ab_mapping_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
