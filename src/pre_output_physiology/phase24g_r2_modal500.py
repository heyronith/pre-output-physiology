"""Phase 24G-R2 — Modal 500-rep full-pipeline corrected diagnostics.

Prospective amendment (recorded before corrected results observed):
The exploratory Phase-24G-R nested resampling target is amended from 1,000 to
500 full-pipeline repetitions solely for computational feasibility. No result
from the corrected nested analysis has yet been observed. All other analysis
rules remain unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pre_output_physiology.phase24c_design import sha256_file
from pre_output_physiology.phase24g_diagnostics import (
    PHASE24F_PRIMARY_HASH,
    classify_diagnostic_category,
)
from pre_output_physiology.phase24g_r_correction import (
    PRIOR_24G_ARTIFACTS,
    assert_no_proxy_in_primary,
    assert_phase24f_immutable,
    category_change_statement,
    compare_proxy_vs_full,
    summarize_lopo,
    summarize_nested,
    verify_prior_24g_unchanged,
)

STARTING_SHA = "be8bbdd8012ed0869d2b4b32c0000a417e45e0b7"
ANALYSIS_SEED = 2408
NESTED_REPS = 500  # prospectively amended from 1000 — do not reduce further
LOPO_N = 28
PROXY_COMPARE_NESTED = 50
N_WORKERS = 8

STATUS = "phase24g_r2_modal500_corrected_diagnostics_complete_awaiting_code_audit"
STATUS_PARTIAL = "phase24g_r2_modal500_running_or_incomplete"

VOLUME_NAME = "pre-output-physiology-phase24g-r2"
APP_NAME = "pre-output-physiology-phase24g-r2-modal500"
MODAL_CPU = 16
MODAL_MEMORY_MIB = 131072  # 128 GiB

AMENDMENT = (
    "The exploratory Phase-24G-R nested resampling target is amended from "
    "1,000 to 500 full-pipeline repetitions solely for computational "
    "feasibility. No result from the corrected nested analysis has yet been "
    "observed. All other analysis rules remain unchanged."
)

GUARANTEE = (
    "PHASE 24G-R2 USED A PROSPECTIVELY AMENDED 500-REPETITION FULL-PIPELINE "
    "NESTED DIAGNOSTIC ON MODAL CPU. THE AMENDMENT WAS MADE FOR COMPUTATIONAL "
    "FEASIBILITY BEFORE CORRECTED RESULTS WERE OBSERVED. PHASE-24F REMAINS "
    "THE FINAL CONFIRMATORY RESULT. NO NEW MODEL DATA, LABEL CHANGES, PROXY "
    "PRIMARY ANALYSIS, SAE ANALYSIS, CAUSAL INTERVENTION, OR PRIMARY-ENDPOINT "
    "REDEFINITION WAS PERFORMED."
)

NESTED_RESULT_SCHEMA = (
    "rep",
    "rep_id",
    "seed",
    "proxy",
    "candidate",
    "discovery_delta",
    "heldout_delta",
    "optimism",
    "split",
)

LOPO_RESULT_SCHEMA = (
    "left_out_prompt",
    "left_out_split",
    "candidate",
    "delta_auroc",
    "exact_t1_l20",
    "in_neighborhood",
    "proxy",
)


def assert_nested_reps_exact() -> None:
    if NESTED_REPS != 500:
        raise RuntimeError(
            f"NESTED_REPS must be exactly 500 for Phase 24G-R2; got {NESTED_REPS}"
        )


def rep_id(rep: int) -> str:
    if not (0 <= int(rep) < NESTED_REPS):
        raise ValueError(f"rep out of range: {rep}")
    return f"{int(rep):03d}"


def seed_for_rep(rep: int, master_seed: int = ANALYSIS_SEED) -> int:
    return int(master_seed) + int(rep)


def nested_path(root: Path, rep: int) -> Path:
    return root / "nested" / f"rep_{rep_id(rep)}.json"


def lopo_path(root: Path, left_out: str, *, proxy: bool = False) -> Path:
    tag = "proxy" if proxy else "full"
    safe = left_out.replace("/", "_")
    return root / "lopo" / tag / f"lopo_{safe}.json"


def atomic_write_json(path: Path, payload: dict[str, Any]) -> str:
    """Write JSON atomically (tmp → fsync → rename). Returns sha256 of bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    # Best-effort directory fsync for durability on POSIX
    try:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass
    return digest


def validate_nested_artifact(blob: dict[str, Any]) -> None:
    for k in NESTED_RESULT_SCHEMA:
        if k not in blob:
            raise ValueError(f"nested artifact missing key: {k}")
    if blob.get("proxy") is not False:
        raise ValueError("primary nested artifact must have proxy=False")
    rid = blob["rep_id"]
    if rid != rep_id(int(blob["rep"])):
        raise ValueError(f"rep_id mismatch: {rid}")
    if int(blob["seed"]) != seed_for_rep(int(blob["rep"])):
        raise ValueError("seed mismatch for rep")


def validate_lopo_artifact(blob: dict[str, Any], *, require_full: bool = True) -> None:
    for k in LOPO_RESULT_SCHEMA:
        if k not in blob:
            raise ValueError(f"lopo artifact missing key: {k}")
    if require_full and blob.get("proxy") is not False:
        raise ValueError("primary LOPO artifact must have proxy=False")


def scan_completed_nested(root: Path) -> dict[str, Any]:
    """Return completed/invalid/missing nested rep IDs under root/nested."""
    completed: list[int] = []
    invalid: list[dict[str, Any]] = []
    for rep in range(NESTED_REPS):
        path = nested_path(root, rep)
        if not path.exists():
            continue
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
            validate_nested_artifact(blob)
            completed.append(rep)
        except (OSError, ValueError, json.JSONDecodeError, TypeError, KeyError) as exc:
            invalid.append({"rep": rep, "path": str(path), "error": str(exc)})
    missing = [r for r in range(NESTED_REPS) if r not in set(completed)]
    return {
        "completed": completed,
        "missing": missing,
        "invalid": invalid,
        "n_completed": len(completed),
        "n_missing": len(missing),
        "n_requested": NESTED_REPS,
    }


def scan_completed_lopo(root: Path, prompts: list[str], *, proxy: bool = False) -> dict[str, Any]:
    completed: list[str] = []
    invalid: list[dict[str, Any]] = []
    for left in prompts:
        path = lopo_path(root, left, proxy=proxy)
        if not path.exists():
            continue
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
            validate_lopo_artifact(blob, require_full=not proxy)
            if blob.get("left_out_prompt") != left:
                raise ValueError("left_out_prompt mismatch")
            completed.append(left)
        except (OSError, ValueError, json.JSONDecodeError, TypeError, KeyError) as exc:
            invalid.append({"left": left, "path": str(path), "error": str(exc)})
    missing = [p for p in prompts if p not in set(completed)]
    return {
        "completed": completed,
        "missing": missing,
        "invalid": invalid,
        "n_completed": len(completed),
        "n_missing": len(missing),
        "n_requested": len(prompts),
        "proxy": proxy,
    }


def refuse_overwrite_completed(path: Path) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite completed artifact: {path}")


def progress_manifest(
    *,
    code_commit: str,
    config_hash: str,
    nested_scan: dict[str, Any],
    lopo_scan: dict[str, Any],
    failed: list[dict[str, Any]],
    running: list[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from pre_output_physiology.provenance import utc_now_iso

    out = {
        "created_at": utc_now_iso(),
        "code_commit": code_commit,
        "config_hash": config_hash,
        "nested_reps_requested": NESTED_REPS,
        "nested_completed": nested_scan.get("completed", []),
        "nested_n_completed": nested_scan.get("n_completed", 0),
        "nested_missing": nested_scan.get("missing", []),
        "nested_invalid": nested_scan.get("invalid", []),
        "lopo_completed": lopo_scan.get("completed", []),
        "lopo_n_completed": lopo_scan.get("n_completed", 0),
        "lopo_missing": lopo_scan.get("missing", []),
        "failed_reps": failed,
        "currently_running_ids": running,
        "amendment": AMENDMENT,
        "n_workers": N_WORKERS,
        "analysis_seed": ANALYSIS_SEED,
        "status": (
            STATUS
            if nested_scan.get("n_completed") == NESTED_REPS
            and lopo_scan.get("n_completed") == LOPO_N
            else STATUS_PARTIAL
        ),
    }
    if extra:
        out.update(extra)
    return out


def load_all_nested_rows(root: Path) -> list[dict[str, Any]]:
    scan = scan_completed_nested(root)
    if scan["n_completed"] != NESTED_REPS:
        raise RuntimeError(
            f"cannot aggregate: nested {scan['n_completed']}/{NESTED_REPS}"
        )
    if scan["invalid"]:
        raise RuntimeError(f"invalid nested artifacts: {scan['invalid']}")
    rows = []
    for rep in range(NESTED_REPS):
        blob = json.loads(nested_path(root, rep).read_text(encoding="utf-8"))
        validate_nested_artifact(blob)
        rows.append(blob)
    return rows


def load_all_lopo_rows(
    root: Path, prompts: list[str], *, proxy: bool = False
) -> list[dict[str, Any]]:
    scan = scan_completed_lopo(root, prompts, proxy=proxy)
    if scan["n_completed"] != len(prompts):
        raise RuntimeError(
            f"cannot aggregate LOPO: {scan['n_completed']}/{len(prompts)} proxy={proxy}"
        )
    rows = []
    for left in sorted(prompts):
        blob = json.loads(lopo_path(root, left, proxy=proxy).read_text(encoding="utf-8"))
        validate_lopo_artifact(blob, require_full=not proxy)
        rows.append(blob)
    return rows


def verify_prior_checkpoint_full_pipeline(path: Path) -> list[dict[str, Any]]:
    """Accept prior 4/28 rows only if proxy=False and schema-valid."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        blob = json.loads(line)
        validate_lopo_artifact(blob, require_full=True)
        rows.append(blob)
    return rows


def config_payload() -> dict[str, Any]:
    return {
        "experiment_id": "phase24g_r2_modal500",
        "starting_sha": STARTING_SHA,
        "nested_reps": NESTED_REPS,
        "lopo_n": LOPO_N,
        "analysis_seed": ANALYSIS_SEED,
        "n_workers": N_WORKERS,
        "modal_cpu": MODAL_CPU,
        "modal_memory_mib": MODAL_MEMORY_MIB,
        "volume_name": VOLUME_NAME,
        "app_name": APP_NAME,
        "amendment": AMENDMENT,
        "proxy_used_in_primary": False,
        "status_target": STATUS,
    }


def config_hash(payload: dict[str, Any] | None = None) -> str:
    blob = payload or config_payload()
    raw = json.dumps(blob, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def collect_input_hashes(repo_root: Path) -> dict[str, Any]:
    """Hashes of frozen inputs required on Modal."""
    files = {
        "phase24f_primary_metrics": repo_root
        / "artifacts/phase24f_confirmation/primary_metrics.json",
        "split_manifest": repo_root / "artifacts/phase24c_design/split_manifest.json",
        "labels_train": repo_root
        / "artifacts/phase24d_collection/by_split/TRAIN/labels.json",
        "labels_validation": repo_root
        / "artifacts/phase24d_collection/by_split/VALIDATION/labels.json",
        "capture_meta": repo_root / "artifacts/phase24d_collection/capture_meta.json",
    }
    out: dict[str, Any] = {"files": {}, "phase24g": {}, "phase24g_r": {}}
    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(str(path))
        out["files"][name] = {
            "path": str(path.relative_to(repo_root)),
            "sha256": sha256_file(str(path)),
        }
    # Phase-24F primary must match frozen constant
    if out["files"]["phase24f_primary_metrics"]["sha256"] != PHASE24F_PRIMARY_HASH:
        raise RuntimeError("Phase-24F primary hash mismatch at collection")
    g_root = repo_root / "artifacts/phase24g_diagnostics"
    for name in PRIOR_24G_ARTIFACTS:
        p = g_root / name
        if p.exists():
            out["phase24g"][name] = sha256_file(str(p))
    r_root = repo_root / "artifacts/phase24g_r_correction"
    for p in sorted(r_root.glob("*")):
        if p.is_file():
            out["phase24g_r"][p.name] = sha256_file(str(p))
    # Activation NPZ inventory (sha of sorted tid list + per-file digests is heavy;
    # record count + aggregate digest of paths+sizes for STOP-on-mismatch).
    act_root = repo_root / "artifacts/phase24e_discovery/activations_local"
    npz_rows = []
    for split in ("TRAIN", "VALIDATION"):
        for p in sorted((act_root / split).glob("*.npz")):
            npz_rows.append(f"{split}/{p.name}:{p.stat().st_size}")
    out["activations"] = {
        "n_npz": len(npz_rows),
        "inventory_sha256": hashlib.sha256("\n".join(npz_rows).encode()).hexdigest(),
    }
    if out["activations"]["n_npz"] < 400:
        raise RuntimeError("activation inventory unexpectedly small")
    return out


def verify_input_hashes(expected: dict[str, Any], repo_root: Path) -> None:
    got = collect_input_hashes(repo_root)
    for section in ("files", "activations"):
        if got[section] != expected[section]:
            raise RuntimeError(
                f"STOP: input hash mismatch in {section}: "
                f"expected={expected[section]} got={got[section]}"
            )
    # phase24g / phase24g_r: require all expected keys present and equal
    for section in ("phase24g", "phase24g_r"):
        for k, v in expected.get(section, {}).items():
            if got.get(section, {}).get(k) != v:
                raise RuntimeError(f"STOP: hash mismatch {section}/{k}")


# Re-exports for convenience
__all__ = [
    "AMENDMENT",
    "ANALYSIS_SEED",
    "APP_NAME",
    "GUARANTEE",
    "LOPO_N",
    "MODAL_CPU",
    "MODAL_MEMORY_MIB",
    "NESTED_REPS",
    "N_WORKERS",
    "PROXY_COMPARE_NESTED",
    "STARTING_SHA",
    "STATUS",
    "VOLUME_NAME",
    "assert_nested_reps_exact",
    "assert_no_proxy_in_primary",
    "assert_phase24f_immutable",
    "atomic_write_json",
    "category_change_statement",
    "classify_diagnostic_category",
    "collect_input_hashes",
    "compare_proxy_vs_full",
    "config_hash",
    "config_payload",
    "load_all_lopo_rows",
    "load_all_nested_rows",
    "lopo_path",
    "nested_path",
    "progress_manifest",
    "refuse_overwrite_completed",
    "rep_id",
    "scan_completed_lopo",
    "scan_completed_nested",
    "seed_for_rep",
    "summarize_lopo",
    "summarize_nested",
    "validate_lopo_artifact",
    "validate_nested_artifact",
    "verify_input_hashes",
    "verify_prior_24g_unchanged",
    "verify_prior_checkpoint_full_pipeline",
]
