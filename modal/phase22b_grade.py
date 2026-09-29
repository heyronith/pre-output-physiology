"""Modal: Phase-22B honesty grading with gpt-4o-2024-08-06 (Apollo template).

Uses Modal secret `openai`. Retries unknown judgements per Apollo logic (max 4).
Checkpoints each shard to disk; resumes by skipping already-graded IDs.

Usage (clean tree; grading authorized):

    uv run modal run modal/phase22b_grade.py --run-id <phase22b_gen_...>
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

import modal

APP_NAME = "pre-output-physiology-phase22b-grade"
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/experiments/phase22b_k20_sampling_validation.yaml"
PROMPTS_PATH = REPO_ROOT / "data/processed/phase22b_k20/prompts.jsonl"
GRADING_PATH = REPO_ROOT / "data/raw/apollo_roleplaying/grading_template.txt"

SHARD_SIZE = 100
OPENAI_TIMEOUT_S = 60.0
PRINT_EVERY = 25

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("openai==1.59.6", "httpx==0.27.2", "pyyaml==6.0.2", "numpy==1.26.4")
    .add_local_python_source("pre_output_physiology")
)
app = modal.App(APP_NAME)


def _ckpt_dir(run_id: str) -> Path:
    return REPO_ROOT / "artifacts/runs" / f"phase22b_grade_ckpt_{run_id}"


def _load_checkpointed(ckpt: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not ckpt.is_dir():
        return out
    for path in sorted(ckpt.glob("shard_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            out[row["continuation_id"]] = row
    return out


def _local_setup(run_id: str) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    from pre_output_physiology.phase21_roleplay import (
        GRADER_MODEL,
        GRADING_CONTENT_SHA256,
        sha256_file,
    )
    from pre_output_physiology.phase22b_sampling import STATUS_GEN_DONE

    dirty = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise SystemExit(f"STOP: dirty:\n{dirty}")
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if cfg.get("status") != STATUS_GEN_DONE:
        raise SystemExit(f"requires {STATUS_GEN_DONE}, got {cfg.get('status')}")
    if cfg["authorizations"].get("grading_api_authorized") is not True:
        raise SystemExit("grading not authorized")
    if sha256_file(GRADING_PATH) != GRADING_CONTENT_SHA256:
        raise SystemExit("grading template SHA mismatch")
    run_dir = REPO_ROOT / "artifacts/runs" / run_id
    cont_path = run_dir / "continuations.jsonl"
    if not cont_path.is_file():
        raise SystemExit(f"missing {cont_path}")
    rows = [
        json.loads(x)
        for x in cont_path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    prompts = {
        json.loads(x)["prompt_id"]: json.loads(x)
        for x in PROMPTS_PATH.read_text(encoding="utf-8").splitlines()
        if x.strip()
    }
    template = GRADING_PATH.read_text(encoding="utf-8")
    payload = []
    for r in rows:
        p = prompts[r["prompt_id"]]
        payload.append(
            {
                "continuation_id": r["continuation_id"],
                "prompt_id": r["prompt_id"],
                "scenario": p["scenario"],
                "question": p["question"],
                "response": r["full_response"],
                "split": r["split"],
                "sample_seed": r["sample_seed"],
                "answer_prefix": r["answer_prefix"],
                "completion": r["completion"],
            }
        )
    meta = {
        "run_id": run_id,
        "grader_model": GRADER_MODEL,
        "n": len(payload),
        "git_commit": subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    return meta, payload, template


@app.function(
    image=image,
    timeout=60 * 60 * 2,
    secrets=[modal.Secret.from_name("openai")],
    memory=8192,
)
def grade_batch(payload_json: str, template: str) -> dict[str, Any]:
    from openai import OpenAI

    from pre_output_physiology.phase21_roleplay import (
        GRADER_MODEL,
        GRADER_SYSTEM,
        grading_user_prompt,
        label_from_judgement,
        parse_judgement,
    )

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY missing from Modal secret")

    client = OpenAI(timeout=OPENAI_TIMEOUT_S)
    try:
        client.models.retrieve(GRADER_MODEL)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"STOP: exact grader model {GRADER_MODEL} unavailable: {e}"
        ) from e

    rows = json.loads(payload_json)
    out: list[dict[str, Any]] = []
    t0 = time.time()
    for i, r in enumerate(rows):
        user = grading_user_prompt(
            template, r["scenario"], r["question"], r["response"]
        )
        judgement: int | str = "unknown"
        raw_text = ""
        last_err = None
        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model=GRADER_MODEL,
                    max_tokens=1000,
                    messages=[
                        {"role": "system", "content": GRADER_SYSTEM},
                        {"role": "user", "content": user},
                    ],
                    timeout=OPENAI_TIMEOUT_S,
                )
                raw_text = resp.choices[0].message.content or ""
                judgement, _ = parse_judgement(raw_text)
                if judgement != "unknown":
                    break
            except Exception as e:  # noqa: BLE001
                last_err = e
                delay = min(30.0, (2**attempt) + random.uniform(0, 0.2))
                print(
                    f"retry continuation={r['continuation_id']} "
                    f"attempt={attempt} err={type(e).__name__}: {e}",
                    flush=True,
                )
                time.sleep(delay)
                continue
            delay = min(30.0, (2**attempt) + random.uniform(0, 0.2))
            time.sleep(delay)
        label = label_from_judgement(judgement)
        out.append(
            {
                "continuation_id": r["continuation_id"],
                "prompt_id": r["prompt_id"],
                "split": r["split"],
                "sample_seed": r["sample_seed"],
                "answer_prefix": r["answer_prefix"],
                "completion": r["completion"],
                "full_response": r["response"],
                "judgement": judgement if judgement != "unknown" else None,
                "label": label if label != "exclude" else "exclude",
                "grader_raw": raw_text,
                "grader_model": GRADER_MODEL,
                "grader_error": str(last_err) if judgement == "unknown" else None,
            }
        )
        if (i + 1) % PRINT_EVERY == 0 or (i + 1) == len(rows):
            print(f"graded {i + 1}/{len(rows)}", flush=True)
    return {
        "rows": out,
        "wall_seconds": time.time() - t0,
        "n": len(out),
        "n_exclude": sum(1 for x in out if x["label"] == "exclude"),
    }


@app.local_entrypoint()
def main(run_id: str) -> None:
    meta, payload, template = _local_setup(run_id)
    ckpt = _ckpt_dir(run_id)
    ckpt.mkdir(parents=True, exist_ok=True)
    done = _load_checkpointed(ckpt)
    remaining = [r for r in payload if r["continuation_id"] not in done]
    print(
        f"Grading with {meta['grader_model']}: "
        f"total={len(payload)} done={len(done)} remaining={len(remaining)}",
        flush=True,
    )

    wall = 0.0
    existing = sorted(ckpt.glob("shard_*.jsonl"))
    shard_idx = len(existing)
    for start in range(0, len(remaining), SHARD_SIZE):
        shard = remaining[start : start + SHARD_SIZE]
        shard_path = ckpt / f"shard_{shard_idx:04d}.jsonl"
        print(
            f"shard {shard_idx} remaining_index {start}:{start + len(shard)} "
            f"n={len(shard)} -> {shard_path.name}",
            flush=True,
        )
        result = grade_batch.remote(json.dumps(shard), template)
        with shard_path.open("w", encoding="utf-8") as f:
            for row in result["rows"]:
                f.write(json.dumps(row, sort_keys=True) + "\n")
                done[row["continuation_id"]] = row
        wall += float(result["wall_seconds"])
        print(
            f"checkpointed {len(result['rows'])} "
            f"(cumulative_done={len(done)} wall_shard={result['wall_seconds']:.1f}s)",
            flush=True,
        )
        shard_idx += 1

    if len(done) != len(payload):
        missing = [r["continuation_id"] for r in payload if r["continuation_id"] not in done]
        raise SystemExit(
            f"STOP: incomplete grading {len(done)}/{len(payload)}; "
            f"missing e.g. {missing[:5]}"
        )

    all_rows = [done[r["continuation_id"]] for r in payload]
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    short = hashlib.sha256(meta["git_commit"].encode()).hexdigest()[:8]
    grade_run = f"phase22b_grade_{ts}_{short}"
    out_dir = REPO_ROOT / "artifacts/runs" / grade_run
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "graded.jsonl").open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    counts: dict[str, int] = {}
    for r in all_rows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    manifest = {
        "grade_run_id": grade_run,
        "source_gen_run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": meta["git_commit"],
        "grader_model": meta["grader_model"],
        "n": len(all_rows),
        "label_counts": counts,
        "wall_seconds": wall,
        "checkpoint_dir": str(ckpt.relative_to(REPO_ROOT)),
        "shard_size": SHARD_SIZE,
        "openai_timeout_s": OPENAI_TIMEOUT_S,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)
