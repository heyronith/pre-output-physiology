"""Modal: Phase-21 honesty grading with gpt-4o-2024-08-06 (Apollo template).

Uses Modal secret `openai`. Retries unknown judgements per Apollo logic (max 4).
Does not see activations, seeds, or split labels beyond what's needed for I/O.

Usage (clean tree; grading authorized):

    uv run modal run modal/phase21_grade.py --run-id <phase21_gen_...>
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

APP_NAME = "pre-output-physiology-phase21-grade"
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/experiments/phase21_roleplay_preoutput_physiology.yaml"
PROMPTS_PATH = REPO_ROOT / "data/processed/phase21_roleplay/prompts.jsonl"
GRADING_PATH = REPO_ROOT / "data/raw/apollo_roleplaying/grading_template.txt"
MATRIX_PATH = REPO_ROOT / "artifacts/phase21a_design/design_matrix.json"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("openai==1.51.0", "pyyaml==6.0.2", "numpy==1.26.4")
    .add_local_python_source("pre_output_physiology")
)
app = modal.App(APP_NAME)


def _local_setup(run_id: str) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    from pre_output_physiology.phase21_roleplay import (
        GRADER_MODEL,
        GRADING_CONTENT_SHA256,
        STATUS_GEN_DONE,
        sha256_file,
    )

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
    timeout=60 * 60 * 6,
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

    # Verify exact model id is available — do not silently substitute.
    client = OpenAI()
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
                )
                raw_text = resp.choices[0].message.content or ""
                judgement, _ = parse_judgement(raw_text)
                if judgement != "unknown":
                    break
            except Exception as e:  # noqa: BLE001
                last_err = e
                delay = (2**attempt) + random.uniform(0, 0.2)
                time.sleep(delay)
                continue
            # unknown judgement — retry
            delay = (2**attempt) + random.uniform(0, 0.2)
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
        if (i + 1) % 100 == 0:
            print(f"graded {i + 1}/{len(rows)}")
    return {
        "rows": out,
        "wall_seconds": time.time() - t0,
        "n": len(out),
        "n_exclude": sum(1 for x in out if x["label"] == "exclude"),
    }


@app.local_entrypoint()
def main(run_id: str) -> None:
    meta, payload, template = _local_setup(run_id)
    print(f"Grading n={len(payload)} with {meta['grader_model']}")
    # Shard to avoid huge single payloads / timeouts
    shard_size = 500
    all_rows: list[dict[str, Any]] = []
    wall = 0.0
    for start in range(0, len(payload), shard_size):
        shard = payload[start : start + shard_size]
        print(f"shard {start}:{start + len(shard)}")
        result = grade_batch.remote(json.dumps(shard), template)
        all_rows.extend(result["rows"])
        wall += float(result["wall_seconds"])

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    short = hashlib.sha256(meta["git_commit"].encode()).hexdigest()[:8]
    grade_run = f"phase21_grade_{ts}_{short}"
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
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2))
