#!/usr/bin/env python3
"""Render Phase 26B production prompts (deterministic; no model inference)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pre_output_physiology.phase26b_render import (
    PARENT_PHASE26A_COMMIT,
    PROTOCOL_VERSION,
    build_template_assignment,
    load_yaml,
    read_jsonl,
    render_all_prompts,
    write_jsonl,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()
    root = Path(args.repo_root)

    blueprints = read_jsonl(root / "data/phase26/scenario_blueprints.jsonl")
    consequences = load_yaml(root / "configs/phase26_consequences.yaml")
    assignment = build_template_assignment(consequences)
    prompts = render_all_prompts(
        blueprints=blueprints,
        consequence_cfg=consequences,
        assignment=assignment,
    )

    out_assign = root / "data/phase26/template_assignment_v1.json"
    out_prompts = root / "data/phase26/production_prompts_v1.jsonl"
    out_assign.parent.mkdir(parents=True, exist_ok=True)
    out_assign.write_text(
        json.dumps(assignment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_jsonl(out_prompts, prompts)

    print(
        f"rendered {len(prompts)} prompts @ {PROTOCOL_VERSION}; "
        f"parent={PARENT_PHASE26A_COMMIT}"
    )
    print(f"wrote {out_prompts}")
    print(f"wrote {out_assign}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
