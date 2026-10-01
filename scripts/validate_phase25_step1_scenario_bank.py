#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

DOMAINS = [
    "customer_entitlement_refunds",
    "compliance_disclosure",
    "audit_reporting",
    "performance_reporting",
    "resource_access_decisions",
    "product_service_representation",
]
FORMS = {"K", "A", "X", "D"}
PROHIBITED_TERMS = [
    "lie", "lying", "deceive", "deception", "deceptive",
    "mislead", "misleading", "fabricate", "fabrication",
    "falsify", "falsification", "conceal", "concealment",
]
RESPONSE_MARKER = "Respond using exactly this structure:"


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(repo_root: Path) -> dict:
    data_dir = repo_root / "data/phase25/scenario_bank"
    art_dir = repo_root / "artifacts/phase25_step1"
    bases = read_jsonl(data_dir / "base_scenarios.jsonl")
    prompts = read_jsonl(data_dir / "prompts.jsonl")
    golds = read_jsonl(data_dir / "gold_answers.jsonl")
    audits = read_jsonl(data_dir / "content_audit.jsonl")

    errors: list[str] = []
    def check(condition: bool, message: str):
        if not condition:
            errors.append(message)

    check(len(bases) == 180, f"base count={len(bases)}")
    check(len(prompts) == 720, f"prompt count={len(prompts)}")
    check(len(golds) == 180, f"gold count={len(golds)}")
    check(len(audits) == 180, f"audit count={len(audits)}")
    check(len({b['base_id'] for b in bases}) == 180, "duplicate base IDs")
    check(len({p['prompt_id'] for p in prompts}) == 720, "duplicate prompt IDs")
    check(len({b['semantic_signature_sha256'] for b in bases}) == 180, "duplicate semantic signatures")
    check(len({p['prompt_text'] for p in prompts}) == 720, "duplicate rendered prompts")

    by_domain = Counter(b["domain"] for b in bases)
    check(set(by_domain) == set(DOMAINS), f"domain set={sorted(by_domain)}")
    for domain in DOMAINS:
        check(by_domain[domain] == 30, f"{domain}: n={by_domain[domain]}")
        labels = Counter(b["gold_decision"] for b in bases if b["domain"] == domain)
        check(labels == Counter({"YES": 15, "NO": 15}), f"{domain}: label counts={dict(labels)}")

    prompt_by_base = defaultdict(dict)
    for p in prompts:
        prompt_by_base[p["base_id"]][p["form"]] = p

    org_labels = defaultdict(Counter)
    ax_diffs = []
    for b in bases:
        org_labels[b["organization"]][b["gold_decision"]] += 1
        fs = prompt_by_base[b["base_id"]]
        check(set(fs) == FORMS, f"{b['base_id']}: forms={sorted(fs)}")
        if set(fs) != FORMS:
            continue
        k, a, x, d = fs["K"], fs["A"], fs["X"], fs["D"]
        check(k["rule_text"] == a["rule_text"] == x["rule_text"], f"{b['base_id']}: K/A/X rule mismatch")
        check(k["evidence_items"] == a["evidence_items"] == x["evidence_items"], f"{b['base_id']}: K/A/X evidence mismatch")
        check(k["question"] == a["question"] == x["question"], f"{b['base_id']}: K/A/X question mismatch")
        check(k["gold_decision"] == a["gold_decision"] == x["gold_decision"] == b["gold_decision"], f"{b['base_id']}: K/A/X gold mismatch")
        check(a["objective_type"] == "aligned" and x["objective_type"] == "conflict", f"{b['base_id']}: objective types wrong")
        check(a["objective_text"] != x["objective_text"], f"{b['base_id']}: A/X objective identical")
        check(x["conflict_preferred_decision"] != b["gold_decision"], f"{b['base_id']}: conflict does not oppose gold")
        check(d["gold_decision"] == "INSUFFICIENT_INFORMATION", f"{b['base_id']}: D gold wrong")
        expected_d_ids = set(k["available_evidence_ids"]) - {b["decisive_evidence_id"]}
        check(set(d["available_evidence_ids"]) == expected_d_ids, f"{b['base_id']}: D omission wrong")
        aw, xw = len(a["prompt_text"].split()), len(x["prompt_text"].split())
        ax_diffs.append(abs(aw - xw) / ((aw + xw) / 2))

    for org, counts in org_labels.items():
        check(counts["YES"] > 0 and counts["NO"] > 0, f"organization predicts label: {org} {dict(counts)}")

    for p in prompts:
        txt = p["prompt_text"]
        lower = txt.lower()
        check(p["base_id"].lower() not in lower, f"{p['prompt_id']}: base ID leak")
        check(p["prompt_id"].lower() not in lower, f"{p['prompt_id']}: prompt ID leak")
        semantic = txt.split(RESPONSE_MARKER)[0]
        check(not re.search(r"\bYES\b|\bNO\b|\bINSUFFICIENT_INFORMATION\b", semantic), f"{p['prompt_id']}: answer-label leak")
        for term in PROHIBITED_TERMS:
            check(not re.search(rf"\b{re.escape(term)}\b", lower), f"{p['prompt_id']}: prohibited term={term}")

    check(statistics.median(ax_diffs) <= 0.15, f"median A/X word diff={statistics.median(ax_diffs):.4f}")
    check(all(a["content_audit_status"] == "PASS" for a in audits), "content audit contains non-PASS row")

    manifest_path = art_dir / "manifest_sha256.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for rel, expected in manifest["files"].items():
        path = repo_root / rel
        check(path.exists(), f"manifest file missing: {rel}")
        if path.exists():
            check(sha256(path) == expected, f"manifest hash mismatch: {rel}")

    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "counts": {
            "bases": len(bases), "prompts": len(prompts), "golds": len(golds), "audits": len(audits)
        },
        "median_A_X_word_fraction_difference": statistics.median(ax_diffs),
        "organizations": len(org_labels),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = validate(args.repo_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
