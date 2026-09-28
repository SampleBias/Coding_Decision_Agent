"""Merge all case files, cap and balance per source, split into train/val/test without leakage,
and (optionally) push the dataset to the Hub.

Split rules
  * A case with `split_hint` (official split of its source) keeps it (test stays test).
  * Otherwise the split is a deterministic hash of (source, group): every case sharing a repo,
    an AgentJudgeBench record, an instruction, ... lands in the same split.
Balance rules
  * `--max-per-source` caps every source (override one with `--cap <substring>=<N>`).
  * Per (source, family) the majority label of a *key question* is capped to
    `--max-majority` of that slice by random subsampling (train only).

Usage:
    python data/merge_and_split.py --in "artifacts/cases_teacher/*.jsonl" --out artifacts/dataset \
        [--push S4MPL3BI4S/coding-decision-cases]
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root

import argparse
import glob
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from data.common import read_jsonl, summarize_cases, write_jsonl
from configs.schema import validate_case

KEY_QUESTION = {
    "code_review": ["likely_correct", "code_quality", "merge_action"],
    "tool_call": ["should_call_tool", "call_verdict", "tool_selection"],
    "agent_trace": ["action"],
    "routing": ["model_tier"],
}


def split_of(case: dict, val_frac: float, test_frac: float) -> str:
    hint = case.get("split_hint")
    if hint in ("train", "val", "test"):
        return hint
    h = hashlib.sha1(f"{case['source']}|{case['group']}".encode()).hexdigest()
    u = int(h[:12], 16) / 16**12
    if u < test_frac:
        return "test"
    if u < test_frac + val_frac:
        return "val"
    return "train"


def key_label(case: dict) -> str:
    for qid in KEY_QUESTION.get(case["family"], []):
        if qid in case["gold"]:
            return case["gold"][qid]["label"]
    return "_"


def source_family_key(case: dict) -> str:
    return f"{case['source'].split('/')[0]}/{case['source'].split('/')[1].split('/')[0]}|{case['family']}"


def balance(cases: List[dict], max_majority: float, rng: random.Random) -> List[dict]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for c in cases:
        groups[source_family_key(c)].append(c)
    out: List[dict] = []
    for key, items in groups.items():
        by_label: Dict[str, List[dict]] = defaultdict(list)
        for c in items:
            by_label[key_label(c)].append(c)
        if len(by_label) <= 1:
            out.extend(items)
            continue
        total = len(items)
        maj_label, maj_items = max(by_label.items(), key=lambda kv: len(kv[1]))
        rest = total - len(maj_items)
        allowed = int(max_majority / (1 - max_majority) * rest)
        if len(maj_items) > allowed:
            rng.shuffle(maj_items)
            print(f"  balance {key}: majority '{maj_label}' {len(maj_items)} -> {allowed}")
            maj_items = maj_items[:allowed]
        for lbl, its in by_label.items():
            out.extend(maj_items if lbl == maj_label else its)
    return out


def cap_sources(cases: List[dict], default_cap: int, caps: Dict[str, int], rng: random.Random) -> List[dict]:
    by_src: Dict[str, List[dict]] = defaultdict(list)
    for c in cases:
        by_src[c["source"].split("/")[0] + "/" + c["source"].split("/")[1]].append(c)
    out = []
    for src, items in by_src.items():
        cap = default_cap
        for sub, n in caps.items():
            if sub.lower() in src.lower():
                cap = n
        if len(items) > cap:
            # cap per group rather than dropping whole groups: shuffle then take
            rng.shuffle(items)
            print(f"  cap {src}: {len(items)} -> {cap}")
            items = items[:cap]
        out.extend(items)
    return out


def to_hub_rows(cases: List[dict], split: str) -> List[dict]:
    return [{
        "id": c["case_id"], "family": c["family"], "source": c["source"], "group": c["group"], "split": split,
        "state": json.dumps(c["state"], ensure_ascii=False),
        "questions": json.dumps(c["questions"], ensure_ascii=False),
        "gold": json.dumps(c["gold"], ensure_ascii=False),
        "meta": json.dumps(c.get("meta", {}), ensure_ascii=False),
    } for c in cases]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inputs", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--max-per-source", type=int, default=30000)
    ap.add_argument("--cap", action="append", default=[], help="<source substring>=<N>, repeatable")
    ap.add_argument("--max-majority", type=float, default=0.60)
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--push", type=str, default=None, help="HF dataset repo id to push to")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    files: List[str] = []
    for pat in args.inputs:
        files.extend(sorted(glob.glob(pat)) or [pat])
    files = [f for f in files if not Path(f).name.startswith("_")]  # skip caches / scratch files
    cases: List[dict] = []
    seen = set()
    for f in files:
        for c in read_jsonl(f):
            if c["needs_teacher"]:
                raise SystemExit(f"{f}: case {c['case_id']} still needs teacher labels; run teacher_vllm.py first")
            validate_case(c)
            if c["case_id"] in seen:
                continue
            seen.add(c["case_id"])
            cases.append(c)
    print(f"loaded {len(cases)} unique cases from {len(files)} files")

    caps = {}
    for spec in args.cap:
        k, v = spec.split("=")
        caps[k] = int(v)
    cases = cap_sources(cases, args.max_per_source, caps, rng)

    splits: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    for c in cases:
        splits[split_of(c, args.val_frac, args.test_frac)].append(c)
    splits["train"] = balance(splits["train"], args.max_majority, rng)
    for s in splits.values():
        rng.shuffle(s)

    args.out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, rows in splits.items():
        write_jsonl(args.out / f"cases_{name}.jsonl", rows)
        stats[name] = summarize_cases(rows)
        print(f"{name}: {len(rows)} cases  families={stats[name]['families']}")
    # leakage check: no group shared between train and test within a source
    tr = {(c["source"], c["group"]) for c in splits["train"]}
    leak = sum(1 for c in splits["test"] if (c["source"], c["group"]) in tr and not c.get("split_hint"))
    stats["leaked_groups_train_test"] = leak
    print(f"leakage check: {leak} test cases share a group with train (expected 0)")
    (args.out / "stats.json").write_text(json.dumps(stats, indent=1))

    if args.push:
        from datasets import Dataset, DatasetDict

        token = os.environ.get("HF_TOKEN")
        dd = DatasetDict({k: Dataset.from_list(to_hub_rows(v, k)) for k, v in splits.items() if v})
        dd.push_to_hub(args.push, token=token, private=args.private)
        print(f"pushed {sum(len(v) for v in splits.values())} cases -> https://huggingface.co/datasets/{args.push}")


if __name__ == "__main__":
    main()
