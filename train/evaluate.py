"""Evaluate a Coding_Decision_Agent checkpoint against a case JSONL split.

Reports, per family and per question: accuracy, soft accuracy (gold · predicted), Brier,
ECE on answer_confidence, and score MAE for `score` questions, plus latency. Baselines:

* majority: most common train-split label for each question (pass --train-cases)
* zero-shot: any other Laya checkpoint, repeated via --baseline name=repo (downloads weights)

    python train/evaluate.py --cases artifacts/dataset/cases_test.jsonl \
        --checkpoint outputs/coding_decision_agent \
        --train-cases artifacts/dataset/cases_train.jsonl \
        --baseline zero_shot=convaiinnovations/laya \
        --baseline typed=convaiinnovations/laya-typed-decisions \
        --out outputs/coding_decision_agent/benchmark_report.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import numpy as np

from data.common import read_jsonl
from configs.schema import option_keys


def gold_probs(case, qid) -> Dict[str, float]:
    return case["gold"][qid]["probabilities"]


def pred_probs(answer, q) -> Dict[str, float]:
    keys = option_keys(q)
    raw = answer.get("probabilities") or {}
    if q["type"] == "noul" and "false" not in raw:
        p_true = float(answer.get("noul", 0.5))
        raw = {"false": 1 - p_true, "true": p_true}
    vals = np.array([float(raw.get(k, 0.0)) for k in keys], dtype=np.float64)
    if vals.sum() <= 0:
        vals[:] = 1.0 / len(keys)
    else:
        vals /= vals.sum()
    return {k: float(v) for k, v in zip(keys, vals)}


def pred_label(answer, q) -> str:
    if q["type"] == "choice":
        return str(answer["choice"])
    if q["type"] == "score":
        probs = pred_probs(answer, q)
        return max(probs, key=probs.get)
    p = pred_probs(answer, q)
    return "true" if p.get("true", 0) >= p.get("false", 0) else "false"


def majority_map(train_cases) -> Dict[str, str]:
    counts = defaultdict(Counter)
    for c in train_cases:
        for qid, g in c["gold"].items():
            counts[f"{c['family']}.{qid}"][g["label"]] += 1
    return {k: v.most_common(1)[0][0] for k, v in counts.items()}


def empty_bucket():
    return {"n": 0, "correct": 0, "soft": 0.0, "brier": 0.0, "score_abs": 0.0, "score_n": 0,
            "conf": [], "hit": []}


def accumulate(bucket, q, gold, answer, latency=None):
    keys = option_keys(q)
    gp = np.array([gold["probabilities"].get(k, 0.0) for k in keys])
    pp = np.array([pred_probs(answer, q).get(k, 0.0) for k in keys])
    label = gold["label"]
    got = pred_label(answer, q)
    bucket["n"] += 1
    bucket["correct"] += int(got == label)
    bucket["soft"] += float((gp * pp).sum())
    bucket["brier"] += float(((pp - gp) ** 2).sum())
    if q["type"] == "score":
        bucket["score_abs"] += abs(keys.index(got) - keys.index(label)) if got in keys and label in keys else 0
        bucket["score_n"] += 1
    conf = float(answer.get("answer_confidence", max(pp)))
    bucket["conf"].append(conf)
    bucket["hit"].append(float(got == label))
    if latency is not None:
        bucket.setdefault("latency", []).append(latency)


def finalize(bucket) -> dict:
    from laya.common import ece_score

    n = max(1, bucket["n"])
    conf = np.array(bucket["conf"]) if bucket["conf"] else np.array([])
    hit = np.array(bucket["hit"]) if bucket["hit"] else np.array([])
    lat = bucket.get("latency") or []
    return {
        "n": bucket["n"],
        "accuracy": bucket["correct"] / n,
        "soft_accuracy": bucket["soft"] / n,
        "brier": bucket["brier"] / n,
        "ece": None if len(conf) == 0 else ece_score(conf, hit),
        "score_mae": None if not bucket["score_n"] else bucket["score_abs"] / bucket["score_n"],
        "latency_ms_p50": None if not lat else float(np.percentile(lat, 50) * 1000),
    }


def evaluate_agent(agent, cases, max_len, head_max_len) -> Dict[str, dict]:
    buckets = defaultdict(empty_bucket)
    for case in cases:
        t0 = time.perf_counter()
        res = agent.predict(case["state"], case["questions"], max_len=max_len, head_max_len=head_max_len)
        dt = time.perf_counter() - t0
        per_q = dt / max(1, len(case["questions"]))
        for qid, q in case["questions"].items():
            ans = res["answers"].get(qid)
            if not ans or qid not in case["gold"]:
                continue
            accumulate(buckets[f"{case['family']}.{qid}"], q, case["gold"][qid], ans, per_q)
            accumulate(buckets[case["family"]], q, case["gold"][qid], ans, per_q)
            accumulate(buckets["ALL"], q, case["gold"][qid], ans, per_q)
    return {k: finalize(v) for k, v in sorted(buckets.items())}


def evaluate_majority(cases, maj) -> Dict[str, dict]:
    buckets = defaultdict(empty_bucket)
    for case in cases:
        for qid, q in case["questions"].items():
            if qid not in case["gold"]:
                continue
            label = maj.get(f"{case['family']}.{qid}", option_keys(q)[0])
            keys = option_keys(q)
            probs = {k: (1.0 if k == label else 0.0) for k in keys}
            ans = {"type": q["type"], "probabilities": probs, "choice": label, "answer_confidence": 1.0}
            if q["type"] == "score":
                ans["score"] = float(keys.index(label)) if label in keys else 0.0
            accumulate(buckets[f"{case['family']}.{qid}"], q, case["gold"][qid], ans)
            accumulate(buckets[case["family"]], q, case["gold"][qid], ans)
            accumulate(buckets["ALL"], q, case["gold"][qid], ans)
    return {k: finalize(v) for k, v in sorted(buckets.items())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--checkpoint", type=str, required=True, help="local dir or HF repo id")
    ap.add_argument("--train-cases", type=Path, default=None)
    ap.add_argument("--baseline", action="append", default=[], help="name=repo_or_path, repeatable")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--head-max-len", type=int, default=None)
    args = ap.parse_args()

    import laya

    cases = list(read_jsonl(args.cases))
    if args.limit:
        cases = cases[: args.limit]
    ckpt = Path(args.checkpoint)
    max_len, head_max_len = args.max_len, args.head_max_len
    if ckpt.is_dir() and (ckpt / "rl_agent_config.json").exists():
        cfg = json.loads((ckpt / "rl_agent_config.json").read_text())
        max_len = max_len or int(cfg.get("max_len", 1024))
        head_max_len = head_max_len or int(cfg.get("head_max_len", 256))
    max_len, head_max_len = max_len or 1024, head_max_len or 256

    report = {"checkpoint": args.checkpoint, "n_cases": len(cases), "max_len": max_len,
              "head_max_len": head_max_len, "models": {}}
    print(f"evaluating {args.checkpoint} on {len(cases)} cases")
    agent = laya.load(args.checkpoint)
    report["models"]["coding_decision_agent"] = evaluate_agent(agent, cases, max_len, head_max_len)
    del agent

    if args.train_cases and args.train_cases.exists():
        maj = majority_map(read_jsonl(args.train_cases))
        report["models"]["majority"] = evaluate_majority(cases, maj)

    for spec in args.baseline:
        name, repo = spec.split("=", 1)
        print(f"baseline {name} = {repo}")
        b = laya.load(repo)
        report["models"][name] = evaluate_agent(b, cases, max_len, head_max_len)
        del b

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")
    for name, metrics in report["models"].items():
        overall = metrics.get("ALL", {})
        print(f"  {name:24s} acc {overall.get('accuracy', float('nan')):.3f}  "
              f"brier {overall.get('brier', float('nan')):.3f}  ece {overall.get('ece')}")


if __name__ == "__main__":
    main()
