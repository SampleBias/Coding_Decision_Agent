"""Build code_review (and routing) cases from nebius/SWE-agent-trajectories.

80k SWE-agent runs on real GitHub issues with a ground-truth `target` (issue resolved or not).
The state is the issue text + the agent's final patch (compacted) + run metadata that a coding
agent runtime legitimately has at review time (exit status, number of steps). `eval_logs` is the
post-hoc grading and is NEVER put in the state.

  * code_review case per trajectory: `likely_correct` from `target` (hard), `merge_action`
    derived from `target` (teacher-blended), `code_quality` teacher-labelled.
  * routing case per issue with >= 2 trajectories from different models: `model_tier` from
    per-tier resolve rates, `task_difficulty` from the overall resolve rate (teacher-blended),
    `skill` teacher-labelled.

Class balance: resolved runs are ~17% of the data, so unresolved runs are subsampled to
`--neg-ratio` x positives (default 1.5).

Usage:
    python data/build_swe_trajectories.py [--limit N] [--max-pos 8000] [--neg-ratio 1.5]
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root

import json
import random
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from data.common import (builder_argparser, default_out, load_hf, maybe_json, stable_id, summarize_cases,
                         write_jsonl)
from data.compaction import code_review_state, routing_state, truncate
from data.tiers import cheapest_tier_distribution, difficulty_from_solve_rate, tier_rates
from configs.schema import make_gold, new_case, one_hot, option_keys, questions_for

SOURCE = "nebius/SWE-agent-trajectories"
_ISSUE_HDR = re.compile(r"^.*?ISSUE:\s*", re.S)
_ISSUE_TAIL = re.compile(r"\n\s*INSTRUCTIONS:.*$", re.S)


def issue_text(trajectory) -> str:
    traj = maybe_json(trajectory) or []
    for e in traj:
        if e.get("role") == "user":
            t = e.get("text") or e.get("content") or ""
            t = _ISSUE_HDR.sub("", t, count=1)
            t = _ISSUE_TAIL.sub("", t)
            return t.strip()
    return ""


def repo_of(instance_id: str) -> str:
    return instance_id.rsplit("-", 1)[0]


def derived_merge(resolved: bool, empty_patch: bool) -> Dict[str, float]:
    if empty_patch:
        return {"accept": 0.01, "request_changes": 0.09, "needs_tests": 0.05, "reject": 0.85}
    if resolved:
        return {"accept": 0.65, "request_changes": 0.12, "needs_tests": 0.20, "reject": 0.03}
    return {"accept": 0.05, "request_changes": 0.45, "needs_tests": 0.15, "reject": 0.35}


def convert(row, cr_q) -> Optional[dict]:
    patch = row.get("generated_patch") or ""
    task = issue_text(row.get("trajectory"))
    if not task:
        return None
    resolved = bool(row["target"])
    traj_len = len(maybe_json(row.get("trajectory")) or [])
    notes = f"agent exit status: {row.get('exit_status')}; steps: {traj_len}; model: {row.get('model_name')}"
    state = code_review_state(task=task, diff=patch, language="python", notes=notes)
    gold = {
        "likely_correct": make_gold(cr_q["likely_correct"],
                                    one_hot(option_keys(cr_q["likely_correct"]), "yes" if resolved else "no", 0.08),
                                    "hard"),
        "merge_action": make_gold(cr_q["merge_action"], derived_merge(resolved, not patch.strip()), "derived"),
    }
    c = new_case(f"swe_{stable_id(row['instance_id'], row['model_name'], patch[:200])}", "code_review", SOURCE,
                 repo_of(row["instance_id"]), state, ["code_quality", "likely_correct", "merge_action"], gold,
                 needs_teacher=["code_quality", "merge_action"])
    c["meta"] = {"instance_id": row["instance_id"], "model_name": row["model_name"], "exit_status": row.get("exit_status"),
                 "resolved": resolved}
    return c


def routing_cases(per_issue: Dict[str, Dict], rt_q) -> List[dict]:
    out = []
    for inst, info in per_issue.items():
        results: List[Tuple[str, float]] = info["results"]
        if len({m for m, _ in results}) < 2:
            continue
        rates = tier_rates(results)
        solve_rate = sum(s for _, s in results) / len(results)
        gold = {
            "model_tier": make_gold(rt_q["model_tier"], cheapest_tier_distribution(rates), "derived"),
            "task_difficulty": make_gold(rt_q["task_difficulty"], difficulty_from_solve_rate(solve_rate, len(results)), "derived"),
        }
        c = new_case(f"swe_route_{stable_id(inst)}", "routing", SOURCE, repo_of(inst),
                     routing_state(info["task"], {"repo": repo_of(inst), "kind": "github issue"}),
                     ["model_tier", "task_difficulty", "skill"], gold, needs_teacher=["task_difficulty", "skill"])
        c["meta"] = {"instance_id": inst, "n_trajectories": len(results), "solve_rate": solve_rate}
        out.append(c)
    return out


def main():
    ap = builder_argparser(__doc__)
    ap.add_argument("--max-pos", type=int, default=8000, help="cap on resolved trajectories")
    ap.add_argument("--neg-ratio", type=float, default=1.5, help="unresolved per resolved")
    ap.add_argument("--no-routing", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    cr_q, rt_q = questions_for("code_review"), questions_for("routing")

    ds = load_hf(SOURCE, split="train", streaming=True, cache_dir=args.cache_dir)
    max_neg = int(args.max_pos * args.neg_ratio)
    n_pos = n_neg = 0
    cases: List[dict] = []
    per_issue: Dict[str, Dict] = defaultdict(lambda: {"task": "", "results": []})
    seen = 0
    for row in ds:
        seen += 1
        if args.limit and seen > args.limit:
            break
        resolved = bool(row["target"])
        # routing statistics use every row we see (cheap), even if the review case is subsampled
        if not args.no_routing:
            info = per_issue[row["instance_id"]]
            info["results"].append((row["model_name"], 1.0 if resolved else 0.0))
        if resolved:
            if n_pos >= args.max_pos:
                continue
        else:
            # reservoir-free subsampling: accept with probability that roughly yields the ratio
            if n_neg >= max_neg or rng.random() > 0.35:
                continue
        c = convert(row, cr_q)
        if c is None:
            continue
        if not args.no_routing and not per_issue[row["instance_id"]]["task"]:
            per_issue[row["instance_id"]]["task"] = truncate(c["state"]["task"], 2500)
        cases.append(c)
        n_pos += resolved
        n_neg += (not resolved)
        if n_pos >= args.max_pos and n_neg >= max_neg and args.no_routing:
            break
    if not args.no_routing:
        cases.extend(routing_cases({k: v for k, v in per_issue.items() if v["task"]}, rt_q))
    out = default_out("swe_trajectories", args)
    n = write_jsonl(out, cases)
    print(f"scanned {seen} rows; wrote {n} cases ({n_pos} resolved / {n_neg} unresolved) -> {out}")
    print(json.dumps(summarize_cases(cases), indent=1)[:3000])


if __name__ == "__main__":
    import os
    import sys

    main()
    # `datasets` streaming leaves background download threads that crash the interpreter during
    # finalisation on some builds; everything is flushed to disk by now, so exit hard and clean.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
