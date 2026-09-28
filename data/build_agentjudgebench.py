"""Build tool_call cases from ServiceNow-AI/AgentJudgeBench.

Each (record x generator x difficulty) triple becomes one case: the user query, the available
tools and the generator's predicted tool calls form the state; the four programmatic scores
(tool_selection, parameter_structure, sequence_accuracy, query_coverage in {0, 0.5, 1}) become
3-level score targets. When judge configs are loaded, the six general-purpose judges' `without_gt`
verdicts are turned into a vote distribution and blended with the programmatic score, which gives
RLCD a genuinely soft target. `call_verdict` is derived from the programmatic scores and marked
for teacher blending.

Usage:
    python data/build_agentjudgebench.py [--limit N] [--no-judges] [--generators a,b]
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root

import json
from collections import defaultdict
from typing import Dict, List

from data.common import (builder_argparser, default_out, load_hf, maybe_json, summarize_cases, take,
                         write_jsonl)
from data.compaction import tool_call_state
from configs.schema import blend, make_gold, new_case, one_hot, option_keys, questions_for

SOURCE = "ServiceNow-AI/AgentJudgeBench"
GENERATORS = ["llama_3_3_70b_instruct", "llama_3_1_8b_instruct", "qwen3_32b", "smollm3_3b", "gpt5_4"]
JUDGES = ["judge_gpt_5_4", "judge_claude_large", "judge_gemini_2_5_pro", "judge_qwq_32b",
          "judge_gpt_oss_20b", "judge_gpt_oss_120b"]  # prometheus2 excluded (weak baseline)
METRICS = ["tool_selection", "parameter_structure", "sequence_accuracy", "query_coverage"]
PROGRAMMATIC_WEIGHT = 0.6  # blend: programmatic one-hot vs judge vote distribution


def level(score: float) -> str:
    """{0, 0.5, 1} -> {"0", "1", "2"}"""
    return str(int(round(float(score) * 2)))


def load_judge_votes(cache_dir, limit=None) -> Dict[tuple, Dict[str, Dict[str, int]]]:
    """(id, generator, difficulty) -> metric -> {level: votes}"""
    votes: Dict[tuple, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for j in JUDGES:
        ds = load_hf(SOURCE, j, split="train", cache_dir=cache_dir)
        for row in take(ds, limit):
            verdict = maybe_json(row.get("without_gt")) or {}
            key = (row["id"], row["generator"], row["difficulty"])
            for m in METRICS:
                v = verdict.get(m) if isinstance(verdict, dict) else None
                if isinstance(v, dict) and v.get("llm_score") is not None:
                    votes[key][m][level(v["llm_score"])] += 1
    return votes


def derive_call_verdict(scores: Dict[str, float], n_generated: int, n_expected: int) -> Dict[str, float]:
    ts, ps = scores["tool_selection"], scores["parameter_structure"]
    sq, qc = scores["sequence_accuracy"], scores["query_coverage"]
    if n_generated == 0 and n_expected > 0:
        return {"execute": 0.02, "fix_args": 0.03, "wrong_tool": 0.85, "abstain": 0.10}
    if ts >= 1 and ps >= 1 and sq >= 1 and qc >= 1:
        return {"execute": 0.90, "fix_args": 0.05, "wrong_tool": 0.03, "abstain": 0.02}
    if ts >= 1 and (ps < 1 or sq < 1):
        return {"execute": 0.10, "fix_args": 0.70, "wrong_tool": 0.15, "abstain": 0.05}
    if ts >= 1 and qc < 1:
        return {"execute": 0.15, "fix_args": 0.35, "wrong_tool": 0.45, "abstain": 0.05}
    if ts == 0.5:
        return {"execute": 0.05, "fix_args": 0.35, "wrong_tool": 0.55, "abstain": 0.05}
    return {"execute": 0.02, "fix_args": 0.10, "wrong_tool": 0.83, "abstain": 0.05}


def convert_row(row, generator, questions, votes, add_should_call: bool):
    tools = maybe_json(row["available_tools"]) or []
    generated = maybe_json(row.get("generated_tool_calls")) or []
    expected = maybe_json(row.get("expected_tool_calls")) or []
    state = tool_call_state(
        request=row["user_query"],
        tools=tools,
        proposed_calls=generated,
        system_prompt="",
    )
    scores = {m: float(row[f"score_{m}"]) for m in METRICS}
    gold = {}
    key = (row["id"], generator, row["difficulty"])
    for m in METRICS:
        q = questions[m]
        keys = option_keys(q)
        prog = one_hot(keys, level(scores[m]), smoothing=0.06)
        v = votes.get(key, {}).get(m) if votes else None
        if v:
            total = sum(v.values())
            vote_dist = {k: v.get(k, 0) / total for k in keys}
            probs = blend(prog, vote_dist, PROGRAMMATIC_WEIGHT, keys)
            signal = "programmatic+judges"
        else:
            probs, signal = prog, "programmatic"
        gold[m] = make_gold(q, probs, signal)
    gold["call_verdict"] = make_gold(questions["call_verdict"],
                                     derive_call_verdict(scores, len(generated), len(expected)), "derived")
    qids = METRICS + ["call_verdict"]
    if add_should_call:
        gold["should_call_tool"] = make_gold(questions["should_call_tool"],
                                             one_hot(option_keys(questions["should_call_tool"]), "call_tool", 0.06),
                                             "hard")
        qids = qids + ["should_call_tool"]
    case = new_case(
        case_id=f"ajb_{row['id']}_{generator}_{row['difficulty']}",
        family="tool_call",
        source=f"{SOURCE}/{generator}",
        group=row["id"],
        state=state,
        qids=qids,
        gold=gold,
        needs_teacher=["call_verdict"],
    )
    case["meta"] = {"dag_type": row["dag_type"], "difficulty": row["difficulty"], "generator": generator,
                    "overall_programmatic_score": float(row["overall_programmatic_score"])}
    return case


def main():
    ap = builder_argparser(__doc__)
    ap.add_argument("--no-judges", action="store_true", help="skip loading judge configs (faster smoke runs)")
    ap.add_argument("--generators", type=str, default=",".join(GENERATORS))
    args = ap.parse_args()
    gens = [g for g in args.generators.split(",") if g]
    questions = questions_for("tool_call")
    votes = {} if args.no_judges else load_judge_votes(args.cache_dir, None if args.limit is None else args.limit * 50)
    cases: List[dict] = []
    for gi, g in enumerate(gens):
        ds = load_hf(SOURCE, g, split="train", cache_dir=args.cache_dir)
        for row in take(ds, args.limit):
            cases.append(convert_row(row, g, questions, votes, add_should_call=(gi == 0)))
    out = default_out("agentjudgebench_tool_call", args)
    n = write_jsonl(out, cases)
    print(f"wrote {n} cases -> {out}")
    print(json.dumps(summarize_cases(cases), indent=1)[:3000])


if __name__ == "__main__":
    main()
