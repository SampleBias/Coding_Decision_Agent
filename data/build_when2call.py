"""Build tool_call cases from nvidia/When2Call: when to call a tool, ask a follow-up, answer
directly, or say the request cannot be fulfilled.

Sources used:
  * train_pref: chosen vs rejected assistant responses. The chosen response type gives
    `should_call_tool`; when either side is a tool call it becomes the proposed call and
    `call_verdict` is execute (chosen) or abstain (rejected).
  * train_sft: the final assistant turn gives `should_call_tool`.
  * test/mcq: explicit `correct_answer` labels; exported with split_hint = test.

Usage:
    python data/build_when2call.py [--limit N]
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root

import json
import re
from typing import Any, Dict, List, Optional

from data.common import (builder_argparser, default_out, load_hf, maybe_json, stable_id, summarize_cases,
                         take, write_jsonl)
from data.compaction import tool_call_state
from configs.schema import make_gold, new_case, one_hot, option_keys, questions_for

SOURCE = "nvidia/When2Call"
MCQ_MAP = {"direct": "answer_directly", "tool_call": "call_tool", "request_for_info": "ask_followup",
           "ask_followup": "ask_followup", "cannot_answer": "cannot_answer", "unable": "cannot_answer"}
_TOOLCALL = re.compile(r"<TOOLCALL>(.*?)</TOOLCALL>", re.S)
_CANNOT = re.compile(r"(unable to|cannot|can't|not able to|don't have (real-time |the )?(access|ability|information)|"
                     r"apolog|i'm sorry|i am sorry|beyond my|not possible for me)", re.I)


def parse_toolcalls(text: str) -> Optional[List[Any]]:
    m = _TOOLCALL.search(text or "")
    stripped = (text or "").strip()
    body = m.group(1) if m else (stripped if stripped.startswith("[{") or
                                 (stripped.startswith("{") and '"name"' in stripped) else None)
    if body is None:
        return None
    try:
        calls = json.loads(body)
        return calls if isinstance(calls, list) else [calls]
    except json.JSONDecodeError:
        return [{"name": "<unparsed>", "arguments": {"raw": body[:200]}}]


def classify(text: str) -> str:
    if parse_toolcalls(text) is not None:
        return "call_tool"
    t = (text or "").strip()
    if t.endswith("?"):
        return "ask_followup"
    if _CANNOT.search(t):
        return "cannot_answer"
    return "answer_directly"


def parse_tools(tools: Any) -> List[Dict[str, Any]]:
    tools = maybe_json(tools) or []
    out = []
    for t in tools:
        t = maybe_json(t)
        if isinstance(t, dict):
            out.append(t)
    return out


def user_request(messages: List[Dict[str, str]]):
    users = [m for m in messages if m.get("role") == "user"]
    req = users[-1]["content"] if users else ""
    history = [m for m in messages if m.get("role") != "system"][:-1] if len(messages) > 1 else None
    return req, history


def build_case(case_id: str, group: str, source: str, request: str, tools, proposed, history,
               should_call: str, verdict: Optional[str], questions, split_hint: Optional[str] = None):
    state = tool_call_state(request=request, tools=tools, proposed_calls=proposed, history=history)
    gold = {"should_call_tool": make_gold(questions["should_call_tool"],
                                         one_hot(option_keys(questions["should_call_tool"]), should_call, 0.06),
                                         "hard")}
    qids = ["should_call_tool"]
    if verdict:
        gold["call_verdict"] = make_gold(questions["call_verdict"],
                                         one_hot(option_keys(questions["call_verdict"]), verdict, 0.08), "derived")
        qids.append("call_verdict")
    case = new_case(case_id, "tool_call", source, group, state, qids, gold)
    if split_hint:
        case["split_hint"] = split_hint
    return case


def from_pref(row, i, questions):
    tools = parse_tools(row["tools"])
    req, hist = user_request(row["messages"])
    chosen = (row.get("chosen_response") or {}).get("content", "")
    rejected = (row.get("rejected_response") or {}).get("content", "")
    label = classify(chosen)
    proposed, verdict = None, None
    if label == "call_tool":
        proposed, verdict = parse_toolcalls(chosen), "execute"
    elif classify(rejected) == "call_tool":
        proposed, verdict = parse_toolcalls(rejected), "abstain"
    return build_case(f"w2c_pref_{stable_id(req, i)}", stable_id(req), f"{SOURCE}/train_pref", req, tools,
                      proposed, hist, label, verdict, questions)


def from_sft(row, i, questions):
    tools = parse_tools(row["tools"])
    msgs = row["messages"]
    assistants = [m for m in msgs if m.get("role") == "assistant"]
    if not assistants:
        return None
    final = assistants[-1]["content"]
    # request = conversation up to the final assistant turn
    idx = max(k for k, m in enumerate(msgs) if m.get("role") == "assistant")
    req, hist = user_request(msgs[:idx])
    label = classify(final)
    proposed = parse_toolcalls(final) if label == "call_tool" else None
    return build_case(f"w2c_sft_{stable_id(req, i)}", stable_id(req), f"{SOURCE}/train_sft", req, tools,
                      proposed, hist, label, "execute" if proposed else None, questions)


def from_mcq(row, questions):
    tools = parse_tools(row["tools"])
    label = MCQ_MAP.get(str(row["correct_answer"]).lower())
    if not label:
        return None
    answers = maybe_json(row.get("answers")) or {}
    proposed = parse_toolcalls(answers.get("tool_call", "")) if answers.get("tool_call") else None
    verdict = ("execute" if label == "call_tool" else "abstain") if proposed else None
    return build_case(f"w2c_mcq_{row['uuid']}", row.get("source_id") or row["uuid"], f"{SOURCE}/test_mcq",
                      row["question"], tools, proposed, None, label, verdict, questions, split_hint="test")


def main():
    ap = builder_argparser(__doc__)
    args = ap.parse_args()
    questions = questions_for("tool_call")
    cases = []
    for i, row in enumerate(take(load_hf(SOURCE, "train_pref", split="train", cache_dir=args.cache_dir), args.limit)):
        cases.append(from_pref(row, i, questions))
    for i, row in enumerate(take(load_hf(SOURCE, "train_sft", split="train", cache_dir=args.cache_dir), args.limit)):
        c = from_sft(row, i, questions)
        if c:
            cases.append(c)
    for row in take(load_hf(SOURCE, "test", split="mcq", cache_dir=args.cache_dir), args.limit):
        c = from_mcq(row, questions)
        if c:
            cases.append(c)
    out = default_out("when2call_tool_call", args)
    n = write_jsonl(out, cases)
    print(f"wrote {n} cases -> {out}")
    print(json.dumps(summarize_cases(cases), indent=1)[:3000])


if __name__ == "__main__":
    main()
