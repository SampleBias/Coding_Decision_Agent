"""Scripted and heuristic stand-ins for the checkpoint.

Scripted rows come from the catalog. Their argmax is the gold label, so a
headless run can prove the harness before any weights exist. The heuristic
covers a request that is not in the catalog. Neither path is the model.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence

from sidecar.scoring import argmax_label, expected_score, tidy

MODEL_SCRIPTED = "mock-scripted"
MODEL_HEURISTIC = "mock-heuristic"


def score_dist(n: int, center: float, sigma: float = 0.55) -> Dict[str, float]:
    keys = [str(i) for i in range(n)]
    weights = [math.exp(-((i - center) ** 2) / (2 * sigma ** 2)) for i in range(n)]
    return tidy({k: w for k, w in zip(keys, weights)}, keys)


def choice_dist(keys: Sequence[str], label: str, mass: float) -> Dict[str, float]:
    if label not in keys:
        raise KeyError(label)
    if not 0.0 < mass < 1.0:
        raise ValueError(mass)
    rest = (1.0 - mass) / (len(keys) - 1)
    return tidy({k: (mass if k == label else rest) for k in keys}, keys)


def answers_for(family_questions: Sequence[Mapping[str, Any]], labels: Mapping[str, str],
                masses: Optional[Mapping[str, float]] = None) -> Dict[str, Dict[str, Any]]:
    """Build a full answer map. Score questions are a gaussian on the level index."""
    masses = masses or {}
    out: Dict[str, Dict[str, Any]] = {}
    for question in family_questions:
        qid = question["id"]
        if qid not in labels:
            raise KeyError(f"missing label for {qid}")
        label = str(labels[qid])
        options = list(question["options"])
        if question["type"] == "score":
            probs = score_dist(len(options), int(label))
        else:
            probs = choice_dist(options, label, float(masses.get(qid, 0.84)))
        got = argmax_label(probs, options)
        if got != label:
            raise RuntimeError(f"{qid}: distribution peaks at {got}, intended {label}")
        item: Dict[str, Any] = {
            "type": question["type"],
            "label": got,
            "probabilities": probs,
            "answer_confidence": max(probs.values()),
        }
        if question["type"] == "choice":
            item["choice"] = got
        else:
            item["score"] = expected_score(probs, options)
        out[qid] = item
    return out


def from_scripted(family_questions: Sequence[Mapping[str, Any]], mock: Mapping[str, Mapping[str, float]]) -> Dict[str, Dict[str, Any]]:
    """Use the probabilities stored on the case, renormalised onto the schema options."""
    out: Dict[str, Dict[str, Any]] = {}
    for question in family_questions:
        qid = question["id"]
        options = list(question["options"])
        probs = tidy(mock[qid], options)
        label = argmax_label(probs, options)
        item: Dict[str, Any] = {
            "type": question["type"],
            "label": label,
            "probabilities": probs,
            "answer_confidence": max(probs.values()),
        }
        if question["type"] == "choice":
            item["choice"] = label
        else:
            item["score"] = expected_score(probs, options)
        out[qid] = item
    return out


def heuristic_labels(family: str, fields: Mapping[str, Any]) -> tuple[Dict[str, str], Dict[str, float]]:
    """A keyword stand-in so an unknown case still returns a full distribution.

    The peaks are deliberate and coarse. They are not a judgement of the patch.
    """
    blob = _blob(fields)
    masses: Dict[str, float] = {}
    if family == "code_review":
        tests = str(fields.get("tests") or "")
        diff = str(fields.get("diff") or "")
        if "failed" in tests.lower() or "assertionerror" in tests.lower():
            labels = {
                "code_quality": "1",
                "instruction_followed": "no",
                "likely_correct": "no",
                "merge_action": "request_changes",
            }
        elif diff.strip() and not tests.strip():
            labels = {
                "code_quality": "2",
                "instruction_followed": "yes",
                "likely_correct": "no",
                "merge_action": "needs_tests",
            }
            masses["merge_action"] = 0.56
        elif "passed" in tests.lower():
            labels = {
                "code_quality": "4",
                "instruction_followed": "yes",
                "likely_correct": "yes",
                "merge_action": "accept",
            }
        else:
            labels = {
                "code_quality": "2",
                "instruction_followed": "yes",
                "likely_correct": "no",
                "merge_action": "needs_tests",
            }
            masses["merge_action"] = 0.5
            masses["likely_correct"] = 0.55
        return labels, masses

    if family == "tool_call":
        names = _tool_names(fields.get("tools"))
        calls = _calls(fields.get("proposed_calls"))
        request = str(fields.get("request") or "").lower()
        general = any(word in request for word in ("capital of", "what is the capital", "explain "))
        empty_arg = any(v == "" for _name, args in calls for v in args.values())
        unknown = any(name not in names for name, _args in calls)
        if not calls and general:
            labels = {
                "tool_selection": "2", "parameter_structure": "2", "sequence_accuracy": "2",
                "query_coverage": "2", "should_call_tool": "answer_directly", "call_verdict": "abstain",
            }
        elif not calls:
            labels = {
                "tool_selection": "0", "parameter_structure": "0", "sequence_accuracy": "0",
                "query_coverage": "0", "should_call_tool": "call_tool", "call_verdict": "wrong_tool",
            }
        elif unknown or (general and calls):
            labels = {
                "tool_selection": "0", "parameter_structure": "2", "sequence_accuracy": "2",
                "query_coverage": "0", "should_call_tool": "answer_directly", "call_verdict": "wrong_tool",
            }
        elif empty_arg or any("query" in args and "get_weather" in name for name, args in calls):
            labels = {
                "tool_selection": "2", "parameter_structure": "0", "sequence_accuracy": "2",
                "query_coverage": "1", "should_call_tool": "call_tool", "call_verdict": "fix_args",
            }
        else:
            labels = {
                "tool_selection": "2", "parameter_structure": "2", "sequence_accuracy": "2",
                "query_coverage": "2", "should_call_tool": "call_tool", "call_verdict": "execute",
            }
        return labels, masses

    if family == "agent_trace":
        if "rm -rf" in blob or "drop table" in blob:
            labels = {
                "action": "stop", "needs_review": "yes", "outcome": "harmful",
                "risk": "3", "urgency": "3",
            }
        elif "traceback" in blob or "failed" in blob:
            labels = {
                "action": "human_review", "needs_review": "yes", "outcome": "failure",
                "risk": "2", "urgency": "2",
            }
        else:
            labels = {
                "action": "continue", "needs_review": "no", "outcome": "success",
                "risk": "0", "urgency": "0",
            }
        return labels, masses

    if family == "routing":
        request = str(fields.get("request") or "").lower()
        if any(word in request for word in ("rename", "typo", "comment")):
            labels = {"model_tier": "small_fast", "task_difficulty": "0", "skill": "code_edit"}
        elif any(word in request for word in ("migrat", "design", "architecture")):
            labels = {"model_tier": "reasoning", "task_difficulty": "3", "skill": "plan"}
        elif "test" in request:
            labels = {"model_tier": "small_fast", "task_difficulty": "1", "skill": "write_tests"}
        else:
            labels = {"model_tier": "mid", "task_difficulty": "2", "skill": "code_edit"}
            masses["model_tier"] = 0.55
        return labels, masses

    raise KeyError(family)


def _blob(fields: Mapping[str, Any]) -> str:
    import json
    return json.dumps(fields, ensure_ascii=False).lower()


def _tool_names(tools: Any) -> set[str]:
    names = set()
    for tool in tools or []:
        if isinstance(tool, dict):
            name = tool.get("name") or tool.get("tool")
            if name:
                names.add(str(name))
    return names


def _calls(calls: Any) -> list[tuple[str, dict]]:
    if calls is None:
        return []
    if isinstance(calls, dict):
        calls = [calls]
    out = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        name = str(call.get("name") or call.get("tool") or "")
        args = call.get("arguments") or call.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        out.append((name, {str(k): "" if v is None else str(v) for k, v in args.items()}))
    return out
