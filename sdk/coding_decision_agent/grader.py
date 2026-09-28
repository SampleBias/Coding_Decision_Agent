"""Load S4MPL3BI4S/Coding_Decision_Agent and ask the four grader families.

The question text is imported from configs/schema.py when the training repo is on the path,
and duplicated in schemas.py so a `pip install` of this package does not need the repo.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from coding_decision_agent.schemas import (
    AGENT_TRACE_QUESTIONS,
    CODE_REVIEW_QUESTIONS,
    ROUTING_QUESTIONS,
    TOOL_CALL_QUESTIONS,
)

DEFAULT_REPO = "S4MPL3BI4S/Coding_Decision_Agent"


class Grade:
    """One forward pass. Attribute access returns the chosen label (or expected score)."""

    def __init__(self, family: str, answers: Dict[str, Dict[str, Any]]):
        self.family = family
        self.answers = answers
        self.probabilities = {qid: a.get("probabilities", {}) for qid, a in answers.items()}
        self.answer_confidence = {qid: a.get("answer_confidence") for qid, a in answers.items()}

    def __getattr__(self, qid: str):
        if qid.startswith("_") or qid not in self.answers:
            raise AttributeError(qid)
        a = self.answers[qid]
        if a.get("type") == "choice":
            return a.get("choice")
        if a.get("type") == "score":
            return a.get("score")
        return a.get("noul")

    def label(self, qid: str) -> str:
        a = self.answers[qid]
        if a.get("type") == "score":
            probs = a.get("probabilities") or {}
            return max(probs, key=probs.get) if probs else "0"
        return str(getattr(self, qid))

    def confident(self, qid: str, threshold: float = 0.7) -> bool:
        c = self.answer_confidence.get(qid)
        return c is not None and c >= threshold


def _state_builders():
    """Use the training-repo compactors when they are importable, else a plain passthrough."""
    try:
        from data.compaction import agent_trace_state, code_review_state, routing_state, tool_call_state
        return code_review_state, tool_call_state, agent_trace_state, routing_state
    except Exception:
        return None


class CodingDecisionAgent:
    def __init__(self, model_id: str = DEFAULT_REPO, device: Optional[str] = None,
                 max_len: int = 2048, head_max_len: int = 320):
        import laya

        self.agent = laya.load(model_id, device=device)
        self.max_len = max_len
        self.head_max_len = head_max_len
        # Honour the budget the checkpoint was trained with when the caller did not override.
        cfg = getattr(self.agent, "cfg", None) or {}
        if max_len == 2048 and cfg.get("max_len"):
            self.max_len = int(cfg["max_len"])
        if head_max_len == 320 and cfg.get("head_max_len"):
            self.head_max_len = int(cfg["head_max_len"])

    def _ask(self, family: str, state: Dict[str, Any], questions: Dict[str, Any],
             only: Optional[Sequence[str]] = None) -> Grade:
        if only:
            questions = {k: questions[k] for k in only}
        res = self.agent.predict(state, questions, max_len=self.max_len, head_max_len=self.head_max_len)
        return Grade(family, res["answers"])

    def grade_code(self, task: str, diff: str = "", language: str = "", tests: str = "",
                   notes: str = "", raw_code: bool = False, only: Optional[Sequence[str]] = None) -> Grade:
        builders = _state_builders()
        if builders:
            state = builders[0](task, diff, language, tests, notes, raw_code)
        else:
            state = {"family": "code_review", "task": task, "language": language,
                     "diff" if not raw_code else "code": diff, "tests": tests, "notes": notes}
        return self._ask("code_review", state, CODE_REVIEW_QUESTIONS, only)

    def grade_tool_call(self, request: str, tools: Sequence[dict], proposed_calls: Any = None,
                        history: Any = None, only: Optional[Sequence[str]] = None) -> Grade:
        builders = _state_builders()
        if builders:
            state = builders[1](request, tools, proposed_calls, history)
        else:
            state = {"family": "tool_call", "request": request, "tools": tools, "proposed_calls": proposed_calls}
        return self._ask("tool_call", state, TOOL_CALL_QUESTIONS, only)

    def grade_trace(self, task: str, agent: Optional[dict] = None, constraints: Any = None,
                    trace_summary: Any = None, recent_steps: Any = None,
                    only: Optional[Sequence[str]] = None) -> Grade:
        builders = _state_builders()
        if builders:
            state = builders[2](task, agent, constraints, trace_summary, recent_steps)
        else:
            state = {"family": "agent_trace", "task": task, "agent": agent, "constraints": constraints,
                     "trace_summary": trace_summary, "recent_steps": recent_steps}
        return self._ask("agent_trace", state, AGENT_TRACE_QUESTIONS, only)

    def route_model(self, request: str, context: Any = None, only: Optional[Sequence[str]] = None) -> Grade:
        builders = _state_builders()
        if builders:
            state = builders[3](request, context)
        else:
            state = {"family": "routing", "request": request, "context": context}
        return self._ask("routing", state, ROUTING_QUESTIONS, only)
