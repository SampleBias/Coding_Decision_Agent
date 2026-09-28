"""Confidence-gated hook for a coding agent.

Call `review_step` after the agent proposes a tool call or a patch. Below the confidence
threshold, or on a negative verdict, escalate to a stronger model instead of executing.

    from examples.coding_agent_hook import AgentGuard

    guard = AgentGuard(threshold=0.7)
    decision = guard.review_tool_call(user_message, tool_schemas, proposed_calls)
    if decision["execute"]:
        run(proposed_calls)
    else:
        escalate(reason=decision["reason"], tier=decision["tier"])
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

from coding_decision_agent import CodingDecisionAgent


class AgentGuard:
    def __init__(self, model_id: str = "S4MPL3BI4S/Coding_Decision_Agent", threshold: float = 0.7,
                 device: Optional[str] = None):
        self.grader = CodingDecisionAgent(model_id, device=device)
        self.threshold = threshold

    def review_tool_call(self, request: str, tools: Sequence[dict], proposed_calls: Any) -> dict:
        g = self.grader.grade_tool_call(request, tools, proposed_calls)
        verdict = g.call_verdict
        confident = g.confident("call_verdict", self.threshold)
        execute = verdict == "execute" and confident
        return {
            "execute": execute,
            "verdict": verdict,
            "confidence": g.answer_confidence.get("call_verdict"),
            "reason": None if execute else f"call_verdict={verdict} confidence={g.answer_confidence.get('call_verdict')}",
            "probabilities": g.probabilities.get("call_verdict"),
            "tier": None,
        }

    def review_patch(self, task: str, diff: str, tests: str = "") -> dict:
        g = self.grader.grade_code(task, diff, tests=tests)
        action = g.merge_action
        confident = g.confident("merge_action", self.threshold)
        accept = action == "accept" and confident
        route = self.grader.route_model(task, context={"has_diff": True, "has_tests": bool(tests)})
        return {
            "accept": accept,
            "merge_action": action,
            "likely_correct": g.likely_correct,
            "code_quality": g.code_quality,
            "confidence": g.answer_confidence.get("merge_action"),
            "reason": None if accept else f"merge_action={action} confidence={g.answer_confidence.get('merge_action')}",
            "tier": route.model_tier,
            "skill": route.skill,
        }
