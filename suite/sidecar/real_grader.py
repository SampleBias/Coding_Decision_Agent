"""Checkpoint backend. Imports Laya only when this module is constructed."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from sidecar.scoring import argmax_label, expected_score, tidy


class RealGrader:
    def __init__(self, checkpoint: str, device: Optional[str] = None):
        from coding_decision_agent import CodingDecisionAgent

        self.checkpoint = checkpoint
        self.device = device
        self.agent = CodingDecisionAgent(checkpoint, device=device)
        self.device_name = device or _torch_device()

    def grade(self, family: str, fields: Mapping[str, Any], questions: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
        if family == "code_review":
            graded = self.agent.grade_code(
                task=str(fields.get("task") or ""),
                diff=str(fields.get("diff") or ""),
                language=str(fields.get("language") or ""),
                tests=str(fields.get("tests") or ""),
                notes=str(fields.get("notes") or ""),
                raw_code=bool(fields.get("raw_code") or False),
            )
        elif family == "tool_call":
            graded = self.agent.grade_tool_call(
                request=str(fields.get("request") or ""),
                tools=list(fields.get("tools") or []),
                proposed_calls=fields.get("proposed_calls"),
                history=fields.get("history"),
            )
        elif family == "agent_trace":
            agent = fields.get("agent") if isinstance(fields.get("agent"), dict) else None
            graded = self.agent.grade_trace(
                task=str(fields.get("task") or ""),
                agent=agent,
                constraints=fields.get("constraints"),
                trace_summary=fields.get("trace_summary"),
                recent_steps=fields.get("recent_steps"),
            )
        elif family == "routing":
            graded = self.agent.route_model(str(fields.get("request") or ""), context=fields.get("context"))
        else:
            raise KeyError(family)
        return answers_from_grade(graded, questions)


def answers_from_grade(graded, questions: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Preserve Laya's choice label. Score labels are the argmax, as in evaluate.py."""
    answers: Dict[str, Any] = {}
    missing = []
    for qid, question in questions.items():
        raw = graded.answers.get(qid)
        if not raw:
            missing.append(qid)
            continue
        options = list(question["options"])
        probs = tidy({k: float((raw.get("probabilities") or {}).get(k, 0.0)) for k in options}, options)
        if question["type"] == "choice" and raw.get("choice") in options:
            label = str(raw["choice"])
        else:
            try:
                stated = str(graded.label(qid))
            except Exception:
                stated = ""
            label = stated if stated in options else argmax_label(probs, options)
        conf = raw.get("answer_confidence")
        item: Dict[str, Any] = {
            "type": question["type"],
            "label": label,
            "probabilities": probs,
            "answer_confidence": float(conf) if conf is not None else max(probs.values()),
        }
        if question["type"] == "choice":
            item["choice"] = label
        else:
            item["score"] = float(raw["score"]) if raw.get("score") is not None else expected_score(probs, options)
        answers[qid] = item
    return {"answers": answers, "missing": missing}


def _torch_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "auto"
