"""Compact a grade request with the training-repo state builders.

The preview in the TUI is this object, which is the same function the SDK uses
when the training repo is on the path. A missing dependency falls back to the
raw fields so the sidecar still starts.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping


def compact_state(family: str, fields: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        from data.compaction import (
            agent_trace_state,
            code_review_state,
            routing_state,
            tool_call_state,
        )
    except Exception as exc:
        return {"family": family, "compaction_error": str(exc)}

    if family == "code_review":
        return code_review_state(
            str(fields.get("task") or ""),
            str(fields.get("diff") or ""),
            str(fields.get("language") or ""),
            str(fields.get("tests") or ""),
            str(fields.get("notes") or ""),
            bool(fields.get("raw_code") or False),
        )
    if family == "tool_call":
        return tool_call_state(
            str(fields.get("request") or ""),
            list(fields.get("tools") or []),
            fields.get("proposed_calls"),
            fields.get("history"),
        )
    if family == "agent_trace":
        return agent_trace_state(
            str(fields.get("task") or ""),
            fields.get("agent") if isinstance(fields.get("agent"), dict) else None,
            fields.get("constraints"),
            fields.get("trace_summary"),
            fields.get("recent_steps"),
        )
    if family == "routing":
        return routing_state(str(fields.get("request") or ""), fields.get("context"))
    raise KeyError(family)
