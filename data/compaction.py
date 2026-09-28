"""State compaction: keep the informative parts of code diffs, tool schemas, tool calls and
agent traces inside Laya's state token budget.

`laya.common.build_sequence` truncates the serialized state from the right. Without compaction a
long diff or a long tool list would push the important part (the change, the proposed call) out
of the window. Every function here is deterministic and budgeted in characters; budgets come
from configs/state_compaction.yaml.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from data.common import compaction_config

_CFG: Optional[Dict[str, Any]] = None


def cfg(section: Optional[str] = None) -> Dict[str, Any]:
    global _CFG
    if _CFG is None:
        _CFG = compaction_config()
    return _CFG[section] if section else _CFG


# --------------------------------------------------------------------------------------------
# Text primitives
# --------------------------------------------------------------------------------------------

def truncate(text: Any, n: int, mode: str = "head", marker: str = " [...] ") -> str:
    """Cut `text` to at most `n` characters. mode: head | tail | middle."""
    if text is None:
        return ""
    s = str(text)
    if n <= 0:
        return ""
    if len(s) <= n:
        return s
    if mode == "tail":
        return marker.strip() + " " + s[-(n - len(marker)):]
    if mode == "middle":
        keep = n - len(marker)
        head = keep * 2 // 3
        return s[:head] + marker + s[-(keep - head):]
    return s[: n - len(marker)] + marker.strip()


def squash_ws(text: str) -> str:
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", text or "")).strip()


# --------------------------------------------------------------------------------------------
# Unified diffs
# --------------------------------------------------------------------------------------------

_FILE_HDR = re.compile(r"^diff --git a/(.+?) b/(.+)$")


def _split_diff_files(diff: str) -> List[Dict[str, Any]]:
    """Split a unified diff into files -> hunks (list of lines)."""
    files: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    hunk: Optional[List[str]] = None
    for line in (diff or "").splitlines():
        m = _FILE_HDR.match(line)
        if m or line.startswith("--- a/") and cur is None:
            if m:
                cur = {"path": m.group(2), "hunks": []}
                files.append(cur)
                hunk = None
                continue
        if line.startswith("--- ") or line.startswith("+++ "):
            if cur is None:
                cur = {"path": line[4:].split("\t")[0].lstrip("ab/"), "hunks": []}
                files.append(cur)
            elif line.startswith("+++ ") and cur.get("path", "").startswith("/dev/null"):
                cur["path"] = line[4:].lstrip("b/")
            continue
        if line.startswith("index ") or line.startswith("new file mode") or line.startswith("deleted file mode") \
                or line.startswith("similarity index") or line.startswith("rename ") or line.startswith("old mode") \
                or line.startswith("new mode") or line.startswith("Binary files"):
            continue
        if line.startswith("@@"):
            if cur is None:
                cur = {"path": "<unknown>", "hunks": []}
                files.append(cur)
            hunk = [line]
            cur["hunks"].append(hunk)
            continue
        if hunk is not None:
            hunk.append(line)
        elif cur is not None and line.strip():
            # stray content before the first hunk header; keep as its own hunk
            hunk = [line]
            cur["hunks"].append(hunk)
    return files


def _compact_hunk(lines: List[str], context: int) -> List[str]:
    """Keep changed lines with `context` unchanged lines around them; elide the rest."""
    if not lines:
        return []
    header, body = lines[0], lines[1:]
    changed = [i for i, l in enumerate(body) if l[:1] in "+-"]
    if not changed:
        return [header] + body[: 2 * context + 1]
    keep = set()
    for i in changed:
        keep.update(range(max(0, i - context), min(len(body), i + context + 1)))
    out = [header]
    last = -1
    for i in sorted(keep):
        if i > last + 1:
            out.append(f"  ... ({i - last - 1} unchanged lines)")
        out.append(body[i])
        last = i
    if last < len(body) - 1:
        out.append(f"  ... ({len(body) - 1 - last} unchanged lines)")
    return out


def compact_diff(diff: str, budget: Optional[int] = None, context_lines: Optional[int] = None,
                 max_files: Optional[int] = None) -> str:
    """Budgeted, hunk-aware compaction of a unified diff.

    Keeps every file header, all +/- lines with a little context, and splits the character
    budget across files proportionally to their size. Files beyond `max_files` are listed by
    name only.
    """
    c = cfg("code_review")
    budget = budget or c["diff_chars"]
    context_lines = c["context_lines_per_hunk"] if context_lines is None else context_lines
    max_files = max_files or c["max_files_listed"]
    if not diff or not diff.strip():
        return ""
    files = _split_diff_files(diff)
    if not files:
        return truncate(diff, budget, "middle")
    rendered = []
    for f in files:
        lines = []
        for h in f["hunks"]:
            lines.extend(_compact_hunk(h, context_lines))
        rendered.append((f["path"], "\n".join(lines)))
    shown, rest = rendered[:max_files], rendered[max_files:]
    total = sum(len(body) for _, body in shown) or 1
    parts = []
    header_cost = sum(len(p) + 12 for p, _ in shown)
    body_budget = max(budget - header_cost, budget // 2)
    for path, body in shown:
        share = max(200, int(body_budget * len(body) / total))
        parts.append(f"### {path}\n{truncate(body, share, 'middle')}")
    if rest:
        parts.append("### other files changed: " + ", ".join(p for p, _ in rest))
    out = "\n".join(parts)
    return truncate(out, budget, "middle")


def diff_stats(diff: str) -> Dict[str, int]:
    files = _split_diff_files(diff or "")
    added = sum(1 for l in (diff or "").splitlines() if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in (diff or "").splitlines() if l.startswith("-") and not l.startswith("---"))
    return {"files": len(files), "added": added, "removed": removed}


# --------------------------------------------------------------------------------------------
# Test output
# --------------------------------------------------------------------------------------------

_FAIL_PAT = re.compile(r"(FAIL|ERROR|Error|error:|Traceback|AssertionError|assert |failed|passed|Exception)", re.I)


def compact_tests(output: str, budget: Optional[int] = None) -> str:
    """Prefer failure lines and the summary tail of a test run."""
    budget = budget or cfg("code_review")["tests_chars"]
    if not output:
        return ""
    lines = [l.rstrip() for l in output.splitlines() if l.strip()]
    if not lines:
        return ""
    tail = "\n".join(lines[-6:])
    fails = [l for l in lines[:-6] if _FAIL_PAT.search(l)]
    head = "\n".join(fails[:20])
    text = (head + "\n" + tail).strip() if head else tail
    return truncate(text, budget, "tail")


# --------------------------------------------------------------------------------------------
# Tool schemas and tool calls
# --------------------------------------------------------------------------------------------

def _tool_fields(tool: Dict[str, Any]) -> Dict[str, Any]:
    """Accept OpenAI ({"type":"function","function":{...}}), plain {"name","description","parameters"}
    and BFCL/xLAM-style dicts."""
    if "function" in tool and isinstance(tool["function"], dict):
        tool = tool["function"]
    name = tool.get("name") or tool.get("tool_name") or tool.get("id") or "tool"
    desc = tool.get("description") or tool.get("desc") or ""
    params = tool.get("parameters") or tool.get("input_schema") or tool.get("args") or {}
    return {"name": str(name), "description": str(desc), "parameters": params}


def compact_tool(tool: Dict[str, Any], max_param_desc: Optional[int] = None, with_param_desc: bool = True,
                 max_desc: int = 140) -> str:
    """One tool -> `name(param: type*, ...) - description` (asterisk = required)."""
    c = cfg("tool_call")
    max_param_desc = max_param_desc or c["max_param_desc_chars"]
    t = _tool_fields(tool)
    params = t["parameters"]
    props = params.get("properties", params) if isinstance(params, dict) else {}
    required = set(params.get("required", []) or []) if isinstance(params, dict) else set()
    items = []
    if isinstance(props, dict):
        for pname, spec in props.items():
            if pname in ("type", "required", "properties") and not isinstance(spec, dict):
                continue
            ptype = spec.get("type", "any") if isinstance(spec, dict) else "any"
            if isinstance(ptype, list):
                ptype = "|".join(map(str, ptype))
            pdesc = spec.get("description", "") if isinstance(spec, dict) else ""
            star = "*" if pname in required else ""
            frag = f"{pname}{star}: {ptype}"
            if pdesc and with_param_desc:
                frag += f" ({truncate(squash_ws(pdesc), max_param_desc)})"
            items.append(frag)
    desc = truncate(squash_ws(t["description"]).split(". ")[0], max_desc)
    return f"{t['name']}({', '.join(items)})" + (f" - {desc}" if desc else "")


def compact_tools(tools: Sequence[Dict[str, Any]], budget: Optional[int] = None,
                  max_tools: Optional[int] = None) -> str:
    """Render tools within `budget`, degrading gracefully: full -> no param descriptions ->
    shorter tool descriptions -> hard truncation. Parameter *names* survive as long as possible
    because a grader must see which arguments exist."""
    c = cfg("tool_call")
    budget = budget or c["tools_chars"]
    max_tools = max_tools or c["max_tools"]
    if not tools:
        return "(no tools available)"
    shown = list(tools)[:max_tools]
    tail = [f"... and {len(tools) - max_tools} more tools"] if len(tools) > max_tools else []
    per = max(80, budget // max(1, len(shown) + len(tail)))
    lines = []
    for t in shown:
        line = compact_tool(t)
        if len(line) > per:
            line = compact_tool(t, with_param_desc=False)
        if len(line) > per:
            line = compact_tool(t, with_param_desc=False, max_desc=60)
        lines.append(truncate(line, per, "middle"))
    text = "\n".join(lines + tail)
    return truncate(text, budget, "head")


def _shorten_value(v: Any, max_str: int = 160) -> Any:
    if isinstance(v, str):
        return truncate(v, max_str, "middle")
    if isinstance(v, dict):
        return {k: _shorten_value(x, max_str) for k, x in list(v.items())[:20]}
    if isinstance(v, list):
        return [_shorten_value(x, max_str) for x in v[:10]] + (["..."] if len(v) > 10 else [])
    return v


def normalize_call(call: Any) -> Dict[str, Any]:
    """Normalise a tool call to {"name": str, "arguments": dict}.

    Accepts OpenAI tool_call objects, {"name","arguments"}, {"tool","args"}, {name: {args}} and
    JSON strings.
    """
    if isinstance(call, str):
        try:
            call = json.loads(call)
        except json.JSONDecodeError:
            return {"name": "<unparsed>", "arguments": {"raw": truncate(call, 200)}}
    if isinstance(call, dict):
        if "function" in call and isinstance(call["function"], dict):
            f = call["function"]
            args = f.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"raw": truncate(args, 200)}
            return {"name": str(f.get("name", "tool")), "arguments": args if isinstance(args, dict) else {"value": args}}
        for nk, ak in (("name", "arguments"), ("name", "args"), ("tool", "args"), ("tool_name", "parameters"),
                       ("name", "parameters"), ("tool", "tool_input"), ("api_name", "arguments")):
            if nk in call:
                args = call.get(ak, {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": truncate(args, 200)}
                return {"name": str(call[nk]), "arguments": args if isinstance(args, dict) else {"value": args}}
        if len(call) == 1:
            (k, v), = call.items()
            return {"name": str(k), "arguments": v if isinstance(v, dict) else {"value": v}}
    return {"name": "<unknown>", "arguments": {"value": _shorten_value(call)}}


def compact_calls(calls: Any, budget: Optional[int] = None) -> str:
    budget = budget or cfg("tool_call")["calls_chars"]
    if calls is None:
        return "(no tool call)"
    if isinstance(calls, (dict, str)):
        calls = [calls]
    norm = [normalize_call(c) for c in calls]
    if not norm:
        return "(no tool call)"
    lines = []
    for i, c in enumerate(norm, 1):
        args = json.dumps(_shorten_value(c["arguments"]), ensure_ascii=False, separators=(",", ":"))
        lines.append(f"{i}. {c['name']}({args})")
    return truncate("\n".join(lines), budget, "middle")


# --------------------------------------------------------------------------------------------
# Agent traces
# --------------------------------------------------------------------------------------------

def compact_steps(steps: Iterable[Dict[str, Any]], max_recent: Optional[int] = None,
                  budget: Optional[int] = None) -> str:
    """Render the last `max_recent` steps as `n. ROLE: text`; earlier steps are counted only."""
    c = cfg("agent_trace")
    max_recent = max_recent or c["max_recent_steps"]
    budget = budget or c["steps_chars"]
    steps = list(steps)
    if not steps:
        return "(no steps)"
    recent = steps[-max_recent:]
    offset = len(steps) - len(recent)
    per = max(120, budget // max(1, len(recent)))
    lines = []
    if offset:
        lines.append(f"({offset} earlier steps omitted)")
    for i, s in enumerate(recent, offset + 1):
        role = str(s.get("role") or s.get("type") or "step").upper()
        text = s.get("content") or s.get("action") or s.get("text") or s.get("observation") or ""
        if isinstance(text, (dict, list)):
            text = json.dumps(_shorten_value(text), ensure_ascii=False, separators=(",", ":"))
        lines.append(f"{i}. {role}: {truncate(squash_ws(str(text)), per, 'middle')}")
    return truncate("\n".join(lines), budget, "tail")


# --------------------------------------------------------------------------------------------
# Family-level state builders
# --------------------------------------------------------------------------------------------

def code_review_state(task: str, diff: str = "", language: str = "", tests: str = "",
                      notes: str = "", raw_code: bool = False) -> Dict[str, Any]:
    """State for the code_review family. `raw_code=True` treats `diff` as a code snippet."""
    c = cfg("code_review")
    state: Dict[str, Any] = {
        "family": "code_review",
        "task": truncate(squash_ws(task), c["task_chars"], "middle"),
    }
    if language:
        state["language"] = str(language)[:40]
    if raw_code:
        state["code"] = truncate(diff, c["diff_chars"], "middle")
    else:
        state["diff_stats"] = diff_stats(diff)
        state["diff"] = compact_diff(diff, c["diff_chars"], c["context_lines_per_hunk"], c["max_files_listed"])
    if tests:
        state["tests"] = compact_tests(tests, c["tests_chars"])
    if notes:
        state["notes"] = truncate(squash_ws(notes), c["notes_chars"])
    return state


def tool_call_state(request: str, tools: Sequence[Dict[str, Any]], proposed_calls: Any = None,
                    history: Any = None, system_prompt: str = "") -> Dict[str, Any]:
    c = cfg("tool_call")
    state: Dict[str, Any] = {
        "family": "tool_call",
        "request": truncate(squash_ws(request), c["request_chars"], "middle"),
        "tools": compact_tools(tools, c["tools_chars"], c["max_tools"]),
        "proposed_calls": compact_calls(proposed_calls, c["calls_chars"]),
    }
    if history:
        if isinstance(history, (list, tuple)):
            history = compact_steps(history, 6, c["history_chars"])
        state["history"] = truncate(str(history), c["history_chars"], "tail")
    if system_prompt:
        state["system_prompt"] = truncate(squash_ws(system_prompt), 300)
    return state


def agent_trace_state(task: str, agent: Optional[Dict[str, Any]] = None, constraints: Any = None,
                      trace_summary: Any = None, recent_steps: Optional[Iterable[Dict[str, Any]]] = None,
                      extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    c = cfg("agent_trace")
    state: Dict[str, Any] = {"family": "agent_trace", "task": truncate(squash_ws(task), c["task_chars"], "middle")}
    if agent:
        state["agent"] = _shorten_value(agent, 80)
    if constraints:
        if isinstance(constraints, (list, tuple)):
            constraints = "; ".join(map(str, constraints))
        state["constraints"] = truncate(str(constraints), c["constraints_chars"])
    if trace_summary is not None:
        if isinstance(trace_summary, dict):
            state["trace_summary"] = _shorten_value(trace_summary, 120)
        else:
            state["trace_summary"] = truncate(squash_ws(str(trace_summary)), c["summary_chars"], "middle")
    if recent_steps:
        state["recent_steps"] = compact_steps(recent_steps, c["max_recent_steps"], c["steps_chars"])
    if extra:
        for k, v in extra.items():
            state.setdefault(k, _shorten_value(v, 120))
    return state


def routing_state(request: str, context: Any = None) -> Dict[str, Any]:
    c = cfg("routing")
    state: Dict[str, Any] = {"family": "routing", "request": truncate(squash_ws(request), c["request_chars"], "middle")}
    if context:
        if isinstance(context, dict):
            state["context"] = _shorten_value(context, 200)
        else:
            state["context"] = truncate(squash_ws(str(context)), c["context_chars"], "middle")
    return state


def state_chars(state: Dict[str, Any]) -> int:
    return len(json.dumps(state, ensure_ascii=False))
