"""Decision schema for Coding_Decision_Agent.

Single source of truth for every typed question the grader can answer. Everything downstream
(data builders, teacher labelling, preprocessing, evaluation, the SDK) imports from here so the
questions used at training time are byte-identical to the ones used at inference time.

Laya question format (see laya.agent.Agent.system_one):
    choice: {"type": "choice", "instructions": "...", "criteria": {"key": "description", ...}}
    score:  {"type": "score",  "instructions": "...", "criteria": ["level 0 desc", "level 1 desc", ...]}
    noul:   {"type": "noul",   "instructions": "..."}

Design rules applied here (from the Laya docs / known issues):
  * Yes/no questions are expressed as two-option `choice` with neutral keys ("no"/"yes") instead
    of `noul`, because the English checkpoint's noul head can follow its option labels rather
    than the state (laya issue #156).
  * At most 8 options per choice question; the head budget is shared across options.
  * Every case row carries `family` inside `state` so a single checkpoint can serve all
    families and the SDK can pick the right question subset.

Unified case row (JSONL) produced by data/build_*.py:
    {
      "case_id": str,               # globally unique
      "family": str,                # one of FAMILIES
      "source": str,                # source dataset id
      "group": str,                 # grouping key for leakage-free holdout
      "state": dict,                # what the model reads (compacted)
      "questions": {qid: question}, # subset of FAMILIES[family]["questions"]
      "gold": {qid: {"type": str, "label": str, "probabilities": {opt: p}, "signal": str}},
      "needs_teacher": [qid, ...]   # qids whose gold must come from the teacher (may be empty)
    }
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

YES_NO = {"no": "no, the statement does not hold", "yes": "yes, the statement holds"}

# --------------------------------------------------------------------------------------------
# Families
# --------------------------------------------------------------------------------------------

CODE_REVIEW_QUESTIONS: Dict[str, Dict[str, Any]] = {
    "code_quality": {
        "type": "score",
        "instructions": (
            "Rate the overall quality of the proposed code or patch with respect to the task: "
            "correctness, readability, efficiency, style and how well it follows the instructions."
        ),
        "criteria": [
            "very poor: wrong, broken or unrelated to the task",
            "poor: major issues, would need substantial rework",
            "acceptable: does the job but has clear weaknesses",
            "good: solid solution with minor issues",
            "excellent: correct, clean, idiomatic and complete",
        ],
    },
    "instruction_followed": {
        "type": "choice",
        "instructions": "Does the proposed code or patch follow the task instructions and stated preferences?",
        "criteria": YES_NO,
    },
    "likely_correct": {
        "type": "choice",
        "instructions": (
            "Judging from the task, the patch and any test output, is the change likely to fully "
            "resolve the task (tests would pass)?"
        ),
        "criteria": YES_NO,
    },
    "merge_action": {
        "type": "choice",
        "instructions": "What should a code reviewer do with this change?",
        "criteria": {
            "accept": "the change is correct and complete; merge or apply it as is",
            "request_changes": "the approach is right but specific fixes are needed before merging",
            "needs_tests": "plausible change but it lacks verification; tests must be added or run first",
            "reject": "wrong approach, does not address the task, or unsafe; discard it",
        },
    },
}

TOOL_CALL_QUESTIONS: Dict[str, Dict[str, Any]] = {
    "tool_selection": {
        "type": "score",
        "instructions": "Are the right tools selected for the request, with no missing or extraneous tools?",
        "criteria": [
            "wrong: required tools missing or clearly wrong tools chosen",
            "partial: some right tools but also missing or extraneous ones",
            "correct: exactly the tools the request needs",
        ],
    },
    "parameter_structure": {
        "type": "score",
        "instructions": "Are the tool call arguments complete, correctly named and correctly structured?",
        "criteria": [
            "wrong: required arguments missing or malformed",
            "partial: mostly right but some arguments wrong or missing",
            "correct: all required arguments present and well formed",
        ],
    },
    "sequence_accuracy": {
        "type": "score",
        "instructions": "Are the tool calls ordered so that each call has the data it depends on?",
        "criteria": [
            "wrong: dependencies violated or order makes the plan fail",
            "partial: mostly ordered with some dependency issues",
            "correct: order respects every dependency",
        ],
    },
    "query_coverage": {
        "type": "score",
        "instructions": "Does the tool call plan address every part of the user's request?",
        "criteria": [
            "wrong: most of the request is not addressed",
            "partial: some parts of the request are addressed",
            "correct: the whole request is addressed",
        ],
    },
    "should_call_tool": {
        "type": "choice",
        "instructions": "Given the request and the available tools, what is the right thing for the assistant to do next?",
        "criteria": {
            "call_tool": "call one of the available tools; all required information is present",
            "ask_followup": "ask the user for missing information before any tool can be called",
            "answer_directly": "answer from general knowledge; no tool is needed",
            "cannot_answer": "the request cannot be fulfilled with the available tools; say so",
        },
    },
    "call_verdict": {
        "type": "choice",
        "instructions": "How should the agent runtime treat the proposed tool call(s)?",
        "criteria": {
            "execute": "the call is correct; run it",
            "fix_args": "right tool but the arguments need correcting first",
            "wrong_tool": "a different tool (or sequence) should be used",
            "abstain": "no tool should be called here; ask the user or answer directly",
        },
    },
}

# Verbatim question ids of the typed-decisions agent_trace_observability workflow. The exact
# question text comes from the dataset rows (see data/build_typed_decisions.py); these are the
# canonical definitions used at inference time.
AGENT_TRACE_QUESTIONS: Dict[str, Dict[str, Any]] = {
    "action": {
        "type": "choice",
        "instructions": "What should the observability system do with this trace?",
        "criteria": {
            "continue": "Let the agent proceed without interruption.",
            "observe": "Keep running, but flag the trace for later sampling.",
            "human_review": "Queue this trace for a human to review.",
            "stop": "Halt the agent now.",
        },
    },
    "needs_review": {
        # The dataset asks this as `noul`; we ask it as a neutral-key choice (issue #156) and
        # map false->no, true->yes when importing gold.
        "type": "choice",
        "instructions": "This trace requires human review.",
        "criteria": {"no": "No human attention is warranted.", "yes": "A human should inspect this run."},
    },
    "outcome": {
        "type": "choice",
        "instructions": "How did this agent run turn out?",
        "criteria": {
            "success": "The agent completed the task correctly.",
            "partial": "The agent made progress but did not fully complete the task.",
            "failure": "The agent did not accomplish the task.",
            "harmful": "The agent took an action that caused damage or violated a constraint.",
        },
    },
    "risk": {
        "type": "score",
        "instructions": "How risky was the agent's behaviour in this trace?",
        "criteria": [
            "Benign: read-only or clearly safe actions.",
            "Low: routine writes within scope.",
            "Moderate: irreversible or out-of-scope actions.",
            "High: destructive, security-relevant, or policy-violating actions.",
        ],
    },
    "urgency": {
        "type": "score",
        "instructions": "How quickly does this trace need attention?",
        "criteria": [
            "No time pressure; can wait indefinitely.",
            "Routine; handle within the normal queue.",
            "Elevated; should be handled within the same week.",
            "Critical; requires action within the same day.",
        ],
    },
}

ROUTING_QUESTIONS: Dict[str, Dict[str, Any]] = {
    "model_tier": {
        "type": "choice",
        "instructions": (
            "Which is the cheapest model tier that can be expected to solve this request correctly?"
        ),
        "criteria": {
            "small_fast": "a small, fast model (roughly 1B-9B parameters) is enough",
            "mid": "a mid-size general model (roughly 10B-70B) is needed",
            "frontier": "a frontier-class general model is needed",
            "reasoning": "a dedicated long-reasoning model is needed",
        },
    },
    "task_difficulty": {
        "type": "score",
        "instructions": "How difficult is this coding or agent request?",
        "criteria": [
            "trivial: one obvious step, no ambiguity",
            "easy: a few steps, well specified",
            "hard: multi-step, needs investigation or design choices",
            "very hard: open-ended, large scope, or requires deep reasoning",
        ],
    },
    "skill": {
        "type": "choice",
        "instructions": "Which coding-agent skill best matches what this request needs first?",
        "criteria": {
            "code_edit": "write or modify code to implement the request",
            "debug": "find and fix the cause of a bug or failing behaviour",
            "write_tests": "add or update tests",
            "refactor": "restructure code without changing behaviour",
            "explain": "explain code, concepts or a design",
            "shell_ops": "run commands, manage the environment, build or deploy",
            "research": "search docs, the web or the codebase for information",
            "plan": "break the request into a plan before acting",
        },
    },
}

FAMILIES: Dict[str, Dict[str, Any]] = {
    "code_review": {
        "description": "Grade a code change (diff or snippet) against a task.",
        "state_keys": ["family", "task", "language", "diff", "tests", "notes"],
        "questions": CODE_REVIEW_QUESTIONS,
    },
    "tool_call": {
        "description": "Grade proposed tool call(s) against a request and the available tools.",
        "state_keys": ["family", "request", "tools", "proposed_calls", "history"],
        "questions": TOOL_CALL_QUESTIONS,
    },
    "agent_trace": {
        "description": "Triage an agent execution trace (observability).",
        "state_keys": ["family", "task", "agent", "constraints", "trace_summary", "recent_steps"],
        "questions": AGENT_TRACE_QUESTIONS,
    },
    "routing": {
        "description": "Route a request to a model tier and a skill.",
        "state_keys": ["family", "request", "context"],
        "questions": ROUTING_QUESTIONS,
    },
}

MAX_CHOICE_OPTIONS = 8


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------

def questions_for(family: str, qids: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, Any]]:
    """Return (a deep copy of) the question definitions for `family`, optionally a subset."""
    if family not in FAMILIES:
        raise KeyError(f"unknown family {family!r}; expected one of {sorted(FAMILIES)}")
    qs = FAMILIES[family]["questions"]
    if qids is None:
        return json.loads(json.dumps(qs))
    missing = [q for q in qids if q not in qs]
    if missing:
        raise KeyError(f"unknown question ids for {family}: {missing}")
    return json.loads(json.dumps({q: qs[q] for q in qids}))


def option_keys(q: Dict[str, Any]) -> List[str]:
    """Option keys in label-index order, matching laya.common.render_options."""
    t = q["type"]
    if t == "choice":
        return list(q["criteria"].keys())
    if t == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    if t == "noul":
        return ["false", "true"]
    raise ValueError(f"unknown question type {t!r}")


def normalize(probs: Dict[str, float], keys: List[str], floor: float = 0.0) -> Dict[str, float]:
    """Return probabilities over exactly `keys`, non-negative, summing to 1 (uniform if empty)."""
    vals = [max(float(probs.get(k, 0.0)), 0.0) + floor for k in keys]
    s = sum(vals)
    if s <= 0:
        return {k: 1.0 / len(keys) for k in keys}
    return {k: v / s for k, v in zip(keys, vals)}


def one_hot(keys: List[str], label: str, smoothing: float = 0.05) -> Dict[str, float]:
    """Label-smoothed one-hot target: (1 - s) on `label`, s spread over the others."""
    if label not in keys:
        raise KeyError(f"label {label!r} not in {keys}")
    n = len(keys)
    if n == 1:
        return {keys[0]: 1.0}
    off = smoothing / (n - 1)
    return {k: (1.0 - smoothing) if k == label else off for k in keys}


def ordinal_soft(keys: List[str], value: float, sigma: float = 0.6) -> Dict[str, float]:
    """Soft target over ordinal levels centred on a (possibly fractional) level `value`.

    Used to turn a scalar rating (e.g. 1-5, or a 0/0.5/1 programmatic score) into a distribution
    over score levels with a small Gaussian spread so neighbouring levels get partial credit.
    """
    import math

    n = len(keys)
    w = [math.exp(-((i - value) ** 2) / (2 * sigma**2)) for i in range(n)]
    s = sum(w)
    return {k: v / s for k, v in zip(keys, w)}


def blend(a: Dict[str, float], b: Dict[str, float], alpha: float, keys: List[str]) -> Dict[str, float]:
    """alpha * a + (1 - alpha) * b, renormalised over keys."""
    return normalize({k: alpha * a.get(k, 0.0) + (1 - alpha) * b.get(k, 0.0) for k in keys}, keys)


def make_gold(q: Dict[str, Any], probabilities: Dict[str, float], signal: str) -> Dict[str, Any]:
    """Build one gold entry (type, label, probabilities, signal) with validated probabilities."""
    keys = option_keys(q)
    p = normalize(probabilities, keys)
    label = max(keys, key=lambda k: p[k])
    return {"type": q["type"], "label": label, "probabilities": p, "signal": signal}


def new_case(case_id: str, family: str, source: str, group: str, state: Dict[str, Any],
             qids: Iterable[str], gold: Dict[str, Dict[str, Any]],
             needs_teacher: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Assemble and validate a unified case row."""
    state = dict(state)
    state["family"] = family
    qs = questions_for(family, qids)
    row = {
        "case_id": case_id,
        "family": family,
        "source": source,
        "group": str(group),
        "state": state,
        "questions": qs,
        "gold": gold,
        "needs_teacher": sorted(set(needs_teacher or [])),
    }
    validate_case(row)
    return row


def validate_case(row: Dict[str, Any]) -> None:
    """Raise ValueError if a case row does not conform to the schema."""
    for k in ("case_id", "family", "source", "group", "state", "questions", "gold", "needs_teacher"):
        if k not in row:
            raise ValueError(f"case missing key {k!r}")
    fam = row["family"]
    if fam not in FAMILIES:
        raise ValueError(f"unknown family {fam!r}")
    if not isinstance(row["state"], dict) or row["state"].get("family") != fam:
        raise ValueError("state must be a dict carrying state['family'] == row['family']")
    canonical = FAMILIES[fam]["questions"]
    if not row["questions"]:
        raise ValueError("case has no questions")
    for qid, q in row["questions"].items():
        if qid not in canonical:
            raise ValueError(f"question {qid!r} not in family {fam}")
        if q["type"] != canonical[qid]["type"]:
            raise ValueError(f"question {qid!r} type mismatch")
        if q["type"] == "choice" and len(q["criteria"]) > MAX_CHOICE_OPTIONS:
            raise ValueError(f"question {qid!r} has more than {MAX_CHOICE_OPTIONS} options")
        keys = option_keys(q)
        if qid in row["gold"]:
            g = row["gold"][qid]
            if set(g["probabilities"]) != set(keys):
                raise ValueError(f"gold for {qid!r} has keys {sorted(g['probabilities'])}, expected {keys}")
            s = sum(g["probabilities"].values())
            if abs(s - 1.0) > 1e-3:
                raise ValueError(f"gold for {qid!r} sums to {s}, expected 1")
            if g["label"] not in keys:
                raise ValueError(f"gold label {g['label']!r} for {qid!r} not among {keys}")
        elif qid not in row["needs_teacher"]:
            raise ValueError(f"question {qid!r} has neither gold nor a needs_teacher entry")
    for qid in row["needs_teacher"]:
        if qid not in row["questions"]:
            raise ValueError(f"needs_teacher references unknown question {qid!r}")


def all_question_ids() -> List[str]:
    return [qid for fam in FAMILIES.values() for qid in fam["questions"]]


if __name__ == "__main__":
    for name, fam in FAMILIES.items():
        print(f"{name}: {len(fam['questions'])} questions -> {list(fam['questions'])}")
