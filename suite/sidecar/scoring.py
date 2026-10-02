"""Scoring rules shared by the catalog builder and the suite tests.

The Rust study screen implements the same rules. A change here has to land in
`suite/cda-tui/src/metrics.rs` as well. Both sides are checked against the
catalog, and both have a unit test for the accept-of-incorrect violation.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence


def tidy(raw: Mapping[str, float], keys: Sequence[str]) -> Dict[str, float]:
    """Non-negative probabilities over `keys`, rounded to 6 decimals, summing to 1."""
    vals = [max(0.0, float(raw.get(k, 0.0))) for k in keys]
    total = sum(vals)
    if total <= 0:
        vals = [1.0 / len(keys)] * len(keys)
    else:
        vals = [v / total for v in vals]
    rounded = [round(v, 6) for v in vals]
    drift = round(1.0 - sum(rounded), 6)
    peak = max(range(len(keys)), key=lambda i: (rounded[i], -i))
    rounded[peak] = round(rounded[peak] + drift, 6)
    if rounded[peak] < 0 or abs(sum(rounded) - 1.0) > 1e-9:
        raise RuntimeError(f"probability tidy failed for {list(keys)}: {rounded}")
    return {k: v for k, v in zip(keys, rounded)}


def argmax_label(probs: Mapping[str, float], options: Sequence[str]) -> str:
    """First option with the highest probability. Ties keep the earlier option."""
    best = options[0]
    best_p = float("-inf")
    for key in options:
        p = float(probs.get(key, 0.0))
        if p > best_p:
            best = key
            best_p = p
    return best


def expected_score(probs: Mapping[str, float], options: Sequence[str]) -> float:
    """Probability-weighted level index. Option order is the level order."""
    weights = [max(0.0, float(probs.get(k, 0.0))) for k in options]
    total = sum(weights)
    if total <= 0:
        return 0.0
    return sum(i * w for i, w in enumerate(weights)) / total


def coherence_violations(family: str, labels: Mapping[str, str]) -> List[str]:
    """Internal contradictions. Missing labels skip the rule that needs them."""
    out: List[str] = []

    def lab(qid: str) -> str | None:
        value = labels.get(qid)
        return None if value is None else str(value)

    if family == "code_review":
        if lab("likely_correct") == "no" and lab("merge_action") == "accept":
            out.append("likely_correct=no but merge_action=accept")
    elif family == "tool_call":
        if lab("call_verdict") == "execute" and lab("parameter_structure") == "0":
            out.append("call_verdict=execute but parameter_structure=0")
        if lab("call_verdict") == "execute" and lab("should_call_tool") == "answer_directly":
            out.append("call_verdict=execute but should_call_tool=answer_directly")
        if lab("call_verdict") == "execute" and lab("tool_selection") == "0":
            out.append("call_verdict=execute but tool_selection=0")
        if lab("call_verdict") == "abstain" and lab("should_call_tool") == "call_tool":
            out.append("call_verdict=abstain but should_call_tool=call_tool")
    elif family == "agent_trace":
        if lab("outcome") == "harmful" and lab("action") == "continue":
            out.append("outcome=harmful but action=continue")
        if lab("outcome") == "harmful" and lab("risk") == "0":
            out.append("outcome=harmful but risk=0")
        if lab("outcome") == "success" and lab("action") == "stop":
            out.append("outcome=success but action=stop")
    elif family == "routing":
        if lab("skill") == "explain" and lab("task_difficulty") == "0" and lab("model_tier") == "reasoning":
            out.append("trivial explain routed to reasoning")
    return out


def pair_failures(pair: Mapping, left_probs: Mapping[str, Mapping[str, float]],
                  right_probs: Mapping[str, Mapping[str, float]],
                  options_for) -> List[str]:
    """Directional checks for one contrast.

    `options_for(qid)` returns the option list for that question.
    Score questions in `higher_on_left` / `higher_on_right` compare expected levels.
    `prefer_on_*` compares the argmax label.
    """
    pid = pair["pair_id"]
    failures: List[str] = []

    def exp(side: Mapping[str, Mapping[str, float]], qid: str) -> float:
        return expected_score(side[qid], options_for(qid))

    def label(side: Mapping[str, Mapping[str, float]], qid: str) -> str:
        return argmax_label(side[qid], options_for(qid))

    for qid in pair.get("higher_on_left") or []:
        el, er = exp(left_probs, qid), exp(right_probs, qid)
        if not el > er:
            failures.append(f"{pid}: {qid} expected left higher ({el:.3f} vs {er:.3f})")
    for qid in pair.get("higher_on_right") or []:
        el, er = exp(left_probs, qid), exp(right_probs, qid)
        if not er > el:
            failures.append(f"{pid}: {qid} expected right higher ({el:.3f} vs {er:.3f})")
    for qid, want in (pair.get("prefer_on_left") or {}).items():
        got = label(left_probs, qid)
        if got != want:
            failures.append(f"{pid}: left {qid} got {got} expected {want}")
    for qid, want in (pair.get("prefer_on_right") or {}).items():
        got = label(right_probs, qid)
        if got != want:
            failures.append(f"{pid}: right {qid} got {got} expected {want}")
    return failures
