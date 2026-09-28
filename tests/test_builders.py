"""Offline tests for the dataset builders' row converters (no network, synthetic rows)."""
import json

from configs.schema import questions_for, validate_case
from data import build_agentjudgebench as AJB
from data import build_codeultrafeedback as CUF
from data import build_swe_trajectories as SWE
from data import build_typed_decisions as TD
from data import build_when2call as W2C
from data.tiers import cheapest_tier_distribution, model_tier, tier_rates


def test_typed_decisions_convert():
    row = {
        "id": "tr_agent_trace_observability_000001",
        "split": "train",
        "state": json.dumps({"agent": {"autonomy": "checkpointed", "model": "m"}, "constraints": ["c"],
                             "task": "do x", "trace_summary": {"steps": 3}}),
        "gold": json.dumps({
            "action": {"probabilities": {"continue": 0.6, "observe": 0.2, "human_review": 0.15, "stop": 0.05}},
            "needs_review": {"probabilities": {"false": 0.7, "true": 0.3}},
            "outcome": {"probabilities": {"success": 0.5, "partial": 0.3, "failure": 0.1, "harmful": 0.1}},
            "risk": {"probabilities": {"0": 0.7, "1": 0.2, "2": 0.1, "3": 0.0}},
            "urgency": {"probabilities": {"0": 0.9, "1": 0.1, "2": 0.0, "3": 0.0}},
        }),
    }
    c = TD.convert_row(row, questions_for("agent_trace"))
    validate_case(c)
    assert c["gold"]["needs_review"]["probabilities"]["no"] == 0.7
    assert c["split_hint"] == "train" and c["needs_teacher"] == []


def test_agentjudgebench_convert_with_votes():
    q = questions_for("tool_call")
    tools = [{"type": "function", "function": {"name": "a", "description": "A", "parameters": {"type": "dict", "properties": {"x": {"type": "string"}}, "required": ["x"]}}}]
    row = {"id": "linear_1", "dag_type": "linear", "difficulty": "easy", "user_query": "do a with x=1",
           "available_tools": tools, "generated_tool_calls": [{"function": {"name": "a", "arguments": {"x": "1"}}}],
           "expected_tool_calls": [{"name": "a", "arguments": {"x": "1"}}],
           "score_tool_selection": 1.0, "score_parameter_structure": 0.5, "score_sequence_accuracy": 1.0,
           "score_query_coverage": 1.0, "overall_programmatic_score": 0.875}
    votes = {("linear_1", "g", "easy"): {"parameter_structure": {"1": 4, "2": 2}}}
    c = AJB.convert_row(row, "g", q, votes, add_should_call=True)
    validate_case(c)
    ps = c["gold"]["parameter_structure"]
    assert ps["signal"] == "programmatic+judges" and ps["label"] == "1"
    assert c["gold"]["call_verdict"]["label"] == "fix_args"
    assert c["gold"]["should_call_tool"]["label"] == "call_tool"
    assert "call_verdict" in c["needs_teacher"]


def test_when2call_classify_and_cases():
    q = questions_for("tool_call")
    assert W2C.classify('<TOOLCALL>[{"name": "f", "arguments": {}}]</TOOLCALL>') == "call_tool"
    assert W2C.classify("Which city do you mean?") == "ask_followup"
    assert W2C.classify("Apologies, but I'm unable to access real-time data.") == "cannot_answer"
    assert W2C.classify("Paris is the capital of France.") == "answer_directly"
    row = {"tools": [json.dumps({"name": "f", "description": "F", "parameters": {"type": "dict", "properties": {}}})],
           "messages": [{"role": "user", "content": "call f"}],
           "chosen_response": {"role": "assistant", "content": "Which f?"},
           "rejected_response": {"role": "assistant", "content": '<TOOLCALL>[{"name": "f", "arguments": {}}]</TOOLCALL>'}}
    c = W2C.from_pref(row, 0, q)
    validate_case(c)
    assert c["gold"]["should_call_tool"]["label"] == "ask_followup"
    assert c["gold"]["call_verdict"]["label"] == "abstain"
    assert c["state"]["proposed_calls"].startswith("1. f(")
    mcq = {"uuid": "u1", "source_id": "s1", "question": "q", "correct_answer": "cannot_answer",
           "answers": {"tool_call": json.dumps({"name": "f", "arguments": {}})}, "tools": []}
    c2 = W2C.from_mcq(mcq, q)
    assert c2["split_hint"] == "test" and c2["gold"]["call_verdict"]["label"] == "abstain"


def test_codeultrafeedback_convert():
    row = {"instruction": "Write a Python function that reverses a list.", "preference": "readability",
           "responses": [{"model": "gpt-4", "response": "```python\ndef rev(x):\n    return x[::-1]\n```"},
                         {"model": "codellama-7b-instruct", "response": "use a loop"}],
           "annotations": [{"model": "gpt-4", "rating": "5"}, {"model": "codellama-7b-instruct", "rating": "2"}]}
    cases = CUF.convert(row, 0, questions_for("code_review"), questions_for("routing"), True)
    for c in cases:
        validate_case(c)
    review = [c for c in cases if c["family"] == "code_review"]
    route = [c for c in cases if c["family"] == "routing"]
    assert len(review) == 2 and len(route) == 1
    assert review[0]["gold"]["code_quality"]["label"] == "4" and review[0]["state"]["language"] == "python"
    assert review[1]["gold"]["merge_action"]["label"] == "reject"
    assert route[0]["gold"]["model_tier"]["label"] == "frontier"  # only gpt-4 solved it


def test_swe_convert_never_leaks_eval_logs():
    traj = [{"role": "system", "system_prompt": "SETTING"},
            {"role": "user", "text": "We're currently solving the following issue within our repository. Here's the issue text:\nISSUE:\nBug in foo\n\nINSTRUCTIONS:\nfix it"},
            {"role": "ai", "text": "ok"}]
    row = {"instance_id": "org__repo-12", "model_name": "swe-agent-llama-70b", "target": True,
           "trajectory": json.dumps(traj), "exit_status": "submitted",
           "generated_patch": "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-x=1\n+x=2\n",
           "eval_logs": "SECRET PASSED"}
    c = SWE.convert(row, questions_for("code_review"))
    validate_case(c)
    assert "SECRET" not in json.dumps(c["state"])
    assert c["state"]["task"] == "Bug in foo"
    assert c["gold"]["likely_correct"]["label"] == "yes" and c["group"] == "org__repo"


def test_tiers():
    assert model_tier("mistralai/mistral-7b-chat") == "small_fast"
    assert model_tier("meta/llama-2-70b-chat") == "mid"
    assert model_tier("mistralai/mixtral-8x7b-chat") == "mid"
    assert model_tier("gpt-4-1106-preview") == "frontier"
    assert model_tier("claude-instant-v1") == "small_fast"
    assert model_tier("o3-mini") == "reasoning"
    rates = tier_rates([("mistral-7b", 0), ("llama-70b", 1), ("gpt-4", 1)])
    d = cheapest_tier_distribution(rates)
    assert d["mid"] > d["frontier"] >= d["reasoning"] and d["small_fast"] < d["mid"]
    d2 = cheapest_tier_distribution(tier_rates([("mistral-7b", 0), ("llama-70b", 0), ("gpt-4", 0)]))
    assert max(d2, key=d2.get) == "reasoning"  # nobody solved it -> strongest tier
    assert abs(sum(d.values()) - 1) < 1e-9
