---
license: apache-2.0
base_model: convaiinnovations/laya
library_name: transformers
tags:
- laya
- modernbert
- rlcd
- code-review
- tool-calling
- llm-routing
- calibrated-decisions
language:
- en
pipeline_tag: text-classification
---

# Coding_Decision_Agent

A [Laya](https://huggingface.co/convaiinnovations/laya) decision model fine-tuned to grade a coding agent's work. It does not write code and it does not emit prose. Given a state (a diff, a tool call, an agent trace, or a user request) and a fixed set of typed questions, it returns a label and a calibrated probability for every option in one forward pass.

The training code is MIT and lives at the GitHub repo linked from the model card sidebar once published. These weights are a derivative of `convaiinnovations/laya` (Apache-2.0), so the checkpoint stays Apache-2.0.

## Results on the held-out test split

| | |
|---|---|
| accuracy | {{ACCURACY}} |
| soft accuracy | {{SOFT_ACCURACY}} |
| Brier (lower is better) | {{BRIER}} |
| ECE (lower is better) | {{ECE}} |
| score MAE | {{SCORE_MAE}} |

{{FAMILY_TABLE}}

`accuracy` is argmax against the gold label. `soft accuracy` is the dot product of the gold distribution and the predicted distribution. Confidence is temperature-scaled per question type on a slice that was held out of training.

## What it grades

| family | questions |
|---|---|
| `code_review` | `code_quality` (0–4), `instruction_followed` (no/yes), `likely_correct` (no/yes), `merge_action` (accept / request_changes / needs_tests / reject) |
| `tool_call` | `tool_selection`, `parameter_structure`, `sequence_accuracy`, `query_coverage` (0–2), `should_call_tool` (call_tool / ask_followup / answer_directly / cannot_answer), `call_verdict` (execute / fix_args / wrong_tool / abstain) |
| `agent_trace` | `action` (continue / observe / human_review / stop), `needs_review` (no/yes), `outcome` (success / partial / failure / harmful), `risk` (0–3), `urgency` (0–3) |
| `routing` | `model_tier` (small_fast / mid / frontier / reasoning), `task_difficulty` (0–3), `skill` (code_edit / debug / write_tests / refactor / explain / shell_ops / research / plan) |

Question wording is fixed. Use the same wording the model was trained on; it ships in the `coding_decision_agent` package and in `configs/schema.py` of the training repo.

## Use it

```bash
pip install laya
```

```python
import laya
from coding_decision_agent import CodingDecisionAgent

grader = CodingDecisionAgent("S4MPL3BI4S/Coding_Decision_Agent")

verdict = grader.grade_code(
    task="Fix the off-by-one in paginate()",
    diff=open("change.diff").read(),
    tests=open("pytest.txt").read(),
)
print(verdict.merge_action, verdict.answer_confidence["merge_action"])

route = grader.route_model("Refactor the billing service to use dependency injection")
print(route.model_tier, route.skill)

# or call Laya directly
agent = laya.load("S4MPL3BI4S/Coding_Decision_Agent")
result = agent.predict(state, questions)
```

Gate on `answer_confidence` (the calibrated one), not on the entropy-style `confidence` field. A threshold of 0.7 is a reasonable starting point; measure it on your own traces before trusting it.

## Training

RLCD on `convaiinnovations/laya` (ModernBERT-large, 421M): a GRPO-style policy gradient against proper scoring rules, plus soft cross-entropy on the teacher distribution. One temperature per question type (`choice`, `score`, `noul`) is fit on a slice removed before training. `temperature_by_options` from the base checkpoint is dropped so it cannot mask the new fit.

Soft targets come from dataset signals (programmatic scores, multi-judge votes, test outcomes, gold distributions) blended with a local teacher (`Qwen/Qwen3-32B-AWQ` served by vLLM). No paid API.

| family | sources |
|---|---|
| agent_trace | `LocalLLaMA/typed-decisions` (`agent_trace_observability`), gold probabilities |
| tool_call | `ServiceNow-AI/AgentJudgeBench`, `nvidia/When2Call` |
| code_review | `coseal/CodeUltraFeedback`, `nebius/SWE-agent-trajectories` |
| routing | `withmartian/routerbench`, optional `ynulihao/LLMRouterBench` |

The case dataset built by this run is `S4MPL3BI4S/coding-decision-cases`.

## Limits

- Context is 2048 tokens (`head_max_len` 320 for the options, the rest for the state). Diffs and tool lists are compacted before they are scored; a huge patch is truncated around the changed lines.
- Yes/no questions are two-option `choice` items with the keys `no` and `yes`, not Laya `noul`, because the English checkpoint can follow the noul option labels instead of the state.
- At most 8 options per question. Skills and model tiers are coarse on purpose.
- Calibrated on this dataset's label style. Check ECE on your own traffic before gating a production agent on the confidence.
