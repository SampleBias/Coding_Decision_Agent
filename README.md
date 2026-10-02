# Coding_Decision_Agent

Training harness for a [Laya](https://huggingface.co/convaiinnovations/laya) grader you can drop into a coding agent. The model does not write code. It scores a diff, a tool call, an agent trace, or a routing decision and returns a label plus a calibrated probability for every option, in one forward pass.

This file is the **GitHub** README. The Hugging Face model card is a separate document: [`huggingface/README.md`](huggingface/README.md). It is uploaded as the model `README.md` by `publish/push_to_hub.py` after training.

| | |
|---|---|
| Code | this repo, MIT |
| Weights (after the RunPod run) | `S4MPL3BI4S/Coding_Decision_Agent`, Apache-2.0, because they are a fine-tune of `convaiinnovations/laya` |
| Cases dataset (after the run) | `S4MPL3BI4S/coding-decision-cases` |
| Base checkpoint | `convaiinnovations/laya` (ModernBERT-large + decision head, 421M) |

The weights are **not on the Hub yet**. Training happens on a RunPod GPU. Start at [`docs/01_RUNPOD_TRAINING.md`](docs/01_RUNPOD_TRAINING.md). What is already done, and what you still run by hand, is in [`docs/00_STATUS.md`](docs/00_STATUS.md).

## What it grades

| family | questions |
|---|---|
| `code_review` | `code_quality` (0–4), `instruction_followed`, `likely_correct`, `merge_action` (accept / request_changes / needs_tests / reject) |
| `tool_call` | `tool_selection`, `parameter_structure`, `sequence_accuracy`, `query_coverage` (0–2), `should_call_tool` (call_tool / ask_followup / answer_directly / cannot_answer), `call_verdict` (execute / fix_args / wrong_tool / abstain) |
| `agent_trace` | `action` (continue / observe / human_review / stop), `needs_review`, `outcome`, `risk`, `urgency` |
| `routing` | `model_tier` (small_fast / mid / frontier / reasoning), `task_difficulty` (0–3), `skill` (code_edit / debug / write_tests / refactor / explain / shell_ops / research / plan) |

The wording is fixed in [`configs/schema.py`](configs/schema.py) and copied into the installable package at [`sdk/coding_decision_agent/schemas.py`](sdk/coding_decision_agent/schemas.py). Train and inference must use the same text.

```python
from coding_decision_agent import CodingDecisionAgent

grader = CodingDecisionAgent()  # S4MPL3BI4S/Coding_Decision_Agent, once published
verdict = grader.grade_code(task="Fix the off-by-one in paginate()", diff=diff_text, tests=pytest_log)
if verdict.merge_action != "accept" or not verdict.confident("merge_action", 0.7):
    escalate()
```

[`sdk/examples/coding_agent_hook.py`](sdk/examples/coding_agent_hook.py) is the confidence-gated hook.

## Pipeline

```
data/build_*.py
  -> data/teacher_vllm.py          local Qwen via vLLM, soft targets
  -> data/merge_and_split.py       group holdout, class balance
  -> data/preprocess.py            laya.common.build_sequence -> items.pt
  -> train/train_rlcd.py           RLCD, torchrun, optional LoRA
  -> train/evaluate.py
  -> publish/push_to_hub.py        model card from huggingface/README.md
```

`bash runpod/run_pipeline.sh all` runs data through eval and **stops before publish**, so you can read `benchmark_report.json` first.

## Repo layout

```
configs/          schema.py, train.yaml (a100_80gb, gpu_24gb, smoke), state_compaction.yaml
data/             builders, compaction, teacher, merge/split, preprocess
train/            train_rlcd.py, calibrate.py, evaluate.py, batching.py
publish/          push_to_hub.py
huggingface/      README.md          the Hub model card, not this file
sdk/              coding_decision_agent package + examples
suite/            terminal test suite: Python sidecar + Rust TUI
runpod/           setup.sh, serve_teacher.sh, run_pipeline.sh
docs/             00_STATUS.md, 01_RUNPOD_TRAINING.md
tests/            CPU tests (pytest). No GPU required.
```

## Local checks (no GPU)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,train]"
pytest
```

This machine has no NVIDIA GPU. Do not start `run_pipeline.sh` here.

## Test suite

The TUI in `suite/cda-tui` never loads weights. A Python sidecar grades each case, and `g` runs Laya-CDA and Jev on the same questions. Cyan is Laya-CDA. Magenta is Jev. Study (press `4`) is the scoreboard, and the line at the bottom of that panel says which model was more accurate.

Mock mode is the default. It replays scripted labels and checks the harness. It does not download a model.

```bash
python3 suite/sidecar/server.py
cargo run --manifest-path suite/cda-tui/Cargo.toml
```

Set `OPENROUTER_API_KEY` before starting the sidecar to include Jev (`~typesafe/jev-latest` on the OpenRouter Decisions API). `JEV_MODEL` overrides that id. Without the key, Laya-CDA still grades and the header reads `Jev off`.

After the checkpoint is published, point the sidecar at it:

```bash
python3 suite/sidecar/server.py --backend real --checkpoint S4MPL3BI4S/Coding_Decision_Agent
```

`cargo run --manifest-path suite/cda-tui/Cargo.toml -- --check` grades the catalog and prints the report. With the Jev key set, that check calls the Decisions API and spends OpenRouter credit.

`?` lists the keys. `b` / `v` hide and reveal grades, `t` cycles the gate threshold, and `e` writes a session log under `suite/sessions/`.

## License

MIT for the code in this repository. See [LICENSE](LICENSE). The fine-tuned weights stay Apache-2.0 with the Laya base. Source datasets keep their own licenses; they are named in the model card.
