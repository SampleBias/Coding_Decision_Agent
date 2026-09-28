# Status

Snapshot of this repo at the point it was pushed. Training has not been run. Nothing has been uploaded to Hugging Face yet.

## Done in this repo

- Decision schema for four families: `code_review`, `tool_call`, `agent_trace`, `routing` ([configs/schema.py](../configs/schema.py)). Yes/no questions are `choice` with keys `no`/`yes`, not Laya `noul`.
- State compaction for diffs, test logs, tool schemas, tool calls and traces ([data/compaction.py](../data/compaction.py)).
- Builders that emit `{state, questions, gold}` JSONL:
  - [data/build_typed_decisions.py](../data/build_typed_decisions.py) — `LocalLLaMA/typed-decisions` agent-trace workflow, gold probabilities kept.
  - [data/build_agentjudgebench.py](../data/build_agentjudgebench.py) — programmatic scores blended with six judge configs.
  - [data/build_when2call.py](../data/build_when2call.py) — when to call a tool.
  - [data/build_codeultrafeedback.py](../data/build_codeultrafeedback.py) — 1–5 ratings to `code_quality`, plus a routing case per instruction.
  - [data/build_swe_trajectories.py](../data/build_swe_trajectories.py) — `target` (issue resolved) to `likely_correct`. Eval logs are never put in the state.
  - [data/build_routerbench.py](../data/build_routerbench.py) — cheapest model tier that solved the prompt. Optional local LLMRouterBench directory.
- Teacher labeller [data/teacher_vllm.py](../data/teacher_vllm.py): letter logprobs from a local vLLM server, cached, resumable. Mock mode (`--mock`) was run on 20-row samples.
- Merge, group holdout, majority cap, Hub row format: [data/merge_and_split.py](../data/merge_and_split.py).
- Tokeniser preprocess: [data/preprocess.py](../data/preprocess.py). Not executed against the real Laya weights on the laptop (no GPU, and the checkpoint download was interrupted). First real run is on the pod.
- RLCD trainer [train/train_rlcd.py](../train/train_rlcd.py): 1..N GPUs via torchrun, profiles in [configs/train.yaml](../configs/train.yaml), calibration slice held out, LoRA folded back into a plain Laya checkpoint, resume from `checkpoint_latest/`.
- Eval [train/evaluate.py](../train/evaluate.py) and publish [publish/push_to_hub.py](../publish/push_to_hub.py).
- SDK: `grade_code`, `grade_tool_call`, `grade_trace`, `route_model`.
- CPU tests: `pytest` (schema, compaction, builder converters, collate, temperature fit, model-card fill).

Each builder was also smoke-tested on about 20 real rows from the Hub (except the full SWE scan, which was capped at 300 streamed rows).

## Not done — this is the RunPod work

Do these in order. Commands are in [01_RUNPOD_TRAINING.md](01_RUNPOD_TRAINING.md).

1. Rent a pod. Preferred: 1x A100 80GB or H100, with at least 150 GB of container disk plus a volume mounted at `/workspace`. Minimum: 24 GB (`TRAIN_PROFILE=gpu_24gb`, LoRA, Qwen3-14B teacher).
2. SSH in, clone this repo, `bash runpod/setup.sh`.
3. Export a **write** `HF_TOKEN` for the Hugging Face account `S4MPL3BI4S`.
4. `bash runpod/run_pipeline.sh data` — downloads the source datasets. SWE-agent-trajectories is an 80k streaming set; expect this stage to take a while.
5. `bash runpod/run_pipeline.sh teacher` — starts vLLM, labels every `needs_teacher` question, stops vLLM so the GPU is free.
6. `bash runpod/run_pipeline.sh merge`
7. `bash runpod/run_pipeline.sh preprocess` — downloads `convaiinnovations/laya` and writes `items_*.pt`. Read `preprocess_summary.json`. If `state_truncated_frac` is high, lower the compaction budgets or raise `max_len` before training.
8. `bash runpod/run_pipeline.sh train`
9. `bash runpod/run_pipeline.sh eval` — writes `benchmark_report.json`. Compare `coding_decision_agent` against majority, zero-shot `convaiinnovations/laya`, and `convaiinnovations/laya-typed-decisions`.
10. If one family is near chance, change `family_weights` in [configs/train.yaml](../configs/train.yaml), delete `$WORKDIR/run/checkpoint_latest`, and run `train` again. One rebalance pass was the plan.
11. `bash runpod/run_pipeline.sh publish` — creates `S4MPL3BI4S/Coding_Decision_Agent` and `S4MPL3BI4S/coding-decision-cases`.

## Intentionally out of scope until after the first checkpoint

- `nebius/SWE-rebench-openhands-trajectories` for step-level tool grading (phase 2 in the plan).
- `ynulihao/LLMRouterBench` unless you download it and pass `--llmrouterbench-dir`.
- Unsloth. Laya's decision head is not a Hugging Face classification head, so Unsloth cannot host it. The 24 GB path is PEFT LoRA inside `train_rlcd.py`.
- Measuring the real checkpoint. There are no accuracy numbers to put in the model card until step 9.
