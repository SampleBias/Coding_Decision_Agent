# Train on RunPod

The laptop this repo was written on has no NVIDIA GPU, 8 GB of RAM and almost no free disk. Every step below runs on the pod. Clone from GitHub; do not copy the laptop checkout.

Repo: `https://github.com/SampleBias/Coding_Decision_Agent`

The Hugging Face account for the upload is `S4MPL3BI4S`. That is separate from the GitHub account `SampleBias`.

## 1. Pod

| | Preferred | 24 GB minimum |
|---|---|---|
| GPU | 1x A100 80GB or H100 | RTX 4090 / A5000 / L4, 24 GB |
| Container disk | 150 GB or more | 100 GB |
| Volume | mount at `/workspace`, 100 GB or more | same |
| Profile | `TRAIN_PROFILE=a100_80gb` | `TRAIN_PROFILE=gpu_24gb` |
| Teacher | `Qwen/Qwen3-32B-AWQ` (picked automatically above 40 GB) | `Qwen/Qwen3-14B-AWQ` |

Template: a PyTorch 2.x image with CUDA 12. `runpod/setup.sh` creates a venv and reuses the image's CUDA torch if `torch.cuda.is_available()` is already true.

Disk budget, roughly:

| thing | size |
|---|---|
| Qwen3-32B-AWQ (teacher, deleted after labelling if you want) | ~20 GB |
| `convaiinnovations/laya` | ~2 GB |
| source datasets + case JSONL | 5–30 GB (SWE trajectories dominate if the stream caches) |
| `items_*.pt` and Adam checkpoints | 5–15 GB |

Keep `WORKDIR` on the volume, not in the container overlay.

## 2. SSH and clone

From the pod's Connect menu, copy the SSH command. Then, on the pod:

```bash
cd /workspace
git clone https://github.com/SampleBias/Coding_Decision_Agent.git
cd Coding_Decision_Agent
```

Public repo, no GitHub token needed to clone.

## 3. Secrets

Create a Hugging Face write token for `S4MPL3BI4S` (Settings, Access Tokens, type Write). On the pod:

```bash
export HF_TOKEN=hf_xxxxxxxx
export HF_MODEL_REPO=S4MPL3BI4S/Coding_Decision_Agent
export HF_DATASET_REPO=S4MPL3BI4S/coding-decision-cases
export TRAIN_PROFILE=a100_80gb          # or gpu_24gb
export WORKDIR=/workspace/cda_artifacts
export NPROC=1                          # GPU count for torchrun
```

Put those lines in `~/.bashrc` if the SSH session will drop. Do not commit the token. `.env` is gitignored; you can also `cp .env.example .env` and edit it, then `set -a; source .env; set +a`.

Check the GPU before installing:

```bash
nvidia-smi
```

You want a GPU name and a memory figure, not "couldn't communicate with the NVIDIA driver".

## 4. Install

```bash
bash runpod/setup.sh
source .venv/bin/activate
python -c "import torch, laya; print(torch.cuda.get_device_name(0), laya.__version__)"
```

`setup.sh` logs in to the Hub when `HF_TOKEN` is set.

## 5. Stages

Run them one at a time so a failure is easy to resume. Each stage is safe to re-run; the teacher cache and `checkpoint_latest/` are what make that true.

```bash
bash runpod/run_pipeline.sh data
bash runpod/run_pipeline.sh teacher
bash runpod/run_pipeline.sh merge
bash runpod/run_pipeline.sh preprocess
bash runpod/run_pipeline.sh train
bash runpod/run_pipeline.sh eval
```

`bash runpod/run_pipeline.sh all` is the same five-plus-eval sequence. It does not publish.

### data

Writes `$WORKDIR/cases/*.jsonl`. Expect on the order of:

| file | rough size |
|---|---|
| typed_decisions | ~400 cases (300 train + 100 test in the agent-trace config) |
| agentjudgebench | ~57k (5 generators x ~3.8k records x 3 difficulties, capped later) |
| when2call | ~9k preference + 15k SFT + ~3.6k MCQ |
| codeultrafeedback | 10k instructions x 4 responses, plus one routing case each |
| swe_trajectories | up to 8k resolved + 1.5x unresolved, plus routing cases |
| routerbench | code evals in full, other English evals capped at 3,000 |

`merge` then caps each source at 30,000 and caps the majority class at 60% of each source/family slice. Override with `--cap substring=N` if you call `data/merge_and_split.py` yourself.

SWE streams from the Hub. If the process dies, run the data stage again; it overwrites that file. The script calls `os._exit` at the end on purpose, because the datasets streaming threads crash the interpreter during shutdown after the file is flushed.

### teacher

`runpod/serve_teacher.sh` starts vLLM on port 8000 and waits until `/v1/models` answers. Labelling is resumable: `$WORKDIR/cases_teacher/_teacher_cache.jsonl`. If SSH drops, run `teacher` again. It skips keys already in the cache.

When it finishes it stops vLLM. Free the teacher weights before training if the disk is tight:

```bash
rm -rf ~/.cache/huggingface/hub/models--Qwen*
```

On a 24 GB card, export this before `teacher`:

```bash
export TEACHER_MODEL=Qwen/Qwen3-14B-AWQ
export TRAIN_PROFILE=gpu_24gb
```

### merge

Writes `$WORKDIR/dataset/cases_{train,val,test}.jsonl` and `stats.json`. The leakage line must say `0`. A test case and a train case never share `(source, group)` unless the source itself marked an official split (`split_hint`).

### preprocess

Downloads Laya and tokenises. Open `$WORKDIR/items/preprocess_summary.json`.

- `dropped_marker_mismatch` should be 0. Non-zero means an option list was crushed by `head_max_len`.
- `state_truncated_frac` is the fraction of questions whose state did not fit. Under about 0.15 is acceptable. If it is much higher, edit `configs/state_compaction.yaml` or the profile's `max_len`, delete `$WORKDIR/items`, and run `preprocess` again.

### train

```bash
bash runpod/run_pipeline.sh train
```

A100 80GB profile: full fine-tune, `max_len` 2048, micro-batch 16, 3 epochs. 24 GB profile: LoRA on `Wqkv`, `Wo`, `Wi`, `max_len` 1024, micro-batch 4.

Time: a few hours for ~100k items on one A100. The official Laya notebook is ~4–5 hours for 30k questions on 2x T4; an A100 is faster than that pair.

Resume: if the pod dies, run `train` again. It reads `$WORKDIR/run/checkpoint_latest/checkpoint_meta.json` and continues at the next epoch. The saved weights are always a plain Laya checkpoint (`model.safetensors`, `encoder/`, `tokenizer/`, `rl_agent_config.json`), including when LoRA is on. LoRA is merged into those weights before save. A resumed LoRA run attaches a new adapter on top of the merged weights; it does not double-count the old adapter.

Logs: per-epoch average loss and a per-family cross-entropy. You want the family losses moving down together. `agent_trace` is upweighted (2.0) because that family is small.

Optional: `export WANDB_API_KEY=...` and `export WANDB=1` before `train`.

### eval

Writes `$WORKDIR/run/benchmark_report.json`. The `ALL` row is the number that matters first, then each family (`code_review`, `tool_call`, `agent_trace`, `routing`).

Read it as:

- `coding_decision_agent` accuracy should beat `majority` and beat `zero_shot` by a wide margin. The base Laya checkpoints are near chance on workflows they were not fine-tuned on.
- `typed` (`laya-typed-decisions`) should be competitive only on `agent_trace`, because that is the workflow it was trained on.
- ECE under ~0.15 on `ALL` means the confidence is usable for a 0.7 gate. If ECE is high, the temperature fit had too little calibration data; check the calib counts in the training log.

### one rebalance, if a family lags

Edit `family_weights` in [configs/train.yaml](../configs/train.yaml). Delete the run directory so training does not think it already finished:

```bash
rm -rf "$WORKDIR/run"
bash runpod/run_pipeline.sh train
bash runpod/run_pipeline.sh eval
```

Do this once. More than that is chasing the test split.

## 6. Publish

Only after the report looks right.

```bash
bash runpod/run_pipeline.sh publish
```

That uploads:

- `https://huggingface.co/S4MPL3BI4S/Coding_Decision_Agent`
- `https://huggingface.co/datasets/S4MPL3BI4S/coding-decision-cases`

The model card is rendered from [huggingface/README.md](../huggingface/README.md). `{{ACCURACY}}` and the family table are filled from `benchmark_report.json`. If you publish without a report, those stay as `TBD`.

Confirm:

```bash
python -c "import laya; a=laya.load('S4MPL3BI4S/Coding_Decision_Agent'); print(a.cfg['model_name'], a.cfg['temperature'])"
```

## 7. If something breaks

| symptom | what to do |
|---|---|
| `No CUDA GPU is visible` | `nvidia-smi`. Wrong image, or the GPU was not attached. |
| teacher never prints `teacher ready` | `tail -f $WORKDIR/logs/teacher_vllm.log`. Usually OOM: set `TEACHER_MODEL=Qwen/Qwen3-14B-AWQ`. |
| `HF_TOKEN is not set` | export it in this shell. `setup.sh` does not persist it. |
| publish 403 | the token is not a write token, or it is not for `S4MPL3BI4S`. |
| train OOM | `TRAIN_PROFILE=gpu_24gb`, or lower `micro_batch` in the profile. |
| preprocess `dropped_marker_mismatch` > 0 | raise `head_max_len` in the profile (320 is the A100 default). |
| disk full during teacher | stop the teacher (`bash runpod/serve_teacher.sh stop`), delete the Qwen cache, restart `teacher`. The cache JSONL keeps what already finished. |
