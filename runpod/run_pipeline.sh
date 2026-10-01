#!/usr/bin/env bash
# End-to-end pipeline. Run from the repo on a RunPod GPU pod after runpod/setup.sh.
#
#   export HF_TOKEN=hf_...                  # write token, S4MPL3BI4S
#   export TRAIN_PROFILE=a100_80gb          # or gpu_24gb
#   export WORKDIR=/workspace/cda_artifacts # data and checkpoints, NOT inside the git clone
#   bash runpod/run_pipeline.sh data
#   bash runpod/run_pipeline.sh teacher
#   bash runpod/run_pipeline.sh merge
#   bash runpod/run_pipeline.sh preprocess
#   bash runpod/run_pipeline.sh train
#   bash runpod/run_pipeline.sh eval
#   bash runpod/run_pipeline.sh publish     # only after you have read the eval report
#
#   bash runpod/run_pipeline.sh all         # data through eval, then stops before publish
#
# See docs/01_RUNPOD_TRAINING.md for disk, time and resume notes.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

export WORKDIR="${WORKDIR:-/workspace/cda_artifacts}"
export TRAIN_PROFILE="${TRAIN_PROFILE:-a100_80gb}"
export HF_MODEL_REPO="${HF_MODEL_REPO:-S4MPL3BI4S/Coding_Decision_Agent}"
export HF_DATASET_REPO="${HF_DATASET_REPO:-S4MPL3BI4S/coding-decision-cases}"
mkdir -p "$WORKDIR/cases" "$WORKDIR/logs"

CASES="$WORKDIR/cases"
TAUGHT="$WORKDIR/cases_teacher"
DATASET="$WORKDIR/dataset"
ITEMS="$WORKDIR/items"
RUN="$WORKDIR/run"

stage_data() {
  echo "== data builders -> $CASES"
  python data/build_typed_decisions.py --out "$CASES/typed_decisions.jsonl"
  python data/build_agentjudgebench.py --out "$CASES/agentjudgebench.jsonl"
  python data/build_when2call.py --out "$CASES/when2call.jsonl"
  python data/build_codeultrafeedback.py --out "$CASES/codeultrafeedback.jsonl"
  python data/build_swe_trajectories.py --out "$CASES/swe_trajectories.jsonl"
  python data/build_routerbench.py --out "$CASES/routerbench.jsonl"
  echo "cases:" && wc -l "$CASES"/*.jsonl
}

stage_teacher() {
  echo "== teacher (vLLM). Training cannot share the GPU with the teacher; this stage stops it at the end."
  if ! python -c "import vllm" 2>/dev/null; then
    python -m pip install -e ".[teacher]"
  fi
  bash runpod/serve_teacher.sh
  python data/teacher_vllm.py --in "$CASES/*.jsonl" --out "$TAUGHT" \
    --base-url "${TEACHER_BASE_URL:-http://127.0.0.1:8000/v1}" \
    --model "${TEACHER_MODEL:-Qwen/Qwen3-32B-AWQ}"
  bash runpod/serve_teacher.sh stop
  # the 32B weights are no longer needed; free the pod disk before training downloads Laya
  echo "teacher done. If disk is tight: rm -rf ~/.cache/huggingface/hub/models--Qwen*"
}

stage_merge() {
  echo "== merge, balance, split -> $DATASET"
  python data/merge_and_split.py --in "$TAUGHT/*.jsonl" --out "$DATASET"
}

stage_preprocess() {
  echo "== tokenize with the Laya tokenizer -> $ITEMS"
  python data/preprocess.py --dataset "$DATASET" --out "$ITEMS" --profile "$TRAIN_PROFILE"
}

stage_train() {
  echo "== RLCD profile=$TRAIN_PROFILE -> $RUN"
  # one GPU by default; set NPROC=2 (or more) on a multi-gpu pod
  # The image torchrun is /usr/local/bin/torchrun and launches system Python, which
  # cannot import the venv's laya. Use the active interpreter instead.
  python -m torch.distributed.run --standalone --nproc_per_node="${NPROC:-1}" train/train_rlcd.py \
    --items "$ITEMS/items_train.pt" --out "$RUN" --profile "$TRAIN_PROFILE" ${WANDB:+--wandb}
}

stage_eval() {
  echo "== eval on the test split"
  python train/evaluate.py \
    --cases "$DATASET/cases_test.jsonl" \
    --train-cases "$DATASET/cases_train.jsonl" \
    --checkpoint "$RUN" \
    --baseline zero_shot=convaiinnovations/laya \
    --baseline typed=convaiinnovations/laya-typed-decisions \
    --out "$RUN/benchmark_report.json"
  echo "read $RUN/benchmark_report.json before publishing"
}

stage_publish() {
  echo "== publish model + dataset"
  python publish/push_to_hub.py \
    --checkpoint "$RUN" \
    --report "$RUN/benchmark_report.json" \
    --dataset "$DATASET" \
    --repo "$HF_MODEL_REPO" \
    --dataset-repo "$HF_DATASET_REPO"
}

cmd="${1:-}"
case "$cmd" in
  data) stage_data ;;
  teacher) stage_teacher ;;
  merge) stage_merge ;;
  preprocess) stage_preprocess ;;
  train) stage_train ;;
  eval) stage_eval ;;
  publish) stage_publish ;;
  all)
    stage_data
    stage_teacher
    stage_merge
    stage_preprocess
    stage_train
    stage_eval
    echo "stopped before publish. Review $RUN/benchmark_report.json, then: bash runpod/run_pipeline.sh publish"
    ;;
  *)
    echo "usage: bash runpod/run_pipeline.sh {data|teacher|merge|preprocess|train|eval|publish|all}"
    exit 1
    ;;
esac
