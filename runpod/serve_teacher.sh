#!/usr/bin/env bash
# Start the local teacher LLM with vLLM (OpenAI-compatible server) in the background.
#
#   bash runpod/serve_teacher.sh            # uses $TEACHER_MODEL (default Qwen/Qwen3-32B-AWQ)
#   bash runpod/serve_teacher.sh stop
#
# Picks sensible defaults per GPU memory:
#   >= 70 GB : Qwen/Qwen3-32B-AWQ, 16k context, 90% memory
#   >= 40 GB : Qwen/Qwen3-32B-AWQ, 8k context
#   else     : Qwen/Qwen3-14B-AWQ (fits 24 GB), 8k context
set -euo pipefail

PORT="${TEACHER_PORT:-8000}"
LOG="${WORKDIR:-/workspace/coding_decision_agent}/logs/teacher_vllm.log"
PIDFILE="${WORKDIR:-/workspace/coding_decision_agent}/logs/teacher_vllm.pid"
mkdir -p "$(dirname "$LOG")"

if [[ "${1:-}" == "stop" ]]; then
  if [[ -f "$PIDFILE" ]]; then
    kill "$(cat "$PIDFILE")" 2>/dev/null || true
    rm -f "$PIDFILE"
    echo "teacher stopped"
  else
    pkill -f "vllm.entrypoints.openai.api_server" || true
  fi
  exit 0
fi

GPU_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [[ -z "${TEACHER_MODEL:-}" ]]; then
  if (( GPU_MB >= 40000 )); then TEACHER_MODEL="Qwen/Qwen3-32B-AWQ"; else TEACHER_MODEL="Qwen/Qwen3-14B-AWQ"; fi
fi
if (( GPU_MB >= 70000 )); then MAXLEN=16384; UTIL=0.90; elif (( GPU_MB >= 40000 )); then MAXLEN=8192; UTIL=0.90; else MAXLEN=8192; UTIL=0.92; fi
export TEACHER_MODEL

echo "starting vLLM: model=$TEACHER_MODEL port=$PORT max_len=$MAXLEN gpu_util=$UTIL (GPU ${GPU_MB} MB)"
nohup python -m vllm.entrypoints.openai.api_server \
  --model "$TEACHER_MODEL" \
  --port "$PORT" \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$UTIL" \
  --max-logprobs 20 \
  --dtype auto \
  --enable-prefix-caching \
  --trust-remote-code \
  > "$LOG" 2>&1 &
echo $! > "$PIDFILE"

echo -n "waiting for server"
for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo; echo "teacher ready at http://127.0.0.1:${PORT}/v1 (log: $LOG)"; exit 0
  fi
  echo -n "."; sleep 5
done
echo; echo "teacher did not come up in time; see $LOG"; exit 1
