#!/usr/bin/env bash
set -euo pipefail
child=
cleanup() {
  if [[ -n "$child" ]]; then
    kill -TERM "$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
: "${MODEL:?Set MODEL to the pinned Qwen3.8-27B MQ4-XT artifact}"
: "${OUT:?Set OUT to a new results directory}"
FLAG_ENV=${FLAG_ENV:-HIPFIRE_MQ4V2_GATEUP_K5120}
case "$FLAG_ENV" in
  HIPFIRE_MQ4V2_GATEUP_K5120|HIPFIRE_GFX12_MQ4V2_GATEUP_K5120) ;;
  *) echo "Unsupported flag: $FLAG_ENV" >&2; exit 2;;
esac
mkdir "$OUT"
OUT=$(cd "$OUT" && pwd)
MODEL=$(realpath "$MODEL")
exec 9>/tmp/hipfire-gpu.lock
flock -n 9
export HIP_VISIBLE_DEVICES=${GPU_ID:-0} HIPFIRE_LOCAL=1
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export HIPFIRE_CLI_BIN="$ROOT/target/release/hipfire"
export HIPFIRE_DAEMON_BIN="$ROOT/target/release/daemon"
export HIPFIRE_KERNEL_CACHE=${HIPFIRE_KERNEL_CACHE:-"$HOME/.hipfire_kernels"}
export HIPFIRE_GFX11_WEIGHT_LOAD_POLICY=buffer HIPFIRE_DPM_WARMUP_SECS=10
PROMPT="$ROOT/benchmarks/prompts/mq4v2_k5120_long_ar.txt"
git rev-parse HEAD > "$OUT/commit.txt"
git diff > "$OUT/working.diff"
md5sum "$HIPFIRE_CLI_BIN" "$HIPFIRE_DAEMON_BIN" "$PROMPT" "$MODEL" > "$OUT/identity.md5"
sha256sum benchmarks/scripts/mq4v2_k5120_abba.sh benchmarks/scripts/mq4v2_k5120_serve.py \
  scripts/serve_harness.py > "$OUT/scripts.sha256"
rocm-smi > "$OUT/gpu-before.txt"
for arm in A1 B1 B2 A2; do
  case "$arm" in A*) flag=0;; B*) flag=1;; esac
  export "$FLAG_ENV=$flag"
  printf 'HIP_VISIBLE_DEVICES=%s\n%s=%s\n' \
    "$HIP_VISIBLE_DEVICES" "$FLAG_ENV" "$flag" > "$OUT/$arm-routing.txt"
  printf '%s %s flag=%s\n' "$(date -Is)" "$arm" "$flag"
  python3 benchmarks/scripts/mq4v2_k5120_serve.py --model "$MODEL" \
    --mode battery --prompt-file "$PROMPT" --thinking off --max-think-tokens 1 \
    --sampling 'json:{"temperature":0,"top_p":1,"repeat_penalty":1.0}' \
    --kv q8 --kv-backend vmm --speculation off --max-seq 8192 --max-tokens 4096 \
    --port "${PORT:-19273}" --home "$OUT/$arm-home" --serve-log "$OUT/$arm-serve.log" \
    --out "$OUT/$arm.json" > "$OUT/$arm-harness.log" 2>&1 &
  child=$!
  wait "$child"
  child=
  sleep 15
done
rocm-smi > "$OUT/gpu-after.txt"
