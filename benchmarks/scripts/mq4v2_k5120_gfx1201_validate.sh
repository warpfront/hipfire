#!/usr/bin/env bash
# Standalone correctness before the unchanged serving ABBA protocol.
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
run_child() {
  "$@" &
  child=$!
  wait "$child"
  child=
}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
: "${MODEL:?Set the pinned Qwen3.8-27B MQ4-XT artifact}"
: "${OUT:?Set a new validation directory}"
mkdir "$OUT"
OUT=$(realpath "$OUT")
MODEL=$(realpath "$MODEL")
export HIP_VISIBLE_DEVICES=${GPU_ID:-0}
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES HIPFIRE_HIPCC_EXTRA_FLAGS
export HIPFIRE_GFX12_WEIGHT_LOAD_POLICY=rt
export HIPFIRE_KERNEL_CACHE=${HIPFIRE_KERNEL_CACHE:-"$HOME/.hipfire_kernels"}
export HIPFIRE_KV_MODE=q8 HIPFIRE_DN_STATE_EF=1
export HIPFIRE_HOME="$OUT/probe-home"
mkdir "$HIPFIRE_HOME"
sha256sum "$MODEL" target/release/daemon target/release/hipfire > "$OUT/identity.sha256"
git rev-parse HEAD > "$OUT/commit.txt"
git diff > "$OUT/working.diff"
rocm-smi > "$OUT/gpu-before.txt"
exec 9>/tmp/hipfire-gpu.lock
flock -n 9
for flag in 0 1; do
  export HIPFIRE_GFX12_MQ4V2_GATEUP_K5120=$flag
  run_child target/release/examples/probe_k5120_logits "$MODEL" "$OUT/logits-$flag.f32" \
    --prefill 64 > "$OUT/logits-$flag.log" 2>&1
  run_child python3 benchmarks/scripts/mq4v2_k5120_replay.py --model "$MODEL" \
    --skip-prefill --decode-context 128 --max-seq 8192 --kv-mode q8 \
    --capture-repeats 2 --measure-repeats 3 --decode-iterations 16 \
    --shadow-iterations 4 --timeout 1800 --pm4 \
    --out "$OUT/replay-$flag.json" --log "$OUT/replay-$flag.log" \
    > "$OUT/replay-$flag.stdout" 2>&1
done
cmp "$OUT/logits-0.f32" "$OUT/logits-1.f32"
sha256sum "$OUT"/*.f32 > "$OUT/logits.sha256"
python3 benchmarks/scripts/mq4v2_k5120_gfx1201_report.py "$OUT" --correctness-only
# The ABBA script reacquires this lock and fails closed if another user got it.
flock -u 9
exec 9>&-
FLAG_ENV=HIPFIRE_GFX12_MQ4V2_GATEUP_K5120 OUT="$OUT/abba" run_child bash benchmarks/scripts/mq4v2_k5120_abba.sh
python3 benchmarks/scripts/mq4v2_k5120_gfx1201_report.py "$OUT"
