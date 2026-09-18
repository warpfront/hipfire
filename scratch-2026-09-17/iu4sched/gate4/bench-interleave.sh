#!/bin/bash
# Gate-4 interleave bench: OFF/ON/OFF/ON, matrix pp 512/2048/8192/32768.
# OFF = wt-lloyd af16ec4bd binary, IU4_PREFILL unset (incumbent route).
# ON = wt-iu4 candidate binary (ea4ee884), IU4_PREFILL=1.
# Record the trailing decode column per run; pair rows only in same state.
# Usage: bench-interleave.sh   (runs all 4 arms sequentially, ~10+ min each)
set -u
W=/home/kaden/ClaudeCode/warpfront/wt-iu4
D=$W/scratch-2026-09-17/iu4sched/gate4
OFF=/home/kaden/ClaudeCode/warpfront/wt-lloyd/target/release/hipfire
ON=$W/target/release/hipfire
export ROCR_VISIBLE_DEVICES=1
export HIPFIRE_KERNEL_CACHE=/home/kaden/.hipfire-homes/ab1/.hipfire_kernels
export HIPFIRE_MODELS_DIR=/home/kaden/.hipfire/models
export HIPFIRE_GRAPH=1 HIPFIRE_LLOYD_GFX12=1
run_arm() {
  local tag=$1 bin=$2 iu4=$3
  echo "== $tag (iu4_prefill=${iu4:-unset}) =="
  if [ -n "$iu4" ]; then export HIPFIRE_IU4_PREFILL="$iu4"; else unset HIPFIRE_IU4_PREFILL; fi
  "$bin" bench qwen3.8:27b-mq4-xt --matrix --pp 512,2048,8192,32768 \
    --ctx 128 --tg 64 --spec off --runs 3 --warmups 1 --kv-mode q8 --json \
    > "$D/bench-$tag.json" 2> "$D/bench-$tag.stderr"
  echo "exit=$?"
  unset HIPFIRE_IU4_PREFILL
}
run_arm off1 "$OFF" ""
run_arm on1 "$ON" 1
run_arm off2 "$OFF" ""
run_arm on2 "$ON" 1
echo "== done =="
