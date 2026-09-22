#!/usr/bin/env bash
# Gate-4 TTFT ABBA: one [Q,W,F,F,W,Q] block, one build, graph on.
set -euo pipefail
WT=/home/kaden/ClaudeCode/warpfront/wt-q8wide
cd "$WT"
export HOME=/home/kaden/.hipfire-homes/ab1
export ROCR_VISIBLE_DEVICES=GPU-e475645fe0200397
export HIP_VISIBLE_DEVICES=GPU-e475645fe0200397
export HIPFIRE_KERNEL_CACHE=/home/kaden/.hipfire-homes/ab1/.hipfire_kernels
export HIPFIRE_MODELS_DIR=/home/kaden/.hipfire/models
export HIPFIRE_GRAPH=1
export HIPFIRE_DAEMON_BIN=$WT/target/release/daemon

run_one() { # tag, knob, kvmode
  ./target/release/hipfire config set kernel.gfx12_q8_fa2_wide "$2" >/dev/null
  rm -f "$HOME/.hipfire/daemon.pid"
  echo "=== ttft $1 (wide=$2 kv=$3) ==="
  ./target/release/hipfire bench qwen3.8:27b-mq4-xt --ttft \
    --prompt-file benchmarks/prompts/ttft_5900.txt --runs 8 --warmups 2 \
    --spec off --kv-mode "$3" --json > "scratch-q8wide/ttft_$1.json" 2>"scratch-q8wide/ttft_$1.log"
  python3 -c "import json; d=json.load(open('scratch-q8wide/ttft_$1.json')); print('saved ok')"
}
run_one Q false q8
run_one W true q8
run_one F false fp8
run_one F2 false fp8
run_one W2 true q8
run_one Q2 false q8
./target/release/hipfire config set kernel.gfx12_q8_fa2_wide false >/dev/null
echo TTFT_ABBA_DONE
