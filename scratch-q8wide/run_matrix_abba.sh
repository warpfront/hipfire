#!/usr/bin/env bash
# Gate-4 matrix ABBA: two [Q,W,F,F,W,Q] blocks, one build, graph on.
# Each run: fresh daemon, typed knob per arm (q8 for Q/W, fp8 for F).
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
  echo "=== matrix $1 (wide=$2 kv=$3) ==="
  ./target/release/hipfire bench qwen3.8:27b-mq4-xt --matrix \
    --pp 512,8192 --ctx 128 --tg 128 --spec off --runs 3 --warmups 1 \
    --kv-mode "$3" --json > "scratch-q8wide/bench_$1.json" 2>"scratch-q8wide/bench_$1.log"
  python3 -c "import json; d=json.load(open('scratch-q8wide/bench_$1.json')); print('saved', len(str(d)), 'bytes')"
}
run_one Q1 false q8
run_one W1 true q8
run_one F1 false fp8
run_one F2 false fp8
run_one W2 true q8
run_one Q2 false q8
run_one Q3 false q8
run_one W3 true q8
run_one F3 false fp8
run_one F4 false fp8
run_one W4 true q8
run_one Q4 false q8
./target/release/hipfire config set kernel.gfx12_q8_fa2_wide false >/dev/null
echo MATRIX_ABBA_DONE
