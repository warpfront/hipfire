#!/bin/bash
# Gate-3 TIME brackets: interleave baseline/candidate oracle-TIME runs.
# Each invocation runs one fresh process per binary; call 3x (a/b/c).
# Usage: time-brackets.sh <a|b|c>
set -u
TAG=${1:?bracket tag a/b/c}
D=/home/kaden/ClaudeCode/warpfront/wt-iu4/scratch-2026-09-17/iu4sched
BASE=/home/kaden/ClaudeCode/warpfront/wt-iu4-base/target/debug/examples/tmp_iu4_gfx12_oracle
CAND=/home/kaden/ClaudeCode/warpfront/wt-iu4/target/debug/examples/tmp_iu4_gfx12_oracle
MODEL=/home/kaden/.hipfire/models/qwen3.8-27b.mq4-xt
export ROCR_VISIBLE_DEVICES=1
export HIPFIRE_KERNEL_CACHE=/home/kaden/.hipfire-homes/ab1/.hipfire_kernels
export HIPFIRE_MODELS_DIR=/home/kaden/.hipfire/models
export TIME=1
echo "== bracket $TAG baseline =="
"$BASE" "$MODEL" > "$D/time-base-$TAG.log" 2>&1
echo "base exit=$?"
echo "== bracket $TAG candidate =="
"$CAND" "$MODEL" > "$D/time-cand-$TAG.log" 2>&1
echo "cand exit=$?"
grep -h "^TIME" "$D/time-base-$TAG.log" "$D/time-cand-$TAG.log"
