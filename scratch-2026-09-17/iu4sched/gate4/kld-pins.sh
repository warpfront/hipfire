#!/bin/bash
# Gate-4 KLD pins under HIPFIRE_IU4_PREFILL=1 (Main's recipe).
# Pins are md5 of the output .bin:
#   c1 032ebad84f2c1dd5e2980fa805215ac0 / c2 dc7e53181662271780374f0a85fd7732
#   c24 slice-mean KLD 0.063410 (bit-for-bit vs prior receipt).
# Usage: kld-pins.sh   (~2 min c1+c2, ~90 s c24; ordinal-1 env set inside)
set -u
W=/home/kaden/ClaudeCode/warpfront/wt-iu4
D=$W/scratch-2026-09-17/iu4sched/gate4
EVAL=$W/target/release/examples/eval_hipfire
MODEL=/home/kaden/.hipfire/models/qwen3.8-27b.mq4-xt
REF=/home/kaden/kldrefs/qwen3.8-27b.ref_wt2.bin
mkdir -p "$D"
bash "$W/scripts/check_fixture.sh" --sha "$MODEL" || exit 2
export ROCR_VISIBLE_DEVICES=1
export HIPFIRE_KERNEL_CACHE=/home/kaden/.hipfire-homes/ab1/.hipfire_kernels
export HIPFIRE_MODELS_DIR=/home/kaden/.hipfire/models
export HIPFIRE_GRAPH=0 HIPFIRE_NORMALIZE_PROMPT=0 HIPFIRE_LLOYD_GFX12=1
export HIPFIRE_IU4_PREFILL=1
for c in 1 2 24; do
  echo "== chunk$c =="
  "$EVAL" --model "$MODEL" --ref "$REF" --kv-mode q8 --kv-v q8 \
    --scoring-mode prefill --max-chunks "$c" --output "$D/c$c.bin" \
    > "$D/c$c.stderr" 2>&1
  echo "exit=$?"
  grep -E "slice-mean KLD" "$D/c$c.stderr" | tail -1
done
echo "== pins (md5) =="
md5sum "$D/c1.bin" "$D/c2.bin"
echo "expect: 032ebad84f2c1dd5e2980fa805215ac0 / dc7e53181662271780374f0a85fd7732"
grep -h "slice-mean KLD" "$D/c24.stderr"
