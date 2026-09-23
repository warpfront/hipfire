#!/bin/bash
# HIPFIRE_IU4_V2B=0 smoke on Halo (ROCR=1/HIP=0): with a fresh kernel cache the
# run must JIT the X5 symfold module and no V2B module.
set -euo pipefail
d=/home/kaden/hipfire-v2bhalo
out=$d/scratch-v2bhalo/gfx1151/killswitch
model=/home/kaden/.hipfire/models/qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq
rm -rf "$out"
mkdir -p "$out/home"
env -i PATH="$PATH" HOME="$out/home" XDG_CONFIG_HOME="$out/home/.config" \
  ROCR_VISIBLE_DEVICES=1 HIP_VISIBLE_DEVICES=0 \
  HIPFIRE_DAEMON_BIN=$d/target/release/daemon HIPFIRE_IU4_V2B=0 \
  $d/target/release/hipfire bench $model --matrix --pp 512,8192 --ctx 128 --tg 16 \
  --spec off --runs 1 --warmups 0 --kv-mode q8 --kv-backend vmm > "$out/bench.log" 2>&1
grep -q 'GPU dev 0: gfx1151' "$out/bench.log"
grep -q 'KV cache: Q8 vmm (' "$out/bench.log"
ls "$out/home/.hipfire_kernels/gfx1151/" | grep -E 'iu4_v2b|x5_symfold' | grep -E '\.hsaco$' > "$out/cache.txt" || true
cat "$out/cache.txt"
grep -q 'x5_symfold' "$out/cache.txt"
! grep -q 'iu4_v2b' "$out/cache.txt"
grep -E 'pp512:|pp8192:' "$out/bench.log"
echo "killswitch PASS: X5 only, no V2B object"
