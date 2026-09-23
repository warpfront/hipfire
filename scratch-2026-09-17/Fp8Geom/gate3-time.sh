#!/bin/sh
# Fp8Geom gate 3: same-session interleaved TIME at N=512, --quick, ordinal 2.
# Control GEOM=128x128 vs new GEOM=128x64w8 (both V2=1, SLABS=2).
# Admission: gate_up(new) <= 1300 us AND no shape worse than control.
# Announce on hub before running; never overlap timing on ordinal 2.
set -eu
WT=/home/kaden/ClaudeCode/warpfront/wt-fp8geom
R=$WT/scratch-2026-09-17/Fp8Geom/time-512
export HOME=/home/kaden/.hipfire-homes/ab2
export ROCR_VISIBLE_DEVICES=2
export HIPFIRE_MODELS_DIR=/home/kaden/.hipfire/models
export HIPFIRE_KERNEL_CACHE=/home/kaden/.hipfire-homes/ab2/.hipfire_kernels
export HIPFIRE_GFX12_MQ4V2_FP8_V2=1
export HIPFIRE_GFX12_MQ4V2_FP8_SLABS=2
BIN=$WT/target/release/examples/tmp_gemm_v2_oracle
for row in base128 new64 base128b new64b; do
  case $row in
    base*) export HIPFIRE_GFX12_MQ4V2_FP8_V2_GEOM=128x128;;
    new*) export HIPFIRE_GFX12_MQ4V2_FP8_V2_GEOM=128x64w8;;
  esac
  mkdir -p "$R-$row"
  TIME=1 "$BIN" --device 0 --quick --out "$R-$row" > "$R-$row.stdout" 2> "$R-$row.stderr"; echo "$row RC=$?"
done
echo "=== TIME medians ==="
grep -h "median" "$R"-*.stdout
