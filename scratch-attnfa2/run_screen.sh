#!/usr/bin/env bash
# hipx standalone screen. usage: run_screen.sh gfx1100|gfx1151 schedule reps [variants...]
# Reference is always prod_<arch> (the production source, runtime JIT flags).
set -euo pipefail
cd "$(dirname "$0")/out"
arch=$1; sched=$2; reps=$3; shift 3
case $arch in
  gfx1100) export ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 ;;
  gfx1151) export ROCR_VISIBLE_DEVICES=1 HIP_VISIBLE_DEVICES=0 ;;
  *) echo "bad arch"; exit 2 ;;
esac
objs=(prod_$arch.hsaco)
for v in "$@"; do objs+=("${v}_$arch.hsaco"); done
./fa2_bench "$sched" "$reps" "${objs[@]}"
