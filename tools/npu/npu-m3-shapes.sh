#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
# npu-m3-shapes.sh — M3 measurement batch as ONE npu-batch.sh run (no nested locks): the deployed whole-array
# design (V6: dynamic TXN, one context reused, every submit verified CPU-exact) at 512x512x64 and the Flash-Next
# expert shapes gate_up [Mx2560]x[2560x1280] and down [Mx640]x[640x2560], M = 512 and 4096, 3 submits each.
# Core program: NPU_M3_CORE (default Serial — the kernel proven exact on hardware in bisect2).
# First (NPU_M3_DIAG=1, default) three Fast-core checks (V2 x2 with status dump, V5) with a 3 s wait.
# Run as the command of one npu-window.sh hold under the exclusive Halo timing lock:
#   ~/pm-wave/hipx-lock.sh timing -- tools/npu/npu-window.sh tools/npu/npu-m3-shapes.sh
# Time bound: each command `timeout --foreground` 14 s + 1 s kill-after, whole batch 75 s + 1 s < 90 s.
# Extra npu-gemm command lines may be appended as arguments (same per-command bound).
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT_DIR=${NPU_M3_OUT:-$ROOT/npu-m3-out/$(date -u +%Y%m%dT%H%M%SZ)-$$}
CORE=${NPU_M3_CORE:-Serial}
CMD_T=14
TOTAL_T=75
WAIT_MS=10000
mkdir -p "$OUT_DIR" || { echo "cannot create $OUT_DIR"; exit 2; }
OUT_DIR=$(cd "$OUT_DIR" && pwd) || { echo "cannot resolve $OUT_DIR"; exit 2; }
echo "npu-m3-shapes: core $CORE, diagnostic bins in $OUT_DIR"

cmds=()
add() {  # add WAIT_MS ARG... : npu-gemm arguments, run from OUT_DIR with the fixed per-command bounds
  local wait=$1 line a; shift
  line=$(printf 'cd %q && timeout --foreground --signal=TERM --kill-after=1 %q %q' "$OUT_DIR" "$CMD_T" "$ROOT/target/release/npu-gemm")
  for a in "$@"; do line+=$(printf ' %q' "$a"); done
  line+=$(printf ' --timeout-ms %q' "$wait")
  cmds+=("$line")
}
# Fast-core checks first (each <= 3 s even when hanging): FastSlowCtl = Fast compute (VLIW wavefront, JNZ
# back edge in place of the hardware ZOL) with Serial-discipline control code; Fast = the same compute with
# the original control code (3-NOP lock spacing, delay-slot work); V5 = the full 8x4 array.
if [ "${NPU_M3_DIAG:-1}" = 1 ]; then
  add 3000 128 64 64 --variant V2 --core FastSlowCtl --iters 1 --dump-status
  add 3000 128 64 64 --variant V2 --core Fast --iters 1 --dump-status
  add 3000 512 512 64 --variant V5 --core FastSlowCtl --iters 1
fi
# Array clock probe (NPU_M3_CLOCK=1; measured 1.80 GHz in window 5): V2 ClockProbe adds N*4014 known core issue cycles before C-full;
# the slope of submit->completion time over N gives the AIE array clock.
if [ "${NPU_M3_CLOCK:-0}" = 1 ]; then
  for n in 0 1000 2000 4000; do
    add 3000 128 64 64 --variant V2 --core ClockProbe --probe-iters "$n" --iters 3
  done
fi
for shape in "512 512 64" "512 1280 2560" "512 2560 640" "4096 1280 2560" "4096 2560 640"; do
  # shellcheck disable=SC2086
  add "$WAIT_MS" $shape --array --core "$CORE" --reuse-ctx --iters 3
done
for extra in "$@"; do
  # shellcheck disable=SC2206
  words=($extra)
  add "$WAIT_MS" "${words[@]}"
done

export NPU_BATCH_BASIC=0
exec timeout --signal=TERM --kill-after=1 "$TOTAL_T" "$ROOT/tools/npu-batch.sh" "${cmds[@]}"
