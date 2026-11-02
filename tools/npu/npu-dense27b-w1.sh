#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
# npu-dense27b-w1.sh — dense Qwen3.8-27B IU4 window (one hold): exactness of the per-K128-epoch int32 partials
# (`npu-gemm --iu4-epochs`, native hipfire operands, every checked submit compared with the CPU reference of the GPU
# contract), timing loops at the real 27B K / feature slices (first + final submit verified), then the fold-free V8
# full-K integer GEMM at 27B shapes (exact int GEMM, NOT the hipfire contract: an upper bound if the fold were free).
#   ~/pm-wave/hipx-lock.sh timing -- env NPU_WINDOW_TIMEOUT=90 tools/npu/npu-window.sh tools/npu/npu-dense27b-w1.sh
# Every step is bounded by `timeout`; the whole script by 80 s. Exit status = failed steps (capped at 100).
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT_DIR=${NPU_PROBE_OUT:-$ROOT/npu-probe-out/$(date -u +%Y%m%dT%H%M%SZ)-$$}
BIN=$ROOT/target/release/npu-gemm
[ -x "$BIN" ] || { echo "PREFLIGHT FAIL: $BIN missing or not executable"; exit 2; }
mkdir -p "$OUT_DIR" && cd "$OUT_DIR" || { echo "cannot use $OUT_DIR"; exit 2; }
START=$(date +%s)
FAILED=0
mark() { echo "mark $(date +%s%N | cut -c1-19) $*"; }
run() {  # run SECONDS NPU-GEMM-ARGS... ; returns the status (124 = timeout, 125 = skipped)
  local t=$1; shift
  if [ $(( $(date +%s) - START + t )) -gt 80 ]; then echo "SKIP (time budget): $*"; return 125; fi
  mark "npu-gemm $*"
  timeout --foreground --signal=TERM --kill-after=1 "$t" "$BIN" "$@"; local rc=$?
  echo "step rc=$rc: $*"; [ "$rc" = 0 ] || FAILED=$((FAILED + 1))
  return "$rc"
}
IU4="--variant V8 --epi i32 --iu4-epochs --timeout-ms 3000"
ok=1
# shellcheck disable=SC2086
{
  run 4 512 512 256 $IU4 --reuse-ctx --iters 3 || ok=0
  run 6 1024 1024 1024 $IU4 --reuse-ctx --iters 2 || ok=0
  run 4 300 700 512 $IU4 --core FastSlowCtl --iters 1 || ok=0
}
if [ $ok = 1 ]; then
  # shellcheck disable=SC2086
  {
    # K=5120 (GDN qkv, zba, FA q/k/v, gate_up), K=6144 (o_proj), K=17408 (down), and the narrow k/v feature count.
    run 8 512 3072 5120 $IU4 --loop 2
    run 8 512 2560 6144 $IU4 --loop 2
    run 8 512 512 17408 $IU4 --loop 2
    run 6 512 1024 5120 $IU4 --loop 1.5
    # Fold-free V8 full-K integer GEMM (int8 / i32 C) at 27B K values; down uses half of K = 17408 (MAX_KC 160).
    run 12 4096 4096 5120 --variant V8 --loop 2 --timeout-ms 3000
    run 12 4096 4096 5120 --variant V8 --epi i32 --loop 2 --timeout-ms 3000
    run 12 4096 5120 6144 --variant V8 --loop 2 --timeout-ms 3000
    run 12 4096 5120 8704 --variant V8 --loop 2 --timeout-ms 3000
  }
fi
mark "end failed_steps=$FAILED"
exit $(( FAILED > 100 ? 100 : FAILED ))
