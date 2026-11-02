#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
# npu-probe4.sh — one npu-window.sh hold: GPU->NPU dma-buf import (`--a-from-hip`: HIP host-VMM A written by the
# GPU, exported as dma-buf, imported by amdxdna as arg0), small and expert shapes, exact against the CPU.
#   ~/pm-wave/hipx-lock.sh timing -- env NPU_WINDOW_TIMEOUT=90 tools/npu/npu-window.sh tools/npu/npu-probe4.sh
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
GU="4096 1280 2560"; DN="4096 2560 640"
if run 15 512 512 64 --variant V8 --a-from-hip --iters 1 --timeout-ms 2000; then
  run 15 512 512 64 --variant V8 --a-from-hip --iters 3 --reuse-ctx --timeout-ms 2000
  run 15 512 512 256 --variant V9 --a-from-hip --iters 2 --timeout-ms 2000
  # shellcheck disable=SC2086
  run 20 $GU --variant V9 --a-from-hip --loop 2 --timeout-ms 3000
  # shellcheck disable=SC2086
  run 20 $DN --variant V9 --a-from-hip --loop 2 --timeout-ms 3000
fi
mark "end failed_steps=$FAILED"
exit $(( FAILED > 100 ? 100 : FAILED ))
