#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
# npu-probe3.sh — one npu-window.sh hold: V9 (resident B) small checks (fresh and reused context), expert-shape
# loops (Fast and FastSlowCtl), M = 8192, DMA-only and compute-repeat timing probes, V8 reference loops.
#   ~/pm-wave/hipx-lock.sh timing -- env NPU_WINDOW_TIMEOUT=90 tools/npu/npu-window.sh tools/npu/npu-probe3.sh
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
ok=1
run 4 512 512 64 --variant V9 --iters 2 --timeout-ms 1000 || ok=0
run 4 1024 1024 128 --variant V9 --iters 2 --timeout-ms 1000 || ok=0
run 4 1536 1024 192 --variant V9 --reuse-ctx --iters 3 --timeout-ms 1000 || ok=0
if [ $ok = 1 ]; then
  for shape in "$GU" "$DN"; do
    # shellcheck disable=SC2086
    run 6 $shape --variant V9 --loop 2 --timeout-ms 3000
    # shellcheck disable=SC2086
    run 6 $shape --variant V9 --core FastSlowCtl --loop 2 --timeout-ms 3000
    # shellcheck disable=SC2086
    run 4 $shape --variant V9 --ctl-probe nocompute --loop 1 --timeout-ms 3000
  done
  run 6 8192 1280 2560 --variant V9 --loop 2 --timeout-ms 3000
  run 6 4096 1280 3456 --variant V9 --loop 2 --timeout-ms 3000
  run 6 4096 2560 2560 --variant V9 --loop 2 --timeout-ms 3000
fi
for shape in "$GU" "$DN"; do
  # shellcheck disable=SC2086
  run 5 $shape --variant V8 --epi int8 --loop 1 --timeout-ms 3000
done
mark "end failed_steps=$FAILED"
exit $(( FAILED > 100 ? 100 : FAILED ))
