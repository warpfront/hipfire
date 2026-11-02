#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
# npu-probe2.sh — one npu-window.sh hold after the branch-target alignment fix: V8 Fast (aligned) small checks and
# expert-shape loops (int8 and i32), V6 Fast small check + expert loops, K = 5120 (gate_up N, MAX_KC 160), and the
# K-slope timing probes with Fast control (DMA only, compute x2/x3).
#   ~/pm-wave/hipx-lock.sh timing -- env NPU_WINDOW_TIMEOUT=90 tools/npu/npu-window.sh tools/npu/npu-probe2.sh
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
for epi in i32 int8; do
  for k in 64 256; do run 4 512 512 $k --variant V8 --epi $epi --iters 2 --timeout-ms 1000 || ok=0; done
done
if [ $ok = 1 ]; then
  for shape in "$GU" "$DN"; do
    # shellcheck disable=SC2086
    run 6 $shape --variant V8 --epi int8 --loop 2 --timeout-ms 3000
    # shellcheck disable=SC2086
    run 6 $shape --variant V8 --epi i32 --reuse-ctx --iters 3 --timeout-ms 3000
  done
  run 6 4096 1280 5120 --variant V8 --epi int8 --loop 2 --timeout-ms 3000
  for shape in "$GU" "$DN"; do
    for knobs in nocompute repeat=2 repeat=3; do
      # shellcheck disable=SC2086
      run 4 $shape --variant V8 --epi int8 --ctl-probe "$knobs" --loop 1 --timeout-ms 3000
    done
  done
fi
run 6 4096 1280 5120 --variant V8 --epi int8 --core FastSlowCtl --loop 2 --timeout-ms 3000
if run 4 512 512 64 --array --core Fast --iters 2 --timeout-ms 1000; then
  for shape in "$GU" "$DN"; do
    # shellcheck disable=SC2086
    run 6 $shape --array --core Fast --loop 2 --timeout-ms 3000
  done
fi
mark "end failed_steps=$FAILED"
exit $(( FAILED > 100 ? 100 : FAILED ))
