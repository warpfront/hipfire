#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
# npu-v8.sh — one npu-window.sh hold for the V8 whole-array design (pair A-sharing through neighbour memory,
# int8 SRS epilogue, two memtile->shim C channels) plus V6 Fast throughput runs.
# Run as the command of one npu-window.sh hold under the exclusive Halo timing lock:
#   ~/pm-wave/hipx-lock.sh timing -- env NPU_WINDOW_TIMEOUT=90 tools/npu/npu-window.sh tools/npu/npu-v8.sh
# Order: small V8 checks per (control, epilogue) with a 3 s wait; expert shapes (M=4096, 3 verified submits on
# one context) only for a control discipline whose small check passed; then --loop runs (first and last submit
# verified) for wall TOPS. V6 Fast runs only if its own small check passes. Every step is bounded by
# `timeout`; the whole script by 80 s. Exit status = number of failed steps (capped at 100).
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT_DIR=${NPU_V8_OUT:-$ROOT/npu-v8-out/$(date -u +%Y%m%dT%H%M%SZ)-$$}
BIN=$ROOT/target/release/npu-gemm
[ -x "$BIN" ] || { echo "PREFLIGHT FAIL: $BIN missing or not executable"; exit 2; }
mkdir -p "$OUT_DIR" && cd "$OUT_DIR" || { echo "cannot use $OUT_DIR"; exit 2; }
START=$(date +%s)
FAILED=0
mark() { echo "mark $(date +%s%N | cut -c1-19) $*"; }
run() {  # run SECONDS NPU-GEMM-ARGS... ; returns the npu-gemm status (124 = timeout)
  local t=$1; shift
  if [ $(( $(date +%s) - START + t )) -gt 80 ]; then echo "SKIP (time budget): $*"; return 125; fi
  mark "npu-gemm $*"
  timeout --foreground --signal=TERM --kill-after=1 "$t" "$BIN" "$@"; local rc=$?
  echo "step rc=$rc: $*"; [ "$rc" = 0 ] || FAILED=$((FAILED + 1))
  return "$rc"
}
GU="4096 1280 2560"; DN="4096 2560 640"
ok_ctl=()
for core in FastSlowCtl Fast; do
  run 6 512 512 64 --variant V8 --epi i32 --core $core --iters 2 --timeout-ms 3000
  run 6 512 512 64 --variant V8 --epi int8 --core $core --iters 2 --timeout-ms 3000 && \
    run 6 512 512 256 --variant V8 --epi int8 --core $core --iters 2 --timeout-ms 3000 && ok_ctl+=("$core")
done
echo "V8 controls passing the small checks: ${ok_ctl[*]:-none}"
for core in "${ok_ctl[@]}"; do
  for shape in "$GU" "$DN"; do
    # shellcheck disable=SC2086
    run 12 $shape --variant V8 --epi int8 --core $core --reuse-ctx --iters 3 --timeout-ms 5000
  done
done
best=${ok_ctl[${#ok_ctl[@]}-1]:-}
if [ -n "$best" ]; then
  for shape in "$GU" "$DN"; do
    # shellcheck disable=SC2086
    run 12 $shape --variant V8 --epi i32 --core $best --reuse-ctx --iters 3 --timeout-ms 5000
  done
  for shape in "$GU" "$DN"; do
    # shellcheck disable=SC2086
    run 8 $shape --variant V8 --epi int8 --core $best --loop 3 --timeout-ms 5000
  done
fi
# V6 (the M3 deployment) with the Fast core (JNZ back edge): throughput only if the small check is exact.
if run 6 512 512 64 --array --core Fast --iters 2 --timeout-ms 3000; then
  for shape in "$GU" "$DN"; do
    # shellcheck disable=SC2086
    run 12 $shape --array --core Fast --reuse-ctx --iters 3 --timeout-ms 5000
    # shellcheck disable=SC2086
    run 8 $shape --array --core Fast --loop 3 --timeout-ms 5000
  done
fi
mark "end failed_steps=$FAILED"
exit $(( FAILED > 100 ? 100 : FAILED ))
