#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
# npu-probe5.sh — one npu-window.sh hold: pair-core data-memory layouts (--ctl-probe layout=N): V9 small and expert
# shapes per layout (exact), and V8 compute-repeat slopes per layout (timing only).
#   ~/pm-wave/hipx-lock.sh timing -- env NPU_WINDOW_TIMEOUT=90 tools/npu/npu-window.sh tools/npu/npu-probe5.sh
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
for l in 1 2 3; do run 4 1024 1024 128 --variant V9 --ctl-probe layout=$l --iters 1 --timeout-ms 1000; done
for l in 0 1 2 3; do
  for shape in "$GU" "$DN"; do
    # shellcheck disable=SC2086
    run 4 $shape --variant V9 --ctl-probe layout=$l --loop 1 --timeout-ms 3000
  done
done
for l in 0 1 3; do
  for r in 2 3; do
    # shellcheck disable=SC2086
    run 4 $GU --variant V8 --epi int8 --ctl-probe layout=$l,repeat=$r --loop 1 --timeout-ms 3000
  done
done
mark "end failed_steps=$FAILED"
exit $(( FAILED > 100 ? 100 : FAILED ))
