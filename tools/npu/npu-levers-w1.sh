#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
# npu-levers-w1.sh — levers-joint Window 1: same-day V9/V10 baselines, E0b (V9 no-compute kc sweep at two M), zero-code
# E0a (V10 no-compute kc sweep: A and B memtile-resident, so the per-chunk slope is memtile->core delivery through the
# normal two-slot core ring), and the V9 gate_up repeat=2/3 compute slope (lever 2 reference).
#   ~/pm-wave/hipx-lock.sh timing -- env NPU_WINDOW_TIMEOUT=90 tools/npu/npu-window.sh tools/npu/npu-levers-w1.sh
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
run 4 1536 1024 192 --variant V9 --reuse-ctx --iters 3 --timeout-ms 1000 || ok=0
run 4 512 512 64 --variant V10 --iters 2 --timeout-ms 1000 || ok=0
if [ $ok = 1 ]; then
  # Baselines (exact) and their no-compute twins.
  # shellcheck disable=SC2086
  run 6 $GU --variant V9 --loop 2 --timeout-ms 3000
  # shellcheck disable=SC2086
  run 6 $DN --variant V9 --loop 2 --timeout-ms 3000
  # shellcheck disable=SC2086
  run 6 $DN --variant V10 --loop 2 --timeout-ms 3000
  for v in "$GU --variant V9" "$DN --variant V9" "$DN --variant V10"; do
    # shellcheck disable=SC2086
    run 4 $v --ctl-probe nocompute --loop 0.7 --timeout-ms 3000
  done
  # E0b: V9 no-compute, N=2560, K=64/128/320/640, M=4096 (40 waves) and M=2048 (20 waves).
  for m in 4096 2048; do for k in 64 128 320 640; do
    run 4 $m 2560 $k --variant V9 --ctl-probe nocompute --loop 0.5 --timeout-ms 3000
  done; done
  # E0a (zero code): V10 no-compute, same grid; DDR reads A and B once per submit.
  for m in 4096 2048; do for k in 64 128 320 640; do
    run 4 $m 2560 $k --variant V10 --ctl-probe nocompute --loop 0.5 --timeout-ms 3000
  done; done
  # Lever 2 reference: V9 gate_up compute slope (timing only).
  for r in 2 3; do
    # shellcheck disable=SC2086
    run 4 $GU --variant V9 --ctl-probe repeat=$r --loop 0.5 --timeout-ms 3000
  done
fi
mark "end failed_steps=$FAILED"
exit $(( FAILED > 100 ? 100 : FAILED ))
