#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
# npu-levers-w6.sh — lever 4 (G80) first silicon window, declared risky (new design): small exact checks, then
# single-submit G80 vs V9 gate_up at expert M = 160/64/256/512 and M = 4096 (exact loops, first + final submit verified,
# interleaved), no-compute twins and repeat=2/3/4 compute slopes; V9 gate_up health check last.
#   ~/pm-wave/hipx-lock.sh timing -- env NPU_WINDOW_TIMEOUT=90 tools/npu/npu-window.sh tools/npu/npu-levers-w6.sh
# Any failing small check stops the window (health check still runs). Exit status = failed steps (capped at 100).
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
  if [ $(( $(date +%s) - START + t )) -gt 76 ]; then echo "SKIP (time budget): $*"; return 125; fi
  mark "npu-gemm $*"
  timeout --foreground --signal=TERM --kill-after=1 "$t" "$BIN" "$@"; local rc=$?
  echo "step rc=$rc: $*"; [ "$rc" = 0 ] || FAILED=$((FAILED + 1))
  return "$rc"
}
GU="4096 1280 2560"
ok=1
run 4 256 1280 128 --variant G80 --iters 1 --timeout-ms 1000 || ok=0
[ $ok = 1 ] && { run 4 256 1280 128 --variant G80 --reuse-ctx --iters 3 --timeout-ms 1000 || ok=0; }
[ $ok = 1 ] && { run 4 200 1280 192 --variant G80 --reuse-ctx --iters 2 --timeout-ms 1000 || ok=0; }
[ $ok = 1 ] && { run 4 160 1280 2560 --variant G80 --reuse-ctx --iters 2 --timeout-ms 1000 || ok=0; }
[ $ok = 1 ] && { run 5 $GU --variant G80 --reuse-ctx --iters 2 --timeout-ms 2000 || ok=0; }
if [ $ok = 1 ]; then
  for m in 160 64 256 512; do for v in V9 G80; do
    run 4 $m 1280 2560 --variant $v --loop 1 --timeout-ms 3000
  done; done
  for pass in 1 2; do for v in V9 G80; do
    # shellcheck disable=SC2086
    run 4 $GU --variant $v --loop 1.5 --timeout-ms 3000
  done; done
  for v in V9 G80; do
    # shellcheck disable=SC2086
    run 4 $GU --variant $v --ctl-probe nocompute --loop 0.5 --timeout-ms 3000
  done
  for v in V9 G80; do for r in 2 3 4; do
    # shellcheck disable=SC2086
    run 4 $GU --variant $v --ctl-probe repeat=$r --loop 0.8 --timeout-ms 3000
  done; done
fi
# Health check (always): V9 gate_up exact.
START=$(( $(date +%s) - 70 ))
# shellcheck disable=SC2086
run 4 $GU --variant V9 --loop 1 --timeout-ms 3000
mark "end failed_steps=$FAILED ok=$ok"
exit $(( FAILED > 100 ? 100 : FAILED ))
