#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
# npu-coop27-w5.sh — one npu-window.sh hold, declared: dense-27B gate_up co-op (coop27 --proj gate_up --mode coop, J IEF15
# commands of 8192x2048x5120 on one context, write-back pack) for each J in $JS (default "1 2 3"), then (PARTS=1) the
# two-partition context probe (coop27 --mode parts, M2 single-core design, --part-tiles 32,16), then (SWITCH=1) the
# one-context-switching cost down <-> gate_up (npu-gemm --switch-with), V9 health last unless a wedge sign appeared.
#   ~/pm-wave/hipx-lock.sh timing -- env NPU_WINDOW_TIMEOUT=90 NPU_FCLK_TOOL=$BIN/npu-tools BIN=.. DATA=.. tools/npu/npu-window.sh tools/npu/npu-coop27-w5.sh
set -u
BIN=${BIN:?BIN}; DATA=${DATA:?DATA}; JS=${JS:-1 2 3}; PARTS=${PARTS:-1}; SWITCH=${SWITCH:-0}
C=$BIN/coop27; GEMM=$BIN/npu-gemm
for b in "$C" "$GEMM"; do [ -x "$b" ] || { echo "PREFLIGHT FAIL: $b missing"; exit 2; }; done
START=$(date +%s); START0=$START; FAILED=0; TIMEDOUT=0; ok=1
mark() { echo "mark $(date +%s%N | cut -c1-19) $*"; }
run() {
  local t=$1; shift
  if [ $(( $(date +%s) - START + t )) -gt 80 ]; then echo "SKIP (time budget): $*"; return 125; fi
  mark "$*"
  timeout --foreground --signal=TERM --kill-after=2 "$t" "$@"; local rc=$?
  echo "step rc=$rc: $*"; [ "$rc" = 0 ] || FAILED=$((FAILED + 1)); { [ "$rc" = 124 ] || [ "$rc" = 137 ]; } && TIMEDOUT=1
  return "$rc"
}
for j in $JS; do
  [ $ok = 1 ] && { run 20 "$C" --mode coop --proj gate_up --layer 2 --xq "$DATA/L2.mlp.a4.bin" --hid $((1024 * j)) --iters 5 --pack wb || ok=0; }
done
[ $ok = 1 ] && [ "$PARTS" = 1 ] && { run 10 "$C" --mode parts --part-tiles 32,16 --iters 20 --timeout-ms 3000 || ok=0; }
[ $ok = 1 ] && [ "$SWITCH" = 1 ] && { run 15 "$GEMM" 8192 1024 17408 --fold-contract ief15 --switch-with 8192,2048,5120 --timeout-ms 3000 || ok=0; }
if [ "$TIMEDOUT" = 1 ] || [ ! -e /dev/accel/accel0 ] || sudo -n journalctl -k --since "@$START0" --no-pager 2>/dev/null | grep -qiE 'smu|amdxdna.*(fail|error|timeout)'; then
  mark "WEDGE SIGN (timeout=$TIMEDOUT accel0=$([ -e /dev/accel/accel0 ] && echo present || echo MISSING)): health check NOT run, stop and report"
  FAILED=$((FAILED + 1))
else
  START=$(( $(date +%s) - 70 ))
  run 6 "$GEMM" 4096 1280 2560 --variant V9 --loop 1 --timeout-ms 3000
fi
mark "end failed_steps=$FAILED ok=$ok"
exit $(( FAILED > 100 ? 100 : FAILED ))
