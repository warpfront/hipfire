#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
# npu-coop27-w3.sh — one npu-window.sh hold: the dense-27B down co-op measurement, fresh processes interleaved
# (coop27 --mode coop, write-back pack), n = 1024 x3 on layer 2 interleaved with n = 768 / 1280 and n = 1024 on layer 42;
# V9 health last unless a wedge sign appeared. Env: BIN, DATA (dir with L2.down.a4.bin, L42.down.a4.bin).
#   ~/pm-wave/hipx-lock.sh timing -- env NPU_WINDOW_TIMEOUT=90 NPU_FCLK_TOOL=$BIN/npu-tools BIN=.. DATA=.. tools/npu/npu-window.sh tools/npu/npu-coop27-w3.sh
set -u
BIN=${BIN:?BIN}; DATA=${DATA:?DATA}
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
for spec in "2 1024" "2 768" "2 1024" "2 1280" "2 1024" "42 1024"; do
  set -- $spec
  [ $ok = 1 ] && { run 12 "$C" --mode coop --layer "$1" --xq "$DATA/L$1.down.a4.bin" --npu-cols "$2" --iters 6 --pack wb || ok=0; }
done
if [ "$TIMEDOUT" = 1 ] || [ ! -e /dev/accel/accel0 ] || sudo -n journalctl -k --since "@$START0" --no-pager 2>/dev/null | grep -qiE 'smu|amdxdna.*(fail|error|timeout)'; then
  mark "WEDGE SIGN (timeout=$TIMEDOUT accel0=$([ -e /dev/accel/accel0 ] && echo present || echo MISSING)): health check NOT run, stop and report"
  FAILED=$((FAILED + 1))
else
  START=$(( $(date +%s) - 70 ))
  run 6 "$GEMM" 4096 1280 2560 --variant V9 --loop 1 --timeout-ms 3000
fi
mark "end failed_steps=$FAILED ok=$ok"
exit $(( FAILED > 100 ? 100 : FAILED ))
