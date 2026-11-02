#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
# npu-ief15-w1.sh — one npu-window.sh hold, the declared IEF15 N0 first silicon window (docs/ief15-n0.md section 7):
#   ~/pm-wave/hipx-lock.sh timing -- env NPU_WINDOW_TIMEOUT=90 tools/npu/npu-window.sh tools/npu/npu-ief15-w1.sh
# Order (stop at the first failure or mismatch):
#   (1) single-core semantics probe `npu-gemm --ief15-probe --dump <prefix>` (437 cases, AxQOS 0 / vendor BD word 5);
#       the expected blob md5 must be 6357278936927f8b8a57c68f4bde43f1 and every case MATCH;
#   (2) the IEF15 kernel: down 8192x1024x17408 --reuse-ctx --iters 3, then --loop 2; gate_up 8192x2048x5120;
#       switch cost down <-> gate_up (--switch-with);
#   (3) V9 health check last unless a wedge sign appeared (timeout, accel0 missing, SMU / amdxdna error): stop and report.
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT_DIR=${NPU_PROBE_OUT:-$ROOT/npu-probe-out/$(date -u +%Y%m%dT%H%M%SZ)-$$}
GEMM=$ROOT/target/release/npu-gemm
EXP_MD5=6357278936927f8b8a57c68f4bde43f1
[ -x "$GEMM" ] || { echo "PREFLIGHT FAIL: $GEMM missing or not executable"; exit 2; }
mkdir -p "$OUT_DIR" && cd "$OUT_DIR" || { echo "cannot use $OUT_DIR"; exit 2; }
START=$(date +%s)
START0=$START
FAILED=0
TIMEDOUT=0
ok=1
mark() { echo "mark $(date +%s%N | cut -c1-19) $*"; }
run() {  # run SECONDS BIN ARGS... ; returns the status (124 = timeout, 125 = skipped)
  local t=$1; shift
  if [ $(( $(date +%s) - START + t )) -gt 78 ]; then echo "SKIP (time budget): $*"; FAILED=$((FAILED + 1)); return 125; fi
  mark "$*"
  timeout --foreground --signal=TERM --kill-after=2 "$t" "$@"; local rc=$?
  echo "step rc=$rc: $*"; [ "$rc" = 0 ] || FAILED=$((FAILED + 1)); { [ "$rc" = 124 ] || [ "$rc" = 137 ]; } && TIMEDOUT=1
  return "$rc"
}
run 15 "$GEMM" --ief15-probe --dump "$OUT_DIR/probe" --timeout-ms 3000 || ok=0
if [ $ok = 1 ]; then
  got=$(md5sum < "$OUT_DIR/probe.expected" | cut -d' ' -f1)
  echo "probe expected md5 $got (want $EXP_MD5)"
  [ "$got" = "$EXP_MD5" ] || { echo "FAIL: probe expected blob md5"; FAILED=$((FAILED + 1)); ok=0; }
fi
[ $ok = 1 ] && { run 15 "$GEMM" 8192 1024 17408 --fold-contract ief15 --reuse-ctx --iters 3 --timeout-ms 3000 || ok=0; }
[ $ok = 1 ] && { run 15 "$GEMM" 8192 1024 17408 --fold-contract ief15 --loop 2 --timeout-ms 3000 || ok=0; }
[ $ok = 1 ] && { run 12 "$GEMM" 8192 2048 5120 --fold-contract ief15 --reuse-ctx --iters 3 --timeout-ms 3000 || ok=0; }
[ $ok = 1 ] && { run 15 "$GEMM" 8192 1024 17408 --fold-contract ief15 --switch-with 8192,2048,5120 --timeout-ms 3000 || ok=0; }
if [ "$TIMEDOUT" = 1 ] || [ ! -e /dev/accel/accel0 ] || sudo -n journalctl -k --since "@$START0" --no-pager 2>/dev/null | grep -qiE 'smu|amdxdna.*(fail|error|timeout)'; then
  mark "WEDGE SIGN (timeout=$TIMEDOUT accel0=$([ -e /dev/accel/accel0 ] && echo present || echo MISSING)): health check NOT run, stop and report"
  FAILED=$((FAILED + 1))
else
  START=$(( $(date +%s) - 70 ))
  run 6 "$GEMM" 4096 1280 2560 --variant V9 --loop 1 --timeout-ms 3000
fi
mark "end failed_steps=$FAILED ok=$ok"
exit $(( FAILED > 100 ? 100 : FAILED ))
