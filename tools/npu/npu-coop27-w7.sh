#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
# npu-coop27-w7.sh — one npu-window.sh hold: the co-op re-measured on the coarse-grained transport (4k anonymous pages,
# NPU userptr BO + hipHostRegister(hipExtHostRegisterCoarseGrained)): gate_up J in $JS, then down n=1024 x $DOWN_RUNS
# (fresh processes), all write-back pack; V9 health last unless a wedge sign appeared. Env: BIN, DATA, JS, DOWN_RUNS, REG.
set -u
BIN=${BIN:?BIN}; DATA=${DATA:?DATA}; JS=${JS:-1 2 3}; DOWN_RUNS=${DOWN_RUNS:-1}; REG=${REG:-coarse}
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
  [ $ok = 1 ] && { run 20 "$C" --mode coop --proj gate_up --layer 2 --xq "$DATA/L2.mlp.a4.bin" --hid $((1024 * j)) --iters 5 --pack wb --reg "$REG" || ok=0; }
done
for _ in $(seq "$DOWN_RUNS"); do
  [ $ok = 1 ] && { run 12 "$C" --mode coop --layer 2 --xq "$DATA/L2.down.a4.bin" --npu-cols 1024 --iters 6 --pack wb --reg "$REG" || ok=0; }
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
