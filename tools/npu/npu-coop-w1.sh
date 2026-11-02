#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
# npu-coop-w1.sh — one npu-window.sh hold, declared new design: grouped G80 (`npu-experts --design G80`, one 256-row wave
# for M <= 256, two waves for M = 512) vs grouped V9, both with lean tail bodies, gate_up 1280x2560, E = 8/32/64 x
# M = 64/128/160/256/512, interleaved V9 then G80 per cell; V9 gate_up health check (npu-gemm) last.
#   ~/pm-wave/hipx-lock.sh timing -- env NPU_WINDOW_TIMEOUT=90 tools/npu/npu-window.sh tools/npu/npu-coop-w1.sh
# Order: insts-only sizes (no device); first silicon sanity G80 m160 E8 (every round of every expert verified); the
# one-wave cells M = 64/128/160/256 for E = 8/32/64; only if all passed: the two-wave sanity G80 m512 E8 (every round
# verified) and the M = 512 cells. Every matrix run checks the first + last round of EVERY expert byte-equal to its eager
# full-TXN oracle (and a CPU sample exact); rounds for ~0.25 s of round time at 10 TOPS (estimate), 2..400. The first
# failing step stops the window; the V9 health check runs last unless a wedge sign appeared (timeout, accel0 missing, SMU
# / amdxdna error in the kernel log): then stop and report. Exit status = failed steps (capped at 100).
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT_DIR=${NPU_PROBE_OUT:-$ROOT/npu-probe-out/$(date -u +%Y%m%dT%H%M%SZ)-$$}
EXP=$ROOT/target/release/npu-experts
GEMM=$ROOT/target/release/npu-gemm
for b in "$EXP" "$GEMM"; do [ -x "$b" ] || { echo "PREFLIGHT FAIL: $b missing or not executable"; exit 2; }; done
mkdir -p "$OUT_DIR" && cd "$OUT_DIR" || { echo "cannot use $OUT_DIR"; exit 2; }
START=$(date +%s)
START0=$START
FAILED=0
TIMEDOUT=0
ok=1
mark() { echo "mark $(date +%s%N | cut -c1-19) $*"; }
run() {  # run SECONDS BIN ARGS... ; returns the status (124 = timeout, 125 = skipped)
  local t=$1; shift
  if [ $(( $(date +%s) - START + t )) -gt 78 ]; then echo "SKIP (time budget): $*"; return 125; fi
  mark "$*"
  timeout --foreground --signal=TERM --kill-after=1 "$t" "$@"; local rc=$?
  echo "step rc=$rc: $*"; [ "$rc" = 0 ] || FAILED=$((FAILED + 1)); [ "$rc" = 124 ] || [ "$rc" = 137 ] && TIMEDOUT=1
  return "$rc"
}
for e in 8 32 64; do for m in 160 512; do
  run 10 "$EXP" --design G80 --proj gate_up --m $m --experts $e --insts-only || ok=0
done; done
[ $ok = 1 ] && { run 15 "$EXP" --design G80 --proj gate_up --m 160 --experts 8 --rounds 3 --verify-every --timeout-ms 3000 || ok=0; }
cell() {  # cell E M : V9 then G80, rounds for ~0.25 s at 10 TOPS (estimate)
  local e=$1 m=$2 ops us rounds d
  ops=$(( 2 * m * 1280 * 2560 * e )); us=$(( ops / 10000000 )); [ "$us" -lt 1 ] && us=1
  rounds=$(( 250000 / us )); [ "$rounds" -lt 2 ] && rounds=2; [ "$rounds" -gt 400 ] && rounds=400
  for d in V9 G80; do
    run 12 "$EXP" --design $d --proj gate_up --m $m --experts $e --rounds $rounds --timeout-ms 3000 || return 1
  done
}
# One-wave G80 cells (M = 64..256) first; the two-wave M = 512 group (newest) only after all of them passed.
if [ $ok = 1 ]; then
  for e in 8 32 64; do for m in 64 128 160 256; do cell $e $m || { ok=0; break 2; }; done; done
fi
[ $ok = 1 ] && { run 15 "$EXP" --design G80 --proj gate_up --m 512 --experts 8 --rounds 3 --verify-every --timeout-ms 3000 || ok=0; }
if [ $ok = 1 ]; then
  for e in 8 32 64; do cell $e 512 || { ok=0; break; }; done
fi
# Health check last: V9 gate_up exact. Skipped on a wedge sign (any timeout, missing accel0, SMU error in the kernel log).
if [ "$TIMEDOUT" = 1 ] || [ ! -e /dev/accel/accel0 ] || sudo -n journalctl -k --since "@$START0" --no-pager 2>/dev/null | grep -qiE 'smu|amdxdna.*(fail|error|timeout)'; then
  mark "WEDGE SIGN (timeout=$TIMEDOUT accel0=$([ -e /dev/accel/accel0 ] && echo present || echo MISSING)): health check NOT run, stop and report"
  FAILED=$((FAILED + 1))
else
  START=$(( $(date +%s) - 70 ))
  run 6 "$GEMM" 4096 1280 2560 --variant V9 --loop 1 --timeout-ms 3000
fi
mark "end failed_steps=$FAILED ok=$ok"
exit $(( FAILED > 100 ? 100 : FAILED ))
