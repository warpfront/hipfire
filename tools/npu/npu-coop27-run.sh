#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
# npu-coop27-run.sh — one npu-window.sh hold running coop27 steps from $STEPS (a file, one coop27 argument list per line,
# `$DATA` expanded; `#` comments), stop at the first failure, then the V9 health check unless a wedge sign appeared (timeout,
# accel0 missing, SMU / amdxdna error in the kernel log). Env: BIN, DATA, STEPS, STEP_TIMEOUT (default 20 s).
#   ~/pm-wave/hipx-lock.sh timing -- env NPU_WINDOW_TIMEOUT=90 NPU_FCLK_TOOL=$BIN/npu-tools BIN=.. DATA=.. STEPS=.. tools/npu/npu-window.sh tools/npu/npu-coop27-run.sh
set -u
BIN=${BIN:?BIN}; DATA=${DATA:?DATA}; STEPS=${STEPS:?STEPS}; STEP_TIMEOUT=${STEP_TIMEOUT:-20}
C=$BIN/coop27; GEMM=$BIN/npu-gemm
for b in "$C" "$GEMM"; do [ -x "$b" ] || { echo "PREFLIGHT FAIL: $b missing"; exit 2; }; done
START=$(date +%s); START0=$START; FAILED=0; TIMEDOUT=0; ok=1
# Halo GPU (0000:bf:00.0) edge temperature and sclk at every mark (thermal throttling near 95 C).
therm() { local h; h=$(ls -d /sys/bus/pci/devices/0000:bf:00.0/hwmon/hwmon* 2>/dev/null | head -1)
  [ -n "$h" ] && echo "gpu_temp_mC=$(cat "$h/temp1_input" 2>/dev/null) sclk_Hz=$(cat "$h/freq1_input" 2>/dev/null)"; }
mark() { echo "mark $(date +%s%N | cut -c1-19) $(therm) $*"; }
run() {
  local t=$1; shift
  if [ $(( $(date +%s) - START + t )) -gt 80 ]; then echo "SKIP (time budget): $*"; return 125; fi
  mark "$*"
  timeout --foreground --signal=TERM --kill-after=2 "$t" "$@"; local rc=$?
  echo "step rc=$rc: $*"; [ "$rc" = 0 ] || FAILED=$((FAILED + 1)); { [ "$rc" = 124 ] || [ "$rc" = 137 ]; } && TIMEDOUT=1
  return "$rc"
}
while IFS= read -r line; do
  line=${line%%#*}; [ -z "${line// }" ] && continue
  [ $ok = 1 ] || break
  # shellcheck disable=SC2086
  run "$STEP_TIMEOUT" "$C" ${line//\$DATA/$DATA} || ok=0
done < "$STEPS"
if [ "$TIMEDOUT" = 1 ] || [ ! -e /dev/accel/accel0 ] || sudo -n journalctl -k --since "@$START0" --no-pager 2>/dev/null | grep -qiE 'smu|amdxdna.*(fail|error|timeout)'; then
  mark "WEDGE SIGN (timeout=$TIMEDOUT accel0=$([ -e /dev/accel/accel0 ] && echo present || echo MISSING)): health check NOT run, stop and report"
  FAILED=$((FAILED + 1))
else
  START=$(( $(date +%s) - 70 ))
  run 6 "$GEMM" 4096 1280 2560 --variant V9 --loop 1 --timeout-ms 3000
fi
mark "end failed_steps=$FAILED ok=$ok"
exit $(( FAILED > 100 ? 100 : FAILED ))
