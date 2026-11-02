#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
# npu-coop27-w2.sh — one npu-window.sh hold, the declared first dense-27B down co-op window (coop27, docs/coop-down27.md):
#   ~/pm-wave/hipx-lock.sh timing -- env NPU_WINDOW_TIMEOUT=90 NPU_FCLK_TOOL=$BIN/npu-tools tools/npu/npu-window.sh tools/npu/npu-coop27-w2.sh
# Env: BIN (coop27 / npu-gemm directory), XQ (captured down A4, layer $LAYER), LAYER (default 2).
# Order (stop at the first failure): (1) GPU-only kernel check incl. the write-back pack variant (no NPU command);
# (2) NPU-only IEF15 down vs columns on the co-op transport (alias check first, in-process); (3) co-op n=1024, streaming
# pack; (4) co-op n=1024, write-back pack; (5) V9 health check unless a wedge sign appeared (timeout, accel0 missing,
# SMU / amdxdna error in the kernel log): then stop and report.
set -u
BIN=${BIN:?BIN}; XQ=${XQ:?XQ}; LAYER=${LAYER:-2}
C=$BIN/coop27; GEMM=$BIN/npu-gemm
for b in "$C" "$GEMM"; do [ -x "$b" ] || { echo "PREFLIGHT FAIL: $b missing"; exit 2; }; done
START=$(date +%s); START0=$START; FAILED=0; TIMEDOUT=0; ok=1
mark() { echo "mark $(date +%s%N | cut -c1-19) $*"; }
run() {  # run SECONDS BIN ARGS...
  local t=$1; shift
  if [ $(( $(date +%s) - START + t )) -gt 80 ]; then echo "SKIP (time budget): $*"; return 125; fi
  mark "$*"
  timeout --foreground --signal=TERM --kill-after=2 "$t" "$@"; local rc=$?
  echo "step rc=$rc: $*"; [ "$rc" = 0 ] || FAILED=$((FAILED + 1)); { [ "$rc" = 124 ] || [ "$rc" = 137 ]; } && TIMEDOUT=1
  return "$rc"
}
A=(--layer "$LAYER" --xq "$XQ")
run 12 "$C" --mode gpu "${A[@]}" --cols 5120 --npu-cols 1024 --reps 3 || ok=0
[ $ok = 1 ] && { run 25 "$C" --mode npu "${A[@]}" --npu-cols 512,768,1024,1280,1536 --iters 5 || ok=0; }
[ $ok = 1 ] && { run 14 "$C" --mode coop "${A[@]}" --npu-cols 1024 --iters 6 --pack stream || ok=0; }
[ $ok = 1 ] && { run 14 "$C" --mode coop "${A[@]}" --npu-cols 1024 --iters 6 --pack wb || ok=0; }
if [ "$TIMEDOUT" = 1 ] || [ ! -e /dev/accel/accel0 ] || sudo -n journalctl -k --since "@$START0" --no-pager 2>/dev/null | grep -qiE 'smu|amdxdna.*(fail|error|timeout)'; then
  mark "WEDGE SIGN (timeout=$TIMEDOUT accel0=$([ -e /dev/accel/accel0 ] && echo present || echo MISSING)): health check NOT run, stop and report"
  FAILED=$((FAILED + 1))
else
  START=$(( $(date +%s) - 70 ))
  run 6 "$GEMM" 4096 1280 2560 --variant V9 --loop 1 --timeout-ms 3000
fi
mark "end failed_steps=$FAILED ok=$ok"
exit $(( FAILED > 100 ? 100 : FAILED ))
