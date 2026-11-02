#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
# npu-levers-w4.sh — lever 3B confirmation (one hold): V10 down prefix-ready B (`--b-prefix`) vs V9 and V10 down,
# four interleaved exact passes (first + final submit verified), then repeat=2/3/4 compute slopes in 2 s loops for V10
# and V10 prefix (timing only).
#   ~/pm-wave/hipx-lock.sh timing -- env NPU_WINDOW_TIMEOUT=90 tools/npu/npu-window.sh tools/npu/npu-levers-w4.sh
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
DN="4096 2560 640"
ok=1
run 4 1536 512 192 --variant V10 --b-prefix --reuse-ctx --iters 3 --timeout-ms 1000 || ok=0
if [ $ok = 1 ]; then
  for pass in 1 2 3 4; do for v in "--variant V9" "--variant V10 --b-prefix" "--variant V10"; do
    # shellcheck disable=SC2086
    run 4 $DN $v --loop 1.5 --timeout-ms 3000
  done; done
  for r in 2 3 4; do for v in "--variant V10" "--variant V10 --b-prefix"; do
    # shellcheck disable=SC2086
    run 5 $DN $v --ctl-probe repeat=$r --loop 2 --timeout-ms 3000
  done; done
fi
mark "end failed_steps=$FAILED"
exit $(( FAILED > 100 ? 100 : FAILED ))
