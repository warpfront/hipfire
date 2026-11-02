#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
# npu-levers-w5.sh — shipped-shape baselines for the lever verdicts (one hold): Flash-Next routed experts at realistic M
# (64/128/160/256/512 rows; gate_up N=1280 K=2560, down N=2560 K=640). gate_up: V9 (V10 cannot hold K=2560 at NW=3).
# down: V9, V10 and V10 `--b-prefix`. Exact loops, first + final submit verified.
#   ~/pm-wave/hipx-lock.sh timing -- env NPU_WINDOW_TIMEOUT=90 tools/npu/npu-window.sh tools/npu/npu-levers-w5.sh
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
ok=1
run 4 160 1280 2560 --variant V9 --reuse-ctx --iters 2 --timeout-ms 1000 || ok=0
run 4 160 2560 640 --variant V10 --b-prefix --reuse-ctx --iters 2 --timeout-ms 1000 || ok=0
if [ $ok = 1 ]; then
  for m in 160 64 128 256 512; do
    run 4 $m 1280 2560 --variant V9 --loop 1 --timeout-ms 3000
    for v in "--variant V9" "--variant V10" "--variant V10 --b-prefix"; do
      # shellcheck disable=SC2086
      run 4 $m 2560 640 $v --loop 1 --timeout-ms 3000
    done
  done
  # Second pass at the average M for same-window variation.
  run 4 160 1280 2560 --variant V9 --loop 1 --timeout-ms 3000
  for v in "--variant V9" "--variant V10" "--variant V10 --b-prefix"; do
    # shellcheck disable=SC2086
    run 4 160 2560 640 $v --loop 1 --timeout-ms 3000
  done
fi
mark "end failed_steps=$FAILED"
exit $(( FAILED > 100 ? 100 : FAILED ))
