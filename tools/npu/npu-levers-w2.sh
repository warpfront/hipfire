#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
# npu-levers-w2.sh — levers-joint lever 2 kill window: balanced pair A order (`--ctl-probe swap`) vs baseline.
# Small exact checks with swap first; then interleaved baseline/swap exact loops at both expert shapes (ABAB for
# same-window variation), V10 down with swap, and repeat=2/3/4 compute slopes (timing only) for V8 and V9.
#   ~/pm-wave/hipx-lock.sh timing -- env NPU_WINDOW_TIMEOUT=90 tools/npu/npu-window.sh tools/npu/npu-levers-w2.sh
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
GU="4096 1280 2560"; DN="4096 2560 640"
ok=1
for epi in i32 int8; do run 4 512 512 128 --variant V8 --epi $epi --ctl-probe swap --iters 2 --timeout-ms 1000 || ok=0; done
run 4 1536 1024 192 --variant V9 --ctl-probe swap --reuse-ctx --iters 3 --timeout-ms 1000 || ok=0
run 4 1536 1024 192 --variant V10 --ctl-probe swap --reuse-ctx --iters 3 --timeout-ms 1000 || ok=0
if [ $ok = 1 ]; then
  for shape in "$GU" "$DN"; do for pass in 1 2; do
    # shellcheck disable=SC2086
    run 4 $shape --variant V9 --loop 1.5 --timeout-ms 3000
    # shellcheck disable=SC2086
    run 4 $shape --variant V9 --ctl-probe swap --loop 1.5 --timeout-ms 3000
  done; done
  # shellcheck disable=SC2086
  run 4 $DN --variant V10 --ctl-probe swap --loop 1.5 --timeout-ms 3000
  for v in V8 V9; do for knob in "" ",swap"; do for r in 2 3 4; do
    # shellcheck disable=SC2086
    run 4 $GU --variant $v --ctl-probe repeat=$r$knob --loop 0.5 --timeout-ms 3000
  done; done; done
fi
mark "end failed_steps=$FAILED"
exit $(( FAILED > 100 ? 100 : FAILED ))
