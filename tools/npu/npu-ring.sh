#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
# npu-ring.sh — one npu-window.sh hold: persistent V9 ring (npu-ring) mechanism latency, 512 and expert-shape runs, negatives.
#   ~/pm-wave/hipx-lock.sh timing -- env NPU_WINDOW_TIMEOUT=90 tools/npu/npu-window.sh tools/npu/npu-ring.sh
# Every step is bounded by `timeout`; the whole script by 80 s. Exit status = failed steps (capped at 100).
# Steps: empty-prepublish S8; empty-paced S8 x50 rounds; 512x512x64 prepublish + paced S8; gate_up 4096x1280x2560 and
# down 4096x2560x640 prepublish S8 (rounds fit into 2 s); gate_up paced S8; negatives skip-done / stale-seq at 512x512x64.
# NPU_RING_PHASE=lean runs only the lean window instead: gate_up/down prepublish S8 --fit-ms 2000 --lean, gate_up paced --lean,
# 512x512x128 prepublish --lean --verify-every, then the full-body gate_up/down prepublish baselines; stops on the first failure.
# All windows use S == R == 8 (multi-round needs S % R == 0). Negatives exit 0 only when the negative is detected.
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT_DIR=${NPU_PROBE_OUT:-$ROOT/npu-probe-out/$(date -u +%Y%m%dT%H%M%SZ)-$$}
BIN=$ROOT/target/release/npu-ring
[ -x "$BIN" ] || { echo "PREFLIGHT FAIL: $BIN missing or not executable"; exit 2; }
mkdir -p "$OUT_DIR" && cd "$OUT_DIR" || { echo "cannot use $OUT_DIR"; exit 2; }
START=$(date +%s)
FAILED=0
mark() { echo "mark $(date +%s%N | cut -c1-19) $*"; }
run() {  # run SECONDS NPU-RING-ARGS... ; returns the status (124 = timeout, 125 = skipped)
  local t=$1; shift
  if [ $(( $(date +%s) - START + t )) -gt 80 ]; then echo "SKIP (time budget): $*"; return 125; fi
  mark "npu-ring $*"
  timeout --foreground --signal=TERM --kill-after=1 "$t" "$BIN" "$@"; local rc=$?
  echo "step rc=$rc: $*"; [ "$rc" = 0 ] || FAILED=$((FAILED + 1))
  return "$rc"
}
GU="4096 1280 2560"; DN="4096 2560 640"
if [ "${NPU_RING_PHASE:-}" = lean ]; then
  # Lean window (<= 80 s): persistent_v9_lean (run 0 full, runs 1.. lean) vs the full body, same shapes, in one hold.
  # Order: full-body gate_up/down prepublish baselines first (proven), then the declared-risky lean runs last:
  # 512x512x128 prepublish --lean --verify-every, gate_up/down S8 prepublish --lean, gate_up paced --lean.
  # Stops on the first failed step.
  # shellcheck disable=SC2086
  run 13 $GU --slots 8 --nslots 8 --mode prepublish --fit-ms 2000 --timeout-ms 3000
  # shellcheck disable=SC2086
  [ "$FAILED" = 0 ] && run 11 $DN --slots 8 --nslots 8 --mode prepublish --fit-ms 2000 --timeout-ms 3000
  [ "$FAILED" = 0 ] && run 7 512 512 128 --slots 8 --nslots 8 --mode prepublish --rounds 20 --lean --verify-every --timeout-ms 2000
  # shellcheck disable=SC2086
  [ "$FAILED" = 0 ] && run 13 $GU --slots 8 --nslots 8 --mode prepublish --fit-ms 2000 --lean --timeout-ms 3000
  # shellcheck disable=SC2086
  [ "$FAILED" = 0 ] && run 11 $DN --slots 8 --nslots 8 --mode prepublish --fit-ms 2000 --lean --timeout-ms 3000
  # shellcheck disable=SC2086
  [ "$FAILED" = 0 ] && run 13 $GU --slots 8 --nslots 8 --mode paced --fit-ms 2000 --lean --timeout-ms 3000
  mark "end phase=lean failed_steps=$FAILED"
  exit $(( FAILED > 100 ? 100 : FAILED ))
fi
run 5 512 512 64 --slots 8 --nslots 8 --mode empty-prepublish --rounds 20 --timeout-ms 2000
run 6 512 512 64 --slots 8 --nslots 8 --mode empty-paced --rounds 50 --timeout-ms 2000
run 7 512 512 64 --slots 8 --nslots 8 --mode prepublish --rounds 20 --timeout-ms 2000
run 7 512 512 64 --slots 8 --nslots 8 --mode paced --rounds 20 --timeout-ms 2000
# shellcheck disable=SC2086
run 13 $GU --slots 8 --nslots 8 --mode prepublish --fit-ms 2000 --timeout-ms 3000
# shellcheck disable=SC2086
run 11 $DN --slots 8 --nslots 8 --mode prepublish --fit-ms 2000 --timeout-ms 3000
# shellcheck disable=SC2086
run 13 $GU --slots 8 --nslots 8 --mode paced --fit-ms 2000 --timeout-ms 3000
run 5 512 512 64 --slots 8 --nslots 8 --mode prepublish --neg skip-done --timeout-ms 2000
run 5 512 512 64 --slots 8 --nslots 8 --mode prepublish --neg stale-seq --rounds 2 --timeout-ms 2000
mark "end failed_steps=$FAILED"
exit $(( FAILED > 100 ? 100 : FAILED ))
