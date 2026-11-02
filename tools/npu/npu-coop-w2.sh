#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
# npu-coop-w2.sh — one npu-window.sh hold, declared new design: the Halo iGPU (PM-native gfx1151 kernels) produces A,
# publishes the persistent-ring seq and consumes the NPU's done line and C through HIP VMM uncached host memory shared
# with the NPU by dma-buf (npu-coop). CPU only arms/re-arms and verifies.
#   ~/pm-wave/hipx-lock.sh timing -- env NPU_WINDOW_TIMEOUT=90 tools/npu/npu-window.sh tools/npu/npu-coop-w2.sh
# Order (stop at the first failure): the transport gate `npu-coop --mode alias` (no NPU command; userptr must alias
# CPU/GPU/NPU, the old HIP VMM dma-buf export path must be detected as non-aliasing), then the empty ring S8/R4 x8
# rounds (GPU publish -> NPU done -> GPU observe, GPU clock),
# then the GEMM ring 512x1280x2560 (gate_up K) S8/R4 x4 rounds and 512x2560x640 (down K) S8/R4 x4 rounds: every run's
# GPU-read C byte-equal to the eager oracle, A arena == GPU-copied A, CPU-direct on rounds 0 and last. A CPU rescue
# is a FAIL. V9 health check last unless a wedge sign appeared (timeout, accel0 missing, SMU / amdxdna error in the
# kernel log): then stop and report.
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT_DIR=${NPU_PROBE_OUT:-$ROOT/npu-probe-out/$(date -u +%Y%m%dT%H%M%SZ)-$$}
COOP=$ROOT/target/release/npu-coop
GEMM=$ROOT/target/release/npu-gemm
for b in "$COOP" "$GEMM"; do [ -x "$b" ] || { echo "PREFLIGHT FAIL: $b missing or not executable"; exit 2; }; done
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
  timeout --foreground --signal=TERM --kill-after=2 "$t" "$@"; local rc=$?
  echo "step rc=$rc: $*"; [ "$rc" = 0 ] || FAILED=$((FAILED + 1)); { [ "$rc" = 124 ] || [ "$rc" = 137 ]; } && TIMEDOUT=1
  return "$rc"
}
run 15 "$COOP" --mode alias --timeout-ms 3000 || ok=0
[ $ok = 1 ] && { run 20 "$COOP" --mode empty --slots 8 --nslots 4 --rounds 8 --timeout-ms 3000 || ok=0; }
[ $ok = 1 ] && { run 20 "$COOP" --mode gemm --shape 512 1280 2560 --slots 8 --nslots 4 --rounds 4 --timeout-ms 3000 || ok=0; }
[ $ok = 1 ] && { run 20 "$COOP" --mode gemm --shape 512 2560 640 --slots 8 --nslots 4 --rounds 4 --timeout-ms 3000 || ok=0; }
if [ "$TIMEDOUT" = 1 ] || [ ! -e /dev/accel/accel0 ] || sudo -n journalctl -k --since "@$START0" --no-pager 2>/dev/null | grep -qiE 'smu|amdxdna.*(fail|error|timeout)'; then
  mark "WEDGE SIGN (timeout=$TIMEDOUT accel0=$([ -e /dev/accel/accel0 ] && echo present || echo MISSING)): health check NOT run, stop and report"
  FAILED=$((FAILED + 1))
else
  START=$(( $(date +%s) - 70 ))
  run 6 "$GEMM" 4096 1280 2560 --variant V9 --loop 1 --timeout-ms 3000
fi
mark "end failed_steps=$FAILED ok=$ok"
exit $(( FAILED > 100 ? 100 : FAILED ))
