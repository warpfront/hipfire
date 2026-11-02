#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
# npu-railgun.sh — one npu-window.sh hold: railgun eager / prepared / chain overhead bench + certification (V9).
#   ~/pm-wave/hipx-lock.sh timing -- env NPU_WINDOW_TIMEOUT=90 tools/npu/npu-window.sh tools/npu/npu-railgun.sh
# Order (priority first): --noop 512 512 64 with G=1/8/32 x eager/prepared/chain (--steps 500); then 512x512x64 with
# G=1/8/32 x eager/prepared/chain (--steps 200), the --neg stale-patch run
# (G=2 prepared), then V9 gate_up 4096 1280 2560 and down 4096 2560 640 with G=1/8 x all modes (--secs 2; G=8 eager --secs 1).
# Every invocation is bounded by `timeout` (capped by what is left of the 80 s script budget, so the whole script
# stays under 80 s); a step that cannot start inside the budget is SKIPPED and counted as failed.
# Exit status = failed steps (capped at 100). Each npu-railgun run ends with RESULT: PASS|FAIL and exits 0 only on PASS.
# NPU_RAILGUN_PHASE=lean: the --lean persistent-submit window (see the phase block below).
# NPU_RAILGUN_PHASE=lean-verify: --lean --verify-every / --tape certification (512x512x128 chain, gate_up/down prepared
#   with --tape, down chain --neg stale-key); stops on the first failed step like lean.
# NPU_RAILGUN_PHASE=prefix: ArrayDesign V9 / V10 / V10-prefix acceptance on G=8 prepared (--variant): down 4096x2560x640
#   (--secs 1.5) and 160x2560x640 (--steps 200). ALL six full runs first (V9, V10, V10-prefix x big, small), then ALL six
#   --lean --verify-every runs (same order); never interleaved. Stops on the first failed/timed-out step; if the shared
#   80 s budget cannot start a step the phase exits 125 immediately (never reports success); every timeout is capped at
#   budget-left minus 1 s so the kill grace keeps the whole phase <= 80 s.
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT_DIR=${NPU_PROBE_OUT:-$ROOT/npu-probe-out/$(date -u +%Y%m%dT%H%M%SZ)-$$}
BIN=$ROOT/target/release/npu-railgun
[ -x "$BIN" ] || { echo "PREFLIGHT FAIL: $BIN missing or not executable"; exit 2; }
mkdir -p "$OUT_DIR" && cd "$OUT_DIR" || { echo "cannot use $OUT_DIR"; exit 2; }
START=$(date +%s)
BUDGET=80
FAILED=0
mark() { echo "mark $(date +%s%N | cut -c1-19) $*"; }
run() {  # run SECONDS NPU-RAILGUN-ARGS... ; returns the status (124 = timeout, 125 = skipped)
  local t=$1; shift
  local left=$(( BUDGET - ($(date +%s) - START) ))
  if [ "$left" -lt 3 ]; then
    echo "SKIP (time budget): $*"; FAILED=$((FAILED + 1))
    if [ "${NPU_RAILGUN_PHASE:-all}" = prefix ]; then mark "STOP prefix: time budget exhausted"; exit 125; fi
    return 125
  fi
  # prefix reserves the 1 s --kill-after grace so a timed-out step cannot overrun BUDGET.
  [ "${NPU_RAILGUN_PHASE:-all}" = prefix ] && left=$(( left - 1 ))
  [ "$t" -gt "$left" ] && t=$left
  mark "npu-railgun $*"
  timeout --foreground --signal=TERM --kill-after=1 "$t" "$BIN" "$@"; local rc=$?
  echo "step rc=$rc: $*"; [ "$rc" = 0 ] || FAILED=$((FAILED + 1))
  # A failed lean/lean-verify run may leave the array mid-GEMM: stop the window rather than submit more work.
  case "${NPU_RAILGUN_PHASE:-all}" in
    lean|lean-verify|prefix) if [ "$rc" != 0 ]; then mark "STOP after failed ${NPU_RAILGUN_PHASE} step"; exit "$rc"; fi ;;
  esac
  return "$rc"
}
MODES="eager prepared chain"
# NPU_RAILGUN_PHASE: small (noop + 512x512x64 + neg), big (gate_up/down), lean (--lean persistent-submit window: see
# above), lean-verify (--lean --verify-every / --tape / --neg stale-key certification, <= 80 s, stops on first
# failure), all (default; small + big, may hit the budget).
PHASE=${NPU_RAILGUN_PHASE:-all}
if [ "$PHASE" = lean ]; then
  # Lean TXN window (<= 80 s): small 512x512x128 G=1/8 x eager/prepared/chain --lean (--steps 200), then V9 gate_up
  # 4096x1280x2560 and down 4096x2560x640 G=1/8 x prepared+chain --lean (--secs 2) plus ONE in-window full-TXN baseline
  # (G=8 chain, no --lean) per big shape. Each --lean run does its own untimed full-TXN warmup; txn_bytes= shows the size.
  for g in 1 8; do
    for mode in $MODES; do
      run 5 512 512 128 --gemms "$g" --mode "$mode" --lean --steps 200 --timeout-ms 2000
    done
  done
  for shape in "4096 1280 2560" "4096 2560 640"; do
    for g in 1 8; do
      for mode in prepared chain; do
        # shellcheck disable=SC2086
        run 9 $shape --gemms "$g" --mode "$mode" --lean --secs 2 --timeout-ms 3000
      done
    done
    # shellcheck disable=SC2086
    run 9 $shape --gemms 8 --mode chain --secs 2 --timeout-ms 3000
  done
  mark "end failed_steps=$FAILED"
  exit $(( FAILED > 100 ? 100 : FAILED ))
fi
if [ "$PHASE" = lean-verify ]; then
  # Lean verify window (<= 80 s): (1) 512x512x128 G=8 chain --lean --verify-every (GEMM, --steps 100); (2) V9 gate_up
  # 4096x1280x2560 and down 4096x2560x640 G=8 prepared --lean --verify-every --tape (--secs 2); (3) down G=8 chain
  # --lean --tape --neg stale-key (--steps 4; --neg stale-key requires --tape). Every step is bounded by run()'s
  # budget-capped timeout and any failure stops the window (see run()).
  run 9 512 512 128 --gemms 8 --mode chain --lean --verify-every --steps 100 --timeout-ms 2000
  for shape in "4096 1280 2560" "4096 2560 640"; do
    # shellcheck disable=SC2086
    run 12 $shape --gemms 8 --mode prepared --lean --verify-every --secs 2 --tape --timeout-ms 3000
  done
  run 9 4096 2560 640 --gemms 8 --mode chain --lean --tape --neg stale-key --steps 4 --timeout-ms 3000
  mark "end failed_steps=$FAILED"
  exit $(( FAILED > 100 ? 100 : FAILED ))
fi
if [ "$PHASE" = prefix ]; then
  # Acceptance window (<= 80 s): variants x {down 4096x2560x640 --secs 1.5, 160x2560x640 --steps 200}, G=8 prepared.
  # Full runs for all variants first, lean --verify-every runs last. run() stops/exits on the first failure or skip.
  for lean in "" "--lean --verify-every"; do
    for variant in V9 V10 V10-prefix; do
      # shellcheck disable=SC2086
      run 8 4096 2560 640 --variant "$variant" --gemms 8 --mode prepared $lean --secs 1.5 --timeout-ms 3000
      # shellcheck disable=SC2086
      run 5 160 2560 640 --variant "$variant" --gemms 8 --mode prepared $lean --steps 200 --timeout-ms 2000
    done
  done
  mark "end failed_steps=$FAILED"
  exit $(( FAILED > 100 ? 100 : FAILED ))
fi
if [ "$PHASE" != big ]; then
# --noop first: minimal 1-write TXN (no GEMM, no CPU reference), pure submission overhead per mode.
for g in 1 8 32; do
  for mode in $MODES; do
    run 5 512 512 64 --noop --gemms "$g" --mode "$mode" --steps 500 --timeout-ms 2000
  done
done
for g in 1 8 32; do
  for mode in $MODES; do
    run 5 512 512 64 --gemms "$g" --mode "$mode" --steps 200 --timeout-ms 2000
  done
done
run 5 512 512 64 --gemms 2 --mode prepared --steps 8 --neg stale-patch --timeout-ms 2000
fi
GU="4096 1280 2560"; DN="4096 2560 640"
[ "$PHASE" = small ] && { mark "end failed_steps=$FAILED"; exit $(( FAILED > 100 ? 100 : FAILED )); }
for shape in "$GU" "$DN"; do
  for g in 1 8; do
    for mode in $MODES; do
      secs=2
      # Budget: G=8 eager is the slowest step (8 serial submit+wait), so it runs --secs 1 instead of being skipped.
      [ "$g" = 8 ] && [ "$mode" = eager ] && secs=1
      # shellcheck disable=SC2086
      run 9 $shape --gemms "$g" --mode "$mode" --secs "$secs" --timeout-ms 3000
    done
  done
done
mark "end failed_steps=$FAILED"
exit $(( FAILED > 100 ? 100 : FAILED ))
