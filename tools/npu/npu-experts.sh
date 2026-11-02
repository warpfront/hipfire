#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
# npu-experts.sh — one npu-window.sh hold: grouped V9 experts (npu-experts) shape matrix.
#   NPU_EXPERTS_PHASE=gate_up ~/pm-wave/hipx-lock.sh timing -- env NPU_WINDOW_TIMEOUT=90 tools/npu/npu-window.sh tools/npu/npu-experts.sh
#   (then the same with NPU_EXPERTS_PHASE=down in a second hold)
# Matrix: proj gate_up (N1280 K2560) / down (N2560 K640) x M 64/128/160/256/512 x E 8/32/64 = 30 runs, ONE npu-experts run
# each. A run is a whole process: operand generation + packing for E experts, E full-TXN eager baseline submits, the
# exact CPU reference (sample of >= 4 experts, all experts when ~2 s allow) and the timed rounds. None of that was
# measured when this script was written; to keep the setup cost of each hold bounded the matrix is split CONSERVATIVELY
# by projection: NPU_EXPERTS_PHASE is REQUIRED (gate_up | down), 15 matrix runs per hold; the two phases together
# preserve all 30 combinations. No other phase exists.
# Order per phase: (0) three `npu-experts --insts-only` calls (E 8/32/64 at m160 of the phase's projection; every M shares
# the same size, no device opened); (1) the required first silicon sanity, ALWAYS gate_up m160 E8 --rounds 3
# --verify-every (in both phases, even the down one: every round of every expert compared); (2) the phase's matrix in
# M-major, E-minor order, each with --rounds chosen for about 0.3 s of ROUND time. The round count is an ESTIMATE from
# useful ops at NPU_EXPERTS_EST_TOPS (positive integer, default 10 TOPS, clamped to 2..400 rounds); no hardware latency was
# measured when this was written, the real time per round is in the printed report.
# The whole script (preflight calls included) stays inside BUDGET=78 s, leaving 2 s for the TERM + kill-after-1 grace of
# `timeout` under the 80 s hold. Every call is bounded by `timeout` (capped by what is left of the budget and by
# NPU_EXPERTS_RUN_TIMEOUT, a positive integer, default 30 s). The FIRST failing, timed-out or skipped (time budget) step
# stops the window: the array may be left mid-GEMM, so nothing more is submitted.
# NPU_EXPERTS_RING=1 adds --ring to every npu-experts call. Exit status: the failing step's status, 2 on usage/preflight.
set -u
PHASE=${NPU_EXPERTS_PHASE:-}
case "$PHASE" in
  gate_up|down) ;;
  *) echo "NPU_EXPERTS_PHASE must be gate_up or down (the matrix is split by projection; run both phases)"; exit 2 ;;
esac
RUN_CAP=${NPU_EXPERTS_RUN_TIMEOUT:-30}
EST_TOPS=${NPU_EXPERTS_EST_TOPS:-10}
posint() { case "$2" in ''|*[!0-9]*|0|0[0-9]*) echo "$1 must be a positive integer, got '$2'"; exit 2 ;; esac; }
posint NPU_EXPERTS_RUN_TIMEOUT "$RUN_CAP"
posint NPU_EXPERTS_EST_TOPS "$EST_TOPS"
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT_DIR=${NPU_PROBE_OUT:-$ROOT/npu-probe-out/$(date -u +%Y%m%dT%H%M%SZ)-$$}
BIN=$ROOT/target/release/npu-experts
[ -x "$BIN" ] || { echo "PREFLIGHT FAIL: $BIN missing or not executable"; exit 2; }
mkdir -p "$OUT_DIR" && cd "$OUT_DIR" || { echo "cannot use $OUT_DIR"; exit 2; }
START=$(date +%s)
BUDGET=78
RING=()
[ "${NPU_EXPERTS_RING:-0}" = 1 ] && RING=(--ring)
mark() { echo "mark $(date +%s%N | cut -c1-19) $*"; }
if [ "$PHASE" = gate_up ]; then N=1280; K=2560; else N=2560; K=640; fi
MS="64 128 160 256 512"
ES="8 32 64"

run() {  # run SECONDS NPU-EXPERTS-ARGS... ; stops the window on the first non-zero status (124 = timeout, 125 = skipped)
  local t=$1; shift
  local left=$(( BUDGET - ($(date +%s) - START) ))
  if [ "$left" -lt 3 ]; then
    echo "SKIP (time budget): $*"; mark "STOP after skipped step"; exit 125
  fi
  [ "$t" -gt "$left" ] && t=$left
  mark "npu-experts $*"
  timeout --foreground --signal=TERM --kill-after=1 "$t" "$BIN" "$@"; local rc=$?
  echo "step rc=$rc: $*"
  if [ "$rc" != 0 ]; then mark "STOP after failed step"; exit "$rc"; fi
}

# (0) Instruction stream sizes (no device).
for e in $ES; do
  run 10 --proj "$PHASE" --m 160 --experts "$e" "${RING[@]}" --insts-only
done
# (1) Required first silicon sanity, always gate_up m160 E8, every round verified.
run "$RUN_CAP" --proj gate_up --m 160 --experts 8 "${RING[@]}" --rounds 3 --verify-every --timeout-ms 3000
# (2) The phase's matrix, rounds ~ 0.3 s at EST_TOPS. E is the outer loop so the largest TXN (E=64) runs last.
for e in $ES; do
  for m in $MS; do
    ops=$(( 2 * m * N * K * e ))
    us=$(( ops / (EST_TOPS * 1000000) )); [ "$us" -lt 1 ] && us=1
    rounds=$(( 300000 / us ))
    [ "$rounds" -lt 2 ] && rounds=2
    [ "$rounds" -gt 400 ] && rounds=400
    run "$RUN_CAP" --proj "$PHASE" --m "$m" --experts "$e" "${RING[@]}" --rounds "$rounds" --timeout-ms 3000
  done
done
mark "end phase=$PHASE"
exit 0
