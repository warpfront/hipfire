#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
# npu-batch.sh — the NPU work for one npu-window.sh hold: hello (3 fresh contexts), then the
# single-core int8 GEMM at 64^3, 128^3, 256^3 (M2), then any extra commands given as arguments
# (each a quoted command line, e.g. "./target/release/npu-gemm 512 512 64 --array").
# NPU_BATCH_BASIC=0 skips hello + single-core. With NPU_METRICS_OUT set, npu-tools soc-metrics samples
# the SMU metrics table (DRAM / NPU bandwidth, NPU clocks; read-only) every 20 ms during the batch.
# Every step runs (one window is expensive); the script exits non-zero if any step failed, and
# refuses to start (exit 2) unless both binaries are present and executable.
# Run from the repo root of a clean worktree with target/release binaries present.
cd "$(dirname "$0")/.."
for bin in target/release/npu-hello target/release/npu-gemm; do
  [ -x "$bin" ] || { echo "PREFLIGHT FAIL: $bin missing or not executable"; exit 2; }
done
if [ -n "${NPU_METRICS_OUT:-}" ]; then
  [ -x target/release/npu-tools ] || { echo "PREFLIGHT FAIL: npu-tools missing or not executable"; exit 2; }
  ./target/release/npu-tools soc-metrics "$NPU_METRICS_OUT" 0.02 & SAMPLER=$!
  trap 'kill $SAMPLER 2>/dev/null' EXIT
fi
FAILED=0
mark() { echo "mark $(date +%s%N | cut -c1-19) $*"; }
step() {  # step LABEL CMD...
  local label=$1; shift
  mark "$label"; "$@"; local rc=$?
  echo "$label rc=$rc"; [ "$rc" = 0 ] || FAILED=$((FAILED + 1))
}
if [ "${NPU_BATCH_BASIC:-1}" = 1 ]; then
  step hello ./target/release/npu-hello run --repeat 3
  for s in 64 128 256; do step "gemm $s" ./target/release/npu-gemm $s $s $s --iters 3; done
fi
for extra in "$@"; do step "extra: $extra" bash -c "$extra"; done
mark "end failed_steps=$FAILED"
[ "$FAILED" = 0 ]
