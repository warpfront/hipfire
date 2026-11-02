#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
# npu-fclk-smoke.sh — one exclusive hold, the fabric-clock guard (crates/railgun/src/npu/fclk.rs) on silicon:
#   ~/pm-wave/hipx-lock.sh timing -- timeout 88 tools/npu/npu-fclk-smoke.sh
# 1. iGPU starts on `auto` (`npu-tools fclk status` exit 1);
# 2. `npu-coop --mode alias`, default guard (`require`), unpinned → refused before any GPU/NPU work (exit 1);
# 3. `NPU_FCLK_GUARD=pin npu-coop --mode alias` → the guard pins, alias PASS, the guard restores `auto` on exit;
# 4. `tools/npu/npu-window.sh npu-coop --mode alias` → the window pins, the guard sees the pin, alias PASS, window restores;
# 5. perf level back to `auto`.
# The EXIT trap writes `auto` back if any step left the pin behind (e.g. a step killed by its timeout).
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
COOP=$ROOT/target/release/npu-coop
TOOLS=$ROOT/target/release/npu-tools
PERF=/sys/bus/pci/devices/0000:bf:00.0/power_dpm_force_performance_level
for b in "$COOP" "$TOOLS"; do [ -x "$b" ] || { echo "PREFLIGHT FAIL: $b missing or not executable"; exit 2; }; done
sudo -n prlimit --pid $$ --memlock=unlimited:unlimited
FAILED=0
mark() { echo "mark $(date +%s%N | cut -c1-19) $*"; }
cleanup() {
  [ "$(cat "$PERF")" = auto ] || { echo "cleanup: perf $(cat "$PERF"), restoring"; "$TOOLS" fclk restore; }
  mark "end failed_steps=$FAILED perf=$(cat "$PERF")"
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP
# expect RC SECONDS PATTERN CMD...: CMD must exit RC and print a line matching PATTERN
expect() {
  local want=$1 t=$2 pat=$3; shift 3
  mark "$*"
  local out rc
  out=$(timeout --signal=TERM --kill-after=2 "$t" "$@" 2>&1); rc=$?
  printf '%s\n' "$out"
  if [ "$rc" = "$want" ] && grep -Eq -- "$pat" <<< "$out"; then echo "step OK rc=$rc"; else echo "step FAIL rc=$rc (want $want, /$pat/)"; FAILED=$((FAILED + 1)); fi
}
expect 1 5 'perf=auto .*pinned=false' "$TOOLS" fclk status
expect 1 15 'refusing concurrent GPU\+NPU work \(npu-coop\)' "$COOP" --mode alias --timeout-ms 3000
expect 1 5 'perf=auto .*pinned=false' "$TOOLS" fclk status
expect 0 20 'restored .*perf=auto' env NPU_FCLK_GUARD=pin "$COOP" --mode alias --timeout-ms 3000
expect 1 5 'perf=auto .*pinned=false' "$TOOLS" fclk status
expect 0 30 'fclk: npu-coop: .* pinned \(perf=manual' env NPU_WINDOW_LOG="$HOME/qcal/npu-logs/npu-window.log" NPU_WINDOW_TIMEOUT=20 \
  "$ROOT/tools/npu-window.sh" "$COOP" --mode alias --timeout-ms 3000
expect 1 5 'perf=auto .*pinned=false' "$TOOLS" fclk status
exit $(( FAILED > 100 ? 100 : FAILED ))
