#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
# npu-ring-probe.sh MODE — first silicon touches of the persistent ring (declared hazardous to Main): exactly ONE empty
# (poll + done only) persistent run per window. MODE = empty-prepublish (1 round) | empty-paced (20 rounds).
#   ~/pm-wave/hipx-lock.sh timing -- env NPU_WINDOW_TIMEOUT=90 tools/npu/npu-window.sh tools/npu/npu-ring-probe.sh empty-prepublish
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BIN=$ROOT/target/release/npu-ring
[ -x "$BIN" ] || { echo "PREFLIGHT FAIL: $BIN missing or not executable"; exit 2; }
case "${1:-}" in
  empty-prepublish) ROUNDS=1 ;;
  empty-paced) ROUNDS=20 ;;
  *) echo "usage: $0 empty-prepublish|empty-paced"; exit 2 ;;
esac
echo "mark $(date +%s%N | cut -c1-19) npu-ring $1 rounds $ROUNDS"
timeout --foreground --signal=TERM --kill-after=1 15 "$BIN" 512 512 64 --slots 8 --nslots 8 --mode "$1" --rounds "$ROUNDS" --timeout-ms 3000
rc=$?; echo "step rc=$rc: $1"
exit "$rc"
