#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
# npu-bisect.sh — the M3 ARRAY hardware-failure bisect as ONE npu-batch.sh run (no nested locks): the
# array variants V1..V6 and V6b, one fresh context each (--iters 1), 3 s wait timeout, K=64:
#   V1,V2: 128x64x64   V3: 512x64x64   V4: 128x512x64   V5,V6,V6b: 512x512x64
# Run it exactly like npu-batch.sh, i.e. as the command of one npu-window.sh hold under the exclusive
# Halo timing lock (the lock/clock-pin/kernel-log protocol is npu-window.sh's; this script does not
# touch it):
#   ~/pm-wave/hipx-lock.sh timing -- tools/npu/npu-window.sh tools/npu/npu-bisect.sh
# Needs target/release/npu-gemm and npu-hello (npu-batch.sh preflight, exit 2 if missing).
#
# Time bound (all 7 variants hanging, every one still attempted): each variant is wrapped in
# `timeout --foreground` 6 s + 1 s kill-after (7 x 7 = 49 s), and the whole batch in `timeout` 55 s + 1 s
# kill-after = 56 s < 60 s. The batch exits non-zero if any variant failed or the total timeout fired
# (rc 124/137). Both bounds are fixed on purpose. Keep NPU_WINDOW_TIMEOUT (default 120 s) above 60 s.
#
# On a non-completed wait npu-gemm writes the raw firmware AIE column status (or, if the query failed or
# returned an empty bitmap, the attempted API buffer, labelled INVALID in its output) to
# aie-status-<variant>-<iter>.bin in the run directory: default <repo>/npu-bisect-out/<UTC stamp>-<pid>
# (unique per run). An explicit NPU_BISECT_OUT directory is used as is and may have same-named files
# overwritten by the variant that produced them; this script never deletes anything.
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT_DIR=${NPU_BISECT_OUT:-$ROOT/npu-bisect-out/$(date -u +%Y%m%dT%H%M%SZ)-$$}
CMD_T=6
TOTAL_T=55
WAIT_MS=3000
mkdir -p "$OUT_DIR" || { echo "cannot create $OUT_DIR"; exit 2; }
echo "npu-bisect: diagnostic bins in $OUT_DIR"

cmds=()
add() {  # add VARIANT M N K
  local line
  line=$(printf 'cd %q && timeout --foreground --signal=TERM --kill-after=1 %q %q %q %q %q --variant %q --iters 1 --timeout-ms %q' \
    "$OUT_DIR" "$CMD_T" "$ROOT/target/release/npu-gemm" "$2" "$3" "$4" "$1" "$WAIT_MS")
  cmds+=("$line")
}
add V1 128 64 64
add V2 128 64 64
add V3 512 64 64
add V4 128 512 64
add V5 512 512 64
add V6 512 512 64
add V6b 512 512 64

export NPU_BATCH_BASIC=0
exec timeout --signal=TERM --kill-after=1 "$TOTAL_T" "$ROOT/tools/npu-batch.sh" "${cmds[@]}"
