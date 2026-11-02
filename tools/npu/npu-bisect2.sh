#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
# npu-bisect2.sh — the SECOND M3 hardware bisect as ONE npu-batch.sh run (no nested locks). Eight commands, one fresh
# context each (--iters 1), 3 s wait timeout, K=64, AIE status decoded + saved after EVERY wait (--dump-status):
#   1. gemm_i8 single core  64x64x64            (control: proven design; native invocation, no --array)
#   2. V2  shape 128x64x64  --core LockOnly     (lock/DMA-only core: C words are the 0xC0DE0000 pattern)
#   3. V2  shape 128x64x64  --core Serial
#   4. V2  shape 128x64x64  --core Fast
#   5. V5  shape 512x512x64 --core LockOnly
#   6. V5  shape 512x512x64 --core Serial
#   7. V1  shape 128x64x64  --core LockOnly     (lane-rule route fix: South0->North0 / South1->North1)
#   8. V1  shape 128x64x64  --core Fast
# Run it exactly like npu-batch.sh / npu-bisect.sh, i.e. as the command of one npu-window.sh hold under the
# exclusive Halo timing lock (the lock/clock-pin/kernel-log protocol is npu-window.sh's; this script does not
# touch it and never nests a window):
#   ~/pm-wave/hipx-lock.sh timing -- tools/npu/npu-window.sh tools/npu/npu-bisect2.sh
# Needs target/release/npu-gemm and npu-hello (npu-batch.sh preflight, exit 2 if missing).
#
# Time bound (every command hanging until its wait timeout, all still attempted): each command is wrapped in
# `timeout --foreground` 5 s + 1 s kill-after (8 x 6 = 48 s), and the whole batch in `timeout` 55 s + 1 s
# kill-after = 56 s < 60 s. The batch exits non-zero if any command failed or the total timeout fired
# (rc 124/137). Both bounds are fixed on purpose. Keep NPU_WINDOW_TIMEOUT (default 120 s) above 60 s.
#
# Each command runs inside the run directory (default <repo>/npu-bisect2-out/<UTC stamp>-<pid>, unique per run;
# an explicit NPU_BISECT2_OUT is used as is) and npu-gemm writes the raw firmware AIE column status of every wait
# to aie-status-<tag>-0.bin there: single, V2-LockOnly, V2-Serial, V2, V5-LockOnly, V5-Serial, V1-LockOnly, V1 (the
# tag carries the core when it is not Fast, so the files do not collide). The decoded status is printed in the batch log. An
# invalid/failed status query is labelled INVALID in that output, never silently replaced. This script never
# deletes anything.
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT_DIR=${NPU_BISECT2_OUT:-$ROOT/npu-bisect2-out/$(date -u +%Y%m%dT%H%M%SZ)-$$}
CMD_T=5
TOTAL_T=55
WAIT_MS=3000
mkdir -p "$OUT_DIR" || { echo "cannot create $OUT_DIR"; exit 2; }
# npu-batch.sh cds to the repo root, so a relative explicit NPU_BISECT2_OUT must be made absolute first.
OUT_DIR=$(cd "$OUT_DIR" && pwd) || { echo "cannot resolve $OUT_DIR"; exit 2; }
echo "npu-bisect2: diagnostic bins in $OUT_DIR"

cmds=()
add() {  # add ARG... : npu-gemm arguments, run from OUT_DIR with the fixed per-command bounds
  local line
  line=$(printf 'cd %q && timeout --foreground --signal=TERM --kill-after=1 %q %q' "$OUT_DIR" "$CMD_T" "$ROOT/target/release/npu-gemm")
  local a
  for a in "$@"; do line+=$(printf ' %q' "$a"); done
  line+=$(printf ' --iters 1 --timeout-ms %q --dump-status' "$WAIT_MS")
  cmds+=("$line")
}
add 64 64 64
add 128 64 64 --variant V2 --core LockOnly
add 128 64 64 --variant V2 --core Serial
add 128 64 64 --variant V2 --core Fast
add 512 512 64 --variant V5 --core LockOnly
add 512 512 64 --variant V5 --core Serial
add 128 64 64 --variant V1 --core LockOnly
add 128 64 64 --variant V1 --core Fast

export NPU_BATCH_BASIC=0
exec timeout --signal=TERM --kill-after=1 "$TOTAL_T" "$ROOT/tools/npu-batch.sh" "${cmds[@]}"
