#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
# npu-mem-path.sh MODE MEMORY [BURST CACHE QOS | DIRECTION]
# Run only inside tools/npu/npu-window.sh under the exclusive hipx timing lock.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
MODE=${1:?mode: sweep|axi|concurrency|v9|v9-all|v9-full|gpu|metrics}
MEMORY=${2:-native}
AVAIL=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
[ "$AVAIL" -ge 16777216 ] || { echo "REFUSE MemAvailable ${AVAIL} KiB < 16 GiB"; exit 2; }
for BIN in target/release/npu-bw target/release/npu-gemm; do
  chmod +x "$BIN"
  test -x "$BIN"
done
printf 'experiment mode=%s memory=%s utc=%s MemAvailable_KiB=%s\n' "$MODE" "$MEMORY" "$(date -u +%FT%TZ)" "$AVAIL"
case "$MODE" in
  sweep) ./target/release/npu-bw --sweep --memory "$MEMORY" --iters 6 --timeout-ms 3000 ;;
  axi) ./target/release/npu-bw --sweep-axi --memory "$MEMORY" --iters 6 --timeout-ms 3000 ;;
  concurrency)
    DIRECTION=${3:?concurrency direction: read|write|mixed}
    ./target/release/npu-bw --sweep-concurrency --direction "$DIRECTION" --memory "$MEMORY" --iters 4 --timeout-ms 3000
    ;;
  v9)
    AXI=(--burst "${3:-3}" --axcache "${4:-2}" --axqos "${5:-0}")
    for SHAPE in '4096 1280 2560' '4096 2560 640'; do
      read -r M N K <<< "$SHAPE"
      ./target/release/npu-gemm "$M" "$N" "$K" --variant V9 --memory "$MEMORY" "${AXI[@]}" --ctl-probe nocompute --loop 1 --timeout-ms 3000
    done
    ./target/release/npu-gemm 512 512 128 --variant V9 --memory "$MEMORY" "${AXI[@]}" --reuse-ctx --iters 3 --timeout-ms 3000
    ;;
  v9-all)
    for TUPLE in '3 2 0' '0 2 0' '1 2 0' '2 2 0'; do
      read -r BURST CACHE QOS <<< "$TUPLE"
      "$0" v9 "$MEMORY" "$BURST" "$CACHE" "$QOS"
    done
    ;;
  v9-full)
    AXI=(--burst "${3:-3}" --axcache "${4:-2}" --axqos "${5:-0}")
    for SHAPE in '4096 1280 2560' '4096 2560 640'; do
      read -r M N K <<< "$SHAPE"
      ./target/release/npu-gemm "$M" "$N" "$K" --variant V9 --memory "$MEMORY" "${AXI[@]}" --loop 1 --timeout-ms 3000
    done
    ;;
  metrics)
    chmod +x target/release/npu-tools
    test -x target/release/npu-tools
    ./target/release/npu-tools soc-metrics --once
    ;;
  gpu)
    chmod +x target/release/npu-tools
    test -x target/release/npu-tools
    for OP in read write; do
      HIP_VISIBLE_DEVICES=1 ./target/release/npu-tools gpu-bw --buf "$MEMORY" --op "$OP" --bytes 1073741824 --iters 20
    done
    ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac
