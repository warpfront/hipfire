#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# npu-coop-bench.sh — GPU<->NPU interop benchmark (halo, gfx1151 + XDNA2): the CPU-free GPU -> NPU -> GPU persistent ring (npu-coop).
# Primary: gemm 512x1280x2560 per-run wall p50 (GPU A copy + publish + NPU GEMM + done + GPU C read), exactness-checked.
# fclk is pinned once for the whole run (npu-tools fclk pin, needs `sudo -n tee` on the sysfs files) and restored on exit.
# Set NPU_IGPU_PCI when the iGPU is not at the hipx default 0000:bf:00.0 (e.g. 0000:c5:00.0); HIP_LIB_DIR overrides
# the libamdhip64 directory.
set -euo pipefail
cd "$(dirname "$0")/../.."
export LD_LIBRARY_PATH=${HIP_LIB_DIR:-$HOME/.hipfire/rocm-merged/lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export NPU_FCLK_GUARD=require

cargo build -q --release -p npu-tools --features lab --bin npu-coop --bin npu-tools
COOP=${CARGO_TARGET_DIR:-target}/release/npu-coop
TOOLS=${CARGO_TARGET_DIR:-target}/release/npu-tools

$TOOLS fclk pin >&2
trap '$TOOLS fclk restore >&2 || true' EXIT

log=$(mktemp)
coop() { timeout --foreground -k 2 30 "$COOP" "$@" --timeout-ms 3000 >"$log" 2>&1 || { cat "$log" >&2; echo "npu-coop $* FAILED" >&2; exit 1; }
         grep -q '^RESULT: PASS' "$log" || { cat "$log" >&2; exit 1; }; }
p50() { grep -m1 "$1" "$log" | sed -E 's/.*'"$2"' p50=([0-9.]+).*/\1/'; }

# median of the per-process p50s over 3 fresh processes
med() { printf '%s\n' "$@" | sort -g | sed -n 2p; }
g=(); d=(); e=(); rt=(); tops=()
for i in 1 2 3; do
  coop --mode gemm --shape 512 1280 2560 --slots 8 --nslots 4 --rounds 16
  g+=("$(p50 '^pipeline:' per_run_wall_us)"); rt+=("$(p50 'pub_to_done_gpu_us' pub_to_done_gpu_us)")
  tops+=("$(grep -m1 '^pipeline:' "$log" | sed -E 's/.*useful_tops_mean=([0-9.]+).*/\1/')")
  coop --mode gemm --shape 512 2560 640 --slots 8 --nslots 4 --rounds 16
  d+=("$(p50 '^pipeline:' per_run_wall_us)")
  coop --mode empty --slots 8 --nslots 4 --rounds 16
  e+=("$(p50 'pub_to_done_gpu_us' pub_to_done_gpu_us)")
done
rm -f "$log"
echo "METRIC gate_up_run_us=$(med "${g[@]}")"
echo "METRIC down_run_us=$(med "${d[@]}")"
echo "METRIC gate_up_pub_to_done_us=$(med "${rt[@]}")"
echo "METRIC gate_up_tops=$(med "${tops[@]}")"
echo "METRIC empty_ring_rtt_us=$(med "${e[@]}")"
