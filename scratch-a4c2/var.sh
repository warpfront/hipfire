#!/bin/bash
# var.sh <quant_variant.hip> <outdir> [CONST...]
# Builds the production concatenation of each CONST from this worktree with
# kernels/src/block_i4_128_quant.hip replaced by <quant_variant.hip>, using
# the production gfx1201 argv plus -DIU4_A4_CANDIDATES=2.
set -e
here=$(cd "$(dirname "$0")" && pwd)
qv=$(realpath "$1"); out=$2; shift 2
consts=${*:-"FUSED_SILU_MUL_MQ_ROTATE_AWQ_I4_HIN_GFX12_SRC FUSED_RMSNORM_MQ_ROTATE_AWQ_I4_GFX12_V2_SRC GATED_NORM_MQ_ROTATE_AWQ_I4_GFX12_V2_SRC SIGMOID_MUL_MQ_ROTATE_X_AWQ_I4_GFX12_SRC"}
tmp=$(mktemp -d)
mkdir -p "$tmp/crates/rdna-compute/src" "$tmp/kernels"
cp "$here/../crates/rdna-compute/src/kernels.rs" "$tmp/crates/rdna-compute/src/"
cp -r "$here/../kernels/src" "$tmp/kernels/src"
cp "$qv" "$tmp/kernels/src/block_i4_128_quant.hip"
for c in $consts; do
  echo -n "$c: "
  "$here/build.sh" "$tmp" "$out" gfx1201 "$c" -DIU4_A4_CANDIDATES=2 | grep -oE 'VGPRs: [0-9]+|Occupancy \[waves/SIMD\]: [0-9]+|error.*' | tr '\n' ' '
  echo
done
rm -rf "$tmp"
