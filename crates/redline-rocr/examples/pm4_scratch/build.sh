#!/usr/bin/env bash
# Build the probe code objects with the product hipcc flags
# (crates/rdna-compute/src/compiler.rs `core_hipcc_flags`).
#   build.sh <arch> <outdir>   e.g. build.sh gfx1100 /path/out
set -euo pipefail
arch=$1
out=$2
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../../.." && pwd)
flags=(--genco "--offload-arch=$arch" -O3 --no-offload-compress -fuse-cuid=none -mcode-object-version=6)
mkdir -p "$out"
hipcc "${flags[@]}" "$here/probe.hip" -o "$out/probe_$arch.co"
# Wave64 twin for gfx10 (gfx10+ can run either wave size).
case "$arch" in gfx10*) hipcc "${flags[@]}" -mwavefrontsize64 "$here/probe.hip" -o "$out/probe_w64_$arch.co";; esac
# Product residual kernel exactly as rdna-compute concatenates it
# (kernels.rs GEMV_MQ4G256V2_RESIDUAL_SRC).
{
  printf '#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n'
  cat "$root/kernels/src/gfx12_weight_cache_policy.inc"
  cat "$root/kernels/src/gemv_mq4g256v2_residual.hip"
} > "$out/mq4_residual_$arch.hip"
hipcc "${flags[@]}" "$out/mq4_residual_$arch.hip" -o "$out/mq4_residual_$arch.co"
