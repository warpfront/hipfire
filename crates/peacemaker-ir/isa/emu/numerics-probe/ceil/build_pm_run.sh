#!/bin/bash
# build_pm_run.sh <gfx1201|gfx1100|gfx1151> [OUTDIR]: CPU-only build of ceil_pm_run.hip (no GPU touched).
set -euo pipefail
ARCH=${1:?arch}
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=${2:-$PWD/ceil_pm_build_$ARCH}
HIPCC=${HIPCC:-/opt/rocm/bin/hipcc}
mkdir -p "$OUT"; cd "$OUT"
export HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1
nice -n19 $HIPCC -O3 -std=c++17 --offload-arch=$ARCH -fuse-cuid=none -ffp-contract=off "$SRC/ceil_pm_run.hip" -o "ceil_pm_run_$ARCH"
sha256sum "ceil_pm_run_$ARCH"
