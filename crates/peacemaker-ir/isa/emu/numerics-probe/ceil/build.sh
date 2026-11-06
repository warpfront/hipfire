#!/bin/bash
# build.sh <gfx1201|gfx1100|gfx1151> [OUTDIR]: CPU-only build of ceil_sweep.hip (no GPU touched) with the product
# hipcc flags of rdna-compute `core_hipcc_flags` minus --genco (a host executable is needed), plus the disassembly.
set -euo pipefail
ARCH=${1:?arch}
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=${2:-$PWD/ceil_build_$ARCH}
HIPCC=${HIPCC:-/opt/rocm/bin/hipcc}
mkdir -p "$OUT"; cd "$OUT"
export HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1
nice -n19 $HIPCC -O3 -std=c++17 --offload-arch=$ARCH -fuse-cuid=none -ffp-contract=off -save-temps "$SRC/ceil_sweep.hip" -o "ceil_sweep_$ARCH"
CO=ceil_sweep.hip-hip-amdgcn-amd-amdhsa.hipfb.amdgcn.$ARCH.img.lto.o
/opt/rocm/llvm/bin/llvm-objdump -d --mcpu=$ARCH "$CO" > "ceil_sweep_$ARCH.disasm.txt"
grep -c "v_ceil_f32" "ceil_sweep_$ARCH.disasm.txt" | sed 's/^/v_ceil_f32 instructions in disasm: /'
sha256sum "ceil_sweep_$ARCH"
