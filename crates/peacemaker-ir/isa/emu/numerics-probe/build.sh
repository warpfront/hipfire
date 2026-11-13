#!/bin/bash
# build.sh <gfx1201|gfx1151> [OUTDIR]   (NP_SRC=<file> builds another probe source of this directory, e.g. fp8_probe.hip)
# CPU-only build of numerics_probe.hip for one architecture (no GPU is touched: every device is hidden), plus a
# disassembly proof that the intended native opcodes are present. Outputs: OUTDIR/numerics_probe_<arch>,
# OUTDIR/numerics_probe_<arch>.disasm.txt, OUTDIR/numerics_probe_<arch>.opcodes.txt, OUTDIR/build.json.
set -euo pipefail
ARCH=${1:?arch gfx1201|gfx1151}
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NP_SRC=${NP_SRC:-numerics_probe.hip}
NP_BIN=${NP_SRC%.hip}
OUT=${2:-$PWD/numerics_build_$ARCH}
HIPCC=${HIPCC:-/opt/rocm/core/bin/hipcc}
LLVM=${LLVM:-/opt/rocm/lib/llvm/bin}
case "$ARCH" in
  gfx1201) DEFS="-DNP_GFX1201" ;;
  gfx1151) DEFS="-DNP_GFX1151" ;;
  *) echo "unsupported arch $ARCH" >&2; exit 2 ;;
esac
mkdir -p "$OUT"
cd "$OUT"
export HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1
FLAGS="-O2 -std=c++17 --offload-arch=$ARCH $DEFS -ffp-contract=off -save-temps"
nice -n19 $HIPCC $FLAGS "$SRC/$NP_SRC" -o "${NP_BIN}_$ARCH"
CO=$(ls $NP_SRC-hip-amdgcn-amd-amdhsa.hipfb.amdgcn.$ARCH.img.lto.o | head -1)
$LLVM/llvm-objdump -d --mcpu=$ARCH "$CO" > "${NP_BIN}_$ARCH.disasm.txt"
grep -oE '^\s+(v_dot[0-9a-z_]+|v_wmma_[0-9a-z_]+|v_swmmac_[0-9a-z_]+|v_cvt_pk_fp8_f32|global_load_tr_b64)' "${NP_BIN}_$ARCH.disasm.txt" | tr -d ' \t' | sort | uniq -c > "${NP_BIN}_$ARCH.opcodes.txt"
printf '{"arch":"%s","hipcc":"%s","flags":"%s","source_sha256":"%s","binary_sha256":"%s"}\n' "$ARCH" "$HIPCC" "$FLAGS" \
  "$(sha256sum "$SRC/$NP_SRC" | cut -d' ' -f1)" "$(sha256sum "${NP_BIN}_$ARCH" | cut -d' ' -f1)" > build.json
cat "${NP_BIN}_$ARCH.opcodes.txt"
echo "build products in $OUT"
