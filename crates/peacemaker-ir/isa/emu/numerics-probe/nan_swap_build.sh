#!/bin/bash
# nan_swap_build.sh [OUTDIR]
# CPU-only build of nan_swap.hip for gfx1201 (no GPU is touched: every device is hidden). Based on build.sh, but
# the opcode proof is specific to this probe: literal presence checks for the six target opcodes (native VOP2
# v_add_f32 / v_mul_f32 / v_fmac_f32 and the dual forms v_dual_add_f32 / v_dual_mul_f32 / v_dual_fmac_f32, with BOTH
# the X and Y half inspected in the exact pair the asm emits). Fails (nonzero) if any check finds nothing.
# Requires a private TMPDIR (not under /tmp). Run it through the build slot; this script never starts a GPU job.
# Outputs in OUTDIR: nan_swap_gfx1201 (binary), host + device objects from -save-temps, nan_swap_gfx1201.disasm.txt,
# nan_swap_gfx1201.opcodes.txt, build.json (flags + source/binary/host-obj/device-obj/disasm sha256).
set -euo pipefail
ARCH=gfx1201
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NP_SRC=nan_swap.hip
BIN=nan_swap_$ARCH
OUT=${1:-$PWD/nan_swap_build_$ARCH}
HIPCC=${HIPCC:-/opt/rocm/core/bin/hipcc}
LLVM=${LLVM:-/opt/rocm/lib/llvm/bin}
: "${TMPDIR:?TMPDIR must be set to a private directory (not /tmp)}"
case "$TMPDIR" in /tmp|/tmp/*) echo "TMPDIR must not be under /tmp" >&2; exit 2 ;; esac
mkdir -p "$TMPDIR" "$OUT"
cd "$OUT"
export HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1
FLAGS="-O2 -std=c++17 --offload-arch=$ARCH -DNP_GFX1201 -ffp-contract=off -save-temps"
nice -n19 "$HIPCC" $FLAGS "$SRC/$NP_SRC" -o "$BIN"
CO=$(ls $NP_SRC-hip-amdgcn-amd-amdhsa.hipfb.amdgcn.$ARCH.img.lto.o | sed -n 1p)
HO=$(ls ${NP_SRC%.hip}-host-*.o 2>/dev/null | sed -n 1p || true)
[ -n "$CO" ] || { echo "device object not found" >&2; exit 1; }
"$LLVM/llvm-objdump" -d --mcpu=$ARCH "$CO" > "$BIN.disasm.txt"
D="$BIN.disasm.txt"
fail=0
count() { grep -cE "$1" "$D" || true; }
check() {  # name regex
  local n; n=$(count "$2")
  printf '%-44s %s\n' "$1" "$n" | tee -a "$BIN.opcodes.txt.tmp"
  if [ "$n" -eq 0 ]; then echo "MISSING: $1" >&2; fail=1; fi
}
: > "$BIN.opcodes.txt.tmp"
# native VOP2 (objdump prints the e32 suffix)
check "v_add_f32 (native)"                    '^\s+v_add_f32(_e32)?\s'
check "v_mul_f32 (native)"                    '^\s+v_mul_f32(_e32)?\s'
check "v_fmac_f32 (native)"                   '^\s+v_fmac_f32(_e32)?\s'
# dual forms, X half and Y half literally present in the emitted pair
check "v_dual_add_f32 X :: v_dual_add_f32 Y"   '^\s+v_dual_add_f32 v4, v0, v2 :: v_dual_add_f32 v5, v1, v3'
check "v_dual_mul_f32 X :: v_dual_mul_f32 Y"   '^\s+v_dual_mul_f32 v4, v0, v2 :: v_dual_mul_f32 v5, v1, v3'
check "v_dual_fmac_f32 X :: v_dual_fmac_f32 Y" '^\s+v_dual_fmac_f32 v4, v0, v2 :: v_dual_fmac_f32 v5, v1, v3'
# literal substring checks for each half (X = first, Y = second operand group)
for op in add mul fmac; do
  nx=$(grep -cF "v_dual_${op}_f32 v4, v0, v2 ::" "$D" || true)
  ny=$(grep -cF ":: v_dual_${op}_f32 v5, v1, v3" "$D" || true)
  printf '%-44s X=%s Y=%s\n' "v_dual_${op}_f32 halves" "$nx" "$ny" | tee -a "$BIN.opcodes.txt.tmp"
  if [ "$nx" -eq 0 ] || [ "$ny" -eq 0 ]; then echo "MISSING dual half for $op" >&2; fail=1; fi
done
printf '%-44s %s\n' "s_setreg_imm32_b32 (MODE writes)" "$(count '^\s+s_setreg_imm32_b32')" | tee -a "$BIN.opcodes.txt.tmp"
printf '%-44s %s\n' "s_getreg_b32 (MODE readback)" "$(count '^\s+s_getreg_b32')" | tee -a "$BIN.opcodes.txt.tmp"
mv "$BIN.opcodes.txt.tmp" "$BIN.opcodes.txt"
h() { if [ -n "${1:-}" ] && [ -f "$1" ]; then sha256sum "$1" | cut -d' ' -f1; else echo ""; fi; }
printf '{"arch":"%s","hipcc":"%s","flags":"%s","tmpdir":"%s","source_sha256":"%s","binary_sha256":"%s","host_obj":"%s","host_obj_sha256":"%s","device_obj":"%s","device_obj_sha256":"%s","disasm_sha256":"%s","opcode_checks_ok":%s}\n' \
  "$ARCH" "$HIPCC" "$FLAGS" "$TMPDIR" "$(h "$SRC/$NP_SRC")" "$(h "$BIN")" "${HO:-}" "$(h "${HO:-}")" "$CO" "$(h "$CO")" "$(h "$D")" \
  "$([ $fail -eq 0 ] && echo true || echo false)" > build.json
cat build.json
echo "build products in $OUT"
exit $fail
