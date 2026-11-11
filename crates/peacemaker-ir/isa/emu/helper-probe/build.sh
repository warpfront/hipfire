#!/bin/bash
# build.sh [OUTDIR]
# Device-only code objects of helper_probe.hip for gfx1201 and gfx1200 (no GPU is touched: every device is hidden),
# the disassembly of each, and the committed fixtures
#   crates/peacemaker-ir/tests/fixtures/helper_probe.gfx1201.hsaco, helper_probe.gfx1200.hsaco
# (copied only when COMMIT=1). Flags are the runtime's `core_hipcc_flags` (rdna-compute/src/compiler.rs), the flags of
# the committed lane_leviathan_hipcc_*.co anchors.
# Toolchain of the committed fixtures (recorded, not required; the committed objects are the fixtures):
#   HIP version: 7.16.26385-0000000
#   AMD clang version 24.0.0git (https://github.com/ROCm/llvm-project.git d6f6cb691863e9258034ab48c1dfce21689d5606)
# Outputs: OUTDIR/helper_probe.<arch>.hsaco, OUTDIR/helper_probe.<arch>.dis, OUTDIR/build.json.
set -euo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SRC/../../../../.." && pwd)"
OUT=${1:-$PWD/helper_probe_build}
HIPCC=${HIPCC:-/opt/rocm/bin/hipcc}
LLVM=${LLVM:-/opt/rocm/lib/llvm/bin}
mkdir -p "$OUT"
export HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1
FLAGS="--genco -O3 --no-offload-compress -fuse-cuid=none -mcode-object-version=6"
printf '{"hipcc":"%s","hipcc_version":%s,"flags":"%s","source_sha256":"%s","objects":{' "$HIPCC" \
  "$("$HIPCC" --version | head -2 | tr '\n' ' ' | sed 's/"/\\"/g; s/^/"/; s/ $/"/')" "$FLAGS" \
  "$(sha256sum "$SRC/helper_probe.hip" | cut -d' ' -f1)" > "$OUT/build.json"
sep=""
for ARCH in gfx1201 gfx1200; do
  CO="$OUT/helper_probe.$ARCH.hsaco"
  nice -n19 "$HIPCC" $FLAGS "--offload-arch=$ARCH" "$SRC/helper_probe.hip" -o "$CO"
  # the bundle holds the device image; extract it for the disassembly
  "$LLVM/clang-offload-bundler" --unbundle --type=o --input="$CO" --output="$OUT/helper_probe.$ARCH.o" \
    --targets="hipv4-amdgcn-amd-amdhsa--$ARCH" --allow-missing-bundles 2>/dev/null || true
  if [ -s "$OUT/helper_probe.$ARCH.o" ]; then
    "$LLVM/llvm-objdump" -d --mcpu="$ARCH" "$OUT/helper_probe.$ARCH.o" > "$OUT/helper_probe.$ARCH.dis"
  else
    "$LLVM/llvm-objdump" -d --mcpu="$ARCH" "$CO" > "$OUT/helper_probe.$ARCH.dis"
  fi
  printf '%s"%s":"%s"' "$sep" "$ARCH" "$(sha256sum "$CO" | cut -d' ' -f1)" >> "$OUT/build.json"
  sep=","
  if [ "${COMMIT:-0}" = 1 ]; then cp "$CO" "$ROOT/crates/peacemaker-ir/tests/fixtures/helper_probe.$ARCH.hsaco"; fi
done
printf '}}\n' >> "$OUT/build.json"
echo "build products in $OUT"
