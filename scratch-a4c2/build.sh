#!/bin/bash
# build.sh <tree> <outdir> <arch> <CONST> [extra hipcc flags...]
# Expands kernels.rs concat!(CONST) from <tree> exactly (gen_prod.py) and
# compiles it with the production compiler.rs argv:
#   hipcc --genco --offload-arch=<arch> -O3 --no-offload-compress
#         <rocm root flags> -I/opt/rocm/include <extra flags> -o out src
# Writes <outdir>/<CONST>.{hip,hsaco,s} and prints VGPR/SGPR/scratch.
set -e
tree=$1; out=$2; arch=$3; name=$4; shift 4
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"
python3 "$here/gen_prod.py" "$tree" "$out" "$name" >/dev/null
hipcc --genco --offload-arch="$arch" -O3 --no-offload-compress \
  --rocm-path=/opt/rocm --hip-path=/opt/rocm -I/opt/rocm/include "$@" \
  -Rpass-analysis=kernel-resource-usage -o "$out/$name.hsaco" "$out/$name.hip" 2>&1 \
  | grep -E 'error|VGPRs:|SGPRs:|ScratchSize|Occupancy' | sed "s|^.*remark: *||" | tr '\n' ' '
echo
/opt/rocm/llvm/bin/clang-offload-bundler --type=o --unbundle --input="$out/$name.hsaco" \
  --output="$out/$name.elf" --targets=hipv4-amdgcn-amd-amdhsa--"$arch" 2>/dev/null
/opt/rocm/llvm/bin/llvm-objdump -d --mcpu="$arch" "$out/$name.elf" > "$out/$name.s"
