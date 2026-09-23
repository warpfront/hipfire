#!/bin/bash
# usage: build_gemm.sh <arch> name:flags ...  -> bin/v2c_<name>_<arch>.hsaco
set -e
cd "$(dirname "$0")"; arch=$1; shift
for v in "$@"; do n=${v%%:*}; f=${v#*:}
  /opt/rocm/bin/hipcc --genco --offload-arch=$arch -O3 --no-offload-compress $f standalone/v2c_var.hip -o bin/v2c_${n}_$arch.hsaco
  /opt/rocm/llvm/bin/clang-offload-bundler --type=o --unbundle --input=bin/v2c_${n}_$arch.hsaco --output=/tmp/v2cae_$n.co --targets=hipv4-amdgcn-amd-amdhsa--$arch
  printf "%-10s %s\n" $n "$(/opt/rocm/llvm/bin/llvm-readelf --notes /tmp/v2cae_$n.co | grep -E '^ +\.(name|vgpr_count|vgpr_spill_count|private_segment_fixed_size):' | awk '{print $2}' | tr '\n' ' ')"
done
