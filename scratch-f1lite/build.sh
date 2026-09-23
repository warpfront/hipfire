#!/bin/bash
# F1Lite scratch build (local only; never compile on hipx).
# Device objects use hipfire's JIT argv (--genco --offload-arch=<arch> -O3
# --no-offload-compress) plus the gfx11 feature flag -DIU4_A4_CANDIDATES=2, and
# the exact source concatenations of crates/rdna-compute/src/kernels.rs.
# "base" objects come from the unmodified base commit (900e52771) sources.
set -euo pipefail
cd "$(dirname "$0")"
ROOT=..
BASE=900e52771
HIPCC=/opt/rocm/bin/hipcc
BUNDLER=/opt/rocm/llvm/bin/clang-offload-bundler
READELF=/opt/rocm/llvm/bin/llvm-readelf
OBJDUMP=/opt/rocm/llvm/bin/llvm-objdump
mkdir -p bin gen
Q=$ROOT/kernels/src/block_i4_128_quant.hip
silu_src() {  # $1 = awq producer source, $2 = extra defines
  printf '#define HIPFIRE_BLOCK_I4_128_QUANT_NO_STANDALONE 1\n'; cat $Q
  printf '#define HIPFIRE_IU4_SIDECAR 1\n%s' "$2"; cat "$1"
}
git -C $ROOT show $BASE:kernels/src/fused_silu_mul_mq_rotate_awq.hip > gen/base_silu_awq.hip
git -C $ROOT show $BASE:kernels/src/gemm_mq4g256v2_residual_iu4_v2c.gfx11.hip > gen/base_v2c.hip
git -C $ROOT show $BASE:kernels/src/gemm_mq4g256v2_residual_iu4_v2b.gfx11.hip > gen/base_v2b.hip
silu_src gen/base_silu_awq.hip '#define HIPFIRE_SILU_MQ_ROTATE_KERNEL fused_silu_mul_mq_rotate_awq_i4
' > gen/base_silu_i4.hip
silu_src $ROOT/kernels/src/fused_silu_mul_mq_rotate_awq.hip '#define HIPFIRE_SILU_MQ_ROTATE_KERNEL fused_silu_mul_mq_rotate_awq_i4
' > gen/silu_i4.hip
silu_src $ROOT/kernels/src/fused_silu_mul_mq_rotate_awq.hip '#define HIPFIRE_SILU_HIN 1
#define HIPFIRE_SILU_MQ_ROTATE_KERNEL fused_silu_mul_mq_rotate_awq_i4_hin
' > gen/silu_hin.hip
cp $ROOT/kernels/src/gemm_mq4g256v2_residual_iu4_v2c.gfx11.hip gen/v2c.hip
cp $ROOT/kernels/src/gemm_mq4g256v2_residual_iu4_v2b.gfx11.hip gen/v2b.hip
OBJS="base_silu_i4 silu_i4 silu_hin base_v2c v2c base_v2b v2b"
for arch in gfx1100 gfx1151; do
  for name in $OBJS; do
    $HIPCC --genco --offload-arch=$arch -O3 --no-offload-compress -DIU4_A4_CANDIDATES=2 \
      -o bin/$name-$arch.hsaco gen/$name.hip
    $BUNDLER --unbundle --type=o --targets=hipv4-amdgcn-amd-amdhsa--$arch \
      --input=bin/$name-$arch.hsaco --output=bin/$name-$arch.elf
    echo "== $name $arch"
    $READELF --notes bin/$name-$arch.elf |
      grep -E '^ +\.name:|\.vgpr_count|\.sgpr_count|spill_count|private_segment_fixed|group_segment_fixed' |
      paste - - - - - - - - | sed 's/  */ /g'
  done
done
# Per-symbol disassembly identity of every pre-existing entry.
dis() { $OBJDUMP -d --no-show-raw-insn --no-leading-addr "bin/$1-$2.elf" |
  awk -v s="<$3>:" '$0 ~ s {p=1; next} p && /^$/ {p=0} p' |
  sed 's|//.*||; s/[[:space:]]*$//' |  # drop trailing inter-function padding
  tac | awk 'f || !/^[[:space:]]*(s_nop 0|s_code_end)$/ {f = 1; print}' | tac; }
for arch in gfx1100 gfx1151; do
  for pair in "base_silu_i4 silu_i4 fused_silu_mul_mq_rotate_awq_i4" \
              "base_v2c v2c gemm_mq4g256v2_residual_iu4_v2c_set_gfx11" \
              "base_v2c v2c gemm_mq4g256v2_residual_iu4_v2c_add_gfx11" \
              "base_v2b v2b gemm_mq4g256v2_residual_iu4_v2b_set_gfx11" \
              "base_v2b v2b gemm_mq4g256v2_residual_iu4_v2b_add_gfx11"; do
    set -- $pair
    a=$(dis $1 $arch $3 | sha256sum | cut -c1-16); b=$(dis $2 $arch $3 | sha256sum | cut -c1-16)
    n=$(dis $1 $arch $3 | wc -l)
    [ "$a" = "$b" ] && r=IDENTICAL || r=DIFFERENT
    echo "DISASM $arch $3 lines=$n base=$a new=$b $r"
  done
done
$HIPCC -O2 -o bin/f1-host f1-host.cpp
sha256sum gen/*.hip f1-host.cpp > bin/SOURCES.sha256
sha256sum bin/*.hsaco bin/f1-host > bin/SHA256SUMS
