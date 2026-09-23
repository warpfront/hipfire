#!/bin/bash
# usage: build_norm.sh <arch> -> bin/norm_base_<arch>.hsaco (this tree's production
# AWQ-i4 producer) and bin/norm_fold_<arch>.hsaco (AddEpilogue's production fold
# variant, fused_rmsnorm_mq_rotate.hip @ gfx11-addepi 726b3410f).
set -e
cd "$(dirname "$0")"; arch=$1; K=../kernels/src
git -C .. show 726b3410f:kernels/src/fused_rmsnorm_mq_rotate.hip > /tmp/v2cae_norm_fold_src.hip
for v in base fold; do
  { echo "#define HIPFIRE_BLOCK_I4_128_QUANT_NO_STANDALONE 1"; cat $K/block_i4_128_quant.hip
    echo "#define HIPFIRE_IU4_SIDECAR 1"; echo "#define HIPFIRE_RMSNORM_AWQ 1"
    if [ $v = fold ]; then echo "#define HIPFIRE_RMSNORM_FOLD 1"; echo "#define HIPFIRE_RMSNORM_KERNEL fused_rmsnorm_mq_rotate_awq_i4_fold"; cat /tmp/v2cae_norm_fold_src.hip
    else echo "#define HIPFIRE_RMSNORM_KERNEL fused_rmsnorm_mq_rotate_awq_i4"; cat $K/fused_rmsnorm_mq_rotate.hip; fi; } > /tmp/v2cae_norm_$v.hip
  /opt/rocm/bin/hipcc --genco --offload-arch=$arch -O3 --no-offload-compress /tmp/v2cae_norm_$v.hip -o bin/norm_${v}_$arch.hsaco
done
