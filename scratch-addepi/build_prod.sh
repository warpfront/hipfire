#!/bin/bash
# Production objects from the committed tree: bin/v2b_prod_<arch>.hsaco and
# bin/norm_{base,fold}_prod_<arch>.hsaco (the kernels.rs AWQ-i4 concatenations).
set -e
# Norm objects take the runtime's gfx11 define -DIU4_A4_CANDIDATES=2 (feature_flags.rs);
# without it block_i4_128_quant.hip defaults to 8 candidates and its bytes differ.
cd "$(dirname "$0")"; arch=$1; K=../kernels/src
/opt/rocm/bin/hipcc --genco --offload-arch=$arch -O3 --no-offload-compress $K/gemm_mq4g256v2_residual_iu4_v2b.gfx11.hip -o bin/v2b_prod_$arch.hsaco
for v in base fold; do
  { echo "#define HIPFIRE_BLOCK_I4_128_QUANT_NO_STANDALONE 1"; cat $K/block_i4_128_quant.hip
    echo "#define HIPFIRE_IU4_SIDECAR 1"; echo "#define HIPFIRE_RMSNORM_AWQ 1"
    if [ $v = fold ]; then echo "#define HIPFIRE_RMSNORM_FOLD 1"; echo "#define HIPFIRE_RMSNORM_KERNEL fused_rmsnorm_mq_rotate_awq_i4_fold"
    else echo "#define HIPFIRE_RMSNORM_KERNEL fused_rmsnorm_mq_rotate_awq_i4"; fi
    cat $K/fused_rmsnorm_mq_rotate.hip; } > /tmp/addepi_prod_$v.hip
  /opt/rocm/bin/hipcc --genco --offload-arch=$arch -O3 --no-offload-compress -DIU4_A4_CANDIDATES=2 /tmp/addepi_prod_$v.hip -o bin/norm_${v}_prod_$arch.hsaco
done
