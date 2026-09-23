#!/bin/bash
# Oracle reference objects from the PRE-LEVER base sources (default 8397738d4 = stack-1):
# bin/v2b_ref_<arch>.hsaco (V2B TU, `_add`) and bin/norm_ref_<arch>.hsaco (the
# kernels.rs AWQ-i4 producer concatenation). usage: build_ref.sh <arch> [base]
set -e
# Norm objects take the runtime's gfx11 define -DIU4_A4_CANDIDATES=2 (feature_flags.rs);
# without it block_i4_128_quant.hip defaults to 8 candidates and its bytes differ.
cd "$(dirname "$0")"; arch=$1; base=${2:-8397738d4}
t=$(mktemp -d)
for f in gemm_mq4g256v2_residual_iu4_v2b.gfx11.hip block_i4_128_quant.hip fused_rmsnorm_mq_rotate.hip; do
  git show $base:kernels/src/$f > $t/$f
done
/opt/rocm/bin/hipcc --genco --offload-arch=$arch -O3 --no-offload-compress $t/gemm_mq4g256v2_residual_iu4_v2b.gfx11.hip -o bin/v2b_ref_$arch.hsaco
{ echo "#define HIPFIRE_BLOCK_I4_128_QUANT_NO_STANDALONE 1"; cat $t/block_i4_128_quant.hip
  echo "#define HIPFIRE_IU4_SIDECAR 1"; echo "#define HIPFIRE_RMSNORM_AWQ 1"
  echo "#define HIPFIRE_RMSNORM_KERNEL fused_rmsnorm_mq_rotate_awq_i4"
  cat $t/fused_rmsnorm_mq_rotate.hip; } > $t/norm_ref.hip
/opt/rocm/bin/hipcc --genco --offload-arch=$arch -O3 --no-offload-compress -DIU4_A4_CANDIDATES=2 $t/norm_ref.hip -o bin/norm_ref_$arch.hsaco
rm -rf $t
ls -la bin/v2b_ref_$arch.hsaco bin/norm_ref_$arch.hsaco
