#!/usr/bin/env bash
# Build TUs with the runtime hipcc flags (hipcc --genco --offload-arch=gfx1201 -O3
# --no-offload-compress) as the runtime concatenation: defines prelude, then
# kernels/src/block_i4_128_quant.hip, then the kernel body.
# Usage: build.sh <name> [-DFOO=1 ...]
#   name=control* body = base_kernel.hip (4bd33ce40 production kernel, unmodified);
#                 CONTROL_BODY=path overrides (e.g. k865.hip = 865fd3d03 kernel)
#   IU4_BODY=path overrides the body (default iu4_var.hip, the scratch screen body)
#   SYM=0         omit IU4_SYMMETRIC_FOLD (asymmetric route); entry names stay *_symfold labels
#   G12R=1        add '#define IU4_G12_RASTER 1' (production lever macro)
#   ISA=1         also emit the device assembly hsaco/<name>.s
set -euo pipefail
cd "$(dirname "$0")"
ROOT=..
name=$1; shift
mkdir -p hsaco src
prelude=''
[[ ${SYM:-1} == 1 ]] && prelude+=$'#define IU4_SYMMETRIC_FOLD 1\n'
[[ ${G12R:-0} == 1 ]] && prelude+=$'#define IU4_G12_RASTER 1\n'
prelude+='#define gemm_mq4g256v2_residual_mmq_iu4 gemm_mq4g256v2_residual_mmq_iu4_symfold
#define gemm_mq4g256v2_residual_mmq_iu4_full_add gemm_mq4g256v2_residual_mmq_iu4_full_add_symfold
#define gemm_mq4g256v2_residual_mmq_iu4_full_set gemm_mq4g256v2_residual_mmq_iu4_full_set_symfold
#define gemm_mq4g256v2_gate_up_silu_mmq_iu4 gemm_mq4g256v2_gate_up_silu_mmq_iu4_symfold'
if [[ $name == control* ]]; then
  body=${CONTROL_BODY:-base_kernel.hip}
else
  body=${IU4_BODY:-iu4_var.hip}
fi
{ printf '%s\n' "$prelude"; cat $ROOT/kernels/src/block_i4_128_quant.hip "$body"; } > src/$name.hip
hipcc --genco --offload-arch=gfx1201 -O3 --no-offload-compress "$@" -o hsaco/$name.hsaco src/$name.hip
if [[ ${ISA:-0} == 1 ]]; then
  hipcc -S --cuda-device-only --offload-arch=gfx1201 -O3 "$@" -o hsaco/$name.s src/$name.hip
fi
echo "built $name $*"
