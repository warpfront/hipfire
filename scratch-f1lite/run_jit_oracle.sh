#!/bin/bash
# F1Lite oracle over the objects hipfire itself JIT-compiled in the pp8192
# traces (radiowave; the h-producer's JIT schedule differs from the local
# hipcc build): C = JIT gate_up_silu + JIT h-producer, A = base SET pair + the
# local production producer, cross-checked against the JIT production producer
# of the HIPFIRE_F1LITE=0 trace. No compile, no model load.
# Usage: run_jit_oracle.sh xtx|halo   (from the hipx scratch dir)
set -uo pipefail
cd "$(dirname "$0")"
card=$1
case $card in
  xtx) arch=gfx1100 stem=v2c rocr=0 on=4096 ;;
  halo) arch=gfx1151 stem=v2b rocr=1 on=8192 ;;
  *) exit 2 ;;
esac
on_dir=trace/$arch-8192/home/.hipfire_kernels/$arch
off_dir=trace/$arch-8192-off/home/.hipfire_kernels/$arch
export EXPECTED_ARCH=$arch ROCR_VISIBLE_DEVICES=$rocr HIP_VISIBLE_DEVICES=0 GEMM_STEM=$stem
export GEMM_BASE_OBJECT=bin/base_$stem-$arch.hsaco
export GEMM_F1_OBJECT=$(ls $on_dir/gemm_mq4g256v2_residual_iu4_${stem}_gfx11.*.hsaco)
export PROD_SILU_OBJECT=bin/base_silu_i4-$arch.hsaco
export PROD_HIN_OBJECT=$(ls $on_dir/fused_silu_mul_mq_rotate_awq_i4_hin.*.hsaco)
export PROD_SILU_JIT_OBJECT=$(ls $off_dir/fused_silu_mul_mq_rotate_awq_i4.*.hsaco)
mkdir -p logs
sha256sum $GEMM_F1_OBJECT $PROD_HIN_OBJECT $PROD_SILU_JIT_OBJECT > logs/$card-jit-objects.sha256
rocm-smi --showpids > logs/$card-jit-pre.txt 2>&1
ORACLE_N=$on ./bin/f1-host oracle > logs/$card-jit-oracle.log 2>&1
echo "jit oracle rc=$? $(tail -1 logs/$card-jit-oracle.log)"
rocm-smi --showpids > logs/$card-jit-post.txt 2>&1
grep -h ORACLE logs/$card-jit-oracle.log | grep -v "differing=0"
