#!/bin/bash
# F1Lite standalone window on hipx (no compile, no model load).
# Usage: run_standalone.sh xtx|halo   (from the hipx scratch dir)
set -uo pipefail
cd "$(dirname "$0")"
card=$1
JIT=/home/kaden/hipfire-prof040/scratch-fusion/bin/jit
case $card in
  xtx) arch=gfx1100 stem=v2c rocr=0 ns="4096 8192" on=4096 ;;
  halo) arch=gfx1151 stem=v2b rocr=1 ns="8192" on=8192 ;;
  *) exit 2 ;;
esac
export EXPECTED_ARCH=$arch ROCR_VISIBLE_DEVICES=$rocr HIP_VISIBLE_DEVICES=0 GEMM_STEM=$stem
export GEMM_BASE_OBJECT=bin/base_$stem-$arch.hsaco GEMM_F1_OBJECT=bin/$stem-$arch.hsaco
export PROD_SILU_OBJECT=bin/base_silu_i4-$arch.hsaco PROD_HIN_OBJECT=bin/silu_hin-$arch.hsaco
export PROD_SILU_JIT_OBJECT=$JIT/silu_prod_jit-$arch.hsaco
mkdir -p logs
sha256sum -c bin/SHA256SUMS --quiet && echo "SHA256SUMS OK"
rocm-smi --showpids > logs/$card-pre.txt 2>&1
{ date -u; rocm-smi --showclocks 2>&1 | grep -E "GPU\[$rocr\]"; } > logs/$card-clocks-pre.txt
ORACLE_N=$on ./bin/f1-host oracle > logs/$card-oracle.log 2>&1
echo "oracle rc=$? $(tail -1 logs/$card-oracle.log)"
for N in $ns; do
  i=0
  for d in fwd rev rev fwd; do
    i=$((i + 1))
    ./bin/f1-host time $d $N > logs/$card-time-N$N-$i-$d.log 2>&1 || echo "time $N $i rc=$?"
  done
done
{ date -u; rocm-smi --showclocks 2>&1 | grep -E "GPU\[$rocr\]"; } > logs/$card-clocks-post.txt
rocm-smi --showpids > logs/$card-post.txt 2>&1
grep -h ORACLE logs/$card-oracle.log | grep -v "differing=0" ; grep -h COVERAGE logs/$card-oracle.log
grep -h RESULT logs/$card-time-*.log | sed 's/ begin=.*//'
