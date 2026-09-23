#!/usr/bin/env bash
# Build the FA2 screen: harness + one code object per (variant, arch).
# Flags mirror the runtime JIT (--genco -O3 --no-offload-compress).
set -euo pipefail
cd "$(dirname "$0")"
HIPCC=/opt/rocm/bin/hipcc
PROD=../kernels/src/attention_q8_0_fa2_gqa.gfx11.hip
mkdir -p out
$HIPCC -O2 -std=c++17 fa2_bench.cpp -o out/fa2_bench
declare -A V=(
  [v0]=""
  [v0u]="-DV_QK_UNROLL=16"
  [abl_noq]="-DV_ABL_NOQ=1"
  [abl_nofill]="-DV_ABL_NOFILL=1"
  [qres2]="-DV_QRES=2"
  [qres4]="-DV_QRES=4"
  [qresx4]="-DV_QRESX=4"
  [qresx8]="-DV_QRESX=8"
  [qpipe1]="-DV_QPIPE=1"
  [qpipe2]="-DV_QPIPE=2"
  [qpipe1w]="-DV_QPIPE=1 -DV_QWRAP=1"
  [vpipe]="-DV_VPIPE=1"
  [vpf]="-DV_VPF=1"
  [qp1vp]="-DV_QPIPE=1 -DV_VPIPE=1"
  [qp1vpf]="-DV_QPIPE=1 -DV_VPF=1"
  [qp1wvpf]="-DV_QPIPE=1 -DV_QWRAP=1 -DV_VPF=1"
  [qresx4qp1]="-DV_QRESX=4 -DV_QPIPE=1"
  [nohelp]="-DV_NOHELP=1"
  [nohelpqp1]="-DV_NOHELP=1 -DV_QPIPE=1"
  [nohelpqp1vp]="-DV_NOHELP=1 -DV_QPIPE=1 -DV_VPIPE=1"
  [nohelpcodeskv]="-DV_NOHELP=1 -DV_CODES=3"
  [codesk]="-DV_CODES=1"
  [codesv]="-DV_CODES=2"
  [codeskv]="-DV_CODES=3"
  [codeskvqp1]="-DV_CODES=3 -DV_QPIPE=1"
)
ONLY=${ONLY:-}
for arch in gfx1100 gfx1151; do
  if [[ -z "$ONLY" ]]; then
    $HIPCC --genco --offload-arch=$arch -O3 --no-offload-compress \
      -DHIPFIRE_FA2_KT=32 -DHIPFIRE_FA2_Q16=1 $PROD -o out/prod_$arch.hsaco
  fi
  for name in "${!V[@]}"; do
    [[ -n "$ONLY" && " $ONLY " != *" $name "* ]] && continue
    $HIPCC --genco --offload-arch=$arch -O3 --no-offload-compress ${V[$name]} \
      -Rpass-analysis=kernel-resource-usage fa2_var.hip -o out/${name}_$arch.hsaco 2> out/${name}_$arch.res &
  done
  wait
done
for arch in gfx1100 gfx1151; do
for f in out/*_$arch.res; do
  n=$(basename "$f" _$arch.res)
  printf '%-12s %-8s %s\n' "$n" "$arch" "$(grep -A12 'Function Name: attention_q8_0_fa2_gqa_gfx11' "$f" | grep -E 'VGPRs:|VGPRs Spill|ScratchSize|Occupancy' | sed -E 's/.*remark: +//; s/ \[-Rpass.*//' | tr '\n' ' ')"
  grep -m3 -i ' error' "$f" || true
done
done
