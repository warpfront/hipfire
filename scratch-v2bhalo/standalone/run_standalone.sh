#!/bin/bash
# Standalone V2B production layout (in-kernel permute) vs native V2B-ms vs X5
# on Halo gfx1151 (HIP 1). usage: run_standalone.sh <tag> <phase...>
# phases: resources oracle F R   (extra args after -- are shape filters)
set -euo pipefail
d=/home/kaden/hipfire-v2bhalo/scratch-v2bhalo
g=/home/kaden/hipfire-prof040/scratch-gemmv2/bin
tag=$1; shift
export ROCR_VISIBLE_DEVICES=1 HIP_VISIBLE_DEVICES=0 EXPECTED_ARCH=gfx1151 SKIP_V2AB=1
# X5 control: the object hipfire mq4-lloyd (land-042 code) JIT-compiled on Halo.
export X5_OBJECT=${X5_OBJECT:-/home/kaden/hipfire-land042/scratch-land042/gfx1151/abba/06-B-home/.hipfire_kernels/gfx1151/gemm_mq4g256v2_residual_mmq_iu4_gfx11_x5_symfold.c4e902d85a7eb0e1.hsaco}
export V2B_OBJECT=$g/v2b-ms-gfx1151.hsaco   # native repack kernel source only
export V2B_MS_OBJECT=$g/v2b-ms-gfx1151.hsaco
export V2BP_OBJECT=${V2BP_OBJECT:-$d/bin/v2bp-gfx1151.hsaco}
mkdir -p $d/logs
sha256sum $X5_OBJECT $V2B_MS_OBJECT $V2BP_OBJECT $d/bin/v2bhalo-host > $d/logs/$tag-objects.sha256
rocm-smi --showpids > $d/logs/$tag-smi-pre.txt 2>&1 || true
phases=(); shapes=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then shift; shapes=("$@"); break; fi
  phases+=("$1"); shift
done
for ph in "${phases[@]}"; do
  case $ph in
    resources) $d/bin/v2bhalo-host resources > $d/logs/$tag-resources.log 2>&1 ;;
    oracle) $d/bin/v2bhalo-host oracle > $d/logs/$tag-oracle.log 2>&1 ;;
    F) $d/bin/v2bhalo-host time fwd "${shapes[@]}" > $d/logs/$tag-F.log 2>&1 ;;
    R) $d/bin/v2bhalo-host time rev "${shapes[@]}" > $d/logs/$tag-R.log 2>&1 ;;
  esac
  echo "phase $ph rc=$?"
done
rocm-smi --showpids > $d/logs/$tag-smi-post.txt 2>&1 || true
