#!/bin/bash
# Standalone V2C in-kernel-permute (prod layout) vs native V2C-ms vs X5 on gfx1100.
# usage: run_standalone.sh <tag> <phase...>   phases: oracle F R
set -euo pipefail
d=/home/kaden/hipfire-gemmv2/scratch-gemmv2integ
g=/home/kaden/hipfire-prof040/scratch-gemmv2/bin
tag=$1; shift
export ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 EXPECTED_ARCH=gfx1100 SKIP_V2AB=1
export X5_OBJECT=/home/kaden/hipfire-x5/scratch-x5/gfx1100-8192/home/.hipfire_kernels/gfx1100/gemm_mq4g256v2_residual_mmq_iu4_gfx11_x5_symfold.936038d287cc3b6f.hsaco
export V2B_OBJECT=$g/v2b-gfx1100.hsaco
export EXTRA_V2C_MS_OBJECT=$g/v2c-ms-gfx1100.hsaco
export EXTRA_ARMS=v2c_ms=EXTRA_V2C_MS_OBJECT:v2c
export V2CP_OBJECT=${V2CP_OBJECT:-$d/bin/v2cp-gfx1100.hsaco}
mkdir -p $d/logs
sha256sum $X5_OBJECT $EXTRA_V2C_MS_OBJECT $V2CP_OBJECT $d/bin/gemmv2integ-host > $d/logs/$tag-objects.sha256
rocm-smi --showpids > $d/logs/$tag-smi-pre.txt 2>&1 || true
for ph in "$@"; do
  case $ph in
    oracle) $d/bin/gemmv2integ-host oracle > $d/logs/$tag-oracle.log 2>&1 ;;
    F) $d/bin/gemmv2integ-host time fwd > $d/logs/$tag-F.log 2>&1 ;;
    R) $d/bin/gemmv2integ-host time rev > $d/logs/$tag-R.log 2>&1 ;;
  esac
  echo "phase $ph rc=$?"
done
rocm-smi --showpids > $d/logs/$tag-smi-post.txt 2>&1 || true
