#!/bin/bash
# AddEpilogue standalone on hipx. usage: run_standalone.sh <tag> <gfx1151|gfx1100> <phase...>
# phases: oracle F R. Objects/host in ./bin (built locally, committed-tree sources).
set -uo pipefail
d=$(cd "$(dirname "$0")" && pwd)
tag=$1 arch=$2; shift 2
if [ $arch = gfx1151 ]; then export ROCR_VISIBLE_DEVICES=1 HIP_VISIBLE_DEVICES=0
else export ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0; fi
export EXPECTED_ARCH=$arch ADDEPI_BIN=${ADDEPI_BIN:-$d/bin} ADDEPI_VARS=${ADDEPI_VARS:-pf4,pf8,pf16l64,pfs}  # ADDEPI_PROD=1 ADDEPI_VARS=prod for the committed kernels
mkdir -p $d/logs
sha256sum $ADDEPI_BIN/*_$arch.hsaco $d/bin/addepi-host > $d/logs/$tag-objects.sha256
rocm-smi --showpids > $d/logs/$tag-smi-pre.txt 2>&1
for ph in "$@"; do
  case $ph in
    oracle) $d/bin/addepi-host oracle > $d/logs/$tag-oracle.log 2>&1 ;;
    F) $d/bin/addepi-host time fwd > $d/logs/$tag-F.log 2>&1 ;;
    R) $d/bin/addepi-host time rev > $d/logs/$tag-R.log 2>&1 ;;
  esac
  echo "phase $ph rc=$?"
done
rocm-smi --showpids > $d/logs/$tag-smi-post.txt 2>&1
