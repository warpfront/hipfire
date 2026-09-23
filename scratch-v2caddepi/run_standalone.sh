#!/bin/bash
# V2CAddEpi standalone on hipx gfx1100. usage: run_standalone.sh <tag> <phase...>
# phases: oracle F R. Objects/host in ./bin (built locally, same hipcc 7.15.26333).
set -uo pipefail
d=$(cd "$(dirname "$0")" && pwd)
tag=$1; shift
export ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 EXPECTED_ARCH=gfx1100
export ADDEPI_BIN=${ADDEPI_BIN:-$d/bin} ADDEPI_VARS=${ADDEPI_VARS:-base,l64d16,l64d8,l64d4,l64d32,l128d16,l128d8,l64d16p4,start}
mkdir -p $d/logs
sha256sum $ADDEPI_BIN/*_gfx1100.hsaco $d/bin/v2c-host > $d/logs/$tag-objects.sha256
rocm-smi --showpids > $d/logs/$tag-smi-pre.txt 2>&1
for ph in "$@"; do
  case $ph in
    oracle) $d/bin/v2c-host oracle > $d/logs/$tag-oracle.log 2>&1 ;;
    F) $d/bin/v2c-host time fwd > $d/logs/$tag-F.log 2>&1 ;;
    R) $d/bin/v2c-host time rev > $d/logs/$tag-R.log 2>&1 ;;
  esac
  echo "phase $ph rc=$?"
done
rocm-smi --showpids > $d/logs/$tag-smi-post.txt 2>&1
