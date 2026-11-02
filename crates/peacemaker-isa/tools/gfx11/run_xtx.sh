#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# gfx1100 (XTX) runs for the builder V2C-equivalent SET GEMM.
#   run_xtx.sh <dir> oracle|time|profile
# <dir> holds bin/{v2c_runtime.hsaco,v2c_pm.hsaco,v2c_pm_prof.co,v2c_bench,pmprof},
# data/{w_gate_b31.bin,xq_b31_n8192.bin,y_a4_b31.bin}; logs go to <dir>/logs.
# Device selection: the XTX by ROCr UUID only (HIP_VISIBLE_DEVICES unset);
# the harnesses assert one device, gfx1100, PCI bus 66.
set -euo pipefail
d=$1; phase=$2
unset HIP_VISIBLE_DEVICES
export ROCR_VISIBLE_DEVICES=GPU-43390a851e296ee5
mkdir -p "$d/logs"
cd "$d"
sha256sum bin/* data/* > "logs/$phase-inputs.sha256"
case $phase in
  oracle)  bin/v2c_bench oracle bin/v2c_runtime.hsaco bin/v2c_pm.hsaco data/w_gate_b31.bin data/xq_b31_n8192.bin data/y_a4_b31.bin 32 2>&1 | tee logs/oracle.log ;;
  time)    bin/v2c_bench time bin/v2c_runtime.hsaco bin/v2c_pm.hsaco data/w_gate_b31.bin data/xq_b31_n8192.bin 10 ABBABAAB 2>&1 | tee logs/time.log ;;
  profile) PM_V2C_W=data/w_gate_b31.bin PM_V2C_XQ=data/xq_b31_n8192.bin PM_RECORDS_PER_PERIOD=12 \
             bin/pmprof v2c bin/v2c_pm.hsaco bin/v2c_pm_prof.co 8192 prof 8 2>&1 | tee logs/profile.log ;;
  *) echo "unknown phase $phase"; exit 2 ;;
esac
