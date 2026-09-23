#!/bin/bash
# Same-binary rocprof pp8192 on card-E: new binary off/on/on/off (ABBA via
# HIPFIRE_G12_A4C2), then the base binary (0f6cea0dd, IXPlan's traced build)
# with the -DIU4_A4_CANDIDATES=2 flag and plain RTN for the flag reference.
set -e
cd "$(dirname "$0")"
run() { local label=$1 bin=$2; shift 2; echo "== $label"; python3 run_trace.py "$label" "$bin" E "$@"; }
run off1 bin/new HIPFIRE_G12_A4C2=0
run on1 bin/new
run on2 bin/new
run off2 bin/new HIPFIRE_G12_A4C2=0
run base-c2 bin/base HIPFIRE_HIPCC_EXTRA_FLAGS=-DIU4_A4_CANDIDATES=2
run base-rtn bin/base
