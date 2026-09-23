#!/bin/bash
# V2CAddEpi gfx1100 proof on the stack-1 port (one gfx1100 lease): pp8192
# rocprof A (HIPFIRE_V2C_ADDEPI=0) and B in the same binaries; oracle of the
# runtime JIT V2C object (trace-B cache, `_add_touch`) vs a hipcc stack-1
# reference; WT2 c24 q8/q8 off vs on (byte identity).
# usage: proof.sh [trace|jit|wt2]...
set -uo pipefail
d=/home/kaden/hipfire-v2caddepi/scratch-v2caddepi
cd $d
for ph in "$@"; do
  case $ph in
    jit)
      # v2c_prod = hipcc V2C TU from stack-1 1fcb53d4d (reference `_add`/`_set`);
      # v2c_touchjit = the daemon's JIT object, its `_add_touch` compared to it.
      src=$d/trace/gfx1100-8192-B/home/.hipfire_kernels/gfx1100; j=$d/binjit
      rm -rf $j && mkdir -p $j && cp bin/v2c_prod_gfx1100.hsaco bin/norm_*_gfx1100.hsaco bin/v2c_l64d16_gfx1100.hsaco $j/
      cp $(ls $src/gemm_mq4g256v2_residual_iu4_v2c_gfx11.*.hsaco) $j/v2c_touchjit_gfx1100.hsaco
      ADDEPI_BIN=$j ADDEPI_VARS=touchjit,l64d16 ./run_standalone.sh J1 oracle F R ;;
    trace)
      python3 run_profile.py gfx1100 8192 A && python3 run_profile.py gfx1100 8192 B && python3 inventory.py && python3 trace_delta.py ;;
    wt2)
      python3 run_wt2.py gfx1100 ;;
  esac
  echo "proof phase $ph rc=$?"
done
rocm-smi --showpids
